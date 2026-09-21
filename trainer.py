"""统一训练循环：骨干/数据/优化器一致，只有每个方法的 loss 不同。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    HAS_AMP = False


def _autocast():
    if config.AMP_DTYPE == "bf16" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", dtype=torch.float16)


def softmax_loss(outputs, targets):
    return F.cross_entropy(outputs, targets)


def _fp32_head_ctx():
    """头与损失在 fp32 下计算（禁用 autocast）。evidential 头的 digamma/log/BCE
    在 bf16 autocast 下会发散成 NaN（rsnn 5 种子 loss=NaN 的根因）。"""
    return torch.autocast("cuda", enabled=False)


def _backbone_features(model, x):
    """骨干前向：CUDA 上走 bf16 autocast 省显存，特征 cast 回 fp32。"""
    if x.is_cuda:
        with torch.autocast("cuda"):
            feats = model.backbone(x)
        return feats.float()
    return model.backbone(x)


def hedl_step(model, x, y, device):
    model.train()
    features = _backbone_features(model, x)
    with _fp32_head_ctx():
        # 与原版一致：edl_HENN 下 features 过 ReLU
        outputs = model.head.forward(features)
        # num_classes 从头自适应（复数面板 n≠10），实数管线不变
        nc = int(getattr(model.head, "num_classes", 10))
        loss = torch.mean(model.head.hedl_ce(
            F.one_hot(y, num_classes=nc).float(), features, outputs, device))
    return loss


def reedl_step(model, x, y, device):
    model.train()
    features = model.backbone(x)
    logits = model.head.forward(features)
    evidence = model.head.compute_evidence(logits)
    from heads.reedl_head import ReEDLLoss
    nc = int(getattr(model.head, "num_classes", 10))
    loss = ReEDLLoss(num_classes=nc, lamb=model.head.lamb)(evidence, y)
    return loss


def rsnn_step(model, x, y, device):
    model.train()
    features = _backbone_features(model, x)
    with _fp32_head_ctx():
        belief = model.head.forward(features)
        loss = model.head.loss(belief, y)
    return loss


def softmax_step(model, x, y, device):
    model.train()
    logits = model(x)
    return softmax_loss(logits, y)


def iedl_step(model, x, y, device):
    model.train()
    features = _backbone_features(model, x)
    with _fp32_head_ctx():
        return model.head.loss(model.head.forward(features), y)


def duq_step(model, x, y, device):
    model.train()
    features = _backbone_features(model, x)
    with _fp32_head_ctx():
        return model.head.loss_from_features(features, y)


def mcdropout_step(model, x, y, device):
    # 与 softmax 训练完全一致（dropout 在 train 模式自动生效）
    model.train()
    logits = model(x)
    return F.cross_entropy(logits, y)


STEP_FN = {
    "softmax": softmax_step,
    "hedl": hedl_step,
    "reedl": reedl_step,
    "rsnn": rsnn_step,
    "iedl": iedl_step,
    "duq": duq_step,
    "mcdropout": mcdropout_step,
}


def train(model, method, train_loader, device, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    use_amp = HAS_AMP and str(device).startswith("cuda") and config.USE_AMP
    use_bf16 = use_amp and config.AMP_DTYPE == "bf16" and torch.cuda.is_bf16_supported()
    scaler = GradScaler() if (use_amp and not use_bf16) else None

    # 可选学习率调度：cosine（DiT 后期 loss 反弹/发散，cosine 衰减防发散）
    sched = None
    if getattr(config, "LR_SCHEDULE", None) == "cosine":
        total_epochs = max(1, config.EPOCHS)
        sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda ep: 0.5 * (1 + math.cos(math.pi * min(ep, total_epochs) / total_epochs)),
        )

    accum = max(1, int(getattr(config, "GRAD_ACCUM", 1)))
    step_fn = STEP_FN[method]
    model.train()
    for epoch in range(config.EPOCHS):
        if method == "iedl":
            # I-EDL 的 KL 正则前 1/10 epoch 线性爬升（与 Re-EDL 官方实现一致）
            model.head.cur_epoch = epoch
        running = 0.0
        n_opt = 0
        optimizer.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if use_bf16:
                with _autocast():
                    loss = step_fn(model, x, y, device)
                loss = loss / accum
                loss.backward()
            elif use_amp:
                with autocast():
                    loss = step_fn(model, x, y, device)
                scaler.scale(loss / accum).backward()
            else:
                loss = step_fn(model, x, y, device)
                (loss / accum).backward()
            running += loss.item() * accum
            if (i + 1) % accum == 0:
                if use_bf16:
                    optimizer.step()
                elif use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                n_opt += 1
        # 处理残余未累积的梯度
        if (i + 1) % accum != 0 and n_opt == 0:
            optimizer.step()
            optimizer.zero_grad()
        if sched is not None:
            sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"  seed {seed} epoch {epoch + 1}/{config.EPOCHS} loss {running / n_opt:.4f}",
                  flush=True)
    return model


@torch.no_grad()
def predict_uncertainty(model, method, loader, device):
    """返回 (preds, uncertainty)，uncertainty 越大越 OOD。

    GRAFT（嫁接算子）骨干含 flash attention 内核，要求 fp16/bf16 输入：
    骨干前向走 bf16 autocast，头与不确定度计算回到 fp32（与训练语义一致）。
    """
    from contextlib import nullcontext
    import config

    model.eval()
    uncs, preds = [], []
    grafted = bool(str(getattr(config, "GRAFT", "")).strip())

    def _bactx():
        return (torch.autocast("cuda", dtype=torch.bfloat16) if grafted
                else nullcontext())

    for x, _ in loader:
        x = x.to(device)
        with _bactx():
            feats = model.backbone(x)
        feats = feats.float()                       # 头与打分全程 fp32
        with _bactx():
            if method == "softmax":
                logits = model(x)
            elif method == "hedl":
                logits = None
            elif method == "reedl":
                logits = model.head.forward(feats)
            elif method == "iedl":
                logits = model.head.forward(feats)
            elif method == "duq":
                logits = None
            elif method == "mcdropout":
                logits = None
            elif method == "rsnn":
                logits = None
        if method == "softmax":
            logits = logits.float()
            prob = F.softmax(logits, dim=-1)
            unc = 1 - prob.max(dim=-1).values
            pred = prob.argmax(dim=-1)
        elif method == "hedl":
            unc = model.head.uncertainty(feats, device)
            sum_belief = model.head.henn_forward(feats, device)
            pred = sum_belief.argmax(dim=-1)
        elif method == "reedl":
            unc = model.head.uncertainty(feats)
            alpha, prob, _ = model.head.alpha_prob_unc(logits.float())
            pred = prob.argmax(dim=-1)
        elif method == "iedl":
            unc = model.head.uncertainty(feats)
            alpha, prob, _ = model.head.alpha_prob_unc(logits.float())
            pred = prob.argmax(dim=-1)
        elif method == "duq":
            prob = model.head.kernel_probs(model.head.normalize(feats))
            unc = 1 - prob.max(dim=-1).values
            pred = prob.argmax(dim=-1)
        elif method == "mcdropout":
            model.head.enable_mc_dropout()
            prob, unc = model.head.mc_predict(feats, t=getattr(config, "MCD_T", 10))
            pred = prob.argmax(dim=-1)
        elif method == "rsnn":
            unc = model.head.uncertainty(feats)
            betp = model.head.betp_from_features(feats)
            pred = betp.argmax(dim=-1)
        else:
            raise ValueError(method)
        uncs.append(unc.cpu())
        preds.append(pred.cpu())
    return torch.cat(preds), torch.cat(uncs)


@torch.no_grad()
def softmax_probs(model, loader, device):
    """导出 softmax 概率矩阵 (N, C)，供 Deep Ensembles 集成与 post-hoc 方法复用。

    嫁接骨干需 bf16 autocast（与 predict_uncertainty 同一语义）。
    """
    from contextlib import nullcontext
    import config

    model.eval()
    grafted = bool(str(getattr(config, "GRAFT", "")).strip())
    out = []
    for x, _ in loader:
        x = x.to(device)
        if grafted:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
        else:
            logits = model(x)
        out.append(F.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(out).numpy()
