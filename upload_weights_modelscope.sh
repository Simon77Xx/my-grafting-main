#!/bin/bash
# 将 4 个嫁接权重上传到 ModelScope（需先提供 SDK Token）。
# 用法: bash upload_weights_modelscope.sh <token 或 token 文件路径>
# 仓库命名: <modelscope用户名>/DiT-XL-2-grafted-uncertainty
set -e
MS=/home/shixisheng/miniforge3/bin/modelscope
PM=/supcon3/shixisheng/xjx/research/grafting-main/pretrained_models
ARG="$1"
[ -n "$ARG" ] || { echo "usage: $0 <token|token_file>"; exit 1; }
if [ -f "$ARG" ]; then TOKEN=$(cat "$ARG"); else TOKEN="$ARG"; fi

echo "== 1. 获取用户名 =="
USER=$(/home/shixisheng/miniforge3/bin/python - "$TOKEN" <<'PY'
import json, sys, urllib.request
tok = sys.argv[1]
req = urllib.request.Request(
    'https://www.modelscope.cn/api/v1/userinfo',
    headers={'Authorization': 'Bearer ' + tok})
d = json.load(urllib.request.urlopen(req, timeout=20))['Data']
print(d['Username'])
PY
)
echo "username = $USER"
REPO="$USER/DiT-XL-2-grafted-uncertainty"

echo "== 2. 创建模型仓库 $REPO =="
$MS create "$REPO" --token "$TOKEN" || echo "(create 失败可能因仓库已存在，继续)"

echo "== 3. 上传 4 个权重（约 30-60 分钟）=="
for f in DiT-XL-2-256x256.pt DiT-XL-2-swa-50p.pt DiT-XL-2-hyena_x-50p.pt DiT-XL-2-mamba_2-50p.pt; do
  echo "--- upload $f"
  $MS upload "$REPO" "$PM/$f" --token "$TOKEN"
done

echo "== 4. 完成，下载链接 =="
for f in DiT-XL-2-256x256.pt DiT-XL-2-swa-50p.pt DiT-XL-2-hyena_x-50p.pt DiT-XL-2-mamba_2-50p.pt; do
  echo "https://www.modelscope.cn/models/$REPO/resolve/master/$f"
done
