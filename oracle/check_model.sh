#!/bin/bash
# 快速验证模型/endpoint 是否可用：bash oracle/check_model.sh
# 需要先 export MODEL=... 以及对应 provider 的 API key
set -u
MODEL=${MODEL:?set MODEL first}
python3 - "$MODEL" <<'PY'
import sys
import litellm

model = sys.argv[1]
r = litellm.completion(
    model=model,
    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    max_tokens=16,
    temperature=0,
)
print("model:", model)
print("content:", r.choices[0].message.content)
print("usage:", r.usage)
PY
