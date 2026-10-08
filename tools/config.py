#!/usr/bin/env python3
"""config.py — 训练超参的【单一来源】

优先级（高 → 低）：
  1. 环境变量        如 GRPO_LR=2e-4 GRPO_STEPS=40
  2. configs/default.yaml
  3. 下面的 DEFAULTS（保底）

设计意图：
  - 超参只在这里定义一次，`configs/default.yaml` 是它的可编辑视图
  - 两者不一致时 `tests/test_config.py` 会失败
  - 本模块不导入 torch / vllm，可以单独导入和测试

用法：
  from config import CFG
  print(CFG["lr"])
"""
from __future__ import annotations

import os
from pathlib import Path

BASE = os.environ.get(
    "SWESMITH_RL_HOME", str(Path(__file__).resolve().parents[1])
)
CONFIG_PATH = Path(
    os.environ.get("GRPO_CONFIG", Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
)

# ─────────────────────────────────────────────────────────────
# 默认值（与 configs/default.yaml 必须一致）
#
# 出处标注：
#   agl        = 对齐 agent-lightning 的 examples/swe_smith
#   single-gpu = 因单卡 96GB（而非参照的 4×192GB）而做的缩放
#   qwen3.5    = Qwen3.5 架构特有的约束
#   self       = 无参照，自行标定
# ─────────────────────────────────────────────────────────────
DEFAULTS: dict = dict(
    # ── rollout ──
    n_rollouts=8,               # agl (rollout.n)：每题采样条数 = 组大小 G
    temperature=1.0,            # agl：训练采样温度
    val_temperature=0.7,        # agl (rollout.val_kwargs)：验证温度
    max_turns=40,               # agl (SMITH_MAX_TURNS)
    max_new_tokens=16384,       # agl (AGL_MAX_TOKENS)

    # ── 训练 ──
    n_prompts=4,                # single-gpu (agl: 16)：每步题数
    steps=20,                   # single-gpu (agl: 1000)
    lr=1e-4,                    # self：LoRA 需要比全参(1e-6)高 100 倍
    mini_batch_size=16,         # agl (ppo_mini_batch_size)
    max_ppo_update_times=2,     # agl：每批做 2 次更新，让 clip 生效
    clip_low=0.2,               # agl
    clip_high=0.28,             # agl (DAPO 非对称)
    max_row_tokens=32768,       # single-gpu (agl: 65536)
    seed=0,

    # ── LoRA ──
    lora_r=16,                  # single-gpu
    lora_alpha=32,              # single-gpu（scaling = alpha/r = 2）

    # ── 评测 ──
    eval_every=10,              # agl (test_freq=16) 的适配
    eval_n=115,                 # single-gpu (agl: 400)：取全部 held-out（实际跑成 113，2 道沙箱失败）

    # ── 显存 ──
    gpu_mem_util=0.45,          # single-gpu (agl: 0.8)
    max_num_seqs=128,           # qwen3.5：Mamba cache 按序列数限制
)


def _flatten(d: dict) -> dict:
    """把 {'rollout': {'n': 8}} 展平成 {'n': 8}（分节只是为了可读）"""
    out: dict = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flatten(v))
        else:
            out[k] = v
    return out


def _coerce(value, like):
    """按 DEFAULTS 里的类型转换环境变量/字符串"""
    if isinstance(like, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(float(value))
    if isinstance(like, float):
        return float(value)
    return str(value)


def load(path: "str | Path | None" = None, env_prefix: str = "GRPO_") -> dict:
    """加载配置：DEFAULTS ← YAML ← 环境变量"""
    cfg = dict(DEFAULTS)

    p = Path(path) if path is not None else CONFIG_PATH
    if p.exists():
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            # ⚠️ 不能静默回退：用户改了 YAML 却不生效，属于静默失败
            raise ImportError(
                f"找到配置文件 {p}，但没装 pyyaml，配置不会被加载。\n"
                f"  解决：pip install pyyaml\n"
                f"  或：GRPO_CONFIG=/nonexistent 用代码里的默认值"
            ) from exc

        raw = _flatten(yaml.safe_load(p.read_text()) or {})
        unknown = sorted(set(raw) - set(DEFAULTS))
        if unknown:
            raise ValueError(
                f"{p} 里有未知的配置键: {unknown}\n"
                f"可用的键: {sorted(DEFAULTS)}"
            )
        cfg.update(raw)

    # 环境变量覆盖（GRPO_LR=2e-4 -> cfg['lr']）
    for key, default in DEFAULTS.items():
        env_val = os.environ.get(env_prefix + key.upper())
        if env_val is not None:
            try:
                cfg[key] = _coerce(env_val, default)
            except (TypeError, ValueError):
                pass

    return cfg


CFG: dict = load()

__all__ = ["CFG", "DEFAULTS", "load", "BASE", "CONFIG_PATH"]
