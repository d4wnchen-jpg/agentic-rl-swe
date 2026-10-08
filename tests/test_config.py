#!/usr/bin/env python3
"""配置一致性测试

防止两类问题：
  ① configs/default.yaml 与 tools/config.py 的 DEFAULTS 漂移
  ② train_grpo.py 用到了 config 里没有的键（运行时 KeyError）
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import config  # noqa: E402

yaml = pytest.importorskip("yaml", reason="需要 pyyaml")


# ═══════════════════════════════════════════════════════════
def test_yaml_exists():
    assert config.CONFIG_PATH.exists(), f"找不到 {config.CONFIG_PATH}"


def test_yaml_keys_are_known():
    """YAML 里不能有 config.py 未定义的键（load() 会抛错，这里显式测）"""
    raw = config._flatten(yaml.safe_load(config.CONFIG_PATH.read_text()) or {})
    unknown = sorted(set(raw) - set(config.DEFAULTS))
    assert not unknown, f"YAML 里有未知键: {unknown}"


def test_yaml_matches_defaults():
    """YAML 的值必须与 DEFAULTS 一致（单一来源的保证）"""
    raw = config._flatten(yaml.safe_load(config.CONFIG_PATH.read_text()) or {})
    mismatched = {
        k: (raw[k], config.DEFAULTS[k])
        for k in raw
        if k in config.DEFAULTS and raw[k] != config.DEFAULTS[k]
    }
    assert not mismatched, (
        "default.yaml 与 config.py 的 DEFAULTS 不一致（改一处要改两处）:\n"
        + "\n".join(f"  {k}: yaml={v[0]!r}  defaults={v[1]!r}" for k, v in mismatched.items())
    )


def test_all_defaults_documented_in_yaml():
    """DEFAULTS 里的每个键都应该在 YAML 里出现（否则用户看不到）"""
    raw = config._flatten(yaml.safe_load(config.CONFIG_PATH.read_text()) or {})
    missing = sorted(set(config.DEFAULTS) - set(raw))
    assert not missing, f"这些键在 default.yaml 里没写: {missing}"


# ═══════════════════════════════════════════════════════════
def test_env_override(monkeypatch):
    monkeypatch.setenv("GRPO_LR", "3e-4")
    monkeypatch.setenv("GRPO_STEPS", "40")
    monkeypatch.setenv("GRPO_SEED", "7")
    cfg = config.load()
    assert cfg["lr"] == pytest.approx(3e-4)
    assert cfg["steps"] == 40
    assert cfg["seed"] == 7


def test_env_override_keeps_types(monkeypatch):
    monkeypatch.setenv("GRPO_N_ROLLOUTS", "4")
    cfg = config.load()
    assert isinstance(cfg["n_rollouts"], int)
    assert cfg["n_rollouts"] == 4


# ═══════════════════════════════════════════════════════════
def test_train_script_only_uses_defined_keys():
    """train_grpo.py 里 CFG["x"] 用到的键都必须在 config 里"""
    src = (ROOT / "tools" / "train_grpo.py").read_text()
    used = set(re.findall(r'CFG\["([a-z_]+)"\]', src))
    missing = sorted(used - set(config.DEFAULTS))
    assert not missing, f"train_grpo.py 用了未定义的配置键: {missing}"


def test_required_keys_present():
    """关键超参一个都不能少"""
    required = {
        "n_rollouts", "n_prompts", "max_turns", "max_row_tokens", "lr",
        "mini_batch_size", "max_ppo_update_times", "clip_low", "clip_high",
        "temperature", "val_temperature", "lora_r", "lora_alpha",
        "steps", "eval_every", "eval_n", "gpu_mem_util",
        "max_new_tokens", "max_num_seqs", "seed",
    }
    assert required <= set(config.DEFAULTS)


def test_sane_values():
    """明显不合理的值要拦住（防止手滑改错）"""
    d = config.DEFAULTS
    assert 0 < d["lr"] < 1e-2, "lr 应该在 (0, 1e-2)"
    assert 0 < d["clip_low"] < 1, "clip_low 应在 (0,1)"
    assert d["clip_high"] >= d["clip_low"], "clip_high 应 >= clip_low"
    assert d["mini_batch_size"] <= d["steps"] * d["n_prompts"], "mini-batch 不该大于总行数"
    assert 0 < d["gpu_mem_util"] <= 1
    assert d["max_num_seqs"] >= 32, "Qwen3.5 Mamba cache 需要足够的序列槽"
    assert 0 < d["val_temperature"] <= d["temperature"] + 1e-9, \
        "验证温度通常不高于训练温度"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_missing_pyyaml_raises(monkeypatch):
    """没装 pyyaml 时必须报错，而不是静默用默认值（否则用户改了 YAML 不生效）"""
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match="pyyaml"):
        config.load()
