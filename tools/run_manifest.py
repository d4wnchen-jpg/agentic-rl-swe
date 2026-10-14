#!/usr/bin/env python3
"""run_manifest.py — 每次训练/评测自动落一份「运行清单」

解决的问题（本项目真实踩过的坑）：
    1. 训练代码没进 git，事后查不到当时用的哪个版本
    2. 只记了命令行，没记环境变量 —— enable_thinking 到底传没传查不出来
    3. 文档写结论时没写统计判据（例如「96% 复读」是 ≥1 个还是 ≥5 个）
    4. 每次跑的「生效配置」没有留痕，只能翻代码猜

用法（在 eval_adapter.py / train_grpo.py 里调一次）：
    from run_manifest import collect, write
    meta = collect(tag="base40", adapter="grpo_adapters/step000", tasks="eval.jsonl")
    write(meta, "eval_base40.meta.json")

设计原则：
    ⭐ 记「实际生效的值」而不是「环境变量」
       —— 从模块里读 HARNESS_V2 / OBS_LIMIT / HIDE_F2P / enable_thinking，
          环境变量只是输入之一，默认值可能才是真正生效的那个。
    ⭐ 未设置的环境变量也记，标成 "<unset>"
       —— 「没设」和「设成 0」是两件事。
    ⭐ 代码版本记 commit + dirty + 逐文件 md5
       —— 训练时如果没提交，md5 是唯一能对齐代码的办法。
    ⭐ 题库记路径 + 行数 + md5
       —— 题目换过的话，两次评测就不可比。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ⭐ 这几个是本项目里会改变行为的开关。新增开关时加到这里。
ENV_KEYS = [
    "SMITH_HARNESS_V1",        # 回退到旧 harness（V1）
    "SMITH_HARNESS_V2",        # 显式启用 V2
    "SMITH_HIDE_F2P",          # 沙箱里移除判分测试
    "SMITH_ENABLE_THINKING",   # 0=False / 1=True / 不设=不传
    "SMITH_OBS_CHAR_CAP",      # 单条 observation 字符上限
    "VLLM_ENFORCE_EAGER",
]

# ⭐ 会被记录 md5 的关键代码文件（改这些 = 改行为）
CODE_FILES = [
    "tools/agent_par.py",
    "tools/agl_util.py",
    "tools/grpo_llm.py",
    "tools/grpo_rollout.py",
    "tools/grpo_util.py",
    "tools/eval_quick.py",
    "tools/eval_adapter.py",
    "tools/tpl_harness_v2.txt",
    "tools/tpl_user_v2.txt",
    "tools/tpl_system.txt",
    "tools/tpl_chat.jinja",
]


def _md5(p: Path) -> str | None:
    try:
        return hashlib.md5(p.read_bytes()).hexdigest()[:12]
    except OSError:
        return None


def _git(base: Path) -> dict:
    """commit + dirty 状态；仓库不存在时全标 unknown（例如工作目录不是仓库）。"""
    def run(*a):
        try:
            r = subprocess.run(["git", *a], cwd=base, capture_output=True,
                               text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def _effective(base: Path) -> dict:
    """⭐ 从模块里读【实际生效】的配置，而不是从环境变量推断。"""
    out: dict = {}
    if str(base / "tools") not in sys.path:
        sys.path.insert(0, str(base / "tools"))
    try:
        import agl_util
        out["HARNESS_V2"] = bool(getattr(agl_util, "HARNESS_V2", None))
        out["OBS_LIMIT"] = getattr(agl_util, "OBS_LIMIT", None)
    except Exception as e:
        out["agl_util_error"] = f"{type(e).__name__}: {e}"
    try:
        import agent_par
        out["HIDE_F2P"] = bool(getattr(agent_par, "HIDE_F2P", None))
        sp = getattr(agent_par, "SYS_PROMPT", "") or ""
        up = getattr(agent_par, "USER_TPL", "") or ""
        # ⭐ 记录 prompt 的特征串 —— 比记文件名可靠（文件可能被改但名字不变）
        out["prompt_marks"] = {
            "sys_has_env_boundaries": "Environment boundaries" in sp,
            "sys_has_very_brief": "VERY brief" in sp,       # ⚠️ 2026-10-14 已删，应为 False
            "user_has_5step": "Create a script to reproduce" in up,
            "user_has_thinking_thorough": "thinking should be thorough" in up,
            "user_has_restored_wording": "restored before the fix" in up,  # ⚠️ 应为 False
        }
    except Exception as e:
        out["agent_par_error"] = f"{type(e).__name__}: {e}"
    try:
        import grpo_llm
        out["enable_thinking_default"] = getattr(grpo_llm, "_ET_DEFAULT", "MISSING")
    except Exception as e:
        out["grpo_llm_error"] = f"{type(e).__name__}: {e}"
    return out


def collect(base: str | Path | None = None, **extra) -> dict:
    base = Path(base or os.environ.get("SWESMITH_RL_HOME") or
                Path(__file__).resolve().parents[1])
    meta = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": platform.node(),
        "cwd": os.getcwd(),
        "base": str(base),
        "python": sys.executable,
        # ⭐ 未设置的也记 —— 「没设」≠「设成 0」
        "env": {k: os.environ.get(k, "<unset>") for k in ENV_KEYS},
        "effective": _effective(base),
        "git": _git(base),
        "code_md5": {f: _md5(base / f) for f in CODE_FILES},
    }
    # ⭐ 题库指纹：换过题就不可比
    for key in ("tasks", "eval_file"):
        t = extra.get(key)
        if t:
            p = Path(t) if Path(t).is_absolute() else base / t
            n = None
            try:
                n = sum(1 for l in open(p) if l.strip())
            except OSError:
                pass
            meta.setdefault("task_file", {})[str(t)] = {"lines": n, "md5": _md5(p)}
    meta.update(extra)
    return meta


def write(meta: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"  运行清单已写 {p}", flush=True)


def _selftest():
    m = collect(tag="selftest")
    print(json.dumps(m, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    _selftest()
