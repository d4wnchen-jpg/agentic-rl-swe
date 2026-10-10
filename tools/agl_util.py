#!/usr/bin/env python3
"""agl_util.py — 对齐 agent-lightning (microsoft/agent-lightning) 的辅助函数

来源：examples/swe_smith/agents/smith_agent.py
- _forbidden_action()  防作弊正则（git / 网络 / 装包 / 篡改测试）
- render_observation() 观察渲染（head + tail + 明确的省略量）
- parse_test_statuses() 判分解析（剥 ANSI + 双模式正则）
"""
from __future__ import annotations
import os
import re

# ⭐ SMITH_HARNESS_V2=1 切到"对齐 agl"的 harness：
#    - OBS_LIMIT 8000 → 10000（agl 的 SMITH_OBS_CHAR_CAP 默认值）
#    - 执行命令时套用 _CMD_ENV（安静 pager / 进度条，减少无关输出）
#    ⚠️ clip_obs() 的 cap 默认参数在【函数定义时】绑定本模块的 OBS_LIMIT，
#       所以必须改这里 —— 改 agent_par.py 里那个同名常量是无效的。
# ⭐ 默认就是 V2（正确配置）。
#    要复现"旧配置下的问题"（跑测试套件 → 输出截断 → 轮数耗尽），
#    显式设 SMITH_HARNESS_V1=1 回退。
_V1 = os.environ.get("SMITH_HARNESS_V1", "").strip() == "1"
_V2 = os.environ.get("SMITH_HARNESS_V2", "").strip() == "1"
HARNESS_V2 = _V2 or not _V1

OBS_LIMIT = int(os.environ.get(
    "SMITH_OBS_CHAR_CAP", "10000" if HARNESS_V2 else "8000"))   # 单条 observation 的字符上限

# 安静 pager / 进度条，避免一条命令淹没观察（照抄 agl 的 _CMD_ENV）
CMD_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}
KEEP_LAST_OBS = 5         # 保留最近 N 个 observation 原文
MAX_FORMAT_ERR = 3        # 格式错误容忍次数

# ═══════════════ 防作弊 ═══════════════
_GIT_INVOKE_RE = re.compile(
    r"""(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:[\w./-]*/)?git(?:-[a-z]+)?(?=\s|$|;|&|\|)""",
    re.I)
_NET_FETCH_RE = re.compile(
    r"""(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:[\w./-]*/)?(?:curl|wget|httpie|http|https|aria2c|scp|sftp|rsync|nc|ncat|netcat|telnet)(?=\s|$|;|&|\|)""",
    re.I)
_PKG_INSTALL_RE = re.compile(
    r"""(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:(?:[\w./-]*/)?(?:pip|pip3|conda|mamba|easy_install|uv)\b[^\n;&|]*?\binstall\b|(?:[\w./-]*/)?python[0-9.]*\s+-m\s+pip\b[^\n;&|]*?\binstall\b)""",
    re.I)
_PY_NET_RE = re.compile(
    r"""urllib\.request|\burlopen\b|\brequests\.(?:get|post|put|head|Session)\b|\bhttpx\.|\bsocket\.(?:socket|create_connection)\b|\burllib3\b""",
    re.I)
_TEST_TAMPER_RE = re.compile(
    r"""(?:>>?|\btee\b(?:\s+-a)?\s+)\s*['"]?[^\s'"|;&<>]*(?:conftest\.py|pytest\.ini|tox\.ini|sitecustomize\.py|usercustomize\.py|setup\.cfg|pyproject\.toml|\.pth)(?=['"\s;&|]|$)""",
    re.I)


def forbidden(action: str):
    """返回拒绝理由；None 表示放行。"""
    if _GIT_INVOKE_RE.search(action):
        return ("git is disabled in this environment. Inspect and edit the source "
                "files under /testbed directly (cat, grep, sed, python).")
    if _NET_FETCH_RE.search(action) or _PY_NET_RE.search(action):
        return ("Network access is disabled. Do not fetch code from the internet "
                "(curl, wget, urllib, requests). Solve the bug using only the "
                "source files already present under /testbed.")
    if _PKG_INSTALL_RE.search(action):
        return ("Installing packages is not allowed. Everything needed is already "
                "installed. Fix the bug by editing the source under /testbed.")
    if _TEST_TAMPER_RE.search(action):
        return ("Modifying test-harness or config files (conftest.py, pytest.ini, "
                "tox.ini, setup.cfg, pyproject.toml, sitecustomize.py, .pth) is not "
                "allowed. Fix the bug in the source under /testbed instead.")
    return None


def tamper_path(path: str) -> bool:
    """判断编辑器的写入目标是不是测试/配置文件或 .git"""
    if not path:
        return False
    if ".git" in path:
        return True
    return bool(re.search(
        r"""(?:conftest\.py|pytest\.ini|tox\.ini|sitecustomize\.py|usercustomize\.py|setup\.cfg|pyproject\.toml|\.pth)$""",
        path.strip(), re.I))


# ═══════════════ 观察渲染（head + tail） ═══════════════
_OVERLONG_WARNING = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can use head, tail or sed to view a smaller "
    "number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, use a more "
    "selective search pattern.\n"
    "If you really need the full output, redirect it to a file and search within it."
)


def clip_obs(out: str, cap: int = OBS_LIMIT) -> str:
    """超长时保留 head + tail，并明确告知省略量（不是静默截断）"""
    if len(out) <= cap:
        return out
    half = cap // 2
    elided = len(out) - cap
    return ("<warning>\n" + _OVERLONG_WARNING + "\n</warning>\n"
            + "<output_head>\n" + out[:half] + "\n</output_head>\n"
            + "<elided_chars>\n" + str(elided) + " characters elided\n</elided_chars>\n"
            + "<output_tail>\n" + out[-half:] + "\n</output_tail>")


def elide_old_obs(messages, keep_last: int = KEEP_LAST_OBS):
    """保留 system / 题面 / 全部 assistant 消息；
    最近 keep_last 个 observation 保留原文，更早的换成显式占位符。"""
    obs_idx = [i for i, m in enumerate(messages)
               if m.get("role") == "user"
               and str(m.get("content", "")).startswith("OBSERVATION:")]
    keep = set(obs_idx[-keep_last:])
    out = []
    for i, m in enumerate(messages):
        c = str(m.get("content", ""))
        if m.get("role") == "user" and c.startswith("OBSERVATION:"):
            if i in keep:
                out.append(m)
            else:
                out.append(dict(m, content=(
                    "OBSERVATION:\n<elided: an earlier environment output ("
                    + str(len(c)) + " chars) was removed to save context. "
                    "It is NOT a file problem.>")))
        else:
            out.append(m)
    return out


FORMAT_HINT = (
    "Format error: your reply contained no valid action.\n"
    "Reply with EXACTLY ONE function call in this format:\n\n"
    "<function=bash>\n<parameter=command>\nyour_command_here\n</parameter>\n</function>\n\n"
    "Do not add any other text after the closing tag.")


# ═══════════════ 判分解析 ═══════════════
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_STATUS_RE = re.compile(
    r"""(?:^|\s)(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)"""
    r"""|(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s|$)""")


def parse_test_statuses(text: str) -> dict:
    """剥掉 ANSI 颜色码后解析测试状态（支持 'PASSED node' 和 'node PASSED' 两种顺序）"""
    out = {}
    for line in _ANSI_RE.sub("", text).splitlines():
        m = _STATUS_RE.search(line)
        if not m:
            continue
        if m.group(1):
            out[m.group(2)] = m.group(1)
        else:
            out[m.group(3)] = m.group(4)
    return out


def f2p_only_filter(f2p, p2p):
    """只保留 F2P 所在测试文件里的 P2P（避免无关跨文件测试拖慢评测）"""
    files = sorted({t.split("::", 1)[0] for t in f2p})
    return [t for t in p2p if any(t.startswith(f) for f in files)]


# 照抄 agl smith_agent.format_error_message 的 finish_reason="length" 分支
LENGTH_HINT = (
    "Your previous response reached the output token limit before you produced "
    "a complete action, so it was cut off. Respond more concisely and provide "
    "exactly one action in the required format. If you need to think more, do so "
    "briefly."
)
