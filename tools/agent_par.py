#!/usr/bin/env python3
"""agent_par.py — 并行 rollout

与 agent.py 的关键区别：
  1. 每个 rollout 独占 /tmp/par/<tag> 工作目录，互不干扰
  2. 模型命令里写死的 /testbed 会被透明替换成真实目录
  3. 工具返回里的真实路径再替换回 /testbed，保持模型看到的世界一致
  4. 支持批量生成（llm.batch_chat）+ 多线程工具执行
"""
from __future__ import annotations
import os, re, json, shutil, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from agl_util import (clip_obs, forbidden, tamper_path, elide_old_obs,
                      FORMAT_HINT, MAX_FORMAT_ERR, parse_test_statuses,
                      f2p_only_filter)

BASE = Path(os.environ.get("SWESMITH_RL_HOME", Path(__file__).resolve().parents[1]))
REPOS = BASE / "probe"
ENVS  = {"mistune": "mistune", "sqlparse": "sqlparse",
         "isodate": "isodate", "boltons": "boltons"}

ACTION_RE = re.compile(r"<function=(\w+)>(.*?)(?:</function>|$)", re.S)
PARAM_RE  = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.S)
OBS_LIMIT, EDIT_LIMIT, CMD_TIMEOUT = 8000, 20000, 60    # 代码阅读需要足够大的观察窗口
STOP_STRINGS = ["</function>"]      # 生成一个动作就停
MAX_CTX_CHARS = 80000               # ~26k token（按 3.05 字符/token 实测），留 6k 给生成

SYS_PROMPT = ((BASE/"tools/tpl_system.txt").read_text()
              if (BASE/"tools/tpl_system.txt").exists() else
              "You are a helpful assistant that can interact with a computer to solve tasks.")

USER_TPL = """<uploaded_files>
/testbed
</uploaded_files>
I've uploaded a python code repository in the directory /testbed. Consider the following PR description:

<pr_description>
{problem}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
"""


def repo_of(task):
    return task["repo"].split("/")[-1].split("__")[1].split(".")[0]

def parse_actions(text):
    out = []
    for m in ACTION_RE.finditer(text):
        name, body = m.group(1), m.group(2)
        params = {p.group(1): p.group(2).strip() for p in PARAM_RE.finditer(body)}
        out.append((name, params))
    return out


# ---------------- 路径双向映射 ----------------
def _to_real(s, wd):  return str(s).replace("/testbed", str(wd))
def _to_virt(s, wd):  return str(s).replace(str(wd), "/testbed")


# ---------------- 工具 ----------------
def run_bash(cmd, wd, timeout=CMD_TIMEOUT):
    try:
        r = subprocess.run(_to_real(cmd, wd), shell=True, cwd=wd,
                           capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        out = f"<command timed out after {timeout}s>"
    out = _to_virt(out.strip() or "<no output>", wd)
    return clip_obs(out)          # agent-lightning 风格：head + tail + 明确省略量


def run_editor(params, wd):
    cmd  = (params.get("command") or "").strip()
    path = params.get("path") or ""
    P    = Path(_to_real(path, wd)) if path else Path(wd)

    if cmd == "view":
        if not path:
            return "<no path>"
        if P.is_dir():
            items = sorted(x.name + ("/" if x.is_dir() else "") for x in P.iterdir())
            return _to_virt("\n".join(items)[:EDIT_LIMIT] or "<empty dir>", wd)
        if not P.exists():
            return f"<file not found: {path}>"

        all_lines = P.read_text(errors="replace").splitlines()
        total     = len(all_lines)
        start, lines = 1, all_lines
        vr = params.get("view_range")
        if vr:
            nums = [int(x) for x in re.findall(r"-?\d+", str(vr))]
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                a = max(1, a if a > 0 else total + a + 1)
                b = total if b == -1 else min(b, total)
                if a <= b:
                    start, lines = a, all_lines[a-1:b]

        # 按字符预算逐行累加，保证截断位置落在行边界
        kept, used = [], 0
        for i, l in enumerate(lines):
            piece = f"{i+start:6}\t{l}\n"
            if used + len(piece) > EDIT_LIMIT and kept:
                break
            kept.append(piece)
            used += len(piece)
        out = "".join(kept).rstrip("\n")
        shown_end = start + len(kept) - 1

        if shown_end < start + len(lines) - 1 or (start > 1 and shown_end < total):
            nxt_a = shown_end + 1
            nxt_b = min(shown_end + 400, total)
            out += (f"\n<truncated: showing lines {start}-{shown_end} of {total} "
                    f"(display limit {EDIT_LIMIT} chars). This is a DISPLAY limit, not a file problem. "
                    f"Use view_range=[{nxt_a}, {nxt_b}] to read the next chunk.>")
        elif start > 1:
            out += f"\n<end of requested range (file has {total} lines total)>"

        return _to_virt(out if out else "<empty file>", wd)

    if cmd == "create":
        if P.exists():
            return f"<file already exists: {path}>"
        P.parent.mkdir(parents=True, exist_ok=True)
        P.write_text(params.get("file_text") or "")
        return f"File created successfully at: {path}"

    if cmd == "str_replace":
        if not P.exists():
            return f"<file not found: {path}>"
        old_s, new_s = params.get("old_str") or "", params.get("new_str") or ""
        txt = P.read_text()
        n = txt.count(old_s)
        if n == 0:
            return f"<old_str not found in {path}>"
        if n > 1:
            return f"<old_str appears {n} times, must be unique>"
        P.write_text(txt.replace(old_s, new_s, 1))
        return f"Edited {path}"

    if cmd == "insert":
        if not P.exists():
            return f"<file not found: {path}>"
        line = int(params.get("insert_line") or 0)
        lines = P.read_text().splitlines()
        lines.insert(line, params.get("new_str") or "")
        P.write_text("\n".join(lines) + "\n")
        return f"Inserted at line {line} in {path}"

    return f"<unknown editor command: {cmd}>"


def run_tools(actions, wd):
    obs = []
    for n, p in actions:
        if n == "bash":
            cmd = p.get("command", "")
            reason = forbidden(cmd)
            obs.append(reason if reason else run_bash(cmd, wd))
        elif n in ("str_replace_editor", "edit"):
            if tamper_path(p.get("path", "")):
                obs.append("<editing test-harness/config files or .git is not allowed>")
            else:
                obs.append(run_editor(p, wd))
        elif n == "submit":
            obs.append("<submitted>")
        else:
            obs.append(f"<unknown tool: {n}>")
    return obs


# ---------------- 沙箱 ----------------
def prep_sandbox(task, wd: Path):
    repo = repo_of(task)
    if wd.exists():
        subprocess.run(["chmod","-R","u+w",str(wd)], capture_output=True)
        shutil.rmtree(wd, ignore_errors=True)
    shutil.copytree(REPOS/repo, wd, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__","*.pyc",".git",
                                                  ".eggs","*.egg-info"))
    subprocess.run(["git","init","-q"], cwd=wd, capture_output=True)
    subprocess.run(["git","add","-A"], cwd=wd, capture_output=True)
    subprocess.run(["git","-c","user.email=a@b","-c","user.name=a",
                    "commit","-qm","base"], cwd=wd, capture_output=True)
    # ⚠️ 补丁放沙箱外面，防止 agent 读到后反向应用（奖励 hack）
    import tempfile
    fd, pth = tempfile.mkstemp(suffix=".patch", dir="/tmp")
    with os.fdopen(fd, "w") as fh: fh.write(task["patch"])
    r = subprocess.run(["git","apply","--whitespace=nowarn",pth], cwd=wd,
                       capture_output=True, text=True)
    os.unlink(pth)
    shutil.rmtree(wd/".git", ignore_errors=True)      # 判分不需要 git
    if r.returncode != 0:
        raise RuntimeError(f"bug patch 失败: {r.stderr[:150]}")


def restore_tests(repo, wd: Path):
    src = REPOS/repo/"tests"
    if not src.exists(): return
    dst = wd/"tests"
    if dst.exists(): shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__","*.pyc"))


def evaluate(task, env, wd: Path, f2p_only=True):
    """对齐 agent-lightning：--color=no -rA（短摘要，不折行）+ 剥 ANSI + f2p_only"""
    f2p = task["FAIL_TO_PASS"]
    p2p = task["PASS_TO_PASS"]
    if f2p_only:
        p2p = f2p_only_filter(f2p, p2p)
    nodes = f2p + p2p
    cmd = ["conda", "run", "-n", env, "--no-capture-output",
           "pytest", "--color=no", "-rA",
           "-p", "no:cacheprovider", "--no-header"] + nodes
    e = dict(os.environ, PYTHONPATH=f"{wd}:{wd}/src", COLUMNS="200")
    try:
        r = subprocess.run(cmd, cwd=wd, env=e, capture_output=True,
                           text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False, {"n_fail": -1, "n_missing": len(nodes),
                       "collected": 0, "note": "timeout"}
    res = parse_test_statuses(r.stdout + r.stderr)
    n_fail    = sum(1 for t in nodes if res.get(t) not in ("PASSED", "XFAIL"))
    n_missing = sum(1 for t in nodes if t not in res)
    return (n_fail == 0 and n_missing == 0), {
        "n_fail": n_fail, "n_missing": n_missing,
        "collected": len(res), "expect": len(nodes), "res": res}


def init_msgs(task):
    return [{"role":"system","content": SYS_PROMPT},
            {"role":"user","content": USER_TPL.format(problem=task["problem_statement"])}]


# ---------------- 并行驱动 ----------------
# 历史管理交给 agl_util.elide_old_obs：
# 保留 system / 题面 / 全部 assistant 消息；最近 N 个 observation 保留原文，
# 更早的换成显式占位符（agent-lightning 不裁历史，我们因 32k 上下文做了温和省略）


def _safe_batch(llm, msgs, stop):
    """批量生成，带降级：整批失败 -> 逐条重试 -> 单条失败返回 None（只废掉它自己）"""
    try:
        return llm.batch_chat(msgs, stop=stop)
    except Exception as e:
        print(f"[warn] 整批生成失败({type(e).__name__}): {str(e)[:100]}", flush=True)
        out = []
        for m in msgs:
            try:
                out += llm.batch_chat([m], stop=stop)
            except Exception as e2:
                print(f"[warn] 单条失败({type(e2).__name__}): {str(e2)[:120]}", flush=True)
                out.append(None)
        return out


def run_parallel(tasks, llm, max_turns=60, workers=8, verbose=False, root="/tmp/par"):
    os.makedirs(root, exist_ok=True)
    stamp = int(time.time())

    states = []
    for i, t in enumerate(tasks):
        wd = Path(root)/f"{stamp}_{i}"
        try:
            prep_sandbox(t, wd)
        except Exception as e:
            print(f"  [{i}] 沙箱准备失败: {e}", flush=True); continue
        states.append({"task": t, "wd": wd, "messages": init_msgs(t), "turn": 0,
                       "done": False, "last_sig": None, "rep": 0, "fmt_err": 0,
                       "resolved": False, "n_fail": -1, "n_missing": -1, "nturns": 0})
    print(f"已准备 {len(states)} 个沙箱", flush=True)

    ex = ThreadPoolExecutor(max_workers=workers)
    t0 = time.time()
    for turn in range(max_turns):
        alive = [s for s in states if not s["done"]]
        if not alive: break

        for s in alive:
            if os.environ.get("ELIDE_OFF", "0") != "1":
                s["messages"] = elide_old_obs(s["messages"])   # ELIDE_OFF=1 时不裁历史
        try:
            texts = _safe_batch(llm, [s["messages"] for s in alive], stop=STOP_STRINGS)
        except TypeError:
            texts = _safe_batch(llm, [s["messages"] for s in alive])

        jobs = []
        for s, text in zip(alive, texts):
            if text is None:
                print(f"  [warn] {s['task']['instance_id'][-24:]} 生成失败，结束该 episode", flush=True)
                s["done"] = True; continue
            s["nturns"] += 1
            s["messages"].append({"role":"assistant","content":text})
            acts = parse_actions(text)
            if verbose:
                print(f"\n--- [{s['task']['instance_id'][:40]}] turn {s['nturns']} ---\n{text[:400]}", flush=True)
            if not acts:
                s["fmt_err"] = s.get("fmt_err", 0) + 1
                if s["fmt_err"] >= MAX_FORMAT_ERR:
                    s["done"] = True; continue
                s["messages"].append({"role": "user", "content": FORMAT_HINT})
                continue
            s["fmt_err"] = 0
            if any(n == "submit" for n, _ in acts):
                s["done"] = True; continue
            sig = json.dumps(acts, ensure_ascii=False)
            if sig == s["last_sig"]:
                s["rep"] += 1
                if s["rep"] >= 3: s["done"] = True; continue
            else:
                s["rep"], s["last_sig"] = 0, sig
            jobs.append((s, acts))

        futs = [(s, ex.submit(run_tools, acts, s["wd"])) for s, acts in jobs]
        for s, f in futs:
            try:   obs = f.result()
            except Exception as e: obs = [f"<tool error: {e}>"]
            s["messages"].append({"role":"user","content":"OBSERVATION:\n" + "\n".join(obs)})

        print(f"[turn {turn+1:3}] 活跃 {len(alive):3}  完成 {sum(1 for s in states if s['done']):3}  "
              f"{time.time()-t0:6.0f}s", flush=True)

    def fin(s):
        repo = repo_of(s["task"])
        restore_tests(repo, s["wd"])
        ok, det = evaluate(s["task"], ENVS[repo], s["wd"])
        s["resolved"]  = ok
        s["n_fail"]    = det.get("n_fail", -1)
        s["n_missing"] = det.get("n_missing", -1)
        return s

    list(ex.map(fin, states))
    ex.shutdown()
    return states
