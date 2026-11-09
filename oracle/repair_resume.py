#!/usr/bin/env python3
"""repair_resume.py — 修复实验：从某个轮次之前的状态继续采样，判断能否修好。

条件（condition）：
  restart : 从原始 base 状态开始（从头重试）
  random  : 回退到随机第 k 轮之前
  teacher : 回退到教师指认的第 p 轮之前
  ours    : 回退到我们验证过的有害步骤之前

所有条件：同样的 agent 循环、同样的步数预算、同样的判分口径。

用法：
  python oracle/repair_resume.py \
    --traj results/d1a/7.traj.json --task SWE-bench_Lite.jsonl \
    --instance-id pallets__flask-5063 \
    --image swebench/sweb.eval.x86_64.pallets_1776_flask-5063:latest \
    --report results/d1a/7.json --teacher results/d1a/7_teacher.json \
    --condition ours --steps 8 \
    --work repair_work --out repair_out_7_ours --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import time
import uuid

GIT = ["git", "-c", "user.email=r@local", "-c", "user.name=repair"]
SYSTEM = ("You are a helpful assistant that can interact with a computer shell to solve "
          "programming tasks.")


def sh(args, cwd=None, timeout=600, input_text=None):
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, input=input_text)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"


def git(args, cwd=None):
    return sh(GIT + args, cwd=cwd)


def dc(cname, cmd, timeout=300):
    return sh(["docker", "exec", cname, "bash", "-lc", cmd], timeout=timeout)


def load_task(path, iid):
    with open(path) as f:
        if path.endswith(".jsonl"):
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    if t["instance_id"] == iid:
                        return t
        else:
            t = json.load(f)
            if t["instance_id"] == iid:
                return t
    raise SystemExit("task 未找到")


def extract_turns(traj):
    turns = []
    for m in traj.get("messages", []):
        if m.get("role") != "assistant":
            continue
        cmds = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                a = json.loads(fn.get("arguments", "{}"))
            except Exception:
                continue
            if fn.get("name") == "bash" and a.get("command"):
                cmds.append(a["command"])
        if cmds:
            turns.append(cmds)
    return turns


def get_base_repo(image, dest):
    rc, out, err = sh(["docker", "create", image])
    if rc != 0:
        raise SystemExit(f"docker create 失败: {err[:300]}")
    cid = out.strip()
    try:
        sh(["docker", "cp", f"{cid}:/testbed", dest], timeout=300)
    finally:
        sh(["docker", "rm", "-f", cid])
    if not os.path.isdir(os.path.join(dest, ".git")):
        sh(GIT + ["init", "-q"], cwd=dest)
    sh(GIT + ["add", "-A"], cwd=dest)
    sh(GIT + ["commit", "-qm", "base"], cwd=dest)
    _, out, _ = sh(GIT + ["rev-parse", "HEAD"], cwd=dest)
    return out.strip()


def call_llm(messages, model):
    """OpenAI 兼容接口调用；REPAIR_API_BASE/REPAIR_API_KEY 可指向硅基流动等。"""
    # 只去掉 provider 前缀，保留 org/model（如 openai/Qwen/Qwen3.5-9B -> Qwen/Qwen3.5-9B）
    for _prefix in ("openai/", "deepseek/", "dashscope/", "openrouter/",
                    "litellm/", "siliconflow/"):
        if model.startswith(_prefix):
            model = model[len(_prefix):]
            break
    from openai import OpenAI
    base = (os.environ.get("REPAIR_API_BASE") or os.environ.get("OPENAI_API_BASE")
            or "https://api.deepseek.com/v1")
    key = (os.environ.get("REPAIR_API_KEY") or os.environ.get("OPENAI_API_KEY")
           or os.environ.get("DEEPSEEK_API_KEY"))
    if not key:
        envp = os.path.expanduser("~/.config/mini-swe-agent/.env")
        if os.path.exists(envp):
            for line in open(envp):
                if line.strip().startswith("DEEPSEEK_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise SystemExit("API key 未设置（REPAIR_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY）")
    client = OpenAI(api_key=key, base_url=base)
    kwargs = {}
    if "qwen" in model.lower() or os.environ.get("THINK", "").lower() == "false":
        kwargs["extra_body"] = {"enable_thinking": False}
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=float(os.environ.get("REPAIR_TEMP", "0.7")),
        max_tokens=int(os.environ.get("REPAIR_MAX_TOKENS", "4096")),
        **kwargs)
    return resp.choices[0].message.content, resp.usage.total_tokens if resp.usage else 0


def _clean_cmd(s):
    s = (s or "").strip()
    s = s.strip('"').strip("'").strip()
    s = s.strip("`").strip()
    s = s.rstrip('"').rstrip("'").rstrip("`").strip()
    return s


def _extract_tool_call(tc):
    """从 Qwen 常见的 <tool_call> 变体里抽命令。"""
    tc = (tc or "").strip()
    m = re.match(r"^(?:bash|shell|sh)\b[\s:]*", tc, re.I)
    if m:
        tc = tc[m.end():]
    try:
        obj = json.loads(tc)
        if isinstance(obj, dict):
            args = obj.get("arguments") or obj.get("args") or {}
            if isinstance(args, dict) and args.get("command"):
                return [_clean_cmd(str(args["command"]))]
            if isinstance(obj.get("command"), str):
                return [_clean_cmd(obj["command"])]
    except Exception:
        pass
    m = re.search(r"command(?:_args)?\s*=\s*(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')", tc, re.S)
    if m:
        val = m.group(1)[1:-1]
        if val.strip():
            return [_clean_cmd(val)]
    m = re.match(r"^command(?:_args)?\s*:?\s*\n?(.*)$", tc, re.S)
    if m:
        val = m.group(1)
        if val.strip():
            return [_clean_cmd(val)]
    return [_clean_cmd(tc)] if tc.strip() else []


def parse_commands(text):
    """支持 mini-swe 的 ```bash 块，以及 Qwen 常见的 <bash>/<bash_command>/<tool_call> 变体。"""
    text = text or ""
    blocks = []
    blocks += re.findall(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", text, re.S)
    for tag in ("bash", "bash_command"):
        blocks += re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)
    closed = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    for tc in closed:
        blocks += _extract_tool_call(tc)
    if "<tool_call>" in text and "</tool_call>" not in text:
        for tc in re.split(r"<tool_call>", text)[1:]:
            blocks += _extract_tool_call(tc)
    out, seen = [], set()
    for b in blocks:
        cmd = _clean_cmd(b)
        if not cmd or cmd in seen:
            continue
        if cmd.startswith("#") and "\n" not in cmd:
            continue
        seen.add(cmd)
        out.append(cmd)
    return out


def eval_in_container(cname, task, base_sha, test_patch, rerun=0):
    f2p = task["FAIL_TO_PASS"]
    p2p = task["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    nodes = list(f2p) + list(p2p)
    files = sorted({n.split("::", 1)[0] for n in nodes})
    dc(cname, "cd /testbed && git config --global --add safe.directory /testbed")
    dc(cname, "cd /testbed && git checkout -q %s -- %s" %
       (base_sha, " ".join(shlex.quote(f) for f in files)))
    if test_patch:
        tp = "/tmp/test_patch.diff"
        # 写 test_patch 到容器
        b64 = __import__("base64").b64encode(test_patch.encode()).decode()
        dc(cname, f"echo {b64} | base64 -d > {tp}")
        dc(cname, f"cd /testbed && git apply --whitespace=nowarn {tp}")
    cmd = ("cd /testbed && pytest --color=no -rA -p no:cacheprovider --no-header "
           + " ".join(shlex.quote(f) for f in files))
    rc, out, err = dc(cname, cmd, timeout=1800)
    statuses = {m.group(2): m.group(1)
                for m in re.finditer(r"^(PASSED|FAILED|ERROR)\s+(\S+)", out + err, re.M)}
    n_fail = sum(1 for n in nodes if statuses.get(n) not in ("PASSED", "XFAIL"))
    f2p_pass = sum(1 for n in f2p if statuses.get(n) == "PASSED")
    return {"n_fail": n_fail, "f2p_pass": f2p_pass, "n_f2p": len(f2p),
            "n_pass": sum(1 for n in nodes if statuses.get(n) == "PASSED")}



def eval_dir(image, repo_dir, task, base_sha, test_patch, tmpdir, tag="x"):
    """把 repo 复制一份，在独立容器里按官方口径评测，返回 n_fail/f2p_pass/failed_nodes。"""
    import base64
    ev_dir = os.path.join(tmpdir, f"evalcopy_{tag}")
    shutil.rmtree(ev_dir, ignore_errors=True)
    shutil.copytree(repo_dir, ev_dir, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".tox",
                                                ".nox", ".venv", "node_modules"))
    f2p = task["FAIL_TO_PASS"]
    p2p = task["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    nodes = list(f2p) + list(p2p)
    files = sorted({n.split("::", 1)[0] for n in nodes})
    b64 = base64.b64encode((test_patch or "").encode()).decode()
    inner = [
        "cd /testbed",
        "git config --global --add safe.directory /testbed",
        "git checkout -q " + base_sha + " -- " + " ".join(shlex.quote(f) for f in files),
        f"echo {b64} | base64 -d > /tmp/tp.diff",
        "git apply --whitespace=nowarn /tmp/tp.diff || true",
        "pytest --color=no -rA -p no:cacheprovider --no-header "
        + " ".join(shlex.quote(f) for f in files),
    ]
    cmd = " && ".join(inner[:-1]) + " ; " + inner[-1]
    rc, out, err = sh(["docker", "run", "--rm", "-v", f"{ev_dir}:/testbed",
                       "-w", "/testbed", image, "bash", "-lc", cmd], timeout=1800)
    statuses = {m.group(2): m.group(1)
                for m in re.finditer(r"^(PASSED|FAILED|ERROR)\s+(\S+)", out + err, re.M)}
    n_fail = sum(1 for n in nodes if statuses.get(n) not in ("PASSED", "XFAIL"))
    f2p_pass = sum(1 for n in f2p if statuses.get(n) == "PASSED")
    failing = [n for n in nodes if statuses.get(n) not in ("PASSED", "XFAIL")]
    shutil.rmtree(ev_dir, ignore_errors=True)
    return {"n_fail": n_fail, "f2p_pass": f2p_pass, "n_f2p": len(f2p),
            "n_pass": sum(1 for n in nodes if statuses.get(n) == "PASSED"),
            "failing_nodes": failing[:20]}

def repo_test_files(test_patch):
    """test_patch 里涉及的测试文件（b/ 路径）。"""
    return sorted({b for _, b in re.findall(r"^diff --git a/(\S+) b/(\S+)", test_patch or "", re.M)})


def build_legal_hint(cname, test_patch):
    """合法提示：仓库自带的测试（agent 自己也能跑），展示当前失败输出。"""
    cands = repo_test_files(test_patch)
    existing = [f for f in cands
                if "YES" in dc(cname, f"cd /testbed && test -f {shlex.quote(f)} && echo YES")[1]]
    if not existing:
        return ""
    cmd = ("cd /testbed && python -m pytest -q --no-header -p no:cacheprovider "
           + " ".join(shlex.quote(f) for f in existing) + " 2>&1 | tail -40")
    _, out, err = dc(cname, cmd, timeout=1800)
    body = (out + err)[-1500:]
    return ("The repository's own test suite (which you can run yourself) currently reports:\n"
            + body + "\n\n")


def build_oracle_hint(ev_start, test_patch):
    """特权提示：隐藏评测的失败用例与测试 patch（agent 原本看不到）。"""
    parts = []
    failing = ev_start.get("failing_nodes") or []
    if failing:
        parts.append("Hidden evaluation tests currently failing:\n- "
                     + "\n- ".join(failing[:20]))
    if test_patch:
        parts.append("Hidden evaluation test patch (not visible in the repo; the grader "
                     "will apply it):\n" + test_patch[:8000])
    return ("\n\n".join(parts) + "\n\n") if parts else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--condition", required=True,
                    choices=["restart", "continue", "revert_random",
                             "revert_teacher", "revert_ours"])
    ap.add_argument("--turn", type=int, default=None, help="ours: 指定回退轮次（默认读 report）")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--model", default=os.environ.get("REPAIR_MODEL", "deepseek-chat"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hint", default="none", choices=["none", "legal", "oracle"],
                    help="续跑时给 agent 的信息：none/legal(仓库自带测试)/oracle(隐藏测试)")
    ap.add_argument("--work", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    traj = json.load(open(args.traj))
    task = load_task(args.task, args.instance_id)
    turns = extract_turns(traj)
    T = len(turns)

    report = json.load(open(args.report)) if args.report and os.path.exists(args.report) else {}
    harmful = [c["turn"] for c in (report.get("counterfactuals") or [])
               if c.get("label") in ("causal", "causal_by_file")]
    teacher_step = None
    if args.teacher and os.path.exists(args.teacher):
        teacher_step = (json.load(open(args.teacher)).get("teacher") or {}).get("pivot_step")

    random.seed(args.seed)
    target = None   # 要撤销的轮次（None 表示不撤销）
    if args.condition == "revert_random":
        target = random.randint(1, T)
    elif args.condition == "revert_teacher":
        target = int(teacher_step) if teacher_step not in (None, 0) else None
    elif args.condition == "revert_ours":
        target = args.turn or (harmful[0] if harmful else None)
        if target is None:
            raise SystemExit("该轨迹没有验证过的有害步骤，不能做 revert_ours 条件")
        target = int(target)

    work = os.path.abspath(args.work or f"/tmp/repair_{uuid.uuid4().hex[:6]}")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    base = os.path.join(work, "base")
    base_sha = get_base_repo(args.image, base)
    replay = os.path.join(work, "replay")
    shutil.copytree(base, replay, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".tox",
                                                ".nox", ".venv", "node_modules"))

    cname = f"repair_{uuid.uuid4().hex[:6]}"
    sh(["docker", "run", "-d", "--name", cname, "-v", f"{replay}:/testbed",
        "-w", "/testbed", args.image, "sleep", "8h"])
    tokens = 0
    try:
        dc(cname, "git config --global --add safe.directory /testbed")
        # 重放到最终状态（restart 除外），并记录每轮结束时的 sha
        shas = [base_sha]
        if args.condition != "restart":
            for i, cmds in enumerate(turns, 1):
                for c in cmds:
                    dc(cname, c, timeout=600)
                dc(cname, "cd /testbed && git add -A")
                dc(cname, "cd /testbed && git -c user.email=r@l -c user.name=r "
                          f"commit --allow-empty -qm turn{i}")
                _, out, _ = dc(cname, "cd /testbed && git rev-parse HEAD")
                sha = out.strip().splitlines()[-1] if out.strip() else shas[-1]
                shas.append(sha)
        # 受控撤销：把目标轮改过的文件回滚到该轮之前的版本
        revert_mode = None
        if target is not None and args.condition.startswith("revert_"):
            if target <= 0 or target > len(turns):
                target = None
            else:
                prev_sha = shas[target - 1]
                cur_sha = shas[target]
                _, names, _ = dc(cname, f"cd /testbed && git diff --name-only {prev_sha} {cur_sha}")
                files = [x for x in names.splitlines() if x.strip()]
                revert_mode = "file_checkout"
                if files:
                    quoted = " ".join(shlex.quote(f) for f in files)
                    dc(cname, f"cd /testbed && git diff {prev_sha} {cur_sha} -- {quoted} > /tmp/turn.diff")
                    rc_apply, out_apply, _ = dc(
                        cname, "cd /testbed && git apply -R --3way /tmp/turn.diff 2>&1; echo RC=$?")
                    if "RC=0" in out_apply:
                        revert_mode = "reverse_patch"
                    else:
                        # 冲突时回退到整文件 checkout（会连带回滚同文件后续修改）
                        dc(cname, "cd /testbed && git reset --hard -q HEAD")
                        dc(cname, "cd /testbed && git checkout %s -- %s" % (prev_sha, quoted))
        dc(cname, f"chown -R {os.getuid()}:{os.getgid()} /testbed")
        ev_start = eval_dir(args.image, replay, task, base_sha,
                            task.get("test_patch", ""), work, tag="start")
        _, diff_before, _ = dc(cname, f"cd /testbed && git diff {base_sha}")
        problem = task.get("problem_statement", "")[:6000]
        hint_block = ""
        if args.hint == "legal":
            hint_block = build_legal_hint(cname, task.get("test_patch", ""))
        elif args.hint == "oracle":
            hint_block = build_oracle_hint(ev_start, task.get("test_patch", ""))
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "<pr_description>\n"
                f"Consider the following PR description:\n{problem}\n"
                "</pr_description>\n\n"
                "<instructions>\n"
                "# Task Instructions\n\n"
                "## Overview\n"
                "You're a software engineer interacting continuously with a computer by "
                "submitting commands. You are continuing a previous attempt that left the "
                "repository at /testbed partially modified; your task is to make the "
                "changes needed to satisfy the PR description, in a way that is general "
                "and consistent with the codebase. Treat the current working tree as "
                "your starting point.\n\n"
                "<IMPORTANT>This is an interactive process where you will think and issue "
                "AT LEAST ONE command, see the result, then think and issue your next "
                "command(s).</IMPORTANT>\n\n"
                "For each response:\n\n"
                "1. Include a THOUGHT section explaining your reasoning and what you're "
                "trying to accomplish\n"
                "2. Provide one or more bash tool calls to execute\n\n"
                "## Important Boundaries\n\n"
                "- MODIFY: Regular source code files in /testbed (this is the working "
                "directory for all your subsequent commands)\n"
                "- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, "
                "etc.)\n\n"
                "## Recommended Workflow\n\n"
                "1. Analyze the codebase, including the uncommitted diff left by the "
                "previous attempt\n"
                "2. Create a script to reproduce the issue\n"
                "3. Edit the source code to resolve the issue\n"
                "4. Verify your fix works by running your script again\n"
                "5. Test edge cases to ensure your fix is robust\n\n"
                "## Command Execution Rules\n\n"
                "1. You issue at least one command\n"
                "2. The system executes the command(s) in a subshell\n"
                "3. You see the result(s)\n"
                "4. You write your next command(s)\n\n"
                "Your response MUST include AT LEAST ONE bash tool call. You can make "
                "MULTIPLE tool calls in a single response when the commands are "
                "independent. Directory or environment variable changes are not "
                "persistent: every action is executed in a new subshell.\n\n"
                "## Current State\n\n"
                "Uncommitted diff left by the previous attempt (truncated):\n"
                f"{diff_before[:6000] or '(empty)'}\n\n"
                "## Step Budget\n\n"
                f"You have at most {args.steps} responses (each may contain multiple "
                "commands). When you believe the fix is complete, run "
                "`git -C /testbed diff` and then submit by replying with exactly:\n"
                "```bash\necho DONE\n```\n"
                + (f"\n## Additional information\n\n{hint_block}\n" if hint_block else "")
                + "</instructions>"}]
        steps_used = 0
        commands_executed = 0
        parse_fail = 0
        conv = []
        for i in range(args.steps):
            text, tk = call_llm(messages, args.model)
            tokens += tk
            conv.append({"role": "assistant", "content": text})
            cmds = parse_commands(text)
            if not cmds:
                parse_fail += 1
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content":
                    "Your reply must contain at least one ```bash ...``` block with your "
                    "command(s), and nothing else outside the reasoning text. Example:\n"
                    "```bash\nls -la /testbed\n```"})
                continue
            messages.append({"role": "assistant", "content": text})
            steps_used += 1
            obs_parts = []
            done = False
            for cmd in cmds:
                if "echo DONE" in cmd:
                    done = True
                    obs_parts.append("(submitted)")
                    break
                rc, out, err = dc(cname, cmd, timeout=600)
                commands_executed += 1
                raw = out + err
                if len(raw) <= 10000:
                    obs = raw
                else:
                    obs = (raw[:5000] + f"\n... [{len(raw) - 10000} chars elided] ...\n"
                           + raw[-5000:])
                obs_parts.append(f"exit={rc}\n{obs}")
                conv.append({"role": "observation", "content": f"exit={rc}\n{obs}"})
            if done:
                break
            messages.append({"role": "user", "content":
                "\n\n".join(obs_parts)
                + f"\n\nStep {steps_used}/{args.steps}. Next command(s)?"})
        dc(cname, f"chown -R {os.getuid()}:{os.getgid()} /testbed")
        _, st, _ = dc(cname, "cd /testbed && git status --short | head -30")
        edited_files = []
        for ln in st.splitlines():
            parts = ln.split(maxsplit=1)
            if len(parts) == 2 and parts[0][0] in "MA":
                edited_files.append(parts[1].strip())
        ev = eval_dir(args.image, replay, task, base_sha,
                      task.get("test_patch", ""), work, tag="final")
    finally:
        sh(["docker", "rm", "-f", cname])
        shutil.rmtree(work, ignore_errors=True)

    result = {"instance_id": args.instance_id, "condition": args.condition,
              "target_turn": target, "n_turns_total": T,
              "harmful_turns": harmful, "teacher_step": teacher_step,
              "steps_budget": args.steps, "steps_used": steps_used,
              "commands_executed": commands_executed,
              "parse_failures": parse_fail, "tokens": tokens, **ev,
              "start_n_fail": ev_start["n_fail"],
              "start_f2p_pass": ev_start["f2p_pass"],
              "start_failing_nodes": ev_start["failing_nodes"],
              "revert_mode": revert_mode, "hint": args.hint, "prompt_mode": "miniswe",
              "edited_files": edited_files[:30], "did_edit": bool(edited_files)}
    try:
        json.dump(conv, open(args.out + ".conv.json", "w"), ensure_ascii=False, indent=1)
    except Exception:
        pass
    json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
