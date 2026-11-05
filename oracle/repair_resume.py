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
SYSTEM = ("You are an autonomous software engineer fixing a bug in a repository at /testbed. "
          "You can run bash commands. Reply with EXACTLY ONE bash command inside a ```bash "
          "code block, then wait for its output. When you believe the fix is complete, run "
          "```bash\necho DONE\n``` and stop.")


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
    model = model.split("/")[-1]
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        envp = os.path.expanduser("~/.config/mini-swe-agent/.env")
        if os.path.exists(envp):
            for line in open(envp):
                if line.strip().startswith("DEEPSEEK_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY 未设置")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
    resp = client.chat.completions.create(model=model, messages=messages,
                                          temperature=0.7, max_tokens=600)
    return resp.choices[0].message.content, resp.usage.total_tokens if resp.usage else 0


def parse_command(text):
    m = re.search(r"```bash\s*\n(.*?)```", text, re.S)
    if m:
        cmd = m.group(1).strip()
        lines = [ln for ln in cmd.splitlines() if ln.strip()]
        return "\n".join(lines[:3]) if lines else ""
    m = re.search(r"```\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    return ""


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--condition", required=True,
                    choices=["restart", "random", "teacher", "ours"])
    ap.add_argument("--turn", type=int, default=None, help="ours: 指定回退轮次（默认读 report）")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--seed", type=int, default=0)
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
    if args.condition == "restart":
        start = 0
    elif args.condition == "random":
        start = random.randint(1, max(1, T))
    elif args.condition == "teacher":
        start = int(teacher_step) if teacher_step not in (None, 0) else 0
    else:
        t = args.turn or (harmful[0] if harmful else None)
        if t is None:
            raise SystemExit("该轨迹没有验证过的有害步骤，不能做 ours 条件")
        start = int(t)

    work = os.path.abspath(args.work or f"/tmp/repair_{uuid.uuid4().hex[:6]}")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    base = os.path.join(work, "base")
    base_sha = get_base_repo(args.image, base)
    replay = os.path.join(work, "replay")
    shutil.copytree(base, replay, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    cname = f"repair_{uuid.uuid4().hex[:6]}"
    sh(["docker", "run", "-d", "--name", cname, "-v", f"{replay}:/testbed",
        "-w", "/testbed", args.image, "sleep", "8h"])
    tokens = 0
    try:
        dc(cname, "git config --global --add safe.directory /testbed")
        # 回放到 start 之前
        for cmds in turns[:start]:
            for c in cmds:
                dc(cname, c, timeout=600)
        dc(cname, f"chown -R {os.getuid()}:{os.getgid()} /testbed")
        _, diff_before, _ = git(["diff", base_sha], replay)
        problem = task.get("problem_statement", "")[:4000]
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                f"Issue:\n{problem}\n\n"
                f"You are continuing from a partially completed attempt. "
                f"Current uncommitted diff (truncated):\n{diff_before[:3000] or '(empty)'}\n\n"
                f"Steps: you have at most {args.steps} commands. "
                f"Start by inspecting the relevant files, then make the fix."}]
        steps_used = 0
        parse_fail = 0
        conv = []
        for i in range(args.steps):
            text, tk = call_llm(messages, args.model)
            tokens += tk
            conv.append({"role": "assistant", "content": text})
            cmd = parse_command(text)
            if not cmd:
                parse_fail += 1
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content":
                    "Your reply must contain EXACTLY ONE ```bash ...``` block with one "
                    "command, and nothing else."})
                continue
            messages.append({"role": "assistant", "content": text})
            if "echo DONE" in cmd:
                break
            rc, out, err = dc(cname, cmd, timeout=600)
            steps_used += 1
            obs = (out + err)[:1500]
            messages.append({"role": "user", "content":
                f"exit={rc}\n{obs}\n\nStep {steps_used}/{args.steps}. Next command?"})
            conv.append({"role": "observation", "content": f"exit={rc}\n{obs}"})
        dc(cname, f"chown -R {os.getuid()}:{os.getgid()} /testbed")
        ev = eval_in_container(cname, task, base_sha, task.get("test_patch", ""))
    finally:
        sh(["docker", "rm", "-f", cname])
        shutil.rmtree(work, ignore_errors=True)

    result = {"instance_id": args.instance_id, "condition": args.condition,
              "start_turn": start, "n_turns_total": T,
              "harmful_turns": harmful, "teacher_step": teacher_step,
              "steps_budget": args.steps, "steps_used": steps_used,
              "parse_failures": parse_fail, "tokens": tokens, **ev}
    try:
        json.dump(conv, open(args.out + ".conv.json", "w"), ensure_ascii=False, indent=1)
    except Exception:
        pass
    json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
