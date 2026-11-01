#!/usr/bin/env python3
"""replay_traj.py — 重放 mini-swe-agent 轨迹，得到每轮的测试状态轨迹。

流程：
  1. 从镜像取出 base repo（或 --repo 指定），确保有 git
  2. 起持久容器，挂载 repo
  3. 按 assistant 消息逐轮执行 bash 命令；每轮结束后把 /testbed 拷出来
  4. 对每一轮的快照：打 test_patch → 跑 pytest → 记录 n_fail/n_pass
  5. 输出状态轨迹 + 候选 pivot（n_fail 上升的轮次）

用法：
  python replay_traj.py \
    --traj mswa_out/psf__requests-3362/psf__requests-3362.traj.json \
    --task SWE-bench_Lite.jsonl --instance-id psf__requests-3362 \
    --image swebench/sweb.eval.x86_64.psf_1776_requests-3362:latest \
    --work /workspaces/agentic-rl-swe/replay_work \
    --out /workspaces/agentic-rl-swe/out_traj
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import time
import uuid

STATUS_RE = re.compile(r"^(PASSED|FAILED|ERROR)\s+(\S+)", re.M)
GIT_ENV = ["git", "-c", "user.email=replay@local", "-c", "user.name=replay"]


def sh(args, cwd=None, timeout=600, check=False):
    t0 = time.perf_counter()
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = 124, "", "TIMEOUT"
    dt = time.perf_counter() - t0
    if check and rc != 0:
        raise RuntimeError(f"{args}\n{out[-500:]}\n{err[-500:]}")
    return rc, out, err, dt


def git(args, cwd, timeout=180):
    return sh(GIT_ENV + args, cwd=cwd, timeout=timeout)


def ensure_git(base):
    if not os.path.isdir(os.path.join(base, ".git")):
        git(["init", "-q"], base)
    git(["add", "-A"], base)
    git(["commit", "-qm", "base"], base)


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
    raise SystemExit(f"task 未找到: {iid}")


def extract_turns(traj):
    turns = []
    for msg in traj.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        cmds = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name") != "bash":
                continue
            try:
                a = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                continue
            if a.get("command"):
                cmds.append(a["command"])
        if cmds:
            turns.append(cmds)
    return turns


def get_base_repo(image, dest):
    rc, out, err, _ = sh(["docker", "create", image], check=True)
    cid = out.strip()
    try:
        sh(["docker", "cp", f"{cid}:/testbed", dest], timeout=300, check=True)
    finally:
        sh(["docker", "rm", "-f", cid])
    ensure_git(dest)


def run_tests(image, workdir, nodes, timeout=600):
    inner = "pytest --color=no -rA -p no:cacheprovider --no-header " + \
            " ".join(shlex.quote(n) for n in nodes)
    rc, out, err, dt = sh(["docker", "run", "--rm", "-v", f"{workdir}:/testbed",
                           "-w", "/testbed", image, "bash", "-lc", inner],
                          timeout=timeout)
    statuses = {m.group(2): m.group(1) for m in STATUS_RE.finditer(out + err)}
    return {"ok": rc == 0, "statuses": statuses, "seconds": round(dt, 2),
            "tail": (out + err).strip().splitlines()[-3:]}


def eval_snapshot(image, snap, ev, test_patch, nodes, out):
    shutil.rmtree(ev, ignore_errors=True)
    shutil.copytree(snap, ev)
    tp_ok, tp_err = True, ""
    if test_patch:
        tp = os.path.join(out, "test_patch.diff")
        open(tp, "w").write(test_patch)
        rc, o, e, _ = git(["apply", "--whitespace=nowarn", tp], ev)
        tp_ok, tp_err = rc == 0, e[:300]
    res = run_tests(image, ev, nodes)
    n_pass = sum(1 for n in nodes if res["statuses"].get(n) == "PASSED")
    n_fail = sum(1 for n in nodes if res["statuses"].get(n) not in ("PASSED", "XFAIL"))
    return {"test_patch_ok": tp_ok, "test_patch_err": tp_err, "n_pass": n_pass,
            "n_fail": n_fail, "test_ok": res["ok"], "test_s": res["seconds"],
            "statuses": res["statuses"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="", help="可选：直接给 base repo（否则从镜像取）")
    ap.add_argument("--skip-pytest-replay", action="store_true")
    ap.add_argument("--cmd-timeout", type=int, default=300)
    args = ap.parse_args()

    traj = json.load(open(args.traj))
    task = load_task(args.task, args.instance_id)
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if isinstance(task.get(k), str):
            task[k] = json.loads(task[k])
    f2p, p2p = task["FAIL_TO_PASS"], task["PASS_TO_PASS"]
    nodes = list(f2p) + list(p2p)
    test_patch = task.get("test_patch")

    work = os.path.abspath(args.work)
    out = os.path.abspath(args.out)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    base = os.path.join(work, "repo")
    if args.repo:
        shutil.copytree(args.repo, base, ignore=shutil.ignore_patterns(".git"))
        ensure_git(base)
    else:
        get_base_repo(args.image, base)

    cname = f"traj_replay_{uuid.uuid4().hex[:8]}"
    rc, _, err, _ = sh(["docker", "run", "-d", "--name", cname, "-v",
                        f"{base}:/testbed", "-w", "/testbed", args.image, "sleep", "6h"],
                       check=True)
    print("container:", cname)

    turns = extract_turns(traj)
    print(f"trajectory: {len(turns)} turns, {sum(len(t) for t in turns)} commands")

    evals = []
    t0 = time.perf_counter()
    try:
        for i, cmds in enumerate(turns, 1):
            logs = []
            for c in cmds:
                if args.skip_pytest_replay and "pytest" in c:
                    logs.append({"skipped_pytest": True, "cmd": c})
                    continue
                rc, o, e, dt = sh(["docker", "exec", cname, "bash", "-lc", c],
                                  timeout=args.cmd_timeout)
                logs.append({"cmd": c, "rc": rc, "seconds": round(dt, 2),
                             "tail": (o + e).strip().splitlines()[-2:]})
            json.dump(logs, open(os.path.join(out, f"turn{i:02d}_replay.json"), "w"),
                      ensure_ascii=False, indent=1)

            snap = os.path.join(work, "current")
            shutil.rmtree(snap, ignore_errors=True)
            sh(["docker", "cp", f"{cname}:/testbed", snap], timeout=300)
            ev = eval_snapshot(args.image, snap, os.path.join(work, "eval"),
                               test_patch, nodes, out)
            shutil.rmtree(snap, ignore_errors=True)
            ev["turn"] = i
            ev["n_cmds"] = len(cmds)
            evals.append(ev)
            print(f"turn {i:02d}: {len(cmds)} cmds → n_fail={ev['n_fail']} "
                  f"n_pass={ev['n_pass']} ({ev['test_s']}s)")
    finally:
        sh(["docker", "rm", "-f", cname])

    pivots = []
    prev = None
    for ev in evals:
        if prev is not None:
            if ev["n_fail"] > prev["n_fail"]:
                pivots.append({"turn": ev["turn"], "type": "regression",
                               "from": prev["n_fail"], "to": ev["n_fail"]})
            elif ev["n_fail"] < prev["n_fail"]:
                pivots.append({"turn": ev["turn"], "type": "improvement",
                               "from": prev["n_fail"], "to": ev["n_fail"]})
        prev = ev

    report = {"instance_id": args.instance_id, "image": args.image,
              "n_turns": len(evals), "evals": evals, "pivots": pivots,
              "eval_seconds": [e["test_s"] for e in evals],
              "replay_seconds": round(time.perf_counter() - t0, 1)}
    json.dump(report, open(os.path.join(out, "traj_report.json"), "w"),
              ensure_ascii=False, indent=1)

    L = [f"# 轨迹重放报告：{args.instance_id}", "",
         f"- turns = {len(evals)}，F2P = {len(f2p)}，P2P = {len(p2p)}",
         f"- 单次判分中位数 = {statistics.median(report['eval_seconds']):.2f}s"
         f"（含容器启动）", "",
         "## 测试状态轨迹", "",
         "| turn | n_fail | n_pass | test_s | changed | replay_s |",
         "|---|---|---|---|---|---|"]
    prev_fail = None
    for ev in evals:
        ch = "" if prev_fail is None else ("↑" if ev["n_fail"] > prev_fail else
                                           ("↓" if ev["n_fail"] < prev_fail else "="))
        L.append(f"| {ev['turn']} | {ev['n_fail']} | {ev['n_pass']} | {ev['test_s']} | "
                 f"{ch} | |")
        prev_fail = ev["n_fail"]
    L += ["", "## 候选 pivot", ""]
    if pivots:
        L += [f"- turn {p['turn']}: {p['type']}（{p['from']} → {p['to']} failing）"
              for p in pivots]
    else:
        L.append("- 无（轨迹单调改善或不变）")
    L += ["", f"重放总耗时：{report['replay_seconds']} s（含判分）"]
    open(os.path.join(out, "TRAJ.md"), "w").write("\n".join(L) + "\n")
    print(f"\n报告 → {out}/TRAJ.md")


if __name__ == "__main__":
    main()
