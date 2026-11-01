#!/usr/bin/env python3
"""replay_traj.py — 重放 mini-swe-agent 轨迹，得到每轮测试状态，并做 edit 级归因。

v2 改动：
  - 每轮在容器内 git commit，记录 sha（状态可精确复现，不用存快照）
  - 可选 --attribute：从最终状态出发，逐轮回滚该轮的编辑，看测试是否变化
    （找到「撤掉它，失败消失」的编辑 = 有害关键编辑）

流程：
  1. 从镜像取出 base repo（或 --repo），确保有 git
  2. 起持久容器挂载 repo；按 assistant 消息逐轮执行 bash 命令
  3. 每轮结束：容器内 git add -A + commit → 记 sha
  4. 对每个 sha：拷 repo → checkout → 打 test_patch → 跑 pytest → 记 n_fail/n_pass
  5. --attribute：对状态变化的轮次，从最终状态回滚该轮编辑 → 重跑测试 → 判断它是否关键

用法：
  python replay_traj.py \
    --traj mswa_out/<iid>/<iid>.traj.json \
    --task SWE-bench_Lite.jsonl --instance-id <iid> \
    --image swebench/sweb.eval.x86_64.<iid_docker>:latest \
    --work replay_work --out out_traj --attribute
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
GIT = ["git", "-c", "user.email=replay@local", "-c", "user.name=replay"]


def sh(args, cwd=None, timeout=600, check=False):
    t0 = time.perf_counter()
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = 124, "", "TIMEOUT"
    dt = time.perf_counter() - t0
    if check and rc != 0:
        raise RuntimeError(f"{args}\n{out[-400:]}\n{err[-400:]}")
    return rc, out, err, dt


def git(args, cwd, timeout=180):
    return sh(GIT + args, cwd=cwd, timeout=timeout)


def ensure_git(base):
    if not os.path.isdir(os.path.join(base, ".git")):
        git(["init", "-q"], base)
    git(["add", "-A"], base)
    git(["commit", "-qm", "base"], base)
    rc, out, _, _ = git(["rev-parse", "HEAD"], base)
    return out.strip()


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
    return ensure_git(dest)


def run_tests(image, workdir, nodes, timeout=600):
    inner = "pytest --color=no -rA -p no:cacheprovider --no-header " + \
            " ".join(shlex.quote(n) for n in nodes)
    rc, out, err, dt = sh(["docker", "run", "--rm", "-v", f"{workdir}:/testbed",
                           "-w", "/testbed", image, "bash", "-lc", inner],
                          timeout=timeout)
    statuses = {m.group(2): m.group(1) for m in STATUS_RE.finditer(out + err)}
    return {"ok": rc == 0, "statuses": statuses, "seconds": round(dt, 2)}


def eval_state(image, base, ev, sha, test_patch, nodes, out):
    shutil.rmtree(ev, ignore_errors=True)
    shutil.copytree(base, ev, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    rc, o, e, _ = git(["checkout", "-q", sha], ev)
    if rc != 0:
        return {"sha": sha, "checkout_ok": False, "err": e[:200],
                "test_patch_ok": False, "n_fail": None, "n_pass": None,
                "test_s": 0.0, "statuses": {}}
    tp_ok, tp_err = True, ""
    if test_patch:
        tp = os.path.join(out, "test_patch.diff")
        open(tp, "w").write(test_patch)
        rc, o, e, _ = git(["apply", "--whitespace=nowarn", tp], ev)
        tp_ok, tp_err = rc == 0, e[:300]
    res = run_tests(image, ev, nodes)
    n_pass = sum(1 for n in nodes if res["statuses"].get(n) == "PASSED")
    n_fail = sum(1 for n in nodes if res["statuses"].get(n) not in ("PASSED", "XFAIL"))
    shutil.rmtree(ev, ignore_errors=True)
    return {"sha": sha, "checkout_ok": True, "test_patch_ok": tp_ok,
            "test_patch_err": tp_err, "n_pass": n_pass, "n_fail": n_fail,
            "test_ok": res["ok"], "test_s": res["seconds"],
            "statuses": res["statuses"]}


def make_report(args, f2p, p2p, evals, pivots, attrs, replay_seconds):
    L = [f"# 轨迹重放报告：{args.instance_id}", "",
         f"- turns = {len(evals)}，F2P = {len(f2p)}，P2P = {len(p2p)}",
         f"- 单次判分中位数 = {statistics.median([e['test_s'] for e in evals]):.2f}s",
         f"- 重放 + 判分总耗时 = {replay_seconds} s", "",
         "## 测试状态轨迹", "",
         "| turn | n_fail | n_pass | test_patch | test_s | changed | sha |",
         "|---|---|---|---|---|---|---|"]
    prev = None
    for ev in evals:
        ch = "" if prev is None else ("↑" if ev["n_fail"] > prev else
                                      ("↓" if ev["n_fail"] < prev else "="))
        L.append(f"| {ev['turn']} | {ev['n_fail']} | {ev['n_pass']} | "
                 f"{ev.get('test_patch_ok')} | {ev['test_s']} | {ch} | `{ev['sha'][:8]}` |")
        prev = ev["n_fail"]
    L += ["", "## 候选 pivot", ""]
    if pivots:
        L += [f"- turn {p['turn']}: {p['type']}（{p['from']} → {p['to']} failing）"
              for p in pivots]
    else:
        L.append("- 无（轨迹单调改善或不变）")
    if attrs:
        L += ["", "## edit 级归因（从最终状态回滚该轮编辑）", "",
              "| turn | revert_ok | n_fail(final) | n_fail(after) | delta | 判断 |",
              "|---|---|---|---|---|---|"]
        for a in attrs:
            if not a.get("revert_ok"):
                L.append(f"| {a['turn']} | ❌ | {a.get('n_fail_final')} | — | — | 回滚失败 |")
                continue
            delta = (a["n_fail_after"] - a["n_fail_final"]) if a["n_fail_after"] is not None else None
            if delta is None:
                verdict = "无数据"
            elif delta > 0:
                verdict = "**有害编辑**（撤掉后失败增加）" if a["n_fail_final"] > 0 else "好编辑"
            elif delta < 0:
                verdict = "好编辑（撤掉后失败减少）"
            else:
                verdict = "无影响（惰性编辑）"
            L.append(f"| {a['turn']} | ✅ | {a['n_fail_final']} | {a['n_fail_after']} | "
                     f"{delta} | {verdict} |")
    open(os.path.join(args.out, "TRAJ.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="")
    ap.add_argument("--skip-pytest-replay", action="store_true")
    ap.add_argument("--cmd-timeout", type=int, default=300)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--attribute", action="store_true")
    ap.add_argument("--attribute-all", action="store_true")
    args = ap.parse_args()

    traj = json.load(open(args.traj))
    task = load_task(args.task, args.instance_id)
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if isinstance(task.get(k), str):
            task[k] = json.loads(task[k])
    f2p, p2p = task["FAIL_TO_PASS"], task["PASS_TO_PASS"]
    nodes = list(f2p) + list(p2p)
    test_patch = task.get("test_patch")

    args.work = os.path.abspath(args.work)
    args.out = os.path.abspath(args.out)
    shutil.rmtree(args.work, ignore_errors=True)
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    base = os.path.join(args.work, "repo")
    if args.repo:
        shutil.copytree(args.repo, base, ignore=shutil.ignore_patterns(".git"))
        base_sha = ensure_git(base)
    else:
        base_sha = get_base_repo(args.image, base)

    cname = f"traj_replay_{uuid.uuid4().hex[:8]}"
    sh(["docker", "run", "-d", "--name", cname, "-v", f"{base}:/testbed",
        "-w", "/testbed", args.image, "sleep", "8h"], check=True)
    print("container:", cname)
    dc(cname, "git config --global --add safe.directory /testbed")

    # 0) baseline：还没执行任何命令时的状态
    base_eval = eval_state(args.image, base, os.path.join(args.work, "eval"),
                           base_sha, test_patch, nodes, args.out)
    base_eval["turn"] = 0
    evals = [base_eval]
    shas = []
    print(f"baseline: n_fail={base_eval['n_fail']} n_pass={base_eval['n_pass']} "
          f"test_patch_ok={base_eval['test_patch_ok']} ({base_eval['test_s']}s)")

    turns = extract_turns(traj)
    if args.max_turns:
        turns = turns[:args.max_turns]
    print(f"trajectory: {len(turns)} turns, {sum(len(t) for t in turns)} commands")

    t0 = time.perf_counter()
    try:
        for i, cmds in enumerate(turns, 1):
            logs = []
            for c in cmds:
                if args.skip_pytest_replay and "pytest" in c:
                    logs.append({"skipped_pytest": True})
                    continue
                rc, o, e, dt = dc(cname, c, timeout=args.cmd_timeout)
                logs.append({"cmd": c, "rc": rc, "seconds": round(dt, 2)})
            json.dump(logs, open(os.path.join(args.out, f"turn{i:02d}_replay.json"), "w"),
                      ensure_ascii=False, indent=1)

            dc(cname, "cd /testbed && git config --global --add safe.directory /testbed")
            dc(cname, "cd /testbed && git add -A")
            rc, _, _, _ = dc(cname, "cd /testbed && git diff --cached --quiet")
            if rc != 0:
                rc2, o2, e2, _ = dc(cname, "cd /testbed && git -c user.email=r@l "
                                          f"-c user.name=r commit -qm turn{i}")
                if rc2 != 0:
                    print(f"  ⚠️ commit 失败: {e2[:200]}")
            rc, out, err, _ = dc(cname, "cd /testbed && git rev-parse HEAD")
            sha = out.strip().splitlines()[-1] if out.strip() else base_sha
            shas.append(sha)
            # 把容器里 root 新建的文件还给宿主用户，避免后续 checkout/apply 权限问题
            dc(cname, f"chown -R {os.getuid()}:{os.getgid()} /testbed")

            ev = eval_state(args.image, base, os.path.join(args.work, "eval"),
                            sha, test_patch, nodes, args.out)
            ev["turn"] = i
            ev["n_cmds"] = len(cmds)
            evals.append(ev)
            print(f"turn {i:02d}: {len(cmds)} cmds → n_fail={ev['n_fail']} "
                  f"n_pass={ev['n_pass']} ({ev['test_s']}s)")
    finally:
        sh(["docker", "rm", "-f", cname])

    pivots, prev = [], None
    for ev in evals:
        if prev is not None and ev["n_fail"] is not None and prev["n_fail"] is not None:
            if ev["n_fail"] > prev["n_fail"]:
                pivots.append({"turn": ev["turn"], "type": "regression",
                               "from": prev["n_fail"], "to": ev["n_fail"]})
            elif ev["n_fail"] < prev["n_fail"]:
                pivots.append({"turn": ev["turn"], "type": "improvement",
                               "from": prev["n_fail"], "to": ev["n_fail"]})
        prev = ev

    attrs = []
    if args.attribute and evals:
        final = evals[-1]
        target_turns = [e["turn"] for e in evals
                        if args.attribute_all or
                        any(p["turn"] == e["turn"] for p in pivots)]
        for tno in target_turns:
            idx = tno - 1
            prev_sha = base_sha if idx == 0 else shas[idx - 1]
            cur_sha = shas[idx]
            rc, patch, _, _ = git(["diff", prev_sha, cur_sha], base)
            if rc != 0 or not patch.strip():
                attrs.append({"turn": tno, "revert_ok": False, "err": "empty diff",
                              "n_fail_final": final["n_fail"], "n_fail_after": None})
                continue
            pp = os.path.join(args.out, f"turn{tno:02d}_edit.diff")
            open(pp, "w").write(patch)
            ev = os.path.join(args.work, "attr")
            shutil.rmtree(ev, ignore_errors=True)
            shutil.copytree(base, ev, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            git(["checkout", "-q", final["sha"]], ev)
            rc, o, e, _ = git(["apply", "-R", "--3way", pp], ev)
            if rc != 0:
                attrs.append({"turn": tno, "revert_ok": False, "err": e[:200],
                              "n_fail_final": final["n_fail"], "n_fail_after": None})
                shutil.rmtree(ev, ignore_errors=True)
                continue
            if test_patch:
                tp = os.path.join(args.out, "test_patch.diff")
                open(tp, "w").write(test_patch)
                git(["apply", "--whitespace=nowarn", tp], ev)
            res = run_tests(args.image, ev, nodes)
            n_fail_after = sum(1 for n in nodes
                               if res["statuses"].get(n) not in ("PASSED", "XFAIL"))
            attrs.append({"turn": tno, "revert_ok": True,
                          "n_fail_final": final["n_fail"],
                          "n_fail_after": n_fail_after,
                          "statuses_after": res["statuses"]})
            shutil.rmtree(ev, ignore_errors=True)
            print(f"attr turn {tno}: final_fail={final['n_fail']} "
                  f"after_revert={n_fail_after}")

    report = {"instance_id": args.instance_id, "image": args.image,
              "base_sha": base_sha, "n_turns": len(evals), "evals": evals,
              "pivots": pivots, "attrs": attrs,
              "eval_seconds": [e["test_s"] for e in evals],
              "replay_seconds": round(time.perf_counter() - t0, 1)}
    json.dump(report, open(os.path.join(args.out, "traj_report.json"), "w"),
              ensure_ascii=False, indent=1)
    make_report(args, f2p, p2p, evals, pivots, attrs, report["replay_seconds"])
    print(f"\n报告 → {args.out}/TRAJ.md")


if __name__ == "__main__":
    main()
