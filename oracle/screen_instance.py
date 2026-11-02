#!/usr/bin/env python3
"""screen_instance.py — 实例准入筛查（跑轨迹之前用）。

判定实例是否值得做：在 **base 状态 + test_patch** 下跑官方语义的整文件 pytest，
检查 baseline 的失败数是否等于 F2P 数（否则标签就是噪声）。

用法：
  python screen_instance.py \
    --task SWE-bench_Lite.jsonl --instance-id pallets__flask-5063 \
    --image swebench/sweb.eval.x86_64.pallets_1776_flask-5063:latest \
    --work /workspaces/agentic-rl-swe/screen_f5063 \
    --out /workspaces/agentic-rl-swe/screen_f5063_out --rerun 1

输出：VERDICT（OK / BAD_LABELS / TEST_PATCH_FAIL）+ 逐项 F2P 状态
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time

STATUS_RE = re.compile(r"^(PASSED|FAILED|ERROR)\s+(\S+)", re.M)
GIT = ["git", "-c", "user.email=screen@local", "-c", "user.name=screen"]


def sh(args, cwd=None, timeout=600):
    t0 = time.perf_counter()
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr, time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT", time.perf_counter() - t0


def git(args, cwd, timeout=180):
    return sh(GIT + args, cwd=cwd, timeout=timeout)


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


def run_pytest(image, workdir, files, timeout=900):
    inner = "pytest --color=no -rA -p no:cacheprovider --no-header " + \
            " ".join(shlex.quote(f) for f in files)
    rc, out, err, dt = sh(["docker", "run", "--rm", "-v", f"{workdir}:/testbed",
                           "-w", "/testbed", image, "bash", "-lc", inner],
                          timeout=timeout)
    statuses = {m.group(2): m.group(1) for m in STATUS_RE.finditer(out + err)}
    return statuses, round(dt, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rerun", type=int, default=0)
    args = ap.parse_args()

    task = load_task(args.task, args.instance_id)
    f2p = task["FAIL_TO_PASS"]
    p2p = task["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    nodes = list(f2p) + list(p2p)
    files = sorted({n.split("::", 1)[0] for n in nodes})

    work = os.path.abspath(args.work)
    out = os.path.abspath(args.out)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    # 从镜像取 base repo
    rc, cid, err, _ = sh(["docker", "create", args.image])
    if rc != 0:
        raise SystemExit(f"docker create 失败（镜像没拉？）: {err[:300]}")
    cid = cid.strip()
    base = os.path.join(work, "repo")
    try:
        sh(["docker", "cp", f"{cid}:/testbed", base], timeout=300)
    finally:
        sh(["docker", "rm", "-f", cid])
    if not os.path.isdir(os.path.join(base, ".git")):
        git(["init", "-q"], base)
    git(["add", "-A"], base)
    git(["commit", "-qm", "base"], base)

    # 官方语义：还原测试文件（这里就是 base）→ 打 test_patch
    tp_ok, tp_err = True, ""
    if task.get("test_patch"):
        tp = os.path.join(out, "test_patch.diff")
        open(tp, "w").write(task["test_patch"])
        rc, o, e, _ = git(["apply", "--whitespace=nowarn", tp], base)
        tp_ok, tp_err = rc == 0, e[:300]
        git(["add", "-A"], base)
        git(["commit", "-qm", "test_patch"], base)

    statuses, test_s = run_pytest(args.image, base, files)
    runs = [statuses]
    for _ in range(args.rerun):
        s2, dt = run_pytest(args.image, base, files)
        runs.append(s2)
        test_s = round(test_s + dt, 2)

    final = {}
    flaky = []
    for n in nodes:
        obs = [r.get(n) for r in runs]
        passed = [s for s in obs if s == "PASSED"]
        failed = [s for s in obs if s not in ("PASSED", "XFAIL")]
        if passed and failed:
            final[n] = "FLAKY"
            flaky.append(n)
        elif passed:
            final[n] = "PASSED"
        elif failed:
            final[n] = obs[0] or "MISSING"
    f2p_fail = [n for n in f2p if final.get(n) not in ("PASSED", "XFAIL", "FLAKY")]
    p2p_fail = [n for n in p2p if final.get(n) not in ("PASSED", "XFAIL", "FLAKY")]

    if not tp_ok:
        verdict = "TEST_PATCH_FAIL"
    elif len(f2p_fail) == len(f2p) and not p2p_fail:
        verdict = "OK"
    else:
        verdict = "BAD_LABELS"

    rep = {"instance_id": args.instance_id, "image": args.image, "verdict": verdict,
           "n_f2p": len(f2p), "n_f2p_fail": len(f2p_fail),
           "n_p2p": len(p2p), "n_p2p_fail": len(p2p_fail),
           "n_flaky": len(flaky), "test_patch_ok": tp_ok,
           "test_patch_err": tp_err, "test_s": test_s,
           "f2p_status": {n: final.get(n, "MISSING") for n in f2p},
           "p2p_failed": p2p_fail[:20], "flaky": flaky[:20]}
    json.dump(rep, open(os.path.join(out, "screen_report.json"), "w"),
              ensure_ascii=False, indent=1)

    print(f"[{args.instance_id}] VERDICT = {verdict}")
    print(f"  F2P: {len(f2p_fail)}/{len(f2p)} 在 base 失败（期望全部失败）")
    print(f"  P2P: {len(p2p_fail)}/{len(p2p)} 在 base 失败（期望 0）")
    print(f"  flaky: {len(flaky)} | test_patch_ok={tp_ok} | 判分 {test_s}s")
    for n in f2p:
        print(f"  {final.get(n, 'MISSING'):8s} {n}")


if __name__ == "__main__":
    main()
