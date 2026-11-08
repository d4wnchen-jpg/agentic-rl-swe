#!/usr/bin/env python3
"""check_fidelity.py — Q1：重放保真度检查。

做的事：
  1. 在容器里重放轨迹的全部 bash 命令（**不评测**，只重建最终状态）
  2. 从干净 base 复制一份，应用 agent 提交的 patch（preds.json 的 model_patch）
  3. 比较「重放最终状态」与「提交 patch 应用后的状态」：
     - 文件集合、逐文件是否一致
  4. 可选 --eval：对重放最终状态跑测试，与 traj_report 的 final n_fail 对比

用法：
  python check_fidelity.py \
    --traj mswa_g2a_1/pallets__flask-4045/pallets__flask-4045.traj.json \
    --task SWE-bench_Lite.jsonl --instance-id pallets__flask-4045 \
    --image swebench/sweb.eval.x86_64.pallets_1776_flask-4045:latest \
    --preds mswa_g2a_1/preds.json \
    --report out_g2a_1/traj_report.json \
    --work fidelity_1 --out fidelity_out_1
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shlex
import shutil
import subprocess
import time
import uuid

GIT = ["git", "-c", "user.email=fid@local", "-c", "user.name=fid"]


def sh(args, cwd=None, timeout=600):
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"


def docker_exec(cname, cmd, timeout=300):
    return sh(["docker", "exec", cname, "bash", "-lc", cmd], timeout=timeout)


def get_base(image, dest):
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


def extract_commands(traj):
    cmds = []
    for m in traj.get("messages", []):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                a = json.loads(fn.get("arguments", "{}"))
            except Exception:
                continue
            if fn.get("name") == "bash" and a.get("command"):
                cmds.append(a["command"])
    return cmds


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


def tree_diff(a, b, ignore=("__pycache__", ".git", ".pytest_cache", "*.pyc"),
              ignore_files=("patch.txt",)):
    """返回不一致的文件列表（相对路径）。"""
    diff = []
    for root, dirs, files in os.walk(a):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", ".pytest_cache")]
        rel = os.path.relpath(root, a)
        for fn in files:
            if fn.endswith(".pyc") or fn in ignore_files:
                continue
            pa = os.path.join(root, fn)
            pb = os.path.join(b, rel, fn)
            if not os.path.exists(pb):
                diff.append(rel + "/" + fn)
                continue
            if not filecmp.cmp(pa, pb, shallow=False):
                diff.append(rel + "/" + fn)
    # b 里多出的文件
    for root, dirs, files in os.walk(b):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", ".pytest_cache")]
        rel = os.path.relpath(root, b)
        for fn in files:
            if fn.endswith(".pyc") or fn in ignore_files:
                continue
            if not os.path.exists(os.path.join(a, rel, fn)):
                diff.append(rel + "/" + fn)
    return sorted(set(diff))


def eval_final(image, workdir, task, timeout=900):
    f2p = task["FAIL_TO_PASS"]
    p2p = task["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    nodes = list(f2p) + list(p2p)
    files = sorted({n.split("::", 1)[0] for n in nodes})
    cmd = "pytest --color=no -rA -p no:cacheprovider --no-header " + \
          " ".join(shlex.quote(f) for f in files)
    rc, out, err = sh(["docker", "run", "--rm", "-v", f"{workdir}:/testbed",
                       "-w", "/testbed", image, "bash", "-lc", cmd], timeout=timeout)
    import re
    statuses = {m.group(2): m.group(1)
                for m in re.finditer(r"^(PASSED|FAILED|ERROR)\s+(\S+)", out + err, re.M)}
    n_fail = sum(1 for n in nodes
                 if statuses.get(n) not in ("PASSED", "XFAIL"))
    return {"n_fail": n_fail, "n_pass": sum(1 for n in nodes if statuses.get(n) == "PASSED"),
            "seconds": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--preds", required=True, help="preds.json（含 model_patch）")
    ap.add_argument("--report", default="", help="traj_report.json（可选，对比 final n_fail）")
    ap.add_argument("--work", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval", action="store_true", help="对重放终态跑测试（慢）")
    args = ap.parse_args()

    traj = json.load(open(args.traj))
    task = load_task(args.task, args.instance_id)
    work = os.path.abspath(args.work or f"/tmp/fidelity_{uuid.uuid4().hex[:6]}")
    out = os.path.abspath(args.out or work + "_out")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    base = os.path.join(work, "base")
    base_sha = get_base(args.image, base)

    # 1) 重放命令
    replay = os.path.join(work, "replay")
    shutil.copytree(base, replay, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".tox",
                                                ".nox", ".venv", "node_modules"))
    cname = f"fid_{uuid.uuid4().hex[:6]}"
    sh(["docker", "run", "-d", "--name", cname, "-v", f"{replay}:/testbed",
        "-w", "/testbed", args.image, "sleep", "2h"])
    cmds = extract_commands(traj)
    fails = 0
    try:
        docker_exec(cname, "git config --global --add safe.directory /testbed")
        for c in cmds:
            rc, o, e = docker_exec(cname, c, timeout=300)
            if rc != 0:
                fails += 1
        docker_exec(cname, f"chown -R {os.getuid()}:{os.getgid()} /testbed")
    finally:
        sh(["docker", "rm", "-f", cname])

    # 2) 干净 base + 应用提交的 patch
    patched = os.path.join(work, "patched")
    shutil.copytree(base, patched, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".tox",
                                                ".nox", ".venv", "node_modules"))
    preds = json.load(open(args.preds))
    patch = (preds.get(args.instance_id) or {}).get("model_patch", "")
    apply_ok, apply_err = False, ""
    if patch.strip():
        pf = os.path.join(out, "submitted.patch")
        open(pf, "w").write(patch)
        rc, o, e = sh(GIT + ["apply", "--whitespace=nowarn", pf], cwd=patched)
        apply_ok, apply_err = rc == 0, e[:300]

    # 3) 比较两棵树
    diff = tree_diff(replay, patched) if apply_ok else []
    same = apply_ok and not diff

    result = {"instance_id": args.instance_id, "n_commands": len(cmds),
              "cmd_fail": fails, "patch_len": len(patch),
              "apply_ok": apply_ok, "apply_err": apply_err,
              "identical": same, "n_diff_files": len(diff),
              "diff_files": diff[:20]}
    if args.eval:
        ev = eval_final(args.image, replay, task)
        result["replay_n_fail"] = ev["n_fail"]
        if args.report and os.path.exists(args.report):
            r = json.load(open(args.report))
            fin = (r.get("evals") or [{}])[-1].get("n_fail")
            result["report_final_n_fail"] = fin
            result["eval_match"] = (fin == ev["n_fail"])

    json.dump(result, open(os.path.join(out, "fidelity.json"), "w"),
              ensure_ascii=False, indent=1)

    print(f"[{args.instance_id}] commands={len(cmds)} cmd_fail={fails} "
          f"patch_len={len(patch)} apply_ok={apply_ok}")
    print(f"  identical={same}  diff_files={len(diff)}")
    for f in diff[:10]:
        print(f"    DIFF {f}")
    if args.eval:
        print(f"  replay_n_fail={result.get('replay_n_fail')} "
              f"report_final={result.get('report_final_n_fail')} "
              f"match={result.get('eval_match')}")
    print(f"结果 → {out}/fidelity.json")


if __name__ == "__main__":
    main()
