#!/usr/bin/env python3
"""replay_oracle.py — L1 oracle 模式。

把数据集的 gold patch 按 hunk 拆成"轮次"，逐轮应用并跑测试，
得到一条真实的「测试状态轨迹」；最后对每个 hunk 单独反向应用并重跑。

用法（公开任务，例如 SWE-bench Lite）：
  python replay_oracle.py \
    --task /path/SWE-bench_Lite.jsonl --instance-id astropy__astropy-12907 \
    --repo /path/to/repo_at_base_commit \
    --test-cmd 'conda run -n <env> pytest --color=no -rA -p no:cacheprovider {nodes}' \
    --out out/

测试命令约定：必须按行打印 `PASSED <name>` / `FAILED <name>`（pytest -rA 就是这样），
脚本据此统计通过/失败；退出码非 0 只表示"有失败"。

本地无 pytest 时可用 demo：--test-cmd 'python3 check.py'（--nodes-optional）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

STATUS_RE = re.compile(r"^(PASSED|FAILED|ERROR)\s+(\S+)", re.M)


def sh(args, cwd=None, check=False):
    t0 = time.perf_counter()
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return r, time.perf_counter() - t0


def git(args, cwd):
    return sh(["git", "-c", "user.email=demo@local", "-c", "user.name=demo"] + args, cwd=cwd)


def split_hunks(patch_text: str):
    lines = patch_text.splitlines(keepends=True)
    header, hunks, cur = [], [], None
    for ln in lines:
        if ln.startswith("@@"):
            if cur is not None:
                hunks.append("".join(header + cur))
            cur = [ln]
        elif cur is None:
            header.append(ln)
        else:
            cur.append(ln)
    if cur is not None:
        hunks.append("".join(header + cur))
    return hunks


def load_task(path, instance_id):
    with open(path) as f:
        if path.endswith(".jsonl"):
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line)
                if t.get("instance_id") == instance_id:
                    return t
            raise SystemExit(f"instance_id 未找到: {instance_id}")
        t = json.load(f)
        if instance_id and t.get("instance_id") != instance_id:
            raise SystemExit(f"instance_id 不匹配: {t.get('instance_id')}")
        return t


def run_tests(cmd: str, cwd: str, timeout: int):
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                           text=True, timeout=timeout)
        out = r.stdout + r.stderr
        ok = r.returncode == 0
    except subprocess.TimeoutExpired:
        out, ok = "TIMEOUT", False
    dt = time.perf_counter() - t0
    statuses = {m.group(2): m.group(1) for m in STATUS_RE.finditer(out)}
    return {"ok": ok, "statuses": statuses, "seconds": round(dt, 2),
            "tail": out.strip().splitlines()[-3:] if out.strip() else []}


def summarize(statuses, f2p, p2p):
    nodes = list(f2p) + list(p2p)
    if not statuses:
        return {"n_expected": len(nodes), "n_pass": None, "n_fail": None}
    n_pass = sum(1 for n in nodes if statuses.get(n) == "PASSED")
    n_fail = sum(1 for n in nodes if statuses.get(n) not in ("PASSED", "XFAIL"))
    return {"n_expected": len(nodes), "n_pass": n_pass, "n_fail": n_fail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="task json / jsonl")
    ap.add_argument("--instance-id", default="")
    ap.add_argument("--repo", required=True, help="干净 repo（缺陷基线状态）")
    ap.add_argument("--work", default="", help="工作目录（默认 /tmp/oracle/<id>）")
    ap.add_argument("--out", default="", help="结果输出目录")
    ap.add_argument("--test-cmd", required=True,
                    help="测试命令；含 {nodes} 时用 F2P+P2P 替换")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--order", default="", help="应用顺序，如 2,1,3（默认顺序）")
    ap.add_argument("--apply-test-patch", action="store_true",
                    help="若 task 有 test_patch，先在基线上应用（SWE-bench 需要）")
    args = ap.parse_args()

    task_path = os.path.abspath(args.task)
    repo = os.path.abspath(args.repo)
    task = load_task(task_path, args.instance_id or None)
    iid = task["instance_id"]
    work = os.path.abspath(args.work or f"/tmp/oracle/{iid}")
    out = os.path.abspath(args.out or work + "_out")
    os.makedirs(out, exist_ok=True)

    patch_text = task.get("patch", "")
    if not patch_text:
        raise SystemExit("task 缺少 patch 字段")
    f2p = task.get("FAIL_TO_PASS", [])
    p2p = task.get("PASS_TO_PASS", [])
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)

    # 1) 建工作区
    if os.path.exists(work):
        shutil.rmtree(work)
    shutil.copytree(repo, work, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    git(["init", "-q"], work)
    git(["add", "-A"], work)
    git(["commit", "-qm", "baseline"], work)

    report = {"instance_id": iid, "f2p": f2p, "p2p": p2p,
              "n_hunks": 0, "turns": [], "reverts": [], "test_patch": None}

    # 2) 可选：应用 test_patch（让 F2P 测试存在）
    if args.apply_test_patch and task.get("test_patch"):
        tp = os.path.join(out, "test_patch.diff")
        open(tp, "w").write(task["test_patch"])
        r, dt = git(["apply", "--whitespace=nowarn", tp], work)
        report["test_patch"] = {"ok": r.returncode == 0, "seconds": round(dt, 2),
                                "err": r.stderr[:300]}
        git(["add", "-A"], work)
        git(["commit", "-qm", "test_patch"], work)

    # 3) 拆 hunk
    hunks = split_hunks(patch_text)
    report["n_hunks"] = len(hunks)
    for i, h in enumerate(hunks, 1):
        open(os.path.join(out, f"hunk{i:02d}.patch"), "w").write(h)

    order = [int(x) for x in args.order.split(",")] if args.order else list(range(1, len(hunks) + 1))

    # 4) 逐轮应用 + 判分
    for turn, idx in enumerate(order, 1):
        hp = os.path.join(out, f"hunk{idx:02d}.patch")
        r, dt = git(["apply", "--whitespace=nowarn", hp], work)
        if r.returncode != 0:
            print("  apply stderr:", r.stderr.strip()[:400])
        cmd = (args.test_cmd.replace("{nodes}", " ".join(list(f2p) + list(p2p)))
               .replace("{work}", work))
        res = run_tests(cmd, work, args.timeout) if r.returncode == 0 else \
            {"ok": False, "statuses": {}, "seconds": 0.0, "tail": [r.stderr[:200]]}
        summary = summarize(res["statuses"], f2p, p2p)
        git(["add", "-A"], work)
        git(["commit", "-qm", f"turn{turn}"], work)
        rec = {"turn": turn, "hunk": idx, "apply_ok": r.returncode == 0,
               "apply_s": round(dt, 2), **summary,
               "test_ok": res["ok"], "test_s": res["seconds"],
               "statuses": res["statuses"], "tail": res["tail"]}
        report["turns"].append(rec)
        print(f"[turn {turn}] hunk{idx:02d} apply={rec['apply_ok']} "
              f"n_fail={summary['n_fail']} test_ok={res['ok']} ({res['seconds']}s)")

    # 5) 从最终状态逐个回滚 hunk
    for idx in range(1, len(hunks) + 1):
        hp = os.path.join(out, f"hunk{idx:02d}.patch")
        r, dt = git(["apply", "-R", hp], work)
        used_3way = False
        if r.returncode != 0:
            r, dt = git(["apply", "-R", "--3way", hp], work)
            used_3way = True
        if r.returncode != 0:
            print(f"  revert stderr(hunk{idx}):", r.stderr.strip()[:400])
        cmd = (args.test_cmd.replace("{nodes}", " ".join(list(f2p) + list(p2p)))
               .replace("{work}", work))
        res = run_tests(cmd, work, args.timeout) if r.returncode == 0 else \
            {"ok": False, "statuses": {}, "seconds": 0.0, "tail": [r.stderr[:200]]}
        summary = summarize(res["statuses"], f2p, p2p)
        rec = {"hunk": idx, "revert_ok": r.returncode == 0, "used_3way": used_3way,
               "revert_s": round(dt, 2), **summary, "statuses": res["statuses"]}
        report["reverts"].append(rec)
        print(f"[revert hunk{idx:02d}] ok={rec['revert_ok']} n_fail={summary['n_fail']}")
        git(["checkout", "--", "."], work)

    json.dump(report, open(os.path.join(out, "oracle_report.json"), "w"),
              ensure_ascii=False, indent=1)

    # 6) markdown 表
    md = [f"# Oracle 报告：{iid}", "",
          f"hunks = {len(hunks)}，F2P = {len(f2p)}，P2P = {len(p2p)}", "",
          "## 逐轮应用", "",
          "| turn | hunk | apply | n_fail | n_pass | test_ok | test_s |",
          "|---|---|---|---|---|---|---|"]
    for t in report["turns"]:
        md.append(f"| {t['turn']} | {t['hunk']} | {t['apply_ok']} | {t['n_fail']} | "
                  f"{t['n_pass']} | {t['test_ok']} | {t['test_s']} |")
    md += ["", "## 单 hunk 回滚（从最终状态）", "",
           "| hunk | revert | 3way | n_fail | n_pass | statuses |",
           "|---|---|---|---|---|---|"]
    for r_ in report["reverts"]:
        md.append(f"| {r_['hunk']} | {r_['revert_ok']} | {r_['used_3way']} | "
                  f"{r_['n_fail']} | {r_['n_pass']} | {r_['statuses']} |")
    open(os.path.join(out, "ORACLE.md"), "w").write("\n".join(md) + "\n")
    print(f"\n结果 → {out}/ORACLE.md")


if __name__ == "__main__":
    main()
