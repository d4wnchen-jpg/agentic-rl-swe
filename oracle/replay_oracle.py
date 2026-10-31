#!/usr/bin/env python3
"""replay_oracle.py — L1 oracle 模式（v2）。

把数据集的 gold patch 按 hunk 拆成"轮次"，逐轮应用并跑测试，
得到一条真实的「测试状态轨迹」；最后对每个 hunk 单独反向应用并重跑。

v2 新增：
  --skip-path     逗号分隔 glob，跳过匹配的 hunk（如 '*.rst,*.md,docs/*'）
  --batch         一次跑多道题（配 --ids a,b,c）
  --invert-patch  patch 是"缺陷注入补丁"，gold 操作是反向应用（SWE-smith）
  报告标注每个 hunk 是否"影响测试"（changed）以及回滚是否为惰性（inert）

测试命令约定：必须按行打印 `PASSED <name>` / `FAILED <name>`（pytest -rA 就是这样）。
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

STATUS_RE = re.compile(r"^(PASSED|FAILED|ERROR)\s+(\S+)", re.M)


def sh(args, cwd=None):
    t0 = time.perf_counter()
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return r, time.perf_counter() - t0


def git(args, cwd):
    return sh(["git", "-c", "user.email=demo@local", "-c", "user.name=demo"] + args, cwd=cwd)


def apply_hunk(work, hp, invert=False):
    flags = ["-R"] if invert else []
    return git(["apply"] + flags + ["--whitespace=nowarn", hp], work)


def revert_hunk(work, hp, invert=False, three_way=False):
    flags = ["-R"] if not invert else []
    extra = ["--3way"] if three_way else []
    return git(["apply"] + flags + extra + ["--whitespace=nowarn", hp], work)


def split_hunks(patch_text: str):
    """先按文件(diff --git)分块，再按 @@ 拆 hunk；每个 hunk 保留自己文件的 header。"""
    blocks, cur = [], []
    for ln in patch_text.splitlines(keepends=True):
        if ln.startswith("diff --git ") and cur:
            blocks.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)

    out = []
    idx = 0
    for block in blocks:
        header, hunks, c = [], [], None
        for ln in block:
            if ln.startswith("@@"):
                if c is not None:
                    hunks.append(header + c)
                c = [ln]
            elif c is None:
                header.append(ln)
            else:
                c.append(ln)
        if c is not None:
            hunks.append(header + c)
        for h in hunks:
            idx += 1
            text = "".join(h)
            out.append({"idx": idx, "path": hunk_path(text), "text": text})
    return out


def hunk_path(text: str) -> str:
    first = text.splitlines()[0] if text.splitlines() else ""
    if first.startswith("diff --git "):
        parts = first.split()
        if len(parts) >= 4:
            return parts[3][2:]
    return "?"


def load_tasks(path, ids):
    rows = {}
    with open(path) as f:
        if path.endswith(".jsonl"):
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    rows[t["instance_id"]] = t
        else:
            t = json.load(f)
            rows[t["instance_id"]] = t
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"instance_id 未找到: {missing}")
    return [rows[i] for i in ids]


def run_tests(cmd: str, cwd: str, timeout: int):
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                           text=True, timeout=timeout)
        out, ok = r.stdout + r.stderr, r.returncode == 0
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


def compile_ok(work, paths):
    files = sorted({p for p in paths if p.endswith(".py") and p != "?"})
    if not files:
        return True, ""
    r, _ = sh([sys.executable, "-m", "py_compile"] + files, cwd=work)
    return r.returncode == 0, r.stderr.strip()[:300]


def run_one(task, args, work, out):
    iid = task["instance_id"]
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if isinstance(task.get(k), str):
            task[k] = json.loads(task[k])
    f2p, p2p = task.get("FAIL_TO_PASS", []), task.get("PASS_TO_PASS", [])
    patch_text = task.get("patch", "")
    if not patch_text:
        raise SystemExit(f"{iid} 缺少 patch 字段")
    nodes_arg = " ".join(list(f2p) + list(p2p))
    os.makedirs(out, exist_ok=True)

    if os.path.exists(work):
        shutil.rmtree(work)
    shutil.copytree(args.repo, work,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    git(["init", "-q"], work)
    git(["add", "-A"], work)
    git(["commit", "-qm", "baseline"], work)

    report = {"instance_id": iid, "f2p": f2p, "p2p": p2p, "n_hunks": 0,
              "skipped": [], "baseline": None, "turns": [], "reverts": [],
              "test_patch": None, "eval_seconds": []}

    tp = task.get("test_patch")
    if args.apply_test_patch and tp:
        tpath = os.path.join(out, "test_patch.diff")
        open(tpath, "w").write(tp)
        r, dt = git(["apply", "--whitespace=nowarn", tpath], work)
        report["test_patch"] = {"ok": r.returncode == 0, "seconds": round(dt, 2),
                                "err": r.stderr[:300]}
        if r.returncode != 0:
            print(f"[{iid}] ⚠️ test_patch 应用失败: {r.stderr[:200]}")
        git(["add", "-A"], work)
        git(["commit", "-qm", "test_patch"], work)

    cmd = args.test_cmd.replace("{nodes}", nodes_arg).replace("{work}", work)
    base = run_tests(cmd, work, args.timeout)
    report["baseline"] = {"statuses": base["statuses"],
                          **summarize(base["statuses"], f2p, p2p),
                          "test_ok": base["ok"], "test_s": base["seconds"]}
    report["eval_seconds"].append(base["seconds"])
    print(f"[{iid}] baseline n_fail={report['baseline']['n_fail']} ({base['seconds']}s)")

    hunks = split_hunks(patch_text)
    report["n_hunks"] = len(hunks)
    kept = [h for h in hunks if not any(fnmatch.fnmatch(h["path"], p) for p in args.skip_path)]
    report["skipped"] = [{"idx": h["idx"], "path": h["path"]}
                         for h in hunks if h not in kept]
    if report["skipped"]:
        print(f"[{iid}] 跳过 {len(report['skipped'])} 个 hunk: "
              f"{[s['path'] for s in report['skipped']]}")
    if not kept:
        raise SystemExit(f"{iid} 所有 hunk 都被过滤掉了")
    for h in kept:
        open(os.path.join(out, f"patch_{h['idx']:02d}.patch"), "w").write(h["text"])

    order = [int(x) for x in args.order.split(",")] if args.order else [h["idx"] for h in kept]
    by_idx = {h["idx"]: h for h in kept}
    for turn, idx in enumerate(order, 1):
        h = by_idx[idx]
        hp = os.path.join(out, f"patch_{idx:02d}.patch")
        r, dt = apply_hunk(work, hp, invert=args.invert_patch)
        if r.returncode != 0:
            print(f"  apply stderr: {r.stderr.strip()[:300]}")
        res = run_tests(cmd, work, args.timeout) if r.returncode == 0 else \
            {"ok": False, "statuses": {}, "seconds": 0.0, "tail": []}
        summary = summarize(res["statuses"], f2p, p2p)
        git(["add", "-A"], work)
        git(["commit", "-qm", f"turn{turn}"], work)
        report["eval_seconds"].append(res["seconds"])
        prev_fail = report["turns"][-1]["n_fail"] if report["turns"] else report["baseline"]["n_fail"]
        rec = {"turn": turn, "hunk": idx, "path": h["path"], "apply_ok": r.returncode == 0,
               "apply_s": round(dt, 2), **summary, "test_ok": res["ok"],
               "test_s": res["seconds"], "statuses": res["statuses"],
               "changed": (summary["n_fail"] != prev_fail)}
        report["turns"].append(rec)
        print(f"[{iid}] turn {turn} hunk{idx:02d} ({h['path']}) apply={rec['apply_ok']} "
              f"n_fail={summary['n_fail']} changed={rec['changed']} ({res['seconds']}s)")

    final = report["turns"][-1] if report["turns"] else report["baseline"]
    final_statuses, final_fail = final["statuses"], final["n_fail"]
    for h in kept:
        hp = os.path.join(out, f"patch_{h['idx']:02d}.patch")
        r, dt = revert_hunk(work, hp, invert=args.invert_patch)
        used_3way = False
        if r.returncode != 0:
            r, dt = revert_hunk(work, hp, invert=args.invert_patch, three_way=True)
            used_3way = True
        res = run_tests(cmd, work, args.timeout) if r.returncode == 0 else \
            {"ok": False, "statuses": {}, "seconds": 0.0, "tail": []}
        summary = summarize(res["statuses"], f2p, p2p)
        syntax_ok, syntax_err = compile_ok(work, [h["path"]])
        inert = (res["statuses"] == final_statuses)
        report["eval_seconds"].append(res["seconds"])
        rec = {"hunk": h["idx"], "path": h["path"], "revert_ok": r.returncode == 0,
               "used_3way": used_3way, "syntax_ok": syntax_ok, "syntax_err": syntax_err,
               "inert": inert, **summary, "statuses": res["statuses"],
               "revert_s": round(dt, 2)}
        report["reverts"].append(rec)
        print(f"[{iid}] revert hunk{h['idx']:02d} ok={rec['revert_ok']} "
              f"n_fail={summary['n_fail']} inert={inert} syntax={syntax_ok}")
        git(["checkout", "--", "."], work)

    report["final_n_fail"] = final_fail
    return report


def write_md(rep, path):
    L = [f"# Oracle 报告：{rep['instance_id']}", "",
         f"- hunks（原始）= {rep['n_hunks']}，跳过 = {len(rep['skipped'])}，"
         f"F2P = {len(rep['f2p'])}，P2P = {len(rep['p2p'])}",
         f"- baseline n_fail = {rep['baseline']['n_fail']}，final n_fail = {rep.get('final_n_fail')}",
         f"- 单次判分中位数 = {statistics.median(rep['eval_seconds']):.2f}s"
         f"（共 {len(rep['eval_seconds'])} 次）", ""]
    if rep["skipped"]:
        L += ["## 跳过的 hunk（文档/非代码）", ""]
        L += [f"- hunk{s['idx']} → `{s['path']}`" for s in rep["skipped"]]
        L.append("")
    L += ["## 逐轮应用", "",
          "| turn | hunk | path | apply | n_fail | n_pass | changed | test_s |",
          "|---|---|---|---|---|---|---|---|"]
    for t in rep["turns"]:
        L.append(f"| {t['turn']} | {t['hunk']} | `{t['path']}` | {t['apply_ok']} | "
                 f"{t['n_fail']} | {t['n_pass']} | {t['changed']} | {t['test_s']} |")
    L += ["", "## 单 hunk 回滚（从最终状态）", "",
          "| hunk | path | revert | 3way | syntax | n_fail | inert |",
          "|---|---|---|---|---|---|---|"]
    for r in rep["reverts"]:
        L.append(f"| {r['hunk']} | `{r['path']}` | {r['revert_ok']} | {r['used_3way']} | "
                 f"{r['syntax_ok']} | {r['n_fail']} | {r['inert']} |")
    L.append("")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance-id", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--work", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--test-cmd", required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--order", default="")
    ap.add_argument("--skip-path", default="", help="逗号分隔 glob，如 '*.rst,*.md,docs/*'")
    ap.add_argument("--apply-test-patch", action="store_true")
    ap.add_argument("--invert-patch", action="store_true")
    args = ap.parse_args()

    ids = [x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else \
          ([args.instance_id] if args.instance_id else [])
    if not ids:
        if not args.task.endswith(".jsonl"):
            with open(args.task) as f:
                ids = [json.load(f)["instance_id"]]
        else:
            raise SystemExit("需要 --instance-id 或 --ids")
    tasks = load_tasks(os.path.abspath(args.task), ids)
    args.repo = os.path.abspath(args.repo)
    args.skip_path = [x.strip() for x in args.skip_path.split(",") if x.strip()]

    reports = []
    for t in tasks:
        iid = t["instance_id"]
        if len(tasks) == 1:
            work = os.path.abspath(args.work or f"/tmp/oracle/{iid}")
            out = os.path.abspath(args.out or work + "_out")
        else:
            work = os.path.join(os.path.abspath(args.work or "/tmp/oracle"), iid)
            out = os.path.join(os.path.abspath(args.out or "/tmp/oracle_out"), iid)
        try:
            rep = run_one(t, args, work, out)
        except SystemExit as e:
            print(f"[{iid}] 失败: {e}")
            reports.append({"instance_id": iid, "error": str(e)})
            continue
        write_md(rep, os.path.join(out, "ORACLE.md"))
        json.dump(rep, open(os.path.join(out, "oracle_report.json"), "w"),
                  ensure_ascii=False, indent=1)
        reports.append(rep)
        print(f"[{iid}] → {out}/ORACLE.md")

    if len(reports) > 1:
        print("\n===== 汇总 =====")
        for r in reports:
            if "error" in r:
                print(f"{r['instance_id']}: ERROR {r['error']}")
                continue
            med = statistics.median(r["eval_seconds"])
            inert = sum(1 for x in r["reverts"] if x["inert"])
            print(f"{r['instance_id']}: hunks={r['n_hunks']} skip={len(r['skipped'])} "
                  f"final_n_fail={r.get('final_n_fail')} eval_med={med:.2f}s "
                  f"inert={inert}/{len(r['reverts'])}")


if __name__ == "__main__":
    main()
