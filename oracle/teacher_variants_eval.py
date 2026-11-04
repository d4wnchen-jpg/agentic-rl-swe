#!/usr/bin/env python3
"""teacher_variants_eval.py — Q4 修正实验：区分「读状态」与「判因果」。

三种中性提问（每轨迹各一次）：
  A 方向题：给出逐轮 n_fail，问哪些步变坏/变好 → 与「状态变化方向」比较
  B 因果题：不给执行，问跳过哪一步会改变最终结果 → 与我们的反事实标签比较
  C 无错题：问是否存在这样的单轮；没有就答 0

用法：
  DEEPSEEK_API_KEY=... python teacher_variants_eval.py --dirs results/g1c results/g2a results/r3a
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
API = "https://api.deepseek.com/v1/chat/completions"
SYSTEM = ("You are an expert software engineer analyzing a SWE-agent trajectory with the "
          "per-step failing-test counts. Answer precisely and only with JSON.")


def truncate(s, n=350):
    s = (s or "").strip()
    return s if len(s) <= n else s[:180] + " ... " + s[-120:]


def build_trace(traj, max_chars=40000):
    turns, step = [], 0
    for m in traj.get("messages", []):
        if m.get("role") == "assistant":
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
                step += 1
                turns.append(f"### Step {step}\n" + "\n".join(f"$ {truncate(c)}" for c in cmds))
        elif m.get("role") == "tool" and turns:
            turns.append("observation: " + truncate(m.get("content", ""), 250))
    return "\n".join(turns)[-max_chars:], step


def call(key, model, user):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 700,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def extract_json(raw):
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results/g1c", "results/g2a", "results/r3a"])
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--out", default="results/teacher_variants")
    args = ap.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("需要 DEEPSEEK_API_KEY")
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for d in args.dirs:
        sub = os.path.basename(d.rstrip("/"))
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
        for rf in sorted(glob.glob(os.path.join(d, "*.json"))):
            b = os.path.basename(rf).split(".")[0]
            if not b.isdigit() or rf.endswith((".traj.json", "_teacher.json", "_preds.json")):
                continue
            trajf = os.path.join(d, f"{b}.traj.json")
            if not os.path.exists(trajf):
                continue
            r = json.load(open(rf))
            traj = json.load(open(trajf))
            evs = r.get("evals") or []
            nfs = [e.get("n_fail") for e in evs]
            # 状态变化的真值方向（相邻轮）
            changes = []
            for i in range(1, len(nfs)):
                if nfs[i] is None or nfs[i-1] is None:
                    continue
                if nfs[i] > nfs[i-1]:
                    changes.append({"step": evs[i]["turn"], "dir": "harmful"})
                elif nfs[i] < nfs[i-1]:
                    changes.append({"step": evs[i]["turn"], "dir": "beneficial"})
            trace, n_steps = build_trace(traj)
            status = ", ".join(f"step{i+1}={v}" for i, v in enumerate(nfs[1:]))
            final = nfs[-1] if nfs else None
            ours = {
                "harmful": sorted({c["turn"] for c in (r.get("counterfactuals") or [])
                                   if c.get("label") in ("causal", "causal_by_file")}),
                "beneficial": sorted({c["turn"] for c in (r.get("counterfactuals") or [])
                                      if c.get("label") in ("beneficial", "beneficial_by_file")}),
                "recovery": sorted({x["pivot_turn"] for x in (r.get("recoveries") or [])}),
            }
            ours["any"] = sorted(set(ours["harmful"]) | set(ours["beneficial"]) | set(ours["recovery"]))
            cachef = os.path.join(args.out, sub, f"{b}.json")
            if os.path.exists(cachef):
                res = json.load(open(cachef))
            else:
                head = f"## Trajectory ({n_steps} steps)\n{trace}\n\n## Per-step n_fail\n{status}\n\n## Final\nn_fail={final}\n"
                qa = head + ('\n## Task A\nFor each step where n_fail CHANGED, label the change: '
                             '"harmful" if n_fail increased, "beneficial" if it decreased.\n'
                             'Return JSON: {"changes": [{"step": <int>, "label": "harmful"|"beneficial"}]}')
                qb = head + ('\n## Task B\nSuppose we skip exactly one step and replay the rest (no new '
                             'actions). Which single step, if skipped, would most improve the FINAL '
                             'test outcome? If none, answer 0.\n'
                             'Return JSON: {"skip_step": <int>, "reason": "<short>"}')
                qc = head + ('\n## Task C\nIs there any single step such that skipping it would CHANGE '
                             'the FINAL test outcome? If none, answer 0.\n'
                             'Return JSON: {"exists": true/false, "step": <int or 0>}')
                try:
                    a = extract_json(call(key, args.model, qa)).get("changes", [])
                    b_ = extract_json(call(key, args.model, qb))
                    c_ = extract_json(call(key, args.model, qc))
                except Exception as e:
                    a, b_, c_ = [], {"error": str(e)[:150]}, {"error": str(e)[:150]}
                res = {"changes": a, "skip": b_, "exists": c_,
                       "status_changes": changes, "ours": ours, "final": final}
                json.dump(res, open(cachef, "w"), ensure_ascii=False, indent=1)
            rows.append((sub, b, res))
            print(f"[{sub}/{b}] A_changes={len(res.get('changes') or [])} "
                  f"B_skip={res.get('skip',{}).get('skip_step')} "
                  f"C={res.get('exists',{}).get('step')} final={final} ours_any={ours['any']}")

    # 指标
    def m():
        return dict(a_correct=0, a_total=0, b_exact=0, b_near=0, b_total=0,
                    c_agree_none=0, c_none_total=0, c_any_found=0, c_any_total=0)
    M = m()
    for sub, b, res in rows:
        changes = res.get("status_changes") or []
        truth = {x["step"]: x["dir"] for x in changes}
        a = res.get("changes") or []
        for x in a:
            s, lab = x.get("step"), x.get("label")
            if s in truth:
                M["a_total"] += 1
                if lab == truth[s]:
                    M["a_correct"] += 1
        ours = res.get("ours") or {}
        any_ = set(ours.get("any") or [])
        skip = (res.get("skip") or {}).get("skip_step")
        if any_:
            M["b_total"] += 1
            if isinstance(skip, int) and skip in any_:
                M["b_exact"] += 1
                M["b_near"] += 1
            elif isinstance(skip, int) and any(abs(skip - o) <= 1 for o in any_):
                M["b_near"] += 1
        st = (res.get("exists") or {}).get("step")
        if not any_:
            M["c_none_total"] += 1
            if st in (0, None):
                M["c_agree_none"] += 1
        else:
            M["c_any_total"] += 1
            if st not in (0, None):
                M["c_any_found"] += 1
    print("\n===== A: 教师读方向（vs 真实状态变化）=====")
    print(f"  方向正确 {M['a_correct']}/{M['a_total']} = "
          f"{M['a_correct']/max(1,M['a_total'])*100:.0f}%")
    print("===== B: 教师猜因果（vs 我们的 cf 标签）=====")
    print(f"  exact {M['b_exact']}/{M['b_total']}, ±1 {M['b_near']}/{M['b_total']}")
    print("===== C: 承认“无单轮可归因” =====")
    print(f"  我们判定无 → 教师也答无: {M['c_agree_none']}/{M['c_none_total']}")
    print(f"  我们有 → 教师也找到: {M['c_any_found']}/{M['c_any_total']}")
    json.dump({"rows": [{"dir": s, "run": b, **res} for s, b, res in rows], "metrics": M},
              open(os.path.join(args.out, "summary.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
