#!/usr/bin/env python3
"""teacher_status_eval.py — Q4 公平性对照（本地跑，仅调 DeepSeek API）。

对比三组：
  A. 教师只看轨迹（已有 *_teacher.json 的结果，直接读）
  B. 教师 + 逐轮测试状态（本脚本新跑）
  C. 随机猜一轮（解析计算）

并与我们的可执行验证标签（harmful/beneficial/recovery pivot）比较：
  - 轮次 exact / ±1
  - 符号：命中的轮次是否仅 beneficial（教师按“失败原因”提问，等价于说成 mistake）
  - 无编辑可归因时是否仍指认

用法：
  DEEPSEEK_API_KEY=... python teacher_status_eval.py --dirs results/g1c results/g2a
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

API = "https://api.deepseek.com/v1/chat/completions"
SYSTEM = ("You are an expert software engineer doing a retrospective analysis of a SWE-agent "
          "attempt. You see the problem, the agent's commands, per-step test status, and the "
          "final result. Be precise and skeptical.")


def truncate(s, n=400):
    s = (s or "").strip()
    return s if len(s) <= n else s[:200] + " ... " + s[-150:]


def build_trace(traj, max_chars=50000):
    turns, step = [], 0
    for m in traj.get("messages", []):
        role = m.get("role")
        if role == "assistant":
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
                turns.append(f"### Step {step}\n" + "\n".join(f"$ {truncate(c, 700)}" for c in cmds))
        elif role == "tool" and turns:
            turns.append("observation: " + truncate(m.get("content", ""), 300))
    return "\n".join(turns)[-max_chars:], step


def call_api(key, model, user):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 500,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def ours_set(r):
    cfs = r.get("counterfactuals") or []
    harmful = {c["turn"] for c in cfs if c.get("label") in ("causal", "causal_by_file")}
    beneficial = {c["turn"] for c in cfs if c.get("label") in ("beneficial", "beneficial_by_file")}
    recov = {x["pivot_turn"] for x in (r.get("recoveries") or [])}
    return harmful, beneficial, recov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results/g1c", "results/g2a"])
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--out", default="results/teacher_status")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    if not key:
        envp = os.path.expanduser("~/.config/mini-swe-agent/.env")
        if os.path.exists(envp):
            for line in open(envp):
                if line.strip().startswith("DEEPSEEK_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY 未设置")

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    rows = []
    for d in args.dirs:
        for rf in sorted(glob.glob(os.path.join(d, "*.json"))):
            b = os.path.basename(rf).split(".")[0]
            if not b.isdigit() or rf.endswith((".traj.json", "_teacher.json", "_preds.json")):
                continue
            r = json.load(open(rf))
            trajf = os.path.join(d, f"{b}.traj.json")
            if not os.path.exists(trajf):
                continue
            traj = json.load(open(trajf))
            ours = ours_set(r)
            harmful, beneficial, recov = ours
            all_ours = sorted(harmful | beneficial | recov)
            evs = r.get("evals") or []
            final = evs[-1].get("n_fail") if evs else None
            statuses = [e.get("n_fail") for e in evs]
            trace, n_steps = build_trace(traj)
            status_line = ", ".join(f"step{i+1}={v}" for i, v in enumerate(statuses[1:]))
            problem = ""
            # problem_statement 不在 report 里，从 task 文件补（由调用方保证 SWE-bench_Lite.jsonl 存在）
            task_file = "SWE-bench_Lite.jsonl"
            if os.path.exists(task_file):
                for line in open(task_file):
                    try:
                        t = json.loads(line)
                    except Exception:
                        continue
                    if t["instance_id"] == r.get("instance_id"):
                        problem = t["problem_statement"]
                        break
            if not problem:
                for m_ in traj.get("messages", []):
                    if m_.get("role") == "user":
                        problem = (m_.get("content") or "")[:3000]
                        break
            question = ("This attempt FAILED. Identify the single most pivotal step (1-indexed) "
                        "that caused the failure. If the failure is because the fix was never "
                        "attempted, answer pivot_step = 0." if (final or 0) > 0 else
                        "This attempt eventually SUCCEEDED. Identify the single most pivotal "
                        "recoverable mistake step (1-indexed) that was later fixed; if none, "
                        "answer 0.")
            user = f"""## Problem
{problem[:3000]}

## Trajectory ({n_steps} steps)
{trace}

## Per-step failing-test count (n_fail after each step)
{status_line}

## Final result
final n_fail={final}

## Question
{question}
Return ONLY JSON: {{"pivot_step": <int>, "reason": "<short>", "correct_action": "<short>"}}
"""
            sub = os.path.basename(d.rstrip("/"))
            os.makedirs(os.path.join(args.out, sub), exist_ok=True)
            cache = os.path.join(args.out, sub, f"{b}_status_teacher.json")
            parsed = None
            raw = ""
            if os.path.exists(cache):
                try:
                    parsed = json.load(open(cache)).get("teacher")
                except Exception:
                    parsed = None
            if parsed is None:
                try:
                    raw = call_api(key, args.model, user)
                    m = re.search(r"\{.*\}", raw, re.S)
                    parsed = json.loads(m.group(0)) if m else {"pivot_step": None}
                except Exception as e:
                    parsed = {"pivot_step": None, "error": str(e)[:200]}
                json.dump({"instance_id": r.get("instance_id"), "teacher": parsed, "raw": raw},
                          open(cache, "w"), ensure_ascii=False, indent=1)
            step = parsed.get("pivot_step")
            # 随机基线：从 1..n_steps 随机挑一轮的命中概率（期望）
            k = len(all_ours)
            chance = (k / n_steps) if n_steps else 0.0
            rows.append(dict(dir=d, run=b, final=final, n_steps=n_steps,
                             ours=all_ours, harmful=sorted(harmful),
                             beneficial=sorted(beneficial), step=step,
                             chance_exact=chance))
            print(f"[{d}/{b}] steps={n_steps} ours={all_ours} teacher_status={step} "
                  f"chance={chance:.2f}")

    # 三组对比
    def stats(use_status, key_attr):
        exact = near = total = sign_bad = sign_hit = noedit_over = noedit = 0
        chance_sum = 0.0
        for x in rows:
            ours = x["ours"]
            step = x.get(key_attr) if use_status else x.get("base_step")
            chance_sum += x["chance_exact"]
            if not ours:
                noedit += 1
                if step not in (None, 0):
                    noedit_over += 1
                continue
            total += 1
            if isinstance(step, int) and step in ours:
                exact += 1
                near += 1
                if step in x["beneficial"] and step not in x["harmful"]:
                    sign_bad += 1
                sign_hit += 1
            elif isinstance(step, int) and any(abs(step - o) <= 1 for o in ours):
                near += 1
        return dict(exact=exact, near=near, total=total, sign_bad=sign_bad,
                    sign_hit=sign_hit, noedit=noedit, noedit_over=noedit_over,
                    chance=chance_sum / max(1, len(rows)))

    print("\n===== 汇总（脚本内 teacher 仅跑了 with-status；baseline 来自已有 *_teacher.json）=====")
    for x in rows:
        t = os.path.join(x["dir"], f"{x['run']}_teacher.json")
        if os.path.exists(t):
            d = json.load(open(t))
            x["base_step"] = (d.get("teacher") or {}).get("pivot_step")
        else:
            x["base_step"] = None
    base = stats(False, "base_step")
    stat = stats(True, "step")
    for name, s in (("A 教师无状态", base), ("B 教师+逐轮状态", stat)):
        print(f"{name}: exact {s['exact']}/{s['total']} (±1 {s['near']}/{s['total']}), "
              f"符号说反 {s['sign_bad']}/{s['sign_hit']}, "
              f"无编辑仍指认 {s['noedit_over']}/{s['noedit']}")
    print(f"C 随机基线 expected exact = {base['chance']:.3f}")

    json.dump({"rows": rows, "base": base, "with_status": stat},
              open(os.path.join(args.out, "summary.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
