#!/usr/bin/env python3
"""teacher_label.py — G2：让教师模型事后指认 pivotal step，与我们验证的标签对比。

教师只看：题目 + 轨迹（命令与输出摘要）+ 最终测试结果。
它回答：哪一步是关键错误（0 = 没有单一错误步 / 从未尝试修复）。

用法：
  python teacher_label.py \
    --traj mswa_g1c_1/pallets__flask-5063/pallets__flask-5063.traj.json \
    --task SWE-bench_Lite.jsonl --instance-id pallets__flask-5063 \
    --report out_g1cv_1/traj_report.json \
    --out out_teacher_1 --model deepseek-chat
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from openai import OpenAI

SYSTEM = ("You are an expert software engineer doing a retrospective analysis of a failed "
          "SWE-agent attempt. You see the problem, the agent's commands, and the final test "
          "results. Be precise and skeptical.")


def truncate(s, n=400):
    s = (s or "").strip()
    return s if len(s) <= n else s[:200] + " ... " + s[-150:]


def build_trace(traj, max_chars=60000):
    turns = []
    step = 0
    for m in traj.get("messages", []):
        role = m.get("role")
        if role == "assistant":
            cmds = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                if fn.get("name") == "bash":
                    try:
                        a = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        continue
                    if a.get("command"):
                        cmds.append(a["command"])
            if cmds:
                step += 1
                turns.append(f"### Step {step}\n" + "\n".join(
                    f"$ {truncate(c, 800)}" for c in cmds))
        elif role == "tool":
            if turns:
                turns.append("observation: " + truncate(m.get("content", ""), 400))
    text = "\n".join(turns)
    return text[-max_chars:], step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--report", default="", help="traj_report.json（用于对比）")
    ap.add_argument("--out", default="out_teacher")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--max-chars", type=int, default=60000)
    args = ap.parse_args()

    task = None
    with open(args.task) as f:
        if args.task.endswith(".jsonl"):
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    if t["instance_id"] == args.instance_id:
                        task = t
                        break
        else:
            t = json.load(f)
            if t["instance_id"] == args.instance_id:
                task = t
    if task is None:
        raise SystemExit("task 未找到")

    traj = json.load(open(args.traj))
    trace, n_steps = build_trace(traj, args.max_chars)

    final_text = "unknown"
    final_n_fail = None
    ours = []
    if args.report and os.path.exists(args.report):
        r = json.load(open(args.report))
        fin = (r.get("evals") or [{}])[-1]
        final_n_fail = fin.get("n_fail")
        final_text = (f"final n_fail={fin.get('n_fail')}, n_pass={fin.get('n_pass')}; "
                      f"F2P failed: {[n for n, s in (fin.get('statuses') or {}).items() if s not in ('PASSED','XFAIL','FLAKY')][:8]}")
        ours = r.get("recoveries") or []
        cfs = r.get("counterfactuals") or []
        fatal = {c.get("turn") for c in cfs
                 if c.get("label") in ("causal", "causal_by_file")}
        our_pivots = sorted(
            {x.get("pivot_turn") for x in ours if x.get("pivot_turn")} | fatal)
    else:
        our_pivots = []

    if final_n_fail == 0:
        question = ("This attempt eventually SUCCEEDED. Identify the single most pivotal "
                    "mistake step (1-indexed) that was made and later recovered, i.e. a step "
                    "that made tests worse before they were fixed. If there was no such "
                    "recoverable mistake, answer pivot_step = 0.")
    else:
        question = ("This attempt FAILED. Identify the single most pivotal step (1-indexed) "
                    "that caused the failure. If the failure is because the fix was never "
                    "attempted / the agent did nothing effective, answer pivot_step = 0. "
                    "If a step made tests worse and they were never recovered, that is a "
                    "pivotal mistake.")

    user = f"""## Problem statement
{task['problem_statement']}

## Agent trajectory (commands + brief observations, {n_steps} steps)
{trace}

## Final test result
{final_text}

## Question
{question}
Return ONLY JSON:
{{"pivot_step": <int>, "reason": "<short>", "correct_action": "<short>"}}
"""

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        envp = os.path.expanduser("~/.config/mini-swe-agent/.env")
        if os.path.exists(envp):
            for line in open(envp):
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY 未设置（环境变量或 ~/.config/mini-swe-agent/.env）")
    client = OpenAI(api_key=api_key,
                    base_url="https://api.deepseek.com/v1")
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0,
        max_tokens=500,
    )
    raw = resp.choices[0].message.content
    m = re.search(r"\{.*\}", raw, re.S)
    parsed = json.loads(m.group(0)) if m else {"pivot_step": None, "raw": raw}

    os.makedirs(args.out, exist_ok=True)
    result = {"instance_id": args.instance_id, "model": args.model,
              "n_steps": n_steps, "teacher": parsed,
              "our_pivot_turns": our_pivots, "raw": raw}
    json.dump(result, open(os.path.join(args.out, "teacher_label.json"), "w"),
              ensure_ascii=False, indent=1)

    t = parsed.get("pivot_step")
    print(f"[teacher] pivot_step={t} reason={parsed.get('reason','')[:120]}")
    print(f"[ours]    verified pivot turns={our_pivots}")
    if ours and isinstance(t, int):
        exact = t in our_pivots
        near = any(abs(t - p) <= 1 for p in our_pivots)
        print(f"[compare] exact={exact} within1={near} (teacher=0 表示它认为没有关键错误步)")


if __name__ == "__main__":
    main()
