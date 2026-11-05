"""pivot_inspector.report — 把验证结果渲染成一页诊断报告（HTML/Markdown）。"""
from __future__ import annotations

import json
import os
from html import escape


LABELS = {
    "causal": ("关键错误步", "#e5484d"),
    "causal_by_file": ("关键错误步(文件级)", "#e5484d"),
    "beneficial": ("有效修复步", "#30a46c"),
    "beneficial_by_file": ("有效修复步(文件级)", "#30a46c"),
}


def load(dirpath, run):
    report = json.load(open(os.path.join(dirpath, f"{run}.json")))
    traj_path = os.path.join(dirpath, f"{run}.traj.json")
    traj = json.load(open(traj_path)) if os.path.exists(traj_path) else None
    teacher = None
    tpath = os.path.join(dirpath, f"{run}_teacher.json")
    if os.path.exists(tpath):
        teacher = json.load(open(tpath))
    return report, traj, teacher


def summarize(report, traj, teacher):
    evs = report.get("evals") or []
    turns = [e.get("turn") for e in evs]
    nfails = [e.get("n_fail") for e in evs]
    final = nfails[-1] if nfails else None
    cfs = report.get("counterfactuals") or []
    recs = report.get("recoveries") or []
    harmful = [c for c in cfs if c.get("label") in ("causal", "causal_by_file")]
    beneficial = [c for c in cfs if c.get("label") in ("beneficial", "beneficial_by_file")]
    ncmds = 0
    if traj:
        for m in traj.get("messages", []):
            if m.get("role") == "assistant":
                ncmds += sum(1 for tc in (m.get("tool_calls") or [])
                             if tc.get("function", {}).get("name") == "bash")
    teacher_step = (teacher or {}).get("teacher", {}).get("pivot_step")
    return dict(turns=turns, nfails=nfails, final=final, harmful=harmful,
                beneficial=beneficial, recs=recs, ncmds=ncmds,
                teacher_step=teacher_step,
                instance=report.get("instance_id"))


def evidence_text(c):
    t = c.get("turn")
    lab = c.get("label", "")
    names = {
        "causal": f"跳过第 {t} 轮 → 最终失败数 {c.get('real_final_n_fail')} → {c.get('cf_final_n_fail')}（减少，说明该步有害）",
        "beneficial": f"跳过第 {t} 轮 → 最终失败数 {c.get('real_final_n_fail')} → {c.get('cf_final_n_fail')}（增加，说明该步有效）",
    }
    if lab in names:
        return names[lab]
    if lab == "causal_by_file":
        return (f"反事实重放失效；把第 {t} 轮改过的文件回滚后，最终失败数 "
                f"{c.get('real_final_n_fail')} → {c.get('fallback_n_fail')}（减少）")
    if lab == "beneficial_by_file":
        return (f"反事实重放失效；把第 {t} 轮改过的文件回滚后，最终失败数 "
                f"{c.get('real_final_n_fail')} → {c.get('fallback_n_fail')}（增加）")
    return f"第 {t} 轮：{lab}"


def bar(n, maxn=6):
    fill = "#" * min(int(n or 0), maxn)
    return fill or "·"


def to_html(dirpath, run, out, title=None):
    report, traj, teacher = load(dirpath, run)
    s = summarize(report, traj, teacher)
    inst = s["instance"] or "unknown"
    title = title or f"PivotInspector · {inst} · run {run}"
    ok = s["final"] == 0
    verdict = "任务成功" if ok else f"任务失败（最终失败测试数 {s['final']}）"
    vcolor = "#30a46c" if ok else "#e5484d"

    # 结论卡片
    cards = []
    if s["harmful"]:
        turns = ", ".join(str(c["turn"]) for c in s["harmful"])
        cards.append((f"关键错误步：{turns}", "#e5484d",
                      "撤销该步，最终结果变好 → 验证过的有害编辑"))
    if s["beneficial"]:
        turns = ", ".join(str(c["turn"]) for c in s["beneficial"])
        cards.append((f"有效修复步：{turns}", "#30a46c",
                      "撤销该步，最终结果变差 → 验证过的有效编辑"))
    if not s["harmful"] and not s["beneficial"]:
        cards.append(("未发现可归因的关键步", "#8b8d98",
                      "移除任何单轮都不改变最终结果 → 失败不是由某个编辑造成"))
    if s["recs"]:
        pairs = ", ".join(f"({r['pivot_turn']}→{r['recovery_turn']})" for r in s["recs"])
        cards.append((f"改坏→修复配对：{pairs}", "#8e4ec6",
                      "中途改坏、后来修回的轮次"))

    # 时间线
    tl_rows = []
    for t, n in zip(s["turns"], s["nfails"]):
        color = "#f1f1f3"
        tag = ""
        for c in s["harmful"]:
            if c.get("turn") == t:
                color, tag = "#ffe0e0", "关键错误"
        for c in s["beneficial"]:
            if c.get("turn") == t:
                color, tag = "#dcf5e7", "有效修复"
        for r in s["recs"]:
            if r.get("pivot_turn") == t:
                color, tag = "#ffe0e0", "改坏"
            if r.get("recovery_turn") == t:
                color, tag = "#f3e8ff", "修复"
        tl_rows.append(
            f"<tr style='background:{color}'><td>第 {t} 轮</td>"
            f"<td>{'▇' * min(int(n or 0), 8) if n else '·'}</td>"
            f"<td>{n if n is not None else '?'}</td><td>{tag}</td></tr>")

    # 证据
    ev_items = "".join(f"<li>{escape(evidence_text(c))}</li>"
                       for c in s["harmful"] + s["beneficial"])
    if not ev_items:
        ev_items = "<li>无（没有通过因果验证的状态变化轮）</li>"

    rec_items = "".join(
        f"<li>第 {r['pivot_turn']} 轮改坏 → 第 {r['recovery_turn']} 轮修复"
        + ("；跳过修复轮 → 最终失败数增加（修复因果成立）"
           if r.get("recovery_causal") else "；跳过修复轮 → 最终失败数不变（未通过验证）")
        + "</li>" for r in s["recs"])

    teacher_txt = ("未运行教师对照" if s["teacher_step"] is None
                   else f"教师事后指认：第 {s['teacher_step']} 轮（我们验证的轮次见上）")

    html = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>{escape(title)}</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      max-width:820px;margin:40px auto;padding:0 20px;color:#1c1d21;line-height:1.55}}
 h1{{font-size:22px;margin:0 0 2px}} .sub{{color:#6b6d76;font-size:13px;margin-bottom:18px}}
 .verdict{{font-size:16px;font-weight:600;color:{vcolor};margin:10px 0 18px}}
 .cards{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:22px}}
 .card{{border:1px solid #e3e3e8;border-left:4px solid var(--c);border-radius:8px;
        padding:10px 14px;min-width:220px;flex:1}}
 .card b{{display:block;font-size:14px;margin-bottom:3px}}
 .card span{{color:#6b6d76;font-size:12.5px}}
 h2{{font-size:15px;margin:26px 0 10px;border-bottom:1px solid #ebebef;padding-bottom:5px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 td,th{{padding:5px 8px;border-bottom:1px solid #f0f0f3;text-align:left}}
 th{{color:#6b6d76;font-weight:500;font-size:12px}}
 code{{background:#f5f5f7;padding:1px 5px;border-radius:4px;font-size:12px}}
 .meta{{font-size:12px;color:#8b8d98;margin-top:26px}}
</style>
<h1>{escape(title)}</h1>
<div class="sub">{escape(inst)} · 共 {len(s['turns']) - 1} 轮 · {s['ncmds']} 条命令</div>
<div class="verdict">{'✅ ' if ok else '❌ '}{escape(verdict)}</div>

<div class="cards">
{''.join(f'<div class="card" style="--c:{c}"><b>{escape(t)}</b><span>{escape(d)}</span></div>' for t, c, d in cards)}
</div>

<h2>每轮测试状态</h2>
<table><tr><th>轮次</th><th>失败数</th><th>n_fail</th><th>标注</th></tr>
{''.join(tl_rows)}
</table>

<h2>因果证据（反事实重放）</h2>
<ul style="font-size:13.5px">{ev_items}</ul>

<h2>改坏 → 修复 配对</h2>
<ul style="font-size:13.5px">{rec_items or '<li>无</li>'}</ul>

<h2>对照</h2>
<p style="font-size:13.5px">{escape(teacher_txt)}</p>
<div class="meta">PivotInspector · 数据 {escape(dirpath)} / run {escape(str(run))} ·
验证方式：跳过该轮（或回滚该轮文件）后重放判分</div>
</html>"""
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    open(out, "w").write(html)
    return s
