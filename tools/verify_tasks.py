#!/usr/bin/env python3
"""验证 SWE-smith 任务：clean → 打 bug patch → F2P应挂/P2P应过 → 反向 → 应全过"""
import json, re, subprocess, sys, tempfile, shutil, os
from pathlib import Path

BASE = Path(os.environ.get("SWESMITH_RL_HOME", Path(__file__).resolve().parents[1]))
REPOS = BASE / "probe"
ENVS  = {"mistune": "mistune", "sqlparse": "sqlparse",
         "isodate": "isodate", "boltons": "boltons"}

def repo_of(i): return i["repo"].split("/")[-1].split("__")[1].split(".")[0]

def run_tests(wd, env, nodes, timeout=180):
    cmd = ["conda","run","-n",env,"--no-capture-output",
           "pytest","-v","--tb=no","-p","no:cacheprovider","--no-header"] + nodes
    env2 = dict(os.environ, PYTHONPATH=f"{wd}:{wd}/src", COLUMNS="200")  # 防止 pytest 折行
    try:
        r = subprocess.run(cmd, cwd=wd, env=env2, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}
    out = {}
    for line in (r.stdout + r.stderr).splitlines():
        m = re.match(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)", line.strip())
        if m: out[m.group(1)] = m.group(2)
    return out

def verify(inst):
    name = repo_of(inst)
    env, src = ENVS.get(name), REPOS / name
    if not env or not src.exists(): return "skip", f"跳过 {name}"

    f2p, p2p = inst["FAIL_TO_PASS"], inst["PASS_TO_PASS"]
    allt = f2p + p2p

    with tempfile.TemporaryDirectory(dir=BASE/"tmp") as td:
        wd = Path(td)/name
        shutil.copytree(src, wd, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__","*.pyc",".eggs","*.egg-info"))

        clean = run_tests(wd, env, allt)
        if [t for t in allt if t not in clean]: return "bad", "测试未收集到"
        if not all(clean[t]=="PASSED" for t in allt):
            bad=[t for t in allt if clean[t]!="PASSED"]
            return "bad", f"clean状态不干净 {bad[:1]}"

        pf = Path(td)/"bug.patch"; pf.write_text(inst["patch"])
        ap = subprocess.run(["git","apply","--whitespace=nowarn",str(pf)],
                            cwd=wd, capture_output=True, text=True)
        if ap.returncode != 0: return "bad", f"bug patch失败: {ap.stderr.strip()[:60]}"

        buggy = run_tests(wd, env, allt)
        ok_f2p = all(buggy.get(t) in ("FAILED","ERROR") for t in f2p)
        ok_p2p = all(buggy.get(t)=="PASSED" for t in p2p)

        subprocess.run(["git","apply","-R","--whitespace=nowarn",str(pf)], cwd=wd,
                       capture_output=True, text=True)
        fixed = run_tests(wd, env, allt)
        ok_fix = all(fixed.get(t)=="PASSED" for t in allt)

        if ok_f2p and ok_p2p and ok_fix: return "ok",""
        why=[]
        if not ok_f2p:
            b=[t for t in f2p if buggy.get(t) not in ("FAILED","ERROR")]
            why.append(f"bug后F2P未挂{b[:1]}")
        if not ok_p2p:
            b=[t for t in p2p if buggy.get(t)!="PASSED"]
            why.append(f"bug后P2P挂{b[:1]}")
        if not ok_fix: why.append("反向修复后未全过")
        return "bad", "; ".join(why)

def main(sf):
    tasks=[json.loads(l) for l in open(sf)]
    print(f"共 {len(tasks)} 道\n", flush=True)
    fok=open(BASE/"tasks_ok.jsonl","w"); fbad=open(BASE/"tasks_bad.jsonl","w")
    st={"ok":0,"bad":0,"skip":0}
    for i,t in enumerate(tasks,1):
        s,m=verify(t); st[s]+=1
        if s=="ok": fok.write(json.dumps(t,ensure_ascii=False)+"\n")
        elif s=="bad": t["_reason"]=m; fbad.write(json.dumps(t,ensure_ascii=False)+"\n")
        if i%20==0 or i==len(tasks):
            print(f"[{i:4}/{len(tasks)}] ok={st['ok']} bad={st['bad']} skip={st['skip']}",flush=True)
    fok.close(); fbad.close()
    print(f"\n=== 结果 ===\n  有效 {st['ok']}\n  无效 {st['bad']}\n  跳过 {st['skip']}")

if __name__=="__main__":
    # 默认接 export_more.py 的产出
    main(sys.argv[1] if len(sys.argv)>1 else str(BASE/"tasks_more.jsonl"))
