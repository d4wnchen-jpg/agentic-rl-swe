#!/usr/bin/env python3
"""生成 oracle 模式的本地 demo：一个含 3 个缺陷的 mini repo + gold patch + task.json。

真跑公开任务时不需要这个脚本；它只是为了在无 pytest/无网络的环境下验证
replay_oracle.py 的代码路径。
"""
import json
import os
import shutil
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "demo_repo")

BUGGY = """def add(a, b):
    return a + b


# ------------------------------------------------------------------
# sub: subtraction helper
# signature mirrors add(); used by check.py
# ------------------------------------------------------------------
def sub(a, b):
    return a + b  # BUG: should be a - b


# ------------------------------------------------------------------
# mul: multiplication helper
# signature mirrors add(); used by check.py
# ------------------------------------------------------------------
def mul(a, b):
    return a + b  # BUG: should be a * b


# ------------------------------------------------------------------
# neg: unary negation
# signature mirrors add(); used by check.py
# ------------------------------------------------------------------
def neg(a):
    return a  # BUG: should be -a
"""

FIXED = """def add(a, b):
    return a + b


# ------------------------------------------------------------------
# sub: subtraction helper
# signature mirrors add(); used by check.py
# ------------------------------------------------------------------
def sub(a, b):
    return a - b


# ------------------------------------------------------------------
# mul: multiplication helper
# signature mirrors add(); used by check.py
# ------------------------------------------------------------------
def mul(a, b):
    return a * b


# ------------------------------------------------------------------
# neg: unary negation
# signature mirrors add(); used by check.py
# ------------------------------------------------------------------
def neg(a):
    return -a
"""

CHECK = """import sys
from calc import add, sub, mul, neg

cases = [
    ("test_add", add(2, 3) == 5),
    ("test_sub", sub(5, 3) == 2),
    ("test_mul", mul(2, 3) == 6),
    ("test_neg", neg(3) == -3),
]
fails = 0
for name, ok in cases:
    print(("PASSED " if ok else "FAILED ") + name)
    fails += 0 if ok else 1
sys.exit(1 if fails else 0)
"""


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main():
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    os.makedirs(REPO)
    open(os.path.join(REPO, "calc.py"), "w").write(BUGGY)
    open(os.path.join(REPO, "check.py"), "w").write(CHECK)

    sh(["git", "init", "-q"], REPO)
    sh(["git", "-c", "user.email=a@b", "-c", "user.name=a", "add", "-A"], REPO)
    sh(["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base"], REPO)

    open(os.path.join(REPO, "calc.py"), "w").write(FIXED)
    diff = sh(["git", "diff"], REPO).stdout
    sh(["git", "checkout", "--", "."], REPO)
    open(os.path.join(BASE, "demo_gold.patch"), "w").write(diff)

    task = {
        "instance_id": "demo__calc",
        "patch": diff,
        "FAIL_TO_PASS": ["test_sub", "test_mul", "test_neg"],
        "PASS_TO_PASS": ["test_add"],
    }
    json.dump(task, open(os.path.join(BASE, "demo_task.json"), "w"), ensure_ascii=False, indent=1)

    # 保留 repo 干净（去掉 .git，模拟真实 base repo 副本）
    shutil.rmtree(os.path.join(REPO, ".git"), ignore_errors=True)
    print("demo ready:", REPO, "| hunks in patch:", diff.count("\n@@") + diff.startswith("@@"))


if __name__ == "__main__":
    main()
