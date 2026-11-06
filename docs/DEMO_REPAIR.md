# 受控清理实验（Controlled Repair / Cleanup Experiment）

**问题**：失败轨迹里存在已验证的有害编辑。如果 agent 在失败后能准确地撤销"那一步"，
状态会不会变好？这是"标签有没有用"的第一个下游检验。

**结论先行**：撤销我们验证过的有害轮，**9/9** 条轨迹的失败数恢复到基线（破坏被精确清除）；
随机撤销 0/9、教师指认撤销 3/9。但 **resolve 没有提升**（所有条件 0/9）——撤销后只剩
原本就难解决的目标测试，8 步 continuation agent 修不出来。详见「局限」。

数据在 `results/repair_d1a/`（主批次）和 `results/repair_d1a_precise/`（精确回退补充实验）。

---

## 1. 数据

flask-5063（F2P=2, P2P=54）用 mini-swe-agent + DeepSeek 生成 50 条轨迹（`gen_many.sh`），
66% 失败（33/50）。`select_cases.py` 从中选出 **9 条失败且含"已验证有害步"的轨迹**：

`demo/cases_d1a.json`：run 6, 14, 15, 23, 24, 25, 28, 32, 40。

| run | 最终 n_fail | 有害轮 | 证据 |
|---|---|---|---|
| 6 | 4 | 14 | 跳过该轮：4→2 |
| 14 | 56 | 14 | 反事实失效（后续命令互相依赖），文件级回退后 2 |
| 15 | 1 | 8 | 跳过该轮：1→0（注意：该轮本身让 n_fail 2→1，但反事实显示它挡住了更好的结果） |
| 23 | 6 | 10 | 跳过该轮：6→3（第 15 轮为 improvement 56→6） |
| 24 | 4 | 5 | 跳过该轮：4→2 |
| 25 | 56 | 8 | 跳过该轮：56→2 |
| 28 | 3 | 13 | 跳过该轮：3→2 |
| 32 | 4 | 6 | 跳过该轮：4→2 |
| 40 | 4 | 4 | 跳过该轮：4→2 |

基线 n_fail = 2（= FAIL_TO_PASS 数量，始终失败的 2 个目标测试）。

---

## 2. 实验设计

对每条轨迹，重建它的最终状态，然后比较四种干预（其余完全相同）：

| 条件 | 干预 |
|---|---|
| `continue` | 不撤销，直接从最终状态继续 |
| `revert_random` | 随机选一个与有害轮无关的轮，回滚它改过的文件 |
| `revert_teacher` | 回滚教师事后指认的那一轮改过的文件 |
| `revert_ours` | 回滚我们反事实验证的有害轮改过的文件 |

干预后统一给同一个 continuation agent（DeepSeek chat，最多 8 条 bash 命令、无隐藏测试反馈）
继续修改，然后按官方 SWE-bench eval 语义在**仓库副本**里评测（避免隐藏测试泄露给 agent 工作区）：

1. 把测试文件还原到 base commit；
2. apply `test_patch`；
3. `pytest -rA <整个测试文件>`；
4. 统计 n_fail（F2P+P2P 中失败数）、f2p_pass。

实现：`oracle/repair_resume.py`（单条）、`oracle/run_repair_batch.sh`（批量，断点续跑 + 自动入库）、
`oracle/stats_repair.py`（配对统计 + bootstrap CI）。

---

## 3. 主批次结果

| run | continue | random | teacher | ours |
|---|---|---|---|---|
| 6 | 4→4 | 4→4 | 4→4 | **2→2** |
| 14 | 56→56 | 56→56 | 56→56 | **2→2** |
| 15 | 1→1 | 1→1 | 1→1 | 2→2 |
| 23 | 6→6 | 6→6 | 6→6 | **2→2** |
| 24 | 4→4 | 4→4 | **2→2** | **2→2** |
| 25 | 56→2 | 56→56 | 56→56 | **2→2** |
| 28 | 3→3 | 3→3 | 3→3 | **2→2** |
| 32 | 4→4 | 4→4 | **2→2** | **2→2** |
| 40 | 4→4 | 4→4 | **2→4** | **2→2** |

（单元格 = 干预后 n_fail → 最终 n_fail；加粗 = 干预后恢复到基线 2）

| 条件 | 干预后 n_fail 均值 | 恢复到基线 | 最终 n_fail 均值 | 最终 f2p_pass 合计 | resolve |
|---|---|---|---|---|---|
| continue | 15.3 | 0/9 | 9.3 | 1 | 0/9 |
| revert_random | 15.3 | 0/9 | 15.3 | 1 | 0/9 |
| revert_teacher | 14.7 | 3/9 | 14.9 | 1 | 0/9 |
| **revert_ours** | **2.0** | **9/9** | **2.0** | 0 | 0/9 |

配对最终 n_fail 差（ours − 其他，bootstrap 5000 次）：

| 对比 | 均值差 | 95% CI |
|---|---|---|
| ours − continue | −7.33 | [−19.22, −0.78] |
| ours − revert_random | −13.33 | [−30.67, −1.44] |
| ours − revert_teacher | −12.89 | [−30.22, −0.78] |

观察：

- 教师只在 24/32/40 三条上恰好指到与我们相同的轮（3/9），其中 case 40 回退后又被
  continuation agent 改坏回 4；
- continue 在 case 25 意外把 56 修回 2（灵光一现的 import 修复），其余无变化；
- 我们的条件在 8/9 条上把状态清回基线；case 15 例外（见下）。

---

## 4. 精确回退补充实验

主批次用 `git checkout <pre-turn-sha> -- <files>` 回滚整文件，会连带回滚同一文件里后续轮的修改。
改成"只反转该轮 diff"（`git apply -R --3way`，冲突才 fallback 整文件）后重跑 9 条 `revert_ours`：

- 精确反转成功 3/9（run 6、15、28），其余 6 条仍因冲突 fallback；
- **结果与整文件回退完全一致**：9/9 干预后 n_fail=2，最终 n_fail=2，f2p_pass=0。

说明粒度不是瓶颈。根因是**回滚的非交换性**：后续轮次的代码是在有害轮改动之上写的，
把有害轮撤掉会产生与"从未执行该轮"不同的状态。case 15 就是典型：

1. 第 8 轮实际让 n_fail 2→1（improvement），但反事实跳过它得到 0（说明它是 harmful）；
2. 从最终状态撤掉第 8 轮 → n_fail 回到 2（而不是反事实的 0）；
3. 后续轮的修改依赖第 8 轮，撤掉后一起失效。

---

## 5. 结论与局限

**能写**：

- 执行验证的标签能因果地定位"自伤编辑"：撤销标注轮后 9/9 恢复基线，random 0/9、teacher 3/9；
- 随机/教师对比 + 配对 CI 排除了"随便撤销也有同样效果"的解释。

**不能写 / 局限**：

- **resolve 无提升**（0/9 全部条件）：撤销后只剩原始 F2P（原始 agent 30–51 轮都没解决，
  8 步续跑 agent 修不出来）。"清除自伤"≠"修好任务"；
- 只有 9 条轨迹、1 个实例；
- 回滚非交换性：撤销后状态 ≠ 从未执行该轮；case 15 因此倒退（1→2），是本方法已知失效模式；
- continuation agent 无隐藏测试反馈（所有条件一致），若给测试反馈结论可能不同，需另做实验。

---

## 6. 复现

```bash
# 生成 50 条轨迹（Codespace）
bash oracle/gen_many.sh d1a pallets__flask-5063 10 50 4

# 回放 + 验证（得到 results/d1a/<run>.json）
bash oracle/run_pipeline.sh pallets__flask-5063 <image> 50 results/d1a ...

# 挑选候选
python oracle/select_cases.py --results-dir results/d1a --out demo/cases_d1a.json

# 主批次（4 条件 × 9 条）
bash oracle/run_repair_batch.sh demo/cases_d1a.json <image> 8 results/repair_d1a results/d1a pallets__flask-5063

# 精确回退补充（只跑 revert_ours）
CONDS="revert_ours" bash oracle/run_repair_batch.sh demo/cases_d1a.json <image> 8 \
  results/repair_d1a_precise results/d1a pallets__flask-5063

# 统计
python oracle/stats_repair.py results/repair_d1a
```
