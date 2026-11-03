# L1 Oracle 模式报告（2026-10-31）

> 目的：用数据集自带的 **gold patch** 当"完美轨迹"，验证「逐轮应用 → 逐轮判分 → 单 hunk 回滚」在真实任务上是否可行。
> 与真实轨迹的区别：oracle 不犯错、单调变好；它测的是**管道**，不是模型分布。

## 1. 代码与产物

| 文件 | 作用 |
|---|---|
| `opd_agent/smoke_test/oracle/replay_oracle.py` | 主脚本（oracle 模式） |
| `opd_agent/smoke_test/oracle/make_demo.py` | 生成本地 demo（无 pytest/无网络时验证代码路径） |
| `opd_agent/smoke_test/oracle/out_demo/` | demo 运行结果（`ORACLE.md` / `oracle_report.json` / 拆分后的 hunk patch） |

## 2. 本地 demo 结果（3 个 hunk、4 个测试）

demo 设计：mini repo 有 3 个独立缺陷（`sub` / `mul` / `neg`），gold patch 恰好 3 个 hunk，每个 hunk 修一个缺陷、对应一个测试。

### 逐轮应用（测试状态轨迹）

| turn | hunk | apply | n_fail | n_pass | test_ok | test_s |
|---|---|---|---|---|---|---|
| 1 | 1（修 sub） | ✅ | 2 | 2 | False | 0.06 |
| 2 | 2（修 mul） | ✅ | 1 | 3 | False | 0.06 |
| 3 | 3（修 neg） | ✅ | 0 | 4 | **True** | 0.06 |

### 单 hunk 回滚（从最终全绿状态）

| 回滚 | revert | n_fail | 只挂掉的测试 |
|---|---|---|---|
| hunk 1 | ✅ | 1 | `test_sub` |
| hunk 2 | ✅ | 1 | `test_mul` |
| hunk 3 | ✅ | 1 | `test_neg` |

**结论**：每个 hunk 的因果角色被测试精确区分——这正是 turn 级方法给不了的信号，也是 hunk 级标签成立的直接证据。

## 3. 在真实公开任务上跑（SWE-bench Lite）

### 前提

1. **repo 处于 `base_commit`**（SWE-bench 数据集的 `repo` + `base_commit` 字段）；
2. 一个能跑该 repo 测试的环境：**Docker 镜像**或**conda env**；
3. 数据集文件：`SWE-bench_Lite.jsonl`（HF: `SWE-bench/SWE-bench_Lite`）。

### 命令（Docker 路线，最省事）

```bash
python replay_oracle.py \
  --task SWE-bench_Lite.jsonl \
  --instance-id astropy__astropy-12907 \
  --repo /path/to/astropy_at_base_commit \
  --apply-test-patch \
  --test-cmd 'docker run --rm -v {work}:/testbed -w /testbed \
      swebench/sweb.eval.x86_64.astropy__astropy-12907:latest \
      pytest --color=no -rA -p no:cacheprovider --no-header {nodes}' \
  --out out/astropy_12907
```

说明：
- `--apply-test-patch` 会先把 `test_patch` 打到基线上（否则 F2P 测试不存在）；
- `{nodes}` 会被替换成 `FAIL_TO_PASS + PASS_TO_PASS` 的 nodeid 列表；
- `{work}` 是工作目录的绝对路径；
- 镜像 tag 以官方 harness 实际输出为准（形如 `swebench/sweb.eval.x86_64.<instance_id>:latest`）。

### 命令（conda 路线）

```bash
--test-cmd 'conda run -n <env> pytest --color=no -rA -p no:cacheprovider --no-header {nodes}'
```

### 预期输出

- 终端：每轮 `[turn k] hunkNN apply=True n_fail=... test_ok=...`
- 文件：`out/<id>/ORACLE.md`（两张表）+ `oracle_report.json` + 拆分好的 `hunkNN.patch`

## 4. 已知限制 / 待补

| 限制 | 影响 | 处理 |
|---|---|---|
| **SWE-smith 的 `patch` 是"注入缺陷"补丁**，gold 修复是它的反向 | 直接跑会把缺陷再注入一次 | 下一步加 `--invert-patch` |
| Docker 首次拉镜像很慢 | 第一次 `test_s` 会把拉镜像时间算进去 | 先手动 `docker pull`，或记住首轮偏大 |
| `test_s` 含容器启动开销（秒级） | 估算真实判分成本偏高 | 记录多次后取中位数，或改用常驻容器 |
| pytest 输出格式依赖 `-rA` | 解析不到状态时 `n_fail=None` | 测试命令里必须保留 `-rA` |
| 只支持单文件多 hunk / 多文件均可，但 hunk 反向应用要求上下文未变 | 后续轮次改同一处会失败 | 脚本已自动回退 `--3way` |

## 5. 下一步（建议顺序）

1. **1 道真实任务**：选 SWE-bench Lite 里 F2P 少的题，跑通上面的命令；记录首轮耗时；
2. **5 道任务**：确认不同 repo 的测试耗时分布（决定后面能采多少轨迹）；
3. **接真实轨迹**：把 `git apply hunk` 换成重放 agent 动作（mini-swe-agent bash-only 最容易）；
4. **加 `--invert-patch`**：支持 SWE-smith 的缺陷注入补丁。

---

## 6. 真实任务结果（2026-10-31，Codespaces）

**任务**：`pallets__flask-4992`（F2P=1，P2P=18）
**环境**：GitHub Codespaces 2 核 / 8 GB；Docker 29.9.2；镜像
`swebench/sweb.eval.x86_64.pallets_1776_flask-4992:latest`（4.23 GB）
**repo**：`pallets/flask` @ `4c288bc9`（base_commit）

### 逐轮应用

| turn | hunk | apply | n_fail | n_pass | test_s |
|---|---|---|---|---|---|
| 1 | 1（签名加 `text`） | ✅ | 1 | 18 | 1.63 |
| 2 | 2（docstring 示例） | ✅ | 1 | 18 | 1.04 |
| 3 | 3（docstring + 函数体） | ✅ | 0 | 19 | 0.79 |

### 单 hunk 回滚（从最终状态）

| 回滚 | n_fail | 解释 |
|---|---|---|
| hunk1 | 3 | 签名去掉 `text` 但函数体仍引用 → TypeError，连坐 3 个测试 |
| hunk2 | 0 | **纯 docstring**，回滚无影响 |
| hunk3 | 1 | 实现被撤掉 → F2P 挂 |

### 三个关键数字

| 数 | 结果 | 意义 |
|---|---|---|
| 单次判分 | 0.79–1.63 s（含容器启动） | 20 轮轨迹 ≈ 30 s；100 条 ≈ 1 h，**可行** |
| 中间状态信号 | 有（1→1→0；回滚可区分） | L1 核心假设成立 |
| 环境准备 | 镜像 4.23 GB + 拉取 | 一次性成本 |

### ⚠️ 最重要的发现：diff hunk ≠ 语义编辑

`flask-4992` 的 gold patch 是**一个功能改动**（给 `from_file` 加 `text` 参数）被 diff 切成 3 块：

- hunk1 = 函数签名；
- hunk2 = docstring 示例（**无功能影响**）；
- hunk3 = docstring + 函数体（真正实现）。

因此：

1. **单独回滚一个 hunk 会破坏语法/语义**，产生“连坐失败”的假信号（hunk1 回滚 → 3 个测试挂）；
2. **文档/注释 hunk 是惰性的**，回滚无影响，会污染“哪一步关键”的判定；
3. 结论：**粒度定义应为「agent 的一次编辑动作（edit）」**，diff hunk 只用于 oracle 模式构造；回滚需过语法与可执行性过滤；文档 hunk 要剔除；
4. 对大编辑，更干净的干预是「把这段编辑替换成教师给出的修正版」（intervention），而不是直接 revert。

### 环境备注

- Codespaces 免费额度 120 core-hours；2 核 ≈ 60 h/月；磁盘 32 GB（拉一个 flask 镜像后剩 6 GB）；
- Codespaces 里 Docker 是 DinD，`--work` 放 `/workspaces/` 下最稳（daemon 路径可见）；
- 数据集 v5 自带 `image` 字段，可直接 `docker pull`，不必调 `make_test_spec`（v5 里它被移到了 `swebench.harness.utils`）。

### 第二个数据点：psf__requests-3362（76 个测试）

| 状态 | n_fail | test_s |
|---|---|---|
| baseline | 1（F2P 挂） | 1.43 |
| turn 1（应用 gold hunk） | 0 | 1.36 |
| revert hunk | 1 | —（inert=False, syntax=True） |

**结论**：76 个测试的单次判分 ≈ **1.4 s**（含容器启动），与 flask 的 1–1.6 s 同一量级。
→ 20 轮轨迹 ≈ 30 s；100 条轨迹约 **50 分钟**（单线程；可多容器并行）。

### 踩坑：nodeid 是 shell 不可信输入

requests 的 P2P 里有 URL 参数化测试 ID，含 `&`、`#`：

```
tests/test_requests.py::TestRequests::test_params_are_added_before_fragment[http://example.com/path#fragment-...?a=b#fragment]
```

用 `shell=True` 拼命令时会被 bash 当成后台/注释，报 `No such file or directory`。
**修法**：全部 `shlex.quote`（已修，commit `d2c1577`）。原 `agentic-rl-swe` 用 list 形式 `subprocess.run` 所以没踩到。

---

## 7. 第一条真实轨迹重放（2026-11-01）

**来源**：mini-swe-agent 2.4.6 + DeepSeek `deepseek-chat`，实例 `psf__requests-3362`
（19 次 API 调用、43 条消息、$0.0038、53 s 解出；提交 patch 与 gold 语义一致）

**重放结果**（`oracle/replay_traj.py`）：

| turn | n_fail | n_pass | 说明 |
|---|---|---|---|
| 1–4 | 1 | 75 | F2P 仍挂 |
| **5** | **0** | 76 | 修复落地（唯一变化点） |
| 6–19 | 0 | 76 | 保持通过 |

- 单次判分 1.49–1.77 s；整条轨迹（19 轮重放 + 19 次判分）**48.3 s**；
- 检测器正确输出唯一 pivot：`turn 5, improvement`；
- 负对照成立：成功轨迹没有 regression。

**结论**：轨迹解析 → 逐轮重放 → 快照 → test_patch → pytest → 状态轨迹 → pivot 检测，
整条链路跑通；下一步需要**失败轨迹**才能看到 regression 型 pivot。

---

## 8. 真实失败轨迹与标签噪声（2026-11-01，psf__requests-2674）

**生成**：mini-swe-agent + DeepSeek `deepseek-chat`（temperature=1.0），11 轮、15 条命令，Submitted 但未解决。

**重放结果**（`replay_traj.py` v5，官方语义：整文件 pytest + F2P 重跑）：

| 项 | 结果 |
|---|---|
| baseline n_fail | **3**，而 F2P = **12**（11 个 F2P 在 base 就 PASSED） |
| 11 轮状态轨迹 | 全部 n_fail=3（agent 的编辑对测试**完全无影响**，inert） |
| flaky 处理 | `--rerun 1` 生效，曲线变干净但无信号 |
| 单次判分 | ≈21 s（154 个测试，含容器启动） |

**结论（两条，都写进方法学）**：

1. **实例标签需要筛**：`baseline n_fail == len(F2P)` 是准入条件。requests 系测试依赖 httpbin/网络，SWE-bench Lite 又未经过 Verified 级人工验证，噪声大 → 弃用该实例。
2. **评测必须复刻官方语义**：按 nodeid 子集跑会因测试顺序/全局状态得到不同的 F2P 结果；官方 `eval_script` 是「还原测试文件 → 打 test_patch → `pytest -rA <整个测试文件>`」。

**下一实例**：`pallets__flask-5063`（F2P 2 / P2P 54，确定性测试）→ 失败轨迹 → 重放 + `--attribute`。

---

## 9. 第一条**可验证的有害编辑**（2026-11-02，pallets__flask-5063）

**实例**：flask-5063（F2P=2 / P2P=54；`screen_instance.py`：VERDICT=OK，基线 2/2 失败、P2P 全过）

**轨迹**：mini-swe-agent + DeepSeek（temperature=1.0），18 轮 / 22 条命令，Submitted 但未解决。

**测试状态轨迹**：

| 轮次 | n_fail | 说明 |
|---|---|---|
| 0（baseline） | 2 | = F2P，标签干净 |
| 1–4 | 2 | 只读探索 |
| **5** | **4** | **regression：弄坏 2 个 P2P** |
| 6–18 | 4 | 再没修回来 |

**turn 5 的编辑**：`src/flask/cli.py` 的 `routes_command` 增加 `Domain` 列（`rule.subdomain`），
行模板从 3 列改成 4 列 → 破坏了 `tests/test_cli.py` 里 2 个依赖原输出格式的测试。

**归因**：
- ❌ 直接回滚补丁失败：后续 13 轮反复改同一文件，`git apply -R --3way` 冲突（`U src/flask/cli.py`）；
- ✅ **反事实重放**（跳过 turn 5 的命令重放整条轨迹）：

```
[cf turn 5] real=4 cf=2 delta=-2
```

**结论**：turn 5 是两次回归的**因果来源**（跳过它，最终失败数 4→2）。
这是本项目第一个 **edit 级、可执行验证的信用标签**——不是教师猜的，是重放+测试验出来的。

**方法学要点**：对“长轨迹 + 同一文件反复编辑”的场景，反事实重放（skip-turn）比补丁回滚更稳；
补丁回滚只适合独立 hunk（oracle 模式）。

---

## 10. G1/G2 正式数据（2026-11-02，flask-5063 × 10 条轨迹）

**设置**：mini-swe-agent + DeepSeek（temperature 1.2）× 10 条；重放按官方 eval 语义（整文件 pytest、还原 test 文件、F2P 重跑 1 次）；
每条对 regression 做反事实重放（skip-turn），对可恢复配对做 skip-recovery 验证；教师（DeepSeek）事后指认 pivot 做对照。
原始数据：`results/g1c/`（10 × report + traj + teacher json + SUMMARY/TEACHER）。

### G1：标签覆盖率

| 指标 | 结果 |
|---|---|
| 轨迹 | 10（成功 3 / 失败 7） |
| 失败中有 regression | 6/7 = **85.7%**（G1a） |
| 含可恢复 pivot 配对 | 7/10 = **70%**（新主指标） |
| 恢复配对数 | 8 对；「跳过恢复→终态变差」= 7/8（原始因果） |
| 致命回归（未恢复） | 2 条；**2/2 归因验证成功**（1 条 cf、1 条文件级回滚兜底） |
| 失败中有因果验证 | 2/7 = 28.6%（G1b，原口径） |

### G2：教师事后指认 vs 可执行验证

| 案例类型 | 数量 | 结果 |
|---|---|---|
| **可比案例**（有验证 pivot） | 5 | 教师 exact **4/5 = 80%**；within-1 4/5 |
| 其中成功轨迹（可恢复错误） | 3 | 3/3 exact |
| 其中失败+致命回归 | 2 | 1/2 exact（1 条教师指到 14 轮，真值 5 轮） |
| **遗漏型失败**（我们无致命 pivot） | 5 | 教师**仍给出 pivot：5/5**（过度归因） |
| 教师回答「没有关键步」 | 0/10 | 从不回答 0 |

### 解读

1. **可恢复配对是最稳、最高产的信号**（70% 轨迹有）；失败多为"没改对"（遗漏型）而非"改坏不修"。
2. **致命回归很稀有（2/10），但一旦存在，执行验证 100% 能归因**。
3. **教师并非一无是处**：存在明确 pivot 时 4/5 命中；但面对遗漏型失败会**编造一个 pivot**（5/5）——这正是"事后叙事"偏差的直接证据，也是本方法的价值所在。
4. 严格有效性判据过严（skip 恢复会让后续命令失败，本属反事实的一部分），已改为「相对原轨迹是否多出失败」（v10）；当前 8 对里原始因果 7 对。

### 限制

- 单一实例（flask-5063）、10 条样本，统计意义有限；
- 教师看到的是截断轨迹（最后 60k 字符），可能低估教师；
- "遗漏型失败"的判定依赖我们的因果关系定义。

---

## 11. Q1 保真度 + G2 修正口径（2026-11-03）

### Q1：重放保真度（g2a × 5）

用 `check_fidelity.py`：重放全部命令 → 与「clean base + agent 提交的 patch」逐文件比对。

| 结果 | 数值 |
|---|---|
| 代码文件完全一致 | **5/5** |
| 唯一差异 | `./patch.txt`（agent 提交时自己生成的中间文件，不属于代码） |
| 重放命令失败 | 0–1 条/轨迹（均不影响最终状态） |

**结论：Q1 通过**——重放能复现 agent 提交的代码状态；`patch.txt` 类提交产物需在比较时忽略。

### G2 修正口径：教师 = 无符号定位器

把「任意已验证编辑」（harmful/beneficial/recovery pivot）作为教师的对照基准：

| 指标 | g1c（10） | g2a（5） | 合计 |
|---|---|---|---|
| 轮次 exact | 4/9 | 5/5 | **9/14 = 64%** |
| 轮次 ±1 | 6/9 | 5/5 | **11/14 = 79%** |
| **符号正确**（命中轮次里，教师把仅 beneficial 的轮说成 mistake） | 0/1 | **5/5 说反** | 5/6 说反 |
| 无编辑可归因时仍指认 | 1/1 | — | 1/1 |
| 教师回答「没有关键步」 | 0/10 | 0/5 | **0/15** |

**解读（论文口径）**：

> 教师能大致定位「轨迹里出问题的那一段」（±1 轮 79%），但它给的是**无符号的事后叙事**：同一轮既可能是错误、也可能是唯一有效编辑，教师无法区分（g2a 5/5 说反）。**因果方向（有害/有益）必须用反事实重放测量**——这是本方法相对教师标签的核心增量。

### 对选题的含义

- 「教师标签不可用」的说法过强，应改为「**教师只能提供无符号的轮次定位，不能提供可用作训练信号的因果符号**」；
- 论文的贡献应聚焦在 **signed（有方向）的可执行验证标签**，以及它在 omission/partial-fix 轨迹上的补充作用。

---

## 12. Q4 公平性对照：教师 + 逐轮测试状态（2026-11-03 晚）

**设置**：对 15 条轨迹（g1c 10 + g2a 5），给 DeepSeek 教师额外提供**逐轮失败数序列**（step k: n_fail），
与「教师原版（只看轨迹）」和「随机猜一轮」对比；基准是我们可执行验证的 pivot 集（harmful/beneficial/recovery）。

| 组 | exact | ±1 | 符号说反 | 无编辑仍指认 | 从不回答"没有" |
|---|---|---|---|---|---|
| A 教师原版 | 9/14 = 64% | 11/14 = 79% | 5/9 | 1/1 | 15/15 |
| **B 教师 + 逐轮状态** | **10/14 = 71%** | 10/14 = 71% | **5/10** | 1/1 | 15/15 |
| C 随机基线 | 4.9% | — | — | — | — |

**解读**：

1. 教师定位轮次远好于随机（64–71% vs 5%），**给它测试状态只带来边际提升**；
2. **给状态也修不好符号**：g2a 的 5 个 beneficial 轮，教师仍全部描述为 "pivotal mistake"；
3. 教师从不回答"没有关键步"（0/15），在无编辑可归因时仍强行指认（1/1）。

**论文口径**：本方法相对教师的核心增量 = **signed（有方向的因果标签）+ 能判定"无可归因编辑"**，
而不是"定位轮次"本身（教师已经能定位）。

数据/脚本：`oracle/teacher_status_eval.py`、`results/teacher_status/summary.json`。

---

## 13. 三实例总表与 G2 修正语义（2026-11-03 晚）

**数据**：flask-5063 × 10、flask-4045 × 5、requests-3362 × 5（合计 20 条；12 失败 / 8 成功）。

### Q2 标签产出

| instance | n | failed | harmful | beneficial | recovery pairs | cf-verified any |
|---|---|---|---|---|---|---|
| flask-5063 | 10 | 7 | 2 | 0 | 8（7 条） | 2/10 |
| flask-4045 | 5 | 5 | 0 | 5 | 0 | 5/5 |
| requests-3362 | 5 | 0 | 0 | 5 | 0 | 5/5 |
| 合计 | **20** | 12 | 2 | 10 | 8 | **12/20 = 60%** |

**标签类型由失败模式决定**：
- 改坏/可恢复型（flask-5063）→ harmful + recovery pairs；
- 只修一半/修好型（flask-4045 / requests-3362）→ beneficial。

### Q4 教师对照（修正语义）

| 组 | 轨迹数 | 教师 exact | ±1 |
|---|---|---|---|
| 有可归因错误（失败+harmful / 成功+recovery） | 5 | **4/5 = 80%** | 4/5 |
| 无可归因错误（omission / 顺利成功） | 15 | — | 教师仍指认 **13/15 = 87%** |

教师 2/20 次回答「没有关键步」。

**论文口径**：教师的增量是“定位”（80%），我们的增量是 **signed 标签 + “无错误”判定**；
在“没有单个错误编辑”的轨迹上，教师 87% 会编一个 pivot。
