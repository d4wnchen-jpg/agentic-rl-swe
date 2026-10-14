# 实验台账

> **为什么有这个文件**：2026-10-15 凌晨查一次实验配置，花了两个小时还没查清 ——
> 训练代码没进 git、命令行没记环境变量、文档写结论没写判据。
> 这个文件就是为了让「下次读一遍就能恢复上下文」，不依赖任何人的记忆。
>
> **配套**：每次跑自动生成的 `eval_<tag>.meta.json`（见 `tools/run_manifest.py`）
> 记录客观事实（环境变量、生效配置、代码 md5、题库指纹）；
> 这个文件记录**意图、解读、下一步** —— 那部分机器记不了。

## 怎么用

**每次实验跑完，追加一条**（最新的在最上面）：

```markdown
## YYYY-MM-DD  <tag> ｜ <一句话说明想验证什么>

**意图**：

**配置** → 详见 eval_<tag>.meta.json
  模型 · harness · enable_thinking · 关键开关 · 题数 × 轮数 × k

**结果**（对照 = <哪个 run>）
  主指标 · 机制指标（行为）

**结论**：

**下一步**：
```

**五项缺一不可**，尤其是「意图」和「下一步」—— 它们让台账能串起来。

**⚠️ 数字必须标来源**：`compare_harness.py` 和自写脚本的判据不同，
同一个「编辑次数」可能给出 2.2/条 和 12.5/条 两个数。不标来源 = 埋雷。

---

## 2026-10-15  v2_aligned ｜ 对齐 SWE-agent 原话后，行为能否回到正轨

**意图**
上一轮（v2_hidden）发现：移除判分测试后行为变了，但方向偏了
（读更多、写脚本更多，却改得更少、改坏更多）。
怀疑是 V2 prompt 自己有问题 —— 后来查出两处：
① 题面写「Do NOT modify the tests... they are restored before the fix is evaluated」
   （自相矛盾：测试已被移除；且泄露判分机制）
② 系统提示词写「Keep your reasoning VERY brief / Do not second-guess / Do not re-read」
   （和题面的 5 步工作流冲突，也和 SWE-agent 原文的
   「Your thinking should be thorough」矛盾）
本轮的意图：把两处都对齐 SWE-agent 原话后，看行为指标是否回升。

**配置** → 详见 `eval_v2_aligned.meta.json`
  模型：基座 Qwen3.5-9B（不挂 adapter）
  harness：V2 **对齐版**（题面用 SWE-agent 原话 + 恢复 thinking thorough）
  `SMITH_HIDE_F2P=1` · `enable_thinking` 不传
  113 道 × 40 轮 × k=1

**状态**：进行中（启动 01:2x，预计约 109 分钟，参考 v2_hidden 的 108 分钟）

**对照**：v2_hidden（同配置，仅 prompt 版本不同）

**判读要点**
  · 「编辑次数」「有编辑的 rollout 比例」是否回升
  · 「首次编辑轮次」是否回到 2.6 附近（而不一直是 4.4）
  · 「改坏 P2P」是否降回 5
  · 写脚本率应保持高位（V2 的 5 步工作流仍应在）

---

## 2026-10-14  v2_hidden ｜ 移除判分测试，模型会不会更爱动手、更早提交

**意图**
训练 20 步后观察到行为退化：编辑 1.9→0.5 次/条、从未编辑 36%→68%、
从未跑 pytest 26%→55%、主动提交 1→19。
假说：测试文件可见 → 模型去跑 pytest → 输出几万字符被 obs_cap 截断 →
拿不到「我完成了」的信号 → 于是少动手、早交工。
本轮把判分测试从沙箱移除（模拟 SWE-bench 原本的设计：F2P 由 test_patch 添加，
baseline 里根本不存在），用**基座**跑（不训练，以隔离「训练效应」）。

**配置** → 详见 `eval_v2_hidden.meta.json`
  模型：基座 Qwen3.5-9B（不挂 adapter）
  harness：V2 **旧版**（题面含「tests are restored」；系统提示词含 VERY brief）
  `SMITH_HIDE_F2P=1` · `enable_thinking` 不传 · 113 道 × 40 轮 × k=1
  耗时 6485s ≈ 108 分钟

**结果**（对照 = base40，同配置仅 harness 不同）

| 指标 | base40（V1·测试可见） | v2_hidden（V2·移除测试） | 来源 |
|---|---|---|---|
| 通过率 | 27/113 = 23.9% | 22/111 = 19.8% | `compare_harness.py` |
| 跑过 pytest | 83/113 (73%) | 44/111 (40%) | `compare_harness.py` |
| 编辑过文件 | 82/113 (73%) | 63/111 (57%) | `compare_harness.py` |
| 编辑命令总数 | 1412 | 892 | 自写脚本（数 str_replace_editor 调用）|
| 平均编辑次数 | 1412/113 = 12.5 | 892/111 = 8.3 | 同上 |
| 首次编辑平均轮次 | 2.6 | 4.4 | 自写脚本 |
| 查看类命令 | 952 | 1161 | 自写脚本 |
| 主动提交 | 1/113 | 1/111 | `compare_harness.py` |
| 复读（阈值 ≥1 个 `</think>`）| 0/113 | 0/111 | 自写脚本 |
| 修好 / 部分修复 / 完全没修 / 改坏 P2P | 28 / 15 / 65 / 5 | 22 / 13 / 69 / 7 | 自写脚本（用 n_f2p_bad / n_p2p_bad）|

**结论**
  · ✅ 环境确实支配行为：跑 pytest −33pp、查看命令 +22%、脚本命令 +19%
  · ❌ 但方向不是预期的「更爱动手」——
    「完全没修」只从 65 增到 69（+4），而编辑命令 −37%、改坏 P2P +2
  · ⭐ 更贴合数据的解释：**失去了「需求锚点」**
    测试文件不只是验证工具，也是「要改什么」的说明书。
    移除后模型读得更多、写脚本更多，却不确定该改到什么程度 → 改得更浅、更不准
  · ⚠️ 通过率 −4.1pp 在 ±6pp 噪声内，不构成结论

**下一步**
V2 prompt 本身有两处缺陷（见 v2_aligned 条目的「意图」）→ 对齐后重跑

---

## 2026-10-09  base40 / s20_40 ｜ 训练到底有没有效果

**意图**
20 步 GRPO（640 条 rollout）之后，基座（step000）和训练后（step020）
在同一批 held-out 上的差异。

**配置** → ⚠️ 当时的 `run_manifest.py` 还不存在。**代码已被事后修改，不能作为依据。**

**⭐ 定案依据（2026-10-15 从会话记录里找到的现场）**
```
10-09 17:06 的原始对话记录（当时正在排查复读）：
  「grpo_llm.py (vLLM, generation side): passes
    chat_template_kwargs={"enable_thinking": False} ✅」

时间顺序：
  10-09 15:30  评测 base40        ← 用当时代码
  10-09 16:47  评测 s20_40        ← 用当时代码
  10-09 17:06  讨论「传 False」    ← 评测后 20 分钟
  10-09 晚 ~ 10-10  改代码：enable_thinking → None（不传）
  10-10 12:43  commit 52f298b "Make the correct configuration the default"

⭐ 所以两次评测都在改代码【之前】→ **都传了 enable_thinking=False（关 think）**
```

**⚠️ 查证时走过的弯路（记下来，别再犯）**
```
读了工作目录的 tools_repo/grpo_llm.py，看到 enable_thinking=None（写死）
→ 据此断定"两次评测都不传"
→ ❌ 错。那是【10-09 之后改过】的版本，评测跑在改动之前。

旁证：llm_client.py 的注释被改过，两段意思相反
  当时： "Qwen3.5 默认会输出 thinking...，用 enable_thinking=False 关掉"
  现在： "enable_thinking 默认【不传】。传 False 会...插入完整的空思考块...模型有概率复读它"

⭐ 教训：**代码会被改，日志不会。** 定案先查现场（对话记录 / 日志 / 产物时间戳），
        再拿代码佐证 —— 并且要确认代码是不是当时的版本。
```
  ```bash
  # 两条在同一脚本里连续跑，参数完全相同，未设任何环境变量
  $PY -u tools_repo/eval_adapter.py --adapter grpo_adapters/step000 --n 115 --k 1 --tag base40 --turns 40
  $PY -u tools_repo/eval_adapter.py --adapter grpo_adapters/step020 --n 115 --k 1 --tag s20_40 --turns 40
  ```
  · 代码：`tools_repo/`（10-09 13:47 的快照；`enable_thinking=None` 写死，环境变量开关尚不存在）
  · harness：V1（当时 V2 还没写）
  · `enable_thinking`：⭐ **传 False（关 think）** —— prompt 尾部 = `<|im_start|>assistant\n<think>\n\n</think>\n\n`
    （定案依据见上方「定案依据」；这一条 2026-10-15 曾被误判为「不传」）
  · 题库：`eval.jsonl` 共 115 行，两次都取前 113 道
  · 采样：温度 0.7（`quick_eval` 默认）

**结果**

| | base40（step000）| s20_40（step020）|
|---|---|---|
| 通过率 | 27/113 = 23.9% | 29/113 = 25.7% |
| 复读（阈值 ≥1）| 0/113 = 0% | 109/113 = 96% |
| 复读（阈值 ≥3）| 0% | 102/113 = 90% |
| 复读（阈值 ≥5）| 0% | 75/113 = 66% |
| 复读（阈值 ≥10）| 0% | 58/113 = 51% |
| 单条最多 `</think>` | 0 个 | 4096 个（= 撞 max_new_tokens 16384）|

**结论**
  · ✅ 两次评测**配置相同**（都不传 `enable_thinking`）→ +1.8pp 的对比成立
  · ⚠️ 但 +1.8pp 落在 ±6pp 噪声内 → 总分看不出效果
    （真正的信号在分层：mixed 层 5/17 → 9/17）
  · ⭐ 复读差异 100% 来自训练：同一配置下基座 0%、step020 96%
    → 复读是**训练放大**出来的行为

**⚠️ 未解之谜（记录在案，避免以后重复考古）**
  · 训练（10-06/07）用的代码**没有进 git**（首次 commit 是 10-08），
    因此训练时 `enable_thinking` 传了什么**无法直接查证**
  · 但 s20_40 复读的内容是**完整空块** `<think>\n\n</think>\n\n`，
    而「不传」时 prompt 尾部只有 `<think>\n`（无空块）
    → 该空块只能来自**训练时**的 prompt → 训练时用了 `False`
  · 也就是说：训练和评测**都是关 think（False）** —— 配置一致
    用户记忆中的「25.7 是关了 think 的结果」**完全正确**

**下一步**
  · 已加 `SMITH_ENABLE_THINKING` 开关（`SMITH_ENABLE_THINKING=0` = 传 False）
  · 如需「训推一致」下的 s20 真实通过率，可用它重跑一次（约 1.5–2.5 小时，
    传 False 会触发复读 → 生成打满 max_new_tokens → 可能更慢）

---

## 待办

- [ ] `run_manifest.py` 同步到 AutoDL（这样以后每次跑自动留痕）
- [ ] 若上述实验推翻/加强任何长期结论，回写到 `docs/LESSONS.md` / `docs/RESULTS.md`
- [ ] `eval_s20_false`（step020 + `SMITH_ENABLE_THINKING=0`）—— 可选，验证训推一致的影响
