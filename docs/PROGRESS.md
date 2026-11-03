# 进度与下次启动（2026-11-01 收工）

> 新会话先读这一页 + [PHASE0_CHARTER.md](PHASE0_CHARTER.md)。
> 目标：**可执行验证的 pivot 信用标签（VPD）**，code agent / SWE 场景；11-10 go/no-go。

## 零、最新更新（2026-11-03 晚）

- **Q1 保真度通过**：5/5 条轨迹重放后的代码与提交 patch 完全一致（仅 `patch.txt` 产物差异）；
- **双向标签成立**：flask-4045 × 5 全部产出 beneficial 标签（G1-any 5/5）；
- **G2 修正口径**：教师是「无符号定位器」——轮次 ±1 命中 11/14（79%），但**把 beneficial 轮说成 mistake 5/5**，从不回答"没有关键步"（0/15）；
- 数据：`results/g1c/`、`results/g2a/`、`results/fidelity_g2a/`；memo 口径见 L1 报告 §11。

## 一、当前状态（2026-11-02 晚，G1/G2 首轮数据已出）

**flask-5063 × 10 条轨迹**（3 成功 / 7 失败）：
- G1a（失败中有 regression）= 6/7 = 85.7%；
- **含可恢复 pivot 配对 = 7/10 = 70%**（超过 60% 门槛）；
- 致命回归 2 条，**2/2 执行验证归因成功**；
- G2：可比案例教师 exact 4/5（80%），但**遗漏型失败 5/5 教师都在编 pivot**，从不回答"没有关键步"。

数据与报告：`results/g1c/`（含 SUMMARY.txt / TEACHER.txt / 每条 report+traj+teacher）。
**下一步：第二个实例复现 + 扩样本 + 修后重放（v10 有效性判据）。**

## 二、已完成（时间线）

| 日期 | 事项 | 产物 |
|---|---|---|
| 10-29 | 方向定稿（VPD） | `PHASE0_CHARTER.md` |
| 10-29 | 引用闭包（37 篇共引 → 20 篇待读） | `notes/reading_closure_vpd.md` |
| 10-31 | L0 冒烟：patch→测试→单 hunk 回滚 | `smoke_test/L0_REPORT.md` |
| 11-01 | L1 oracle：flask-4992、requests-3362 真实任务跑通 | `smoke_test/L1_ORACLE_REPORT.md` |
| 11-01 | 真实轨迹重放：mini-swe-agent + DeepSeek（requests-3362 成功轨迹） | `replay_traj.py` + L1 报告 §7 |
| 11-01 | 发现并处理：diff hunk≠edit、nodeid 需 shlex、官方 eval 语义、实例标签噪声 | charter §2/§3、L1 报告 §8 |
| 11-02 | 实例筛查工具 + 官方 eval 语义对齐（v5） | `screen_instance.py` |
| 11-02 | **第一条可验证有害编辑**：flask-5063 turn5，反事实 `4→2` | L1 报告 §9 |
| 11-02 | 反事实归因（skip-turn）+ G1 批量驱动/汇总 | `replay_traj.py --counterfactual-pivots`、`run_g1.sh`、`summarize_g1.py` |
| 11-02 | G1/G2 首轮：10 条 flask-5063 | `results/g1c/` |
| 11-02 | 管线化 + 自动入库 + 教师标注 | `run_pipeline.sh`、`teacher_label.py`、`summarize_g2.py` |

## 三、关键数字

| 项 | 值 |
|---|---|
| 小 repo 单次判分 | 1.5–2 s（含容器启动） |
| requests 大 repo | ~21 s（154 个测试） |
| 一条 19 轮成功轨迹重放+判分 | 48 s |
| 一条 11 轮失败轨迹重放（rerun=1） | ~5 min |
| DeepSeek 解题成本 | $0.004/条（requests-3362） |

## 四、代码位置

- 仓库：`github.com/d4wnchen-jpg/agentic-rl-swe`，目录 `oracle/`
  - `replay_oracle.py`：gold patch → 逐轮应用/判分/回滚（L1-oracle）
  - `replay_traj.py`：mini-swe-agent 轨迹 → 逐轮重放/判分/归因（v5，官方 eval 语义 + flaky rerun）
  - `make_demo.py`：本地 demo
- 本地留档：`opd_agent/smoke_test/oracle/`（与仓库同步）
- 最新 commit：`4300bad`（v5）

## 五、Codespace 使用要点

- **空闲 30 分钟自动停止**（默认；停止后不计计算时长，但存储仍计 15 GB-month 免费额度）；
- 关浏览器**不会立刻停**；`nohup` 后台任务会在停止时被杀；
- 手动停：UI「Stop codespace」或 `gh codespace stop -c <name>`；
- **重启后**：文件都在，但终端环境变量丢失（要重新 `export DEEPSEEK_API_KEY=...`）；
- 磁盘 32 G 是硬约束：跑完一个实例先 `docker rmi`，再拉下一个。

## 六、下次第一步（按顺序）

1. 打开 Codespace → `cd /workspaces/agentic-rl-swe && git pull`；
2. （可选）若 requests-2674 镜像还在，跑官方脚本确认标签噪声：
   ```bash
   python - <<'EOF'
   import json
   v=[json.loads(l) for l in open("SWE-bench_Lite.jsonl") if json.loads(l)["instance_id"]=="psf__requests-2674"][0]
   open("/tmp/eval_2674.sh","w").write(v["eval_script"])
   EOF
   docker run --rm -v /tmp/eval_2674.sh:/tmp/eval.sh \
     swebench/sweb.eval.x86_64.psf_1776_requests-2674:latest bash /tmp/eval.sh 2>&1 | tail -40
   ```
   若官方也是 11/12 通过 → 确认标签噪声，按 §7 筛实例。
3. 清无用镜像：`docker rmi ...` + `docker system prune -af` + `df -h /`；
4. **生成 flask-5063 失败轨迹**：
   ```bash
   export DEEPSEEK_API_KEY='...'   # 注意：之前泄漏过，建议换新 key
   nohup mini-extra swebench --subset lite --split test \
     --filter 'pallets__flask-5063' --workers 1 \
     -m deepseek/deepseek-chat -c swebench.yaml \
     -c model.model_kwargs.temperature=1.0 \
     -o mswa_out > mswa_flask5063.log 2>&1 &
   ```
5. 重放 + 归因：
   ```bash
   python oracle/replay_traj.py \
     --traj mswa_out/pallets__flask-5063/pallets__flask-5063.traj.json \
     --task SWE-bench_Lite.jsonl --instance-id pallets__flask-5063 \
     --image swebench/sweb.eval.x86_64.pallets_1776_flask-5063:latest \
     --work replay_work_f5063 --out out_traj_f5063 --attribute --rerun 1
   ```
6. **验收**：baseline `n_fail == len(F2P)=2`；状态轨迹出现 `↑`；归因表出现 `delta`。

## 七、实例筛选标准（新增，写进方法学）

1. **baseline n_fail == len(F2P)**，否则标签不可信，丢弃；
2. 优先确定性 repo（flask/pytest），避开网络型（requests/httpbin）；
3. 评测按官方语义：还原测试文件 → 打 test_patch → `pytest -rA <整个测试文件>`；
4. F2P 失败重跑 1 次，仍失败才算 fail，否则标 flaky。

## 八、时间线与风险

| 日期 | 事项 |
|---|---|
| 11-02 ~ 11-05 | flask 失败轨迹 → edit 级标签；再扩 5–10 条算 **G1 覆盖率** |
| 11-05 ~ 11-06 | 教师（DeepSeek）事后指认 vs 验证标签 → **G2 分歧率** |
| **11-10** | **go/no-go**（G1≥60%、G2≥25%、G3≥30%） |
| 11-10 ~ 11-30 | ARR reviews 优先，项目暂停 |

**风险**：DeepSeek 把题都解出来（→ 提高温度/降 step_limit）；flaky（→ rerun）；磁盘（→ 一次一个镜像）；Codespace 空闲自动停（→ 后台任务别跑太久）。

## 九、11-03 ~ 11-06 计划（G1/G2 已出首轮）

1. **换第二个实例**（`pallets__flask-4045` 或 `pytest-dev__pytest-7168`）：先跑 `screen_instance.py` 准入，再跑 `run_pipeline.sh ... N=10`，看 G1-rec 是否还在 60% 以上；
2. **扩样本**：同实例再加 10 条（prefix g1d），把 G1/G2 的 n 提到 20+；
3. **修后重放**：用 v10 有效性判据重放 `results/g1c/*.traj.json`，更新恢复配对的因果数（原始 7/8 → 预期更高）；
4. **11-06**：写半页 memo（G1-rec、fatal 归因、教师过度归因），交 11-10 go/no-go。
