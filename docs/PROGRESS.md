# 进度与下次启动（2026-11-01 收工）

> 新会话先读这一页 + [PHASE0_CHARTER.md](PHASE0_CHARTER.md)。
> 目标：**可执行验证的 pivot 信用标签（VPD）**，code agent / SWE 场景；11-10 go/no-go。

## 一、当前状态（一句话）

**管道全部打通，开始产出真实数据**：oracle 模式已在 2 个公开任务上验证；mini-swe-agent + DeepSeek 生成的轨迹能逐轮重放、逐轮判分、定位变化点、做 edit 级归因。**还缺一条“标签干净 + 有 regression”的失败轨迹**（11-06 前必须拿到）。

## 二、已完成（时间线）

| 日期 | 事项 | 产物 |
|---|---|---|
| 10-29 | 方向定稿（VPD） | `PHASE0_CHARTER.md` |
| 10-29 | 引用闭包（37 篇共引 → 20 篇待读） | `notes/reading_closure_vpd.md` |
| 10-31 | L0 冒烟：patch→测试→单 hunk 回滚 | `smoke_test/L0_REPORT.md` |
| 11-01 | L1 oracle：flask-4992、requests-3362 真实任务跑通 | `smoke_test/L1_ORACLE_REPORT.md` |
| 11-01 | 真实轨迹重放：mini-swe-agent + DeepSeek（requests-3362 成功轨迹） | `replay_traj.py` + L1 报告 §7 |
| 11-01 | 发现并处理：diff hunk≠edit、nodeid 需 shlex、官方 eval 语义、实例标签噪声 | charter §2/§3、L1 报告 §8 |

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
