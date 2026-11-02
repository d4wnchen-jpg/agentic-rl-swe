# 轨迹重放报告：pallets__flask-5063

- turns = 19，F2P = 2，P2P = 54
- 单次判分中位数 = 1.36s
- 重放 + 判分总耗时 = 90.9 s

## 测试状态轨迹

| turn | n_fail | n_pass | flaky | test_patch | test_s | changed | sha |
|---|---|---|---|---|---|---|---|
| 0 | 2 | 54 | 0 | True | 1.33 |  | `c538a6f8` |
| 1 | 2 | 54 | 0 | True | 1.53 | = | `8c1a7b73` |
| 2 | 2 | 54 | 0 | True | 1.32 | = | `1bf1fbb4` |
| 3 | 2 | 54 | 0 | True | 1.25 | = | `cdb5a5bd` |
| 4 | 2 | 54 | 0 | True | 1.36 | = | `d50406ce` |
| 5 | 4 | 52 | 0 | True | 1.33 | ↑ | `744b05dd` |
| 6 | 4 | 52 | 0 | True | 1.37 | = | `f740694f` |
| 7 | 4 | 52 | 0 | True | 1.43 | = | `301c7cb5` |
| 8 | 4 | 52 | 0 | True | 1.41 | = | `a5b7472c` |
| 9 | 4 | 52 | 0 | True | 1.28 | = | `b3f15d4b` |
| 10 | 4 | 52 | 0 | True | 1.4 | = | `0f701a31` |
| 11 | 4 | 52 | 0 | True | 1.42 | = | `3ac14cd8` |
| 12 | 4 | 52 | 0 | True | 1.43 | = | `8077173f` |
| 13 | 4 | 52 | 0 | True | 1.29 | = | `2dd01aef` |
| 14 | 4 | 52 | 0 | True | 1.3 | = | `a7736200` |
| 15 | 4 | 52 | 0 | True | 1.46 | = | `304681d3` |
| 16 | 4 | 52 | 0 | True | 1.38 | = | `a79049a5` |
| 17 | 4 | 52 | 0 | True | 1.27 | = | `ef9577a0` |
| 18 | 4 | 52 | 0 | True | 1.31 | = | `c6799357` |

## baseline 的 F2P 状态（0/2 在 base 就通过）

| status | nodeid |
|---|---|
| FAILED | `tests/test_cli.py::TestRoutes::test_subdomain` |
| FAILED | `tests/test_cli.py::TestRoutes::test_host` |

## 候选 pivot

- turn 5: regression（2 → 4 failing）

## edit 级归因（从最终状态回滚该轮编辑）

| turn | revert_ok | n_fail(final) | n_fail(after) | delta | 判断 |
|---|---|---|---|---|---|
| 5 | ❌ | 4 | — | — | 回滚失败 |

## 反事实重放（跳过该轮，评测终态）

| turn | 真实终态 n_fail | 跳过该轮后 | delta | 判断 |
|---|---|---|---|---|
| 5 | 4 | 2 | -2 | **有害编辑**（跳过它，失败减少） |
