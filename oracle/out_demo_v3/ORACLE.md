# Oracle 报告：demo__calc

- hunks（原始）= 4，跳过 = 0，F2P = 3，P2P = 1
- baseline n_fail = 3，final n_fail = 0
- 单次判分中位数 = 0.06s（共 9 次）

## 逐轮应用

| turn | hunk | path | apply | n_fail | n_pass | changed | test_s |
|---|---|---|---|---|---|---|---|
| 1 | 1 | `calc.py` | True | 2 | 2 | True | 0.06 |
| 2 | 2 | `calc.py` | True | 1 | 3 | True | 0.06 |
| 3 | 3 | `calc.py` | True | 0 | 4 | True | 0.06 |
| 4 | 4 | `CHANGES.rst` | True | 0 | 4 | False | 0.06 |

## 单 hunk 回滚（从最终状态）

| hunk | path | revert | 3way | syntax | n_fail | inert |
|---|---|---|---|---|---|---|
| 1 | `calc.py` | True | False | True | 1 | False |
| 2 | `calc.py` | True | False | True | 1 | False |
| 3 | `calc.py` | True | False | True | 1 | False |
| 4 | `CHANGES.rst` | True | False | True | 0 | True |

