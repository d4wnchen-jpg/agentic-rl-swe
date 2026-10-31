# Oracle 报告：demo__calc

hunks = 3，F2P = 3，P2P = 1

## 逐轮应用

| turn | hunk | apply | n_fail | n_pass | test_ok | test_s |
|---|---|---|---|---|---|---|
| 1 | 1 | True | 2 | 2 | False | 0.06 |
| 2 | 2 | True | 1 | 3 | False | 0.06 |
| 3 | 3 | True | 0 | 4 | True | 0.06 |

## 单 hunk 回滚（从最终状态）

| hunk | revert | 3way | n_fail | n_pass | statuses |
|---|---|---|---|---|---|
| 1 | True | False | 1 | 3 | {'test_add': 'PASSED', 'test_sub': 'FAILED', 'test_mul': 'PASSED', 'test_neg': 'PASSED'} |
| 2 | True | False | 1 | 3 | {'test_add': 'PASSED', 'test_sub': 'PASSED', 'test_mul': 'FAILED', 'test_neg': 'PASSED'} |
| 3 | True | False | 1 | 3 | {'test_add': 'PASSED', 'test_sub': 'PASSED', 'test_mul': 'PASSED', 'test_neg': 'FAILED'} |
