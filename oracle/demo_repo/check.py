import sys
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
