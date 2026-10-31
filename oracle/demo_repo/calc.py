def add(a, b):
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
