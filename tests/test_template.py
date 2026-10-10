#!/usr/bin/env python3
"""tools/tpl_chat.jinja 的回归测试（纯 CPU，不需要模型/GPU）

这个模板是为了解决 Qwen3.5 原生模板的三个问题而写的：

  1. 生成点在 prompt 尾部插入完整的空思考块 → 训练后模型复读它
  2. 生成提示符与历史渲染是两套逻辑       → 相邻两轮不构成 token 前缀链
  3. 历史渲染剥离 assistant 内容里的 think 块 → 模型看不到自己上一轮的推理

模板的核心写法只有一条：**生成提示符 = 内容为空的历史渲染**。

本文件守住这三条性质。改模板时如果破坏其中任何一条，测试会失败。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
jinja2 = pytest.importorskip("jinja2", reason="需要 jinja2")

TEMPLATE_PATH = ROOT / "tools" / "tpl_chat.jinja"
TEMPLATE_AGL_PATH = ROOT / "tools" / "tpl_chat_agl.jinja"


def _mk_render(path):
    tpl = jinja2.Environment().from_string(path.read_text())

    def _render(messages, add_generation_prompt=True, **kw):
        return tpl.render(messages=messages,
                          add_generation_prompt=add_generation_prompt, **kw)

    return _render


@pytest.fixture(scope="module")
def render():
    """我们推荐的模板：两处都不插 think 结构"""
    return _mk_render(TEMPLATE_PATH)


@pytest.fixture(scope="module")
def render_agl():
    """参照实现风格：enable_thinking=False 时两处都插同一个块"""
    return _mk_render(TEMPLATE_AGL_PATH)


REASON = "问题可能出在 tokenizer 的分词逻辑上"
ACTION = "<function=bash><parameter=command>ls</parameter></function>"
R1 = f"<think>\n{REASON}\n</think>\n\n{ACTION}"

BASE = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "修复 bug"}]
FULL = BASE + [{"role": "assistant", "content": R1},
               {"role": "user", "content": "OBSERVATION: ok"}]


def test_template_exists():
    assert TEMPLATE_PATH.exists(), f"找不到 {TEMPLATE_PATH}"


def test_generation_prompt_has_no_think_tag(render):
    """性质 1：生成点不插入任何 think 标签

    原生模板在这里插入 <think>\\n（或完整的空块），导致训练后模型复读。
    """
    p = render(BASE)
    assert "<think>" not in p, f"生成提示符里出现了 think 标签：{p[-80:]!r}"


def test_history_keeps_thinking(render):
    """性质 2：历史渲染保留思考内容

    原生模板会把 <think>…</think> 整个删掉，模型因此看不到自己上一轮的推理。
    """
    p = render(FULL)
    assert REASON in p, "思考内容在历史渲染时被丢弃了"
    assert ACTION in p, "动作在历史渲染时被丢弃了"


def test_prefix_chain_is_continuous(render):
    """性质 3：生成提示符 + 生成内容，是下一轮 prompt 的前缀

    这保证多轮可以拼成一条训练行（不需要 LCP 修补）。
    """
    p1 = render(BASE)
    p2 = render(FULL)
    assert p2.startswith(p1 + R1), (
        "前缀链断裂：\n"
        f"  p1        = {p1[-60:]!r}\n"
        f"  p1 + R1   = {(p1 + R1)[-60:]!r}\n"
        f"  p2        = {p2[:len(p1) + 80]!r}"
    )


def test_generation_prompt_ends_with_assistant(render):
    """生成提示符必须停在 assistant 角色上（模型从这里开始写）"""
    p = render(BASE)
    assert p.rstrip("\n").endswith("assistant"), f"结尾异常：{p[-40:]!r}"


def test_no_generation_prompt(render):
    """add_generation_prompt=False 时不应追加生成提示符"""
    p = render(FULL, add_generation_prompt=False)
    assert not p.rstrip().endswith("assistant"), "不该追加生成提示符"


def test_system_message_not_duplicated(render):
    """system 消息只出现一次"""
    p = render(BASE)
    assert p.count("<|im_start|>system") == 1, "system 消息被重复渲染"


def test_non_string_content(render):
    """content 不是字符串时不应崩（例如带 tool_calls 的消息）"""
    msgs = BASE + [{"role": "assistant", "content": None}]
    p = render(msgs)
    assert "<|im_start|>assistant" in p


def test_multi_turn_chain(render):
    """多轮连续：每一轮都接得上"""
    msgs = list(BASE)
    for i in range(3):
        p = render(msgs)
        r = f"<think>\n第{i}轮思考\n</think>\n\n{ACTION}"
        msgs = msgs + [{"role": "assistant", "content": r},
                       {"role": "user", "content": f"OBSERVATION: {i}"}]
        p_next = render(msgs)
        assert p_next.startswith(p + r), f"第 {i + 1} 轮前缀链断裂"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ═══════════════════════════════════════════════════════════
# tpl_chat_agl.jinja —— 参照实现风格（两处都插，靠"一致"保证前缀链）
# ═══════════════════════════════════════════════════════════

def test_agl_template_exists():
    assert TEMPLATE_AGL_PATH.exists()


def test_agl_inserts_block_at_both_sites(render_agl):
    """enable_thinking=False 时，生成点和历史渲染都插同一个块"""
    p1 = render_agl(BASE, enable_thinking=False)
    p2 = render_agl(FULL, enable_thinking=False)
    assert "<think>\n\n</think>\n\n" in p1
    # 历史里那条 assistant 消息也带块
    seg = p2.split("<|im_start|>assistant")[1].split("<|im_end|>")[0]
    assert "<think>\n\n</think>\n\n" in seg


def test_agl_prefix_chain_with_block(render_agl):
    """两处一致 → 前缀链仍然连续（这是参照实现的保证）"""
    p1 = render_agl(BASE, enable_thinking=False)
    p2 = render_agl(FULL, enable_thinking=False)
    assert p2.startswith(p1 + R1)


def test_agl_no_block_when_not_passed(render_agl):
    """不传 enable_thinking 时两处都不插 —— 行为等价于 tpl_chat.jinja"""
    p1 = render_agl(BASE)
    assert "<think>" not in p1
    p2 = render_agl(FULL)
    assert p2.startswith(p1 + R1)


def test_both_templates_keep_prefix_chain(render, render_agl):
    """两个模板在各自推荐用法下都必须保证前缀链连续"""
    for r, kw in ((render, {}), (render_agl, {"enable_thinking": False})):
        p1 = r(BASE, **kw)
        p2 = r(FULL, **kw)
        assert p2.startswith(p1 + R1)
