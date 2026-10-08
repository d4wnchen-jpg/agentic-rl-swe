#!/usr/bin/env python3
"""grpo_util.py — GRPO 的两个核心工具

1. build_row_lcp     : 把多轮 rollout 拼成一条连续训练行（LCP 对齐）
                       同时给出 loss_mask 和 old_logprobs
2. masked_logprobs   : 只对 mask 位置计算 lm_head，避免 [S, vocab] 巨型 logits
"""
from __future__ import annotations

# torch 只在 masked_logprobs 里用到，做成延迟导入：
# 这样 build_row_lcp（纯 Python 算法）可以在没有 torch 的环境里被导入和测试。


# ═══════════════════════════════════════════════════════════
# ① LCP 对齐
# ═══════════════════════════════════════════════════════════
def _lcp(a, b):
    """最长公共前缀长度"""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _find_subseq(hay, needle, start=0):
    """在 hay 中从 start 起找 needle，返回起始下标；找不到返回 -1"""
    L, N = len(hay), len(needle)
    if N == 0 or N > L:
        return -1
    for i in range(start, L - N + 1):
        if hay[i:i + N] == needle:
            return i
    return -1


def build_row_lcp(turns, max_tokens=16384, min_loss_tokens=1):
    """turns: [[prompt_ids, response_ids, response_logprobs], ...]

    返回 (input_ids, loss_mask, old_logprobs) 或 None
      loss_mask[i]=1  -> 该 token 是模型生成的（要算 loss）
      old_logprobs[i] -> 对应的【采样时】logprob（mask=0 处为 0.0）

    算法（因果 LM 的前缀链）：
      row = P0 + R0
      对第 k 轮:
          L = LCP(row, P_k)          # 用 P_k 作权威，丢弃被模板改写的 token
          row = row[:L] + P_k[L:]
          把 P_k 中等于历史 response 的片段标回 mask=1，并还原其 logprob
          row += R_k
    """
    if not turns:
        return None

    def unpack(t):
        P = list(t[0]) if len(t) > 0 and t[0] else []
        R = list(t[1]) if len(t) > 1 and t[1] else []
        LP = list(t[2]) if len(t) > 2 and t[2] else []
        if len(LP) != len(R):
            LP = (LP + [0.0] * len(R))[:len(R)]
        return P, R, LP

    P0, R0, L0 = unpack(turns[0])
    if not P0 or not R0:
        return None

    row = list(P0) + list(R0)
    mask = [0] * len(P0) + [1] * len(R0)
    olp = [0.0] * len(P0) + list(L0)
    spans = [(len(P0), len(R0), L0)]

    for k in range(1, len(turns)):
        P, R, LP = unpack(turns[k])
        if not P or not R:
            continue
        L = _lcp(row, P)
        new_row = row[:L] + list(P[L:])
        new_mask = mask[:L] + [0] * (len(P) - L)
        new_olp = olp[:L] + [0.0] * (len(P) - L)

        for (pos, ln, lp_hist) in spans:
            r = row[pos:pos + ln]
            st = _find_subseq(P, r, start=L)
            if st >= 0:
                for i in range(st, min(st + ln, len(new_mask))):
                    new_mask[i] = 1
                    new_olp[i] = lp_hist[i - st] if (i - st) < len(lp_hist) else 0.0

        new_row += list(R)
        new_mask += [1] * len(R)
        new_olp += LP
        spans.append((len(new_row) - len(R), len(R), LP))
        row, mask, olp = new_row, new_mask, new_olp

    if sum(mask) < min_loss_tokens:
        return None

    if len(row) > max_tokens:
        head = list(P0)
        keep = max_tokens - len(head)
        if keep <= 0:
            return None
        row = head + row[-keep:]
        mask = [0] * len(head) + mask[-keep:]
        olp = [0.0] * len(head) + olp[-keep:]

    return row, mask, olp


# ═══════════════════════════════════════════════════════════
# ② 只对 mask 位置算 logits
# ═══════════════════════════════════════════════════════════
def find_lm_head(model):
    """返回所有以 lm_head 结尾的模块 [(name, module), ...]（按 id 去重）"""
    seen, out = set(), []
    for n, m in model.named_modules():
        if n.split(".")[-1] == "lm_head" and id(m) not in seen:
            seen.add(id(m))
            out.append((n, m))
    if not out:
        raise RuntimeError("找不到 lm_head")
    return out


def masked_logprobs(model, input_ids, loss_mask, chunk=512):
    """对 loss_mask==1 的位置计算 logprob（带梯度）

    两处显存优化（对应 agent-lightning 的两个配置）：
      1. 分块计算 lm_head
         <- 对应 agl 的 `entropy_from_logits_with_chunking: True`
      2. 分块内用 torch.utils.checkpoint 做【反向重算】
         <- 对应 agl 的 `use_fused_kernels: True`（前向不物化，反向重算）

    不做 2 的话，每个 chunk 的 [chunk, V] fp32 中间结果都要留到反向：
      chunk=512, V=248320 -> 0.5GB x 13 chunk x 2 = 13GB ❌
    做了 2 之后只留 chunk 的输入（很小）✅

    ⚠️ 因果 LM 错位：预测 position i 的 logits 来自 position i-1 的隐藏状态
    ⚠️ 必须 patch【被 forward 实际调用的】lm_head 的 forward 方法
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    ids = torch.as_tensor(input_ids, dtype=torch.long, device="cuda").unsqueeze(0)
    m = torch.as_tensor(loss_mask, dtype=torch.long, device="cuda")

    pred_pos = torch.nonzero(m[1:] == 1, as_tuple=False).squeeze(-1) + 1
    if pred_pos.numel() == 0:
        return None, None
    head_pos = pred_pos - 1
    targets = ids[0][pred_pos]

    heads = find_lm_head(model)
    saved = []
    for name, mod in heads:
        orig_fwd = mod.forward
        saved.append((mod, orig_fwd))

        def _make(orig_fwd):
            def _seg(x_seg, tg_seg):
                lg = orig_fwd(x_seg)                       # [1, c, V]
                lp = F.log_softmax(lg.float(), dim=-1)
                return lp.gather(-1, tg_seg.view(1, -1, 1)).squeeze(-1)

            def _chunked(x):
                outs = []
                n = head_pos.numel()
                for i in range(0, n, chunk):
                    idx = head_pos[i:i + chunk]
                    tg = targets[i:i + chunk]
                    if x.requires_grad:
                        o = checkpoint(_seg, x[:, idx, :], tg, use_reentrant=False)
                    else:
                        o = _seg(x[:, idx, :], tg)
                    outs.append(o)
                return torch.cat(outs, dim=1)
            return _chunked
        mod.forward = _make(orig_fwd)

    try:
        out = model(input_ids=ids, use_cache=False)
        o = out.logits
        # 分块 head 返回 [1, N]（logprob）；未被拦截时是 [1, S, V] 或 [S, V]
        if o.dim() == 3 and o.shape[0] == 1:
            o = o[0]                              # [S, V]
        if o.dim() == 2 and o.shape[0] == 1:
            o = o[0]                              # [N]  <- 分块输出
        if o.dim() != 1 or o.shape[0] != head_pos.numel():
            raise RuntimeError(
                f"lm_head 未被拦截或形状异常！out.logits.shape={tuple(out.logits.shape)}，"
                f"期望 [1, {head_pos.numel()}]。候选={[n for n, _ in heads]}")
        logits = o
    finally:
        for mod, f in saved:
            mod.forward = f

    return logits, pred_pos
