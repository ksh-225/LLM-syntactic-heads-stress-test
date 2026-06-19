"""
A.5 — Fix 1: Bidirectional Arc Coverage.

GPT-2's causal mask blocks 63% of English UD arcs and 94.6% of Hindi UD arcs
(any arc where the dependent precedes the head in linear order is unreachable
under a purely causal mask). For right-pointing arcs we reverse the query
direction: the head attends back to the dependent instead.
"""


def arc_attention_weight(attn, dep_pos, head_pos):
    """
    attn: attention tensor of shape [batch, heads, seq, seq]
    dep_pos, head_pos: token indices for the UD dependent / head

    Left arc (dependent after head): standard causal direction.
    Right arc (Fix 1): query direction reversed, head attends back to dep.
    """
    src = dep_pos if head_pos < dep_pos else head_pos
    tgt = head_pos if head_pos < dep_pos else dep_pos
    return float(attn[:, :, src, tgt].mean())
