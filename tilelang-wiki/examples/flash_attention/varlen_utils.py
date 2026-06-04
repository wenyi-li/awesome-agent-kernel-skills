# ruff: noqa
import math
import torch
from einops import rearrange, repeat
from bert_padding import pad_input, unpad_input


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    return padding_mask


def generate_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(output_unpad, indices_q, batch_size, seqlen_q)
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device)
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(output_unpad, "(b s) h d -> b s h d", b=batch_size)

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device)
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def padded_varlen_attention_reference(q, k, v, query_padding_mask=None, key_padding_mask=None, causal=False, groups=1):
    batch_size, seqlen_q, q_heads, dim = q.shape
    _, seqlen_k, kv_heads, _ = k.shape
    assert q_heads == kv_heads * groups, f"Expected q_heads == kv_heads * groups, got {q_heads}, {kv_heads}, {groups}"

    output = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(dim)

    for batch_idx in range(batch_size):
        q_len = int(query_padding_mask[batch_idx].sum().item()) if query_padding_mask is not None else seqlen_q
        k_len = int(key_padding_mask[batch_idx].sum().item()) if key_padding_mask is not None else seqlen_k
        if q_len == 0 or k_len == 0:
            continue

        q_valid = q[batch_idx, :q_len].float()
        k_valid = k[batch_idx, :k_len].float()
        v_valid = v[batch_idx, :k_len].float()

        if groups != 1:
            k_valid = k_valid.repeat_interleave(groups, dim=1)
            v_valid = v_valid.repeat_interleave(groups, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q_valid, k_valid) * scale
        if causal:
            q_positions = torch.arange(q_len, device=q.device)
            k_positions = torch.arange(k_len, device=q.device)
            offset = k_len - q_len
            causal_mask = (q_positions[:, None] + offset) >= k_positions[None, :]
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
            visible_rows = causal_mask.any(dim=1)
            scores = torch.where(visible_rows.view(1, q_len, 1), scores, torch.zeros_like(scores))
        else:
            visible_rows = None

        attn = torch.softmax(scores, dim=-1)
        if visible_rows is not None:
            attn = torch.where(visible_rows.view(1, q_len, 1), attn, torch.zeros_like(attn))
        output[batch_idx, :q_len] = torch.einsum("hqk,khd->qhd", attn, v_valid).to(q.dtype)

    return output
