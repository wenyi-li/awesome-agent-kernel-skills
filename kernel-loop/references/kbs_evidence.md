# KBS Evidence Template

Use this file to write `<output_dir>/vK/kbs_evidence.md` after NCU profiling and before `hypothesis.txt`.

Keep it short. Record only evidence that affects the next one-variable decision.

KBS queries must use space-separated semantic terms, not concatenated benchmark ids. Bad: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1`. Good: `DSA sparse attention h16 ckv512 kpe64 topk2048`, then broaden to `DeepSeek sparse attention topk` or `sparse MLA` if needed.

```markdown
# KBS Evidence - vK

## NCU Facts

- Runtime:
- Bottleneck:
- Key symptoms:
- Metrics driving next decision:

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `<kernel + backend + arch terms>` | kernel-specific tactic | `<top ids or 0 results>` |
| 2 | `<bottleneck symptom + technique terms>` | bottleneck tactic | `<top ids or 0 results>` |
| 3 | `<broader alias query if prior result was 0>` | fallback | `<top ids or 0 results>` |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| `<doc-id>` | `<canonical path>` | `<label>` | `<one sentence mapping to NCU symptom>` |

## Rejected / Limits

- `<doc-id or query>`: rejected because `<arch/layout/dtype/metric mismatch>`.

## Decision Link

`<NCU symptom>` -> `<KBS pattern/doc-id>` -> `<one-variable change>` -> `<expected metric movement>`
```
