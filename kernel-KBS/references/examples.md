---
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
---

# Query Playbook

These examples show how to translate user questions into SQLite queries, evidence reads, and final answers. Default mode is read-only; create, update, delete, rebuild, and optimize actions require explicit human permission first.

Basic loop:

```bash
python3 scripts/kbs.py query "<terms>" --limit 10
python3 scripts/kbs.py get <doc-id> --follow-sources --include-code
```

## 1. Broad Hardware Question

User: "How do I use Blackwell tcgen05 / UMMA?"

```bash
python3 scripts/kbs.py query tcgen05 --architecture sm100 --limit 10
python3 scripts/kbs.py get hw-tcgen05-mma --follow-sources --include-code
python3 scripts/kbs.py get hw-tmem --follow-sources
```

Answer shape:

- Start from `hw-tcgen05-mma`, not random PR snippets.
- Explain issue model, TMEM accumulator, fences, and SMEM descriptor constraints.
- Follow `sources` to official docs / CUTLASS PR before making authoritative claims.

## 2. Fast GEMM On B200

User: "How do I write a fast GEMM kernel on B200?"

```bash
python3 scripts/kbs.py query --kind wiki-kernel --tag gemm --architecture sm100 --limit 10
python3 scripts/kbs.py query tcgen05 tmem gemm --repo cutlass --limit 10
python3 scripts/kbs.py get kernel-fp8-block-scale-gemm --follow-sources --include-code
python3 scripts/kbs.py get kernel-nvfp4-gemm --follow-sources --include-code
```

Answer shape:

- Use kernel pages for synthesis and PR pages for concrete implementation provenance.
- Discuss tile shape, TMA pipeline, warp specialization, TMEM epilogue, and CLC only when supported by sources.
- If reporting speed, use `performance_claims` fields from `get --json`.

## 3. Low SM Utilization

User: "B200 kernel SM utilization is low. What should I inspect?"

```bash
python3 scripts/kbs.py query --symptom low-sm-utilization --limit 10
python3 scripts/kbs.py get pattern-low-sm-utilization --follow-sources
python3 scripts/kbs.py query clc persistent tile-scheduling --architecture sm100 --limit 10
```

Answer shape:

- Treat it as diagnosis, not immediate optimization advice.
- Check tail effect, load imbalance, and launch/work queue behavior.
- Recommend candidate pages such as persistent kernels, tile scheduling, and CLC only after reading the pattern page.

## 4. Memory-Bound NVFP4/GEMV

User: "How should I optimize a memory-bound NVFP4 GEMV?"

```bash
python3 scripts/kbs.py query --symptom memory-bound --limit 10
python3 scripts/kbs.py query nvfp4 gemv --architecture sm100 --limit 10
python3 scripts/kbs.py get pattern-memory-bound --follow-sources
python3 scripts/kbs.py get kernel-nvfp4-gemv --follow-sources --include-code
```

Answer shape:

- Prioritize bandwidth, coalescing, vectorized loads, cache policy, and register budgeting.
- Do not over-focus on MMA throughput if the evidence says the kernel is memory-bound.
- Cite blog/source pages when using contest or community-derived techniques.

## 5. Repo-Specific Implementation Search

User: "Where does CUTLASS use tcgen05.mma?"

```bash
python3 scripts/kbs.py query tcgen05 --repo cutlass --limit 20
python3 scripts/kbs.py query tcgen05 mma --repo cutlass --has-code --limit 20
python3 scripts/kbs.py get <selected-pr-id> --follow-sources --include-code
```

Answer shape:

- Do not hard-code PR counts; run `stats` or trust the query result.
- Prefer PR rows with relevant `changed_paths`, artifact metadata, or code bundle.
- Cite `pr-...` id and canonical `store/docs/sources/prs/...` path.

## 6. FlashAttention-4 Evidence

User: "What are the FlashAttention-4 implementation details and performance evidence?"

```bash
python3 scripts/kbs.py get kernel-flash-attention-4 --follow-sources --include-code
python3 scripts/kbs.py get kernel-flash-attention-4 --json
python3 scripts/kbs.py query flashattention tcgen05 --repo flashinfer --limit 10
```

Answer shape:

- Use `kernel-flash-attention-4` for synthesis.
- Use `performance_claims` for numbers: GPU, dtype, shape, metric, value, utilization, source_id.
- Follow source links before summarizing claims from paper/blog/PR.

## 7. Hopper To Blackwell Migration

User: "What is the difference between wgmma and tcgen05?"

```bash
python3 scripts/kbs.py get migration-wgmma-to-tcgen05 --follow-sources
python3 scripts/kbs.py get migration-register-to-tmem --follow-sources
python3 scripts/kbs.py get hw-tcgen05-mma --follow-sources
```

Answer shape:

- Separate instruction model, accumulator storage, synchronization, and epilogue changes.
- Do not apply Hopper-only advice to SM100 unless the page has `blackwell_relevance`.

## 8. Triton On Blackwell

User: "Does Triton support Blackwell tcgen05/TMEM now?"

```bash
python3 scripts/kbs.py query triton blackwell --language triton --limit 10
python3 scripts/kbs.py get lang-triton --follow-sources --include-code
python3 scripts/kbs.py query triton sm100 --repo vllm --limit 10
```

Answer shape:

- Mention that this KBS tracks a version-sensitive claim, `vs-triton-3.6-blackwell-tcgen05`.
- For current/latest upstream behavior, verify outside the local KBS before giving a definitive latest-version answer.
- Distinguish native Triton lowering, Gluon-only paths, and downstream PR evidence.

## 9. MoE / FlashInfer / FP8 Search

User: "Find FlashInfer PRs related to FP8 MoE."

```bash
python3 scripts/kbs.py query fp8 moe --repo flashinfer --limit 20
python3 scripts/kbs.py query --repo flashinfer --tag moe --limit 20
python3 scripts/kbs.py query --repo flashinfer --tag fp8 --limit 20
python3 scripts/kbs.py get <selected-pr-id> --follow-sources --include-code
```

Answer shape:

- Use combined free text first, then exact tag filters.
- If results include tests/refactors, say so; do not imply every result is a kernel implementation.
- Open selected PR pages before summarizing changed behavior.

## 10. Contest-Derived Techniques

User: "What techniques did the winning GPU Mode NVFP4 Hackathon solutions use?"

```bash
python3 scripts/kbs.py query --kind source-contest --tag nvfp4 --limit 20
python3 scripts/kbs.py query nvfp4 hackathon --limit 20
python3 scripts/kbs.py get <contest-id> --follow-sources --include-code
```

Answer shape:

- Treat contest pages as source notes, not canonical hardware docs.
- Separate official submissions, author-published posthoc code, reconstructed notes, and unavailable code.
- Cite submission truth/provenance when recommending a technique.

## 11. Raw Relationship Inspection

Use this only when the CLI result is not enough and a direct relationship table query is useful:

```bash
sqlite3 store/indexes/kernel_kbs.sqlite \
  "SELECT d.id, l.rel_type, l.target_id
   FROM docs d
   JOIN doc_links l ON l.doc_pk = d.doc_pk
   WHERE d.id = 'kernel-flash-attention-4';"
```

Prefer `get --follow-sources` for normal use. Raw SQL is for auditing or compound analysis.

## 12. Permissioned CRUD Examples

Only run these after the user explicitly approves the mutation.

Create/update from Markdown:

```bash
python3 scripts/kbs.py put store/docs/wiki/techniques/example.md
python3 scripts/kbs.py doctor
python3 scripts/kbs.py query example --limit 3
```

Metadata update:

```bash
python3 scripts/kbs.py update <doc-id> --set confidence=source-reported --reason "correct evidence level"
python3 scripts/kbs.py get <doc-id> --frontmatter-only
```

Soft delete and restore:

```bash
python3 scripts/kbs.py delete <doc-id>
python3 scripts/kbs.py query <terms> --include-deleted --limit 5
python3 scripts/kbs.py restore <doc-id>
```

Bulk Markdown changes:

```bash
python3 scripts/kbs.py build-db --reset
python3 scripts/kbs.py doctor
python3 scripts/kbs.py check all
```

## Anti-Patterns

- Do not answer from this playbook alone; use it to choose commands.
- Do not recurse through Markdown files as the retrieval strategy.
- Do not cite a performance number unless `performance_claims` has enough fields.
- Do not treat `source-pr` test/refactor pages as kernel implementation evidence without checking `changed_paths` and artifacts.
- Do not mutate SQLite, Markdown, ledgers, state, or artifacts without explicit human permission.
