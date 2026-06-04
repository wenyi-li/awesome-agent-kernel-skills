# FlashInfer Sampling Reference

Sources:
- [Sampling API](https://docs.flashinfer.ai/api/sampling.html)
- [Sorting-Free Sampling](https://flashinfer.ai/2025/03/10/sampling.html)

## Core Sampling Operations

### Sampling from Probabilities

**Basic sampling**:
```python
# Sample from probability distribution
next_tokens = flashinfer.sampling_from_probs(
    probs=probs,                # (batch_size, vocab_size)
    uniform_samples=torch.rand(batch_size, device='cuda'),
    deterministic=False,
)
```

### Top-K Sampling

**Purpose**: Sample from top K most likely tokens

```python
next_tokens = flashinfer.top_k_sampling_from_probs(
    probs=probs,                # (batch_size, vocab_size)
    top_k=40,                   # Keep top 40 tokens
    uniform_samples=torch.rand(batch_size, device='cuda'),
)

# From logits (pre-softmax)
next_tokens = flashinfer.top_k_sampling_from_logits(
    logits=logits,
    top_k=40,
    uniform_samples=torch.rand(batch_size, device='cuda'),
)
```

### Top-P Sampling (Nucleus)

**Purpose**: Sample from smallest set of tokens whose cumulative probability ≥ p

```python
next_tokens = flashinfer.top_p_sampling_from_probs(
    probs=probs,
    top_p=0.9,                  # 90% cumulative probability
    uniform_samples=torch.rand(batch_size, device='cuda'),
)
```

### Min-P Sampling

**Purpose**: Remove tokens below minimum probability threshold

```python
next_tokens = flashinfer.min_p_sampling_from_probs(
    probs=probs,
    min_p=0.05,                 # Remove tokens with prob < 5%
    uniform_samples=torch.rand(batch_size, device='cuda'),
)
```

### Combined Top-K and Top-P

**Apply both filters**:
```python
next_tokens = flashinfer.top_k_top_p_sampling_from_logits(
    logits=logits,
    top_k=40,
    top_p=0.9,
    uniform_samples=torch.rand(batch_size, device='cuda'),
)

# Or from probabilities
next_tokens = flashinfer.top_k_top_p_sampling_from_probs(
    probs=probs,
    top_k=40,
    top_p=0.9,
    uniform_samples=torch.rand(batch_size, device='cuda'),
)
```

## Sorting-Free Algorithm

### Dual Pivot Rejection Sampling

**Traditional Approach**:
1. Sort entire vocabulary: O(V log V)
2. Select top-k/top-p: O(k)
3. Sample: O(1)

**FlashInfer Approach (Sorting-Free)**:
1. Rejection sampling with dual pivots: O(log V)
2. No explicit sorting required
3. Logarithmic time complexity

### Algorithm

```python
def dual_pivot_rejection_sampling(probs, top_k, top_p):
    # Initialize pivots
    lower_pivot = 0.0
    upper_pivot = 1.0
    
    # Binary search for valid range
    while True:
        # Sample candidate
        u = uniform(0, 1)
        candidate_idx = select_token(u, lower_pivot, upper_pivot)
        
        # Check acceptance criteria
        if meets_top_k_criteria(candidate_idx, top_k) and \
           meets_top_p_criteria(candidate_idx, top_p):
            return candidate_idx
        
        # Update pivots based on rejection
        if candidate_idx < top_k:
            lower_pivot = update_lower(candidate_idx)
        else:
            upper_pivot = update_upper(candidate_idx)
```

### Performance

**Complexity**:
- Traditional: O(V log V)
- FlashInfer: O(log V)

**Speedup**: 5-10x faster for large vocabularies

## Probability Renormalization

### Top-K Renormalization

```python
# Renormalize after top-k filtering
renormalized_probs = flashinfer.top_k_renorm_probs(
    probs=probs,
    top_k=40,
)

# Mask logits (zero out bottom tokens)
masked_logits = flashinfer.top_k_mask_logits(
    logits=logits,
    top_k=40,
)
```

### Top-P Renormalization

```python
renormalized_probs = flashinfer.top_p_renorm_probs(
    probs=probs,
    top_p=0.9,
)
```

## Speculative Decoding

### Chain Speculative Sampling

**Purpose**: Verify draft model tokens against target model

```python
# Draft model generates multiple tokens
draft_tokens = draft_model.generate(prompt, num_tokens=5)
draft_probs = draft_model.get_probs(draft_tokens)

# Target model verifies
target_probs = target_model.get_probs(draft_tokens)

# Accept/reject tokens
accepted_tokens = flashinfer.chain_speculative_sampling(
    draft_probs=draft_probs,      # (batch, num_draft, vocab)
    draft_tokens=draft_tokens,      # (batch, num_draft)
    target_probs=target_probs,     # (batch, num_draft+1, vocab)
    uniform_samples=torch.rand(batch, num_draft+1, device='cuda'),
)
```

### Acceptance Criteria

```python
def speculative_acceptance(draft_prob, target_prob, uniform_sample):
    # Accept if target >= draft
    if target_prob >= draft_prob:
        return True
    
    # Probabilistic acceptance
    accept_prob = target_prob / draft_prob
    return uniform_sample < accept_prob
```

## Advanced Sampling Patterns

### Temperature Sampling

```python
# Apply temperature before sampling
temperature = 0.7
scaled_logits = logits / temperature

next_tokens = flashinfer.top_k_top_p_sampling_from_logits(
    logits=scaled_logits,
    top_k=40,
    top_p=0.9,
    uniform_samples=torch.rand(batch_size, device='cuda'),
)
```

### Repetition Penalty

```python
# Penalize repeated tokens
def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    for i, tokens in enumerate(generated_tokens):
        for token in tokens:
            if logits[i, token] > 0:
                logits[i, token] /= penalty
            else:
                logits[i, token] *= penalty
    return logits

logits = apply_repetition_penalty(logits, generated_tokens)
next_tokens = flashinfer.sampling_from_probs(
    probs=torch.softmax(logits, dim=-1),
    uniform_samples=torch.rand(batch_size, device='cuda'),
)
```

### Presence and Frequency Penalties

```python
def apply_penalties(logits, generated_tokens, presence=0.5, frequency=0.5):
    for i, tokens in enumerate(generated_tokens):
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            # Presence penalty (binary)
            logits[i, token] -= presence
            
            # Frequency penalty (count-based)
            logits[i, token] -= frequency * count
    
    return logits
```

## Batch Sampling

### Heterogeneous Sampling

**Different parameters per request**:
```python
# Per-request top-k and top-p
batch_size = 8
top_k_per_request = torch.tensor([40, 50, 30, 40, 40, 60, 40, 50], device='cuda')
top_p_per_request = torch.tensor([0.9, 0.8, 0.95, 0.9, 0.85, 0.9, 0.9, 0.8], device='cuda')

# Sample with per-request parameters
next_tokens = []
for i in range(batch_size):
    token = flashinfer.top_k_top_p_sampling_from_probs(
        probs=probs[i:i+1],
        top_k=top_k_per_request[i].item(),
        top_p=top_p_per_request[i].item(),
        uniform_samples=torch.rand(1, device='cuda'),
    )
    next_tokens.append(token)

next_tokens = torch.cat(next_tokens)
```

## Top-K Selection (No Sampling)

```python
from flashinfer.topk import select_top_k

# Get top-k tokens and probabilities (no sampling)
top_k_probs, top_k_indices = flashinfer.topk.select_top_k(
    probs=probs,
    top_k=5,
)

# Returns:
# top_k_probs: (batch, k)
# top_k_indices: (batch, k)
```

## Beam Search Integration

```python
class BeamSearchSampler:
    def __init__(self, beam_width, vocab_size):
        self.beam_width = beam_width
        self.vocab_size = vocab_size
    
    def step(self, beam_probs, new_logits):
        # Expand beams
        expanded_probs = beam_probs.unsqueeze(-1) + torch.log_softmax(new_logits, dim=-1)
        
        # Select top-k from all beams
        top_probs, top_indices = flashinfer.topk.select_top_k(
            probs=expanded_probs.flatten(-2),
            top_k=self.beam_width,
        )
        
        # Map back to beam and token indices
        beam_indices = top_indices // self.vocab_size
        token_indices = top_indices % self.vocab_size
        
        return top_probs, beam_indices, token_indices
```

## Logits Processors

### Standard Processors

```python
class LogitsProcessor:
    def temperature(self, logits, temperature):
        return logits / temperature
    
    def repetition_penalty(self, logits, tokens, penalty):
        return apply_repetition_penalty(logits, tokens, penalty)
    
    def top_k_filter(self, logits, top_k):
        return flashinfer.top_k_mask_logits(logits, top_k)
    
    def process(self, logits, **kwargs):
        # Apply all processors in sequence
        logits = self.temperature(logits, kwargs.get('temperature', 1.0))
        logits = self.repetition_penalty(logits, kwargs['tokens'], kwargs.get('penalty', 1.0))
        logits = self.top_k_filter(logits, kwargs.get('top_k', 50))
        return logits
```

## Best Practices

1. **Use sorting-free kernels**: 5-10x faster than traditional sampling
2. **Batch sampling**: Process multiple requests together
3. **Preallocate uniform samples**: Reuse random numbers
4. **Combine top-k and top-p**: Better quality than either alone
5. **Apply temperature carefully**: Lower values (<0.7) for factual, higher (>1.0) for creative
6. **Use speculative decoding**: 2-3x speedup with minimal quality loss
7. **Profile sampling overhead**: Should be <5% of total inference time
