# Dependency & Patching Debugging on ROCm

## Rule: Spend at most 2 tool calls diagnosing. If not obvious, revert your changes and re-run.

## 4-Step Diagnostic Protocol

### Step 1: Map the dependency landscape
```bash
pip show transformers torch torchvision 2>/dev/null | grep -E "^(Name|Version|Location)"
python3 -c "import sys; print(f'Python: {sys.executable}')"
```
Compare with what the repo expects in `pyproject.toml` / `requirements.txt`.

### Step 2: Verify what's actually loaded at runtime
```python
import inspect
from transformers.models.gemma.modeling_gemma import GemmaModel
print(f"Loaded from: {inspect.getfile(GemmaModel)}")
```
The installed version may differ from what's loaded (patching, sys.path manipulation).

### Step 3: Identify the mismatch

| Symptom | Likely Cause | Diagnostic |
|---|---|---|
| `ImportError: cannot import name 'X'` | API removed/moved in newer version | `pip show pkg` → compare to `pyproject.toml` |
| `TypeError: unexpected keyword 'X'` | Function signature changed | `inspect.signature(module.function)` |
| `expected scalar type Float but found BFloat16` | Dtype conversion broken | Trace WHERE the wrong dtype originates (Step 4) |
| Patch runs but doesn't fix anything | Patch copies to wrong path or runs AFTER import | `inspect.getfile(TheClass)` to check source |

### Step 4: Trace the root cause — do NOT patch the crash site

For dtype mismatches, add forward hooks to find where dtype changes:
```python
for name, mod in model.named_modules():
    mod.register_forward_hook(lambda n: (lambda m, i, o: print(f"{n}: {o.dtype}") if isinstance(o, torch.Tensor) else None)(name))
```

For patching failures, verify files were actually copied:
```python
import pathlib, transformers
target = pathlib.Path(transformers.__file__).resolve().parent / "models" / "gemma" / "modeling_gemma.py"
print(f"Exists: {target.exists()}")
# Check for a signature unique to the custom version:
if target.exists():
    print("custom" if "some_custom_marker" in target.read_text() else "STOCK FILE - patch did NOT apply")
```

## Common Fix: `pathlib.Path.parents[N]` Trap

```python
# For: /usr/lib/python3/site-packages/transformers/__init__.py
p.parent      # → .../transformers/        ← the package dir (usually correct)
p.parents[1]  # → .../site-packages/       ← one level too high!
```
If a patching script uses `.parents[1]`, the copy lands in the wrong directory.

## Dependency Fix Ladder (in order of preference)

1. **Install exact version with `--no-deps`**: `pip install transformers==4.53.2 --no-deps`
2. **Check what depends on it**: `pip show transformers | grep "Required-by"`
3. **Always verify PyTorch survived**: `python3 -c "import torch; print(torch.__version__, torch.version.hip)"`
4. **If nothing works, use sys.path insertion** (custom code without touching site-packages):
   ```python
   sys.path.insert(0, "/path/to/repo/src/custom_models_parent/")
   ```

## CRITICAL Anti-Pattern: Never Fake Checks or Disable Features

**Never bypass a verification check to make code "run." Fix the underlying issue.**

Bad examples:
- Faking a patch check with `importlib.util` while patched files were never installed
- Commenting out `set_use_aiter_attention(True)` because the function doesn't exist (means custom model file wasn't patched) — results in 2-4x slower attention
- Rewriting dtype conversion order to mask a version mismatch

Every time you comment out a feature or fake a check, you hide a root cause that costs 2-10x performance.

## Dtype Mismatches (float32 vs bfloat16)

Common in vision-language models. Root causes:
1. `model.to(bfloat16)` doesn't propagate to all submodules
2. Custom dtype handling removes the global cast
3. Wrong transformers version loaded

**Fix**: Cast at boundaries between float32 and bfloat16 submodules:
```python
vision_output = self.vision_tower(pixel_values.float())
vision_output = vision_output.to(self.language_model.dtype)  # cast at boundary
```
Never modify the global model dtype logic.
