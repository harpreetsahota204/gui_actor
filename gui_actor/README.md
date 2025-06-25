# GUI-Actor FiftyOne Integration - Code Implementation

This folder contains the implementation code for integrating Microsoft's GUI-Actor models with FiftyOne.

The original code resides here: https://github.com/microsoft/GUI-Actor/tree/main/src/gui_actor

## Files Overview

- `modeling_qwen25vl.py` - Custom Qwen2.5-VL model with pointer generation capabilities
- `inference.py` - Inference pipeline and attention processing functions
- `constants.py` - Model constants and configuration

## Recent Updates & Compatibility Fixes

### Transformers Library Compatibility

This implementation has been updated to work with newer versions of the Hugging Face Transformers library. Key changes include:

#### 1. Position Encoding Fix
**Issue**: `AttributeError` for missing `get_rope_index` method in custom model class.
**Solution**: Copied the `get_rope_index` function from the parent `Qwen2_5_VLModel` class into our custom implementation.

```python
# Added to custom model class
def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
    # Implementation copied from parent class
    # Ensures position encoding works correctly
```

#### 2. Embedding Access Modernization

**Issue**: Direct attribute access (`model.embed_tokens`) deprecated in newer Transformers versions.
**Solution**: Updated to use official API methods for better compatibility.

```python
# OLD (deprecated)
embeddings = model.embed_tokens(input_ids)

# NEW (recommended)
embeddings = self.model.get_input_embeddings()(input_ids)
```

#### 3. Debug Code Cleanup
**Removed**: Unused training utilities and verbose print statements
- Removed `from gui_actor.trainer import rank0_print` import
- Cleaned up debug prints that used `rank0_print`
- Simplified code paths for inference-only usage

### FiftyOne Integration Fixes

#### 4. LogitsProcessor State Isolation
**Issue**: The `ForceFollowTokensLogitsProcessor` maintains internal state (`self.force_queue`) that persists between inference calls, potentially causing token generation from one sample to affect the next.

**Solution**: Reset processor state for each inference call and fix mutable default arguments.

```python
# In inference function - reset state for each call
if logits_processor is None:
    logits_processor = ForceFollowTokensLogitsProcessor(...)
logits_processor.force_queue = []  # Reset state

# In LogitsProcessor class - fix mutable default
def __init__(self, token_a_id, forced_sequence=None):
    if forced_sequence is None:
        forced_sequence = [DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN]
    self.forced_sequence = forced_sequence
    self.force_queue = []
```

**Why this matters**: Stateful components can cause unexpected behavior when processing multiple samples, leading to inconsistent predictions and hard-to-debug issues.

## Key Lessons Learned

### 1. Method Inheritance in Custom Models
When extending transformer model classes, some methods may not be automatically inherited and must be explicitly added to your custom class. Always verify that required methods are available, especially for:
- Position encoding (`get_rope_index`)
- Attention mechanisms
- Custom forward pass logic

### 2. Future-Proof Embedding Access
Using official APIs instead of direct attribute access ensures:
- Compatibility with library updates
- Proper error handling
- Consistent behavior across model versions

### 3. State Management in ML Pipelines
When building model integrations:
- **Avoid instance state modifications** during prediction
- **Reset stateful components** between samples
- **Use local variables** for sample-specific data
- **Be mindful of mutable defaults** in class constructors

### 4. Dataset Processing Considerations
FiftyOne's flexible data model allows:
- Multiple samples with the same image but different metadata
- Batch processing that can expose state persistence issues
- Field-based prompting that requires careful state management

### 5. Code Maintenance Best Practices
- Remove unused imports and training-specific code for inference deployments
- Use official methods over internal attributes
- Keep parent class functionality synchronized when copying methods
- Test with realistic datasets that exercise edge cases
