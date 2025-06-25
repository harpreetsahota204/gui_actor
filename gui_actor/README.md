# GUI-Actor FiftyOne Integration - Code Implementation

This folder contains the implementation code for integrating Microsoft's GUI-Actor models with FiftyOne.

The original code resides here: https://github.com/microsoft/GUI-Actor/tree/main/src/gui_actor

## Files Overview

- `modeling_qwen25vl.py` - Custom Qwen2.5-VL model with pointer generation capabilities
- `inference.py` - Inference pipeline and attention processing functions
- `constants.py` - Model constants and configuration

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

### 3. Code Maintenance Best Practices
- Remove unused imports and training-specific code for inference deployments
- Use official methods over internal attributes
- Keep parent class functionality synchronized when copying methods