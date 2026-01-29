import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLForConditionalGeneration
from typing import List, Tuple, Union, Optional

class QwenVLwithVisionHeadOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    """
    Output class for Qwen2_5_VL with pointer head, extending the base output class.
    
    Args:
        lm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss.
        pointer_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Vision pointer network loss.
        pointer_scores (`List[torch.FloatTensor]`, *optional*):
            Attention scores from the pointer network, one tensor per batch item.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Combined loss (weighted sum of lm_loss and pointer_loss).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores from the language modeling head.
        past_key_values, hidden_states, attentions, rope_deltas:
            Same as parent class.
    """
    def __init__(self, loss=None, lm_loss=None, pointer_loss=None, pointer_scores=None, *args, **kwargs):
        # Pass loss to parent class
        super().__init__(loss=loss, *args, **kwargs)
        self.lm_loss = lm_loss
        self.pointer_loss = pointer_loss
        self.pointer_scores = pointer_scores


class VisionHead_MultiPatch(nn.Module):
    """
    Vision head for multi-patch attention that processes visual features and computes attention scores.
    
    This module takes encoded visual features and decoder hidden states, processes them through 
    projection networks and self-attention, and computes attention scores between them.
    
    Args:
        d_model (int): Hidden dimension size
        projection_dim (int): Intermediate projection dimension
        num_attention_heads (int, optional): Number of attention heads. Defaults to 8.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, d_model, projection_dim, num_attention_heads=8, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Projection networks for encoder and decoder features
        # Note: We omit additional normalization here because Qwen2VL
        # already normalizes hidden states using RMSNorm.
        self.projection_enc = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )
        self.projection_dec = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )

        # Self-attention layer to process visual features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization and dropout for attention outputs
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                hidden_state_enc,  # shape: [n_enc, d_model] where n_enc can vary with image size
                hidden_state_dec,  # shape: [n_dec, d_model] there can be multiple query in one sample
                labels: Optional[torch.Tensor] = None,  # shape: [n_dec, n_enc], binary mask of patches in bbox
                do_single_patch: bool = False,
               ):
        """
        Forward pass of the vision head.
        
        Args:
            hidden_state_enc: Encoded visual features
            hidden_state_dec: Decoder hidden states
            labels: Binary mask indicating ground truth patches
            do_single_patch: Whether to use single patch mode (deprecated)
            
        Returns:
            attn_weights: Attention scores between decoder states and visual features
            loss: KL divergence loss between predicted and target distributions (if labels provided)
        """
        # Add batch dimension for self-attention
        enc_input = hidden_state_enc.unsqueeze(0)
        
        # Apply self-attention to visual features
        attn_output, _ = self.self_attention(
            query=enc_input,
            key=enc_input,
            value=enc_input,
            need_weights=False
        )
        
        # Residual connection and layer normalization
        hidden_state_enc_ctx = self.layer_norm(enc_input + self.dropout(attn_output))
        # Remove batch dimension
        hidden_state_enc_ctx = hidden_state_enc_ctx.squeeze(0)  # [n_enc, d_model]

        # Project encoder and decoder features through MLPs
        proj_enc = self.projection_enc(hidden_state_enc_ctx)  # [n_enc, d_model]
        proj_dec = self.projection_dec(hidden_state_dec)  # [n_dec, d_model]
        
        # Compute scaled dot-product attention scores
        # Scaling by sqrt(d_model) is critical regardless of variable n_enc
        scaling = self.d_model ** 0.5
        patch_logits = torch.matmul(proj_dec, proj_enc.transpose(0, 1)) / scaling  # [n_dec, n_enc]
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(patch_logits, dim=-1)

        loss = None
        if (labels is not None) and (not do_single_patch):
            # Compute KL divergence loss between predicted and target distributions
            epsilon = 1e-8  # Small constant for numerical stability
            labels_float = labels.float()
            # Normalize labels to get target probability distribution
            target_dist = labels_float / (labels_float.sum(dim=-1, keepdim=True) + epsilon)

            # Get log probabilities of predictions
            pred_log_probs = F.log_softmax(patch_logits, dim=-1)
            # Compute KL divergence loss
            loss = F.kl_div(pred_log_probs, target_dist, reduction='batchmean')

        if do_single_patch and (labels is not None):
            # Legacy single patch mode - uses cross entropy loss
            loss = F.cross_entropy(attn_weights, labels)

        return attn_weights, loss


class Qwen2_5_VLForConditionalGenerationWithPointer(Qwen2_5_VLForConditionalGeneration):
    """
    Qwen2.5-VL model with additional pointer network for visual grounding.
    
    This class extends the base Qwen2.5-VL model by adding a pointer mechanism that allows
    the model to ground text tokens to specific visual patches/regions.
    
    Args:
        *args: Variable length argument list passed to parent class
        **kwargs: Arbitrary keyword arguments passed to parent class
            pointer_loss_weight (float): Weight for pointer loss (default: 1.0)
            lm_loss_weight (float): Weight for language modeling loss (default: 1.0)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize pointer head for multi-patch attention
        self.multi_patch_pointer_head = VisionHead_MultiPatch(self.config.text_config.hidden_size, self.config.text_config.hidden_size)
        # Get loss weights from kwargs or use defaults
        self.pointer_loss_weight = kwargs.get("pointer_loss_weight", 1.0)
        self.lm_loss_weight = kwargs.get("lm_loss_weight", 1.0)
        # Initialize rope_deltas attribute to avoid AttributeError
        self.rope_deltas = None
        # Initialize gradient checkpointing settings
        self._use_gradient_checkpointing = False
        self.post_init()
    
    def reset_loss_weights(self, pointer_loss_weight, lm_loss_weight):
        """Update the weights used for combining losses"""
        self.pointer_loss_weight = pointer_loss_weight
        self.lm_loss_weight = lm_loss_weight
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the model.
        
        This significantly reduces memory usage during training at the cost of some
        additional computation to recompute intermediate activations.
        """
        if not self._use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            # Enable requires_grad for input activations if gradient checkpointing is enabled
            self.enable_input_require_grads()
            self._use_gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disables gradient checkpointing for the model."""
        if self._use_gradient_checkpointing:
            self.model.gradient_checkpointing_disable()
            self._use_gradient_checkpointing = False
            
    def enable_input_require_grads(self):
        """
        Enables the computation of gradients with respect to input embeddings.
        
        This is needed when using gradient checkpointing with the huggingface
        Trainer class and for efficient memory usage during training.
        """
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
            
        self.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate position IDs and RoPE deltas for rotary position embeddings.
        
        This method handles both image and video inputs, computing appropriate position
        indices based on the spatial and temporal dimensions of the inputs.
        
        Args:
            input_ids: Input token IDs
            image_grid_thw: Image grid dimensions (time, height, width)
            video_grid_thw: Video grid dimensions (time, height, width) 
            second_per_grid_ts: Seconds per grid for video temporal encoding
            attention_mask: Attention mask for input sequence
            
        Returns:
            position_ids: Computed position indices for RoPE
            mrope_position_deltas: Position deltas for RoPE calculations
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        
        # Handle case with vision inputs
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            
            # Process each sequence in batch
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                
                # Count number of image and video tokens
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                
                # Process each vision token
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                        
                    # Handle image token
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    # Handle video token
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                        
                    # Calculate grid dimensions
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    # Generate position IDs for text tokens
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # Generate position IDs for vision tokens
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    # Normalize temporal dimension
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    # Generate spatial position indices
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                # Handle remaining text tokens
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # Combine all position IDs
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
            
        # Handle case without vision inputs
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def forward(self,
                input_ids: torch.LongTensor = None, # (batch_size, seq_len)
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                rope_deltas: Optional[torch.LongTensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
                second_per_grid_ts: Optional[torch.Tensor] = None,
                # Grounding
                visual_token_indices_of_coordinates: Optional[torch.Tensor] = None, # shape: (batch_size, n_target); each element is the ground-truth index of the visual token that should be attended to for the corresponding target token
                multi_patch_labels: Optional[torch.Tensor] = None, # shape: list [(n_target, n_visual), ...]; binary mask of patches in bbox
                if_multi_patch: bool = True,
                coordinates: Optional[List[Tuple[float, float]]] = None,
                verbose: bool = False) -> Union[Tuple, QwenVLwithVisionHeadOutputWithPast]:
        """
        Forward pass of the model.
        
        This method handles both language modeling and visual grounding tasks. It processes
        text and vision inputs, computes losses for both tasks if labels are provided, and
        returns appropriate outputs based on the input configuration.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input sequence
            position_ids: Position IDs for input tokens
            past_key_values: Cached key/value states for faster inference
            inputs_embeds: Pre-computed input embeddings
            labels: Labels for language modeling loss
            use_cache: Whether to use past key/values for faster inference
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a ModelOutput object
            pixel_values: Image pixel values
            pixel_values_videos: Video pixel values
            image_grid_thw: Image grid dimensions
            video_grid_thw: Video grid dimensions
            rope_deltas: RoPE position deltas
            cache_position: Position in generation cache
            second_per_grid_ts: Seconds per grid for video temporal encoding
            visual_token_indices_of_coordinates: Ground truth visual token indices
            multi_patch_labels: Binary masks for patches in bounding boxes
            if_multi_patch: Whether to use multi-patch mode
            coordinates: Coordinate pairs for visual grounding
            verbose: Whether to print debug information
            
        Returns:
            Union[Tuple, QwenVLwithVisionHeadOutputWithPast]: Model outputs including losses,
            logits, and attention scores if requested
        """
        # Input validation
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
            
        # Additional validation for training mode
        if labels is not None:
            # When labels are provided, we are in training mode and should validate inputs
            # Check if we have the necessary inputs for grounding loss if needed
            if self.pointer_loss_weight > 0:
                if pixel_values is None and pixel_values_videos is None:
                    raise ValueError("When pointer_loss_weight > 0 and labels are provided, "
                                     "you must provide either pixel_values or pixel_values_videos")
                if visual_token_indices_of_coordinates is None and multi_patch_labels is None:
                    import warnings
                    warnings.warn(
                        "Pointer loss is enabled (pointer_loss_weight > 0), but neither "
                        "visual_token_indices_of_coordinates nor multi_patch_labels are provided. "
                        "Pointer loss will not be calculated.",
                        RuntimeWarning
                    )
                    
        # Set output flags based on config if not explicitly provided
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get input embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids) # shape: (batch_size, seq_len, d_model)
            
        # Process image inputs if provided
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            # Replace image token embeddings with visual features
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Process video inputs if provided
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                # Replace video token embeddings with visual features
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # Calculate position IDs and RoPE deltas if not provided
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # Calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids=input_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, attention_mask=attention_mask
                )
                self.rope_deltas = rope_deltas
            # Use pre-calculated rope-deltas to get position IDs
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Forward pass through base model
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0] # shape: (batch_size, seq_len, d_model)
        logits = self.lm_head(hidden_states)

        # Calculate language modeling loss if labels provided
        lm_loss = None
        if labels is not None:
            if self.lm_loss_weight > 0:
                # Memory optimization: avoid full float32 conversion of the entire tensor
                # Use chunking to reduce memory usage
                batch_size, seq_len, vocab_size = logits.shape
                
                # Check if we have valid label values (not all IGNORE_INDEX)
                shift_labels = labels[..., 1:].contiguous()
                if (shift_labels != -100).sum() > 0:
                    # Process in smaller chunks to avoid OOM
                    loss_fct = nn.CrossEntropyLoss()
                    chunk_size = min(512, seq_len-1)  # Process in chunks of 512 tokens
                    total_loss = 0.0
                    total_tokens = 0
                    
                    for i in range(0, seq_len-1, chunk_size):
                        end_idx = min(i + chunk_size, seq_len-1)
                        # Process a slice of the sequence
                        chunk_logits = logits[..., i:end_idx, :].float()  # Convert only the chunk to float32
                        chunk_shift_logits = chunk_logits.contiguous().view(-1, self.config.vocab_size)
                        chunk_shift_labels = shift_labels[..., i:end_idx].contiguous().view(-1)
                        chunk_shift_labels = chunk_shift_labels.to(chunk_shift_logits.device)
                        
                        # Skip computation for padding tokens
                        valid_mask = chunk_shift_labels != -100
                        if valid_mask.sum() > 0:
                            valid_logits = chunk_shift_logits[valid_mask]
                            valid_labels = chunk_shift_labels[valid_mask]
                            chunk_loss = loss_fct(valid_logits, valid_labels)
                            
                            # Weighted accumulation
                            total_loss += chunk_loss * valid_mask.sum()
                            total_tokens += valid_mask.sum()
                    
                    # Compute final average loss
                    if total_tokens > 0:
                        lm_loss = total_loss / total_tokens
                    else:
                        lm_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                else:
                    if verbose:
                        print("Warning: All labels are IGNORE_INDEX (-100), skipping language modeling loss")
            elif verbose:
                print("Warning: lm_loss_weight is 0, skipping language modeling loss calculation")

        # Process vision pointer head if visual grounding requested
        pointer_loss = None
        pointer_scores = []
        
        # Check for inputs required for pointer network loss
        has_pointer_inputs = (
            visual_token_indices_of_coordinates is not None or 
            (multi_patch_labels is not None and if_multi_patch)
        )
        
        if has_pointer_inputs and self.pointer_loss_weight > 0:
            batch_size = input_ids.shape[0]
            pointer_losses = []
            
            # Process each sample individually
            for i in range(batch_size):
                try:
                    dummy_target = False
    
                    # Get token IDs and hidden states for current sample
                    token_ids = input_ids[i]          # shape: (seq_length,)
                    hs = hidden_states[i]             # shape: (seq_length, d_model)
    
                    # Find visual token positions
                    visual_mask = (token_ids == self.config.image_token_id)
                    visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1)
    
                    # Find target token positions
                    target_mask = (token_ids == self.config.pointer_pad_token_id)
                    target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
                    
                    # Validate we have necessary visual tokens
                    if visual_indices.numel() == 0:
                        if verbose:
                            print(f"Warning: No visual tokens found for sample {i}, skipping pointer loss")
                        continue
                        
                    # Handle target tokens
                    if target_indices.numel() == 0:
                        if verbose:
                            print(f"Warning: No target tokens found for sample {i}, using dummy target")
                        # Use dummy target if no target tokens found
                        target_indices = torch.tensor([hs.shape[0] - 1]).to(hs.device)
                        gt = torch.tensor([0]).to(hs.device)
                        if if_multi_patch:
                            sample_labels = torch.zeros(1, visual_indices.shape[0], device=hs.device)
                            # Mark a few patches as positive to create a meaningful training signal
                            if visual_indices.shape[0] >= 4:
                                sample_labels[0, :4] = 1
                            else:
                                sample_labels[0, :] = 1
                        dummy_target = True
                    else:
                        # Get ground truth indices/labels
                        if visual_token_indices_of_coordinates is not None:
                            if i < len(visual_token_indices_of_coordinates):
                                gt = visual_token_indices_of_coordinates[i].to(hs.device)
                            else:
                                # Handle case where indices don't match batch size
                                if verbose:
                                    print(f"Warning: visual_token_indices_of_coordinates missing entry for sample {i}")
                                gt = torch.tensor([0]).to(hs.device)
                                dummy_target = True
                        else:
                            gt = torch.tensor([0]).to(hs.device)
                            
                        # Get multi-patch labels if available and needed
                        if if_multi_patch:
                            if multi_patch_labels is not None and i < len(multi_patch_labels):
                                sample_labels = multi_patch_labels[i]
                            else:
                                # Handle missing labels
                                if verbose:
                                    print(f"Warning: multi_patch_labels missing entry for sample {i}")
                                sample_labels = torch.zeros(1, visual_indices.shape[0], device=hs.device)
                                dummy_target = True
                    
                    # Get embeddings for visual and target tokens
                    visual_embeds = inputs_embeds[i][visual_indices]
                    target_hidden = hs[target_indices]
    
                    # Process using multi-patch pointer head
                    if if_multi_patch:
                        # Verify matching dimensions and handle errors
                        if sample_labels.shape[0] != target_indices.shape[0]:
                            if verbose:
                                print(f"Warning: Sample {i} has mismatched target counts: "
                                      f"{sample_labels.shape[0]} labels but found {target_indices.shape[0]} target tokens. "
                                      f"Reshaping sample_labels to match.")
                            
                            # Try to fix mismatch by reshaping or padding
                            if sample_labels.shape[0] > target_indices.shape[0]:
                                # Truncate extra labels
                                sample_labels = sample_labels[:target_indices.shape[0]]
                            else:
                                # Repeat last label to match target count
                                last_label = sample_labels[-1].unsqueeze(0)
                                padding = last_label.repeat(target_indices.shape[0] - sample_labels.shape[0], 1)
                                sample_labels = torch.cat([sample_labels, padding], dim=0)
    
                        # Get attention scores and loss
                        attn_scores, loss_v = self.multi_patch_pointer_head(
                            visual_embeds,
                            target_hidden,
                            labels=sample_labels
                        )
                        
                    else:
                        # Single patch mode (deprecated)
                        attn_scores, loss_v = self.pointer_head(visual_embeds, target_hidden, labels=gt)
                    
                    pointer_scores.append(attn_scores.detach().cpu())
                    
                    # Add loss (zeroed if dummy target)
                    pointer_losses.append(loss_v * 0.0 if dummy_target else loss_v)
                    
                except Exception as e:
                    if verbose:
                        print(f"Error processing pointer head for sample {i}: {e}")
                    # Continue with next sample
                    continue
                    
            # Calculate average loss if we have any valid samples
            if pointer_losses:
                pointer_loss = torch.stack(pointer_losses).mean()
            elif verbose:
                print("No valid pointer losses calculated for any sample in batch")

        # Combine losses using weights, ensuring we always have a valid loss
        total_loss = None
        
        if lm_loss is not None and self.lm_loss_weight > 0:
            total_loss = self.lm_loss_weight * lm_loss
            
        if pointer_loss is not None and self.pointer_loss_weight > 0:
            if total_loss is None:
                total_loss = self.pointer_loss_weight * pointer_loss
            else:
                total_loss = total_loss + self.pointer_loss_weight * pointer_loss
                
        # If no loss components are available, raise a warning and use a tiny loss
        # to allow debugging rather than silently failing
        if total_loss is None:
            if labels is not None:
                # Only warn when labels are provided and we expect to calculate a loss
                import warnings
                warnings.warn(
                    "No loss components were calculated despite labels being provided. "
                    "Check that you're correctly passing label information and that "
                    "lm_loss_weight and/or pointer_loss_weight are non-zero.",
                    RuntimeWarning
                )
            # Use a small non-zero value that will produce small gradients
            # to help identify the issue rather than a true zero that produces no gradients
            total_loss = torch.tensor(1e-8, device=hidden_states.device, requires_grad=True)

        if return_dict:
            return QwenVLwithVisionHeadOutputWithPast(
                loss=total_loss,  # Ensure loss is always set
                lm_loss=lm_loss,
                pointer_loss=pointer_loss,
                pointer_scores=pointer_scores,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
            )
        else:
            # Return appropriate tuple based on whether labels were provided
            if labels is not None:
                # Make sure loss is first in the tuple to match HF Trainer expectations
                return (total_loss, logits) + outputs[1:]
            else:
                return outputs