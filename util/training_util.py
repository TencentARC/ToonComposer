from typing import Union
import torch
import random
import numpy as np
import cv2
import os


def create_random_mask(batch_size, num_frames, height, width, device, dtype, shape_type=None):
    """
    Create random masks for sketch frames.
    
    Args:
        batch_size: Batch size
        num_frames: Number of frames to mask
        height, width: Image dimensions
        device: Device for tensor
        dtype: Data type for tensor
        mask_area_ratio: Ratio of area to mask (0-1)
        shape_type: Type of shape for masking ('square', 'circle', 'random'). If None, one is randomly selected.
    
    Returns:
        Mask tensor in [b, 1, num_frames, height, width] where 0 indicates areas to mask (inverse of previous implementation)
    """
    # Initialize with ones (unmasked)
    masks = torch.ones(batch_size, 1, num_frames, height, width, device=device, dtype=dtype)
    
    for b in range(batch_size):
        for f in range(num_frames):
            # Randomly select shape type if not specified
            if shape_type is None:
                shape_type = random.choice(['square', 'circle', 'random'])
            
            # Create numpy mask for easier shape drawing
            mask = np.zeros((height, width), dtype=np.float32)
            
            if shape_type == 'square':
                # Random squares
                num_squares = random.randint(1, 5)
                for _ in range(num_squares):
                    # Random square size (proportional to image dimensions)
                    max_size = min(height, width)
                    size = random.randint(max_size // 4, max_size)
                    
                    # Random position
                    x = random.randint(0, width - size)
                    y = random.randint(0, height - size)
                    
                    # Draw square
                    mask[y:y+size, x:x+size] = 1.0
                    
            elif shape_type == 'circle':
                # Random circles
                num_circles = random.randint(1, 5)
                for _ in range(num_circles):
                    # Random radius (proportional to image dimensions)
                    max_radius = min(height, width) // 2
                    radius = random.randint(max_radius // 4, max_radius)
                    
                    # Random center
                    center_x = random.randint(radius, width - radius)
                    center_y = random.randint(radius, height - radius)
                    
                    # Draw circle
                    cv2.circle(mask, (center_x, center_y), radius, 1.0, -1)
                    
            elif shape_type == 'random':
                # Create connected random shape with cv2
                num_points = random.randint(5, 16)
                points = []
                
                # Generate random points
                for _ in range(num_points):
                    x = random.randint(0, width - 1)
                    y = random.randint(0, height - 1)
                    points.append([x, y])
                
                # Convert to numpy array for cv2
                points = np.array(points, dtype=np.int32)
                
                # Draw filled polygon
                cv2.fillPoly(mask, [points], 1.0)
            
            # Convert numpy mask to tensor and subtract from ones (inverse the mask)
            masks[b, 0, f] = 1.0 - torch.from_numpy(mask).to(device=device, dtype=dtype)
    
    return masks


@torch.no_grad()
def extract_img_to_sketch(_sketch_model, _img, model_name="random"):
    """
    Return sketch: [-1, 1]
    """
    orig_shape = (_img.shape[-2], _img.shape[-1])
    with torch.amp.autocast(dtype=torch.float32, device_type="cuda"):
        reshaped_img = torch.nn.functional.interpolate(_img, (2048, 2048))
        sketch = _sketch_model(reshaped_img, model_name=model_name)
        sketch = torch.nn.functional.interpolate(sketch, orig_shape)
    if sketch.shape[1] == 1:
        sketch = sketch.repeat(1, 3, 1, 1)
    return sketch


def video_to_frame_and_sketch(
    sketch_model,
    original_video,
    max_num_preserved_sketch_frames=2,
    max_num_preserved_image_frames=1,
    min_num_preserved_sketch_frames=2,
    min_num_preserved_image_frames=1,
    model_name=None,
    detach_image_and_sketch=False,
    equally_spaced_preserve_sketch=False,
    apply_sketch_mask=False,
    sketch_mask_ratio=0.2,
    sketch_mask_shape=None,
    no_first_sketch: Union[bool, float] = False,
    video_clip_names=None,
    is_flux_sketch_available=None,
    is_evaluation=False,
):
    """
    Args:
        sketch_model: torch.nn.Module, a sketch pool for extracting sketches from images
        original_video: torch.Tensor, shape=(batch_size, num_channels, num_frames, height, width)
        max_num_preserved_sketch_frames: int, maximum number of preserved sketch frames
        max_num_preserved_image_frames: int, maximum number of preserved image frames
        min_num_preserved_sketch_frames: int, minimum number of preserved sketch frames
        min_num_preserved_image_frames: int, minimum number of preserved image frames
        model_name: str, name of the sketch model. If None, randomly select from ["lineart", "lineart_anime", "anime2sketch"]. Default: None.
        equally_spaced_preserve_sketch: bool, whether to preserve sketches at equally spaced intervals. Default: False.
        apply_sketch_mask: bool, whether to apply random masking to sketch frames. Default: False.
        sketch_mask_ratio: float, ratio of frames to mask (0-1). Default: 0.2.
        sketch_mask_shape: str, shape type for masking ('square', 'circle', 'random'). If None, randomly selected. Default: None.
    Returns:
        conditional_image: torch.Tensor, shape=(batch_size, num_frames, num_channels, height, width)
        preserving_image_mask: torch.Tensor, shape=(batch_size, num_frames, 1, height, width)
        full_sketch_frames: torch.Tensor, shape=(batch_size, num_frames, num_channels, height, width)
        sketch_local_mask: torch.Tensor, shape=(batch_size, 1, num_frames, height, width) or None if apply_sketch_mask=False
    """
    video_shape = original_video.shape
    video_dtype = original_video.dtype
    video_device = original_video.device

    if min_num_preserved_sketch_frames is None or min_num_preserved_sketch_frames < 2:
        min_num_preserved_sketch_frames = 2  # Minimum num: 2 (the first and the last)
    num_preserved_sketch_frames = random.randint(min_num_preserved_sketch_frames, max_num_preserved_sketch_frames)
    num_preserved_sketch_frames = min(num_preserved_sketch_frames, video_shape[2])
    
    # Always include first and last frames
    if video_clip_names is not None and is_flux_sketch_available is not None:
        if is_flux_sketch_available[0]:
            num_preserved_sketch_frames = 2
    
    if isinstance(no_first_sketch, float):
        no_first_sketch = random.random() < no_first_sketch
    
    if equally_spaced_preserve_sketch:
        preserved_sketch_indices = torch.linspace(0, video_shape[2] - 1, num_preserved_sketch_frames).long().tolist()
        if no_first_sketch:
            preserved_sketch_indices = preserved_sketch_indices[1:]
    else:
        if no_first_sketch:
            preserved_sketch_indices = [video_shape[2] - 1] 
        else:   
            preserved_sketch_indices = [0, video_shape[2] - 1] 
        # If we need more frames than just first and last
        if num_preserved_sketch_frames > 2 and video_shape[2] > 4:
            # Create set of all valid candidates (excluding first, last and their adjacent frames)
            # Exclude indices adjacent to first and last
            candidates = set(range(2, video_shape[2] - 2))
            
            # Determine how many additional frames to select
            additional_frames_needed = min(num_preserved_sketch_frames - 2, len(candidates))
            
            # Keep selecting frames until we have enough or run out of candidates
            additional_indices = []
            while len(additional_indices) < additional_frames_needed and candidates:
                # Convert set to list for random selection
                candidate_list = list(candidates)
                # Select a random candidate
                idx = random.choice(candidate_list)
                additional_indices.append(idx)
                
                # Remove selected index and adjacent indices from candidates
                candidates.remove(idx)
                if idx - 1 in candidates:
                    candidates.remove(idx - 1)
                if idx + 1 in candidates:
                    candidates.remove(idx + 1)
            
            preserved_sketch_indices.extend(additional_indices)
            preserved_sketch_indices.sort()
            
    # Indices to preserve has been determined. 
    # Later code will not care the number of preserved frames but rely on the indices only.
    preserved_image_indices = [0]
    if max_num_preserved_image_frames is not None and max_num_preserved_image_frames > 1:
        max_num_preserved_image_frames -= 1
        if min_num_preserved_image_frames is None or min_num_preserved_image_frames < 1:
            min_num_preserved_image_frames = 1
        min_num_preserved_image_frames -= 1
        other_indices = torch.tensor([i for i in range(video_shape[2]) if i not in preserved_sketch_indices])
        max_num_preserved_image_frames = min(max_num_preserved_image_frames, len(other_indices))
        min_num_preserved_image_frames = min(min_num_preserved_image_frames, max_num_preserved_image_frames)
        num_preserved_image_frames = random.randint(min_num_preserved_image_frames, max_num_preserved_image_frames)
        other_indices = other_indices[torch.randperm(len(other_indices))]
        if num_preserved_image_frames > 0:
            preserved_image_indices.extend(other_indices[:num_preserved_image_frames])
    
    preserved_condition_mask = torch.zeros(size=(video_shape[0], video_shape[2]), dtype=video_dtype, device=video_device)  # [b, t]
    masked_condition_video = torch.zeros_like(original_video)   # [b, c, t, h, w]
    full_sketch_frames = torch.zeros_like(original_video)  # [b, c, t, h, w]
    
    if detach_image_and_sketch:
        preserved_condition_mask_sketch = torch.zeros_like(preserved_condition_mask)
        masked_condition_video_sketch = torch.zeros_like(masked_condition_video)
        if 0 not in preserved_sketch_indices and not no_first_sketch:
            preserved_sketch_indices.append(0)
    else:
        preserved_condition_mask_sketch = None
        masked_condition_video_sketch = None

    for _idx in preserved_image_indices:
        preserved_condition_mask[:, _idx] = 1.0
        masked_condition_video[:, :, _idx, :, :] = original_video[:, :, _idx, :, :]
    
    # Set up sketch_local_mask if masking is applied
    sketch_local_mask = None
        
    if apply_sketch_mask:
        # Create a full-sized mask initialized to all ones (unmasked)
        sketch_local_mask = torch.ones(
            video_shape[0], video_shape[2], video_shape[3], video_shape[4],
            device=video_device,
            dtype=video_dtype
        ).unsqueeze(1)  # Add channel dimension to get [b, 1, t, h, w]
        
        if not is_evaluation and random.random() < sketch_mask_ratio:
            # For preserved frames, apply random masking
            for i, frame_idx in enumerate(preserved_sketch_indices):
                if i == 0:
                    # First frame is not masked
                    continue
                # Create masks only for preserved frames
                frame_masks = create_random_mask(
                    batch_size=video_shape[0],
                    num_frames=1,  # Just one frame at a time
                    height=video_shape[3],
                    width=video_shape[4],
                    device=video_device,
                    dtype=video_dtype,
                    # mask_area_ratio=0.4 * random.random() + 0.1,
                    shape_type=sketch_mask_shape
                )
                
                # Set the mask for this preserved frame
                sketch_local_mask[:, :, frame_idx:frame_idx+1, :, :] = frame_masks
    
    # Produce sketches for preserved frames
    # Sketches can either be 1) calculated from sketch pool or 2) loaded from the flux sketch directory
    if is_flux_sketch_available is not None and is_flux_sketch_available[0]:
        should_use_flux_sketch = random.random() < 0.75 if not is_evaluation else True
    else:
        should_use_flux_sketch = False
        
    cur_model_name = "flux" if should_use_flux_sketch else random.choice(["lineart", "lineart_anime", "anime2sketch"]) if model_name is None else model_name # "anime2sketch"
    # cur_model_name = "anyline"
    for _idx in preserved_sketch_indices:
        sketch_frame = None
        if should_use_flux_sketch:
            # Load flux sketch
            sketech_path = f"/group/40005/gzhiwang/iclora/linearts/{video_clip_names[0]}/{_idx}.lineart.png"
            print(f"Loading flux sketch from {sketech_path}...")
            if os.path.exists(sketech_path):
                sketch_frame = cv2.imread(sketech_path)
                sketch_frame = cv2.cvtColor(sketch_frame, cv2.COLOR_BGR2RGB)
                # resize to 480p
                sketch_frame = cv2.resize(sketch_frame, (video_shape[4], video_shape[3]))
                sketch_frame = torch.from_numpy(sketch_frame).to(video_device, dtype=video_dtype)
                # Normalize to [-1, 1]
                sketch_frame = sketch_frame / 255.0 * 2.0 - 1.0
                sketch_frame = sketch_frame.permute(2, 0, 1)
                sketch_frame = sketch_frame.unsqueeze(0)
            else:
                print(f"FLUX Sketch path {sketech_path} does not exist. Falling back to sketch pool.")
            #     raise ValueError(f"FLUX Sketch path {sketech_path} does not exist.")
        if sketch_frame is None:
            # Calculate sketch from sketch pool
            sketch_frame = extract_img_to_sketch(
                    sketch_model, original_video[:, :, _idx, :, :].float(),
                    model_name=cur_model_name).to(video_device, dtype=video_dtype)
        # Convert white BG (from sketch pool or loaded from flux sketch files) to black BG (for training)
        sketch_frame = -torch.clip(sketch_frame, -1, 1)
        full_sketch_frames[:, :, _idx, :, :] = sketch_frame

    if len(preserved_sketch_indices) > 0:
        _mask_to_add = preserved_condition_mask_sketch if detach_image_and_sketch else preserved_condition_mask
        _video_to_add = masked_condition_video_sketch if detach_image_and_sketch else masked_condition_video
        if not detach_image_and_sketch:
            preserved_sketch_indices = preserved_sketch_indices[1:]
        
        # Apply masking to sketch frames if required
        if apply_sketch_mask and sketch_local_mask is not None:
            # sketch_local_mask: [b, 1, t, h, w]
            for _idx in preserved_sketch_indices:
                _mask_to_add[:, _idx] = 1.0 if detach_image_and_sketch else -1.0
                _video_to_add[:, :, _idx, :, :] = torch.where(sketch_local_mask[:, 0:1, _idx, :, :] == 0, -1.0, full_sketch_frames[:, :, _idx, :, :])
        else:
            for _idx in preserved_sketch_indices:
                _mask_to_add[:, _idx] = 1.0 if detach_image_and_sketch else -1.0
                _video_to_add[:, :, _idx, :, :] = full_sketch_frames[:, :, _idx, :, :]
                     
    return masked_condition_video, preserved_condition_mask, masked_condition_video_sketch, preserved_condition_mask_sketch, full_sketch_frames, sketch_local_mask, cur_model_name


