import torch
import numpy as np
from PIL import Image
from tooncomposer import ToonComposer, get_base_model_paths
import argparse
import json
import os
import tempfile
import cv2
import gradio as gr
from einops import rearrange
from datetime import datetime
from typing import Optional, List, Dict
from huggingface_hub import snapshot_download

os.environ["GRADIO_TEMP_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "gradio_cache"))
gradio_version = gr.__version__
if gradio_version != "5.25.2":
    print(f"[WARNING] ToonComposer app is not tested on gradio=={gradio_version}. Please install gradio==5.25.2 if you encounter any issues.")

# -----------------------------------------------------------------------------
# Weights resolution and download helpers
# -----------------------------------------------------------------------------

WAN_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
TOONCOMPOSER_REPO_ID = "TencentARC/ToonComposer"

def _path_is_dir_with_files(dir_path: str, required_files: List[str]) -> bool:
    if not dir_path or not os.path.isdir(dir_path):
        return False
    for f in required_files:
        if not os.path.exists(os.path.join(dir_path, f)):
            return False
    return True

def resolve_wan_model_root(preferred_dir: Optional[str] = None, hf_token: Optional[str] = None) -> str:
    """Return a directory containing Wan2.1-I2V-14B-480P weights.

    Resolution order:
    1) preferred_dir arg (if valid)
    2) WAN21_I2V_DIR env var (if valid)
    3) HF local cache (no download) via snapshot_download(local_files_only=True)
    4) HF download to cache via snapshot_download()
    """
    # Required filenames relative to the model root
    expected = get_base_model_paths("Wan2.1-I2V-14B-480P", format='dict', model_root=".")
    required_files = []
    required_files.extend([os.path.basename(p) for p in expected["dit"]])
    required_files.append(os.path.basename(expected["image_encoder"]))
    required_files.append(os.path.basename(expected["text_encoder"]))
    required_files.append(os.path.basename(expected["vae"]))

    # 1) preferred_dir arg
    if _path_is_dir_with_files(preferred_dir or "", required_files):
        return os.path.abspath(preferred_dir)

    # 2) environment variable
    env_dir = os.environ.get("WAN21_I2V_DIR")
    if _path_is_dir_with_files(env_dir or "", required_files):
        return os.path.abspath(env_dir)

    # 3) try local cache without network
    try:
        cached_dir = snapshot_download(repo_id=WAN_REPO_ID, local_files_only=True)
        return cached_dir
    except Exception:
        pass

    # 4) download (may be large)
    cached_dir = snapshot_download(repo_id=WAN_REPO_ID, token=hf_token)
    return cached_dir

def resolve_tooncomposer_repo_dir(preferred_dir: Optional[str] = None, hf_token: Optional[str] = None) -> str:
    """Return a directory containing ToonComposer repo with 480p/608p subdirs."""
    # Quick validity check: ensure either a subdir 480p or 608p exists with required files
    def has_resolution_dirs(base_dir: str) -> bool:
        if not base_dir or not os.path.isdir(base_dir):
            return False
        ok = False
        for res in ["480p", "608p"]:
            d = os.path.join(base_dir, res)
            if os.path.isdir(d):
                ckpt = os.path.join(d, "tooncomposer.ckpt")
                cfg = os.path.join(d, "config.json")
                if os.path.exists(ckpt) and os.path.exists(cfg):
                    ok = True
        return ok

    # 1) preferred_dir arg
    if has_resolution_dirs(preferred_dir or ""):
        return os.path.abspath(preferred_dir)

    # 2) environment variable
    env_dir = os.environ.get("TOONCOMPOSER_DIR")
    if has_resolution_dirs(env_dir or ""):
        return os.path.abspath(env_dir)

    # 3) try local cache first
    try:
        cached_dir = snapshot_download(repo_id=TOONCOMPOSER_REPO_ID, local_files_only=True)
        return cached_dir
    except Exception:
        pass

    # 4) download repo to cache
    cached_dir = snapshot_download(repo_id=TOONCOMPOSER_REPO_ID, token=hf_token)
    return cached_dir

def build_checkpoints_by_resolution(tooncomposer_base_dir: str) -> Dict[str, Dict[str, object]]:
    """Construct resolution mapping from a base repo dir that contains 480p/608p.

    The ToonComposer HF repo stores, inside each resolution dir:
      - tooncomposer.ckpt
      - config.json (model configuration)
    """
    mapping = {}
    # Known target sizes
    res_to_hw = {
        "480p": (480, 832),
        "608p": (608, 1088),
    }
    for res, (h, w) in res_to_hw.items():
        res_dir = os.path.join(tooncomposer_base_dir, res)
        mapping[res] = {
            "target_height": h,
            "target_width": w,
            "snapshot_args_path": os.path.join(res_dir, "config.json"),
            "checkpoint_path": os.path.join(res_dir, "tooncomposer.ckpt"),
        }
    return mapping

# Will be populated in main() after resolving ToonComposer repo directory
checkpoints_by_resolution = {}

def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames

def _load_model_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r") as f:
        data = json.load(f)
    return data

def _merge_with_defaults(cfg: Dict[str, object]) -> Dict[str, object]:
    # Provide safe defaults for optional fields used at inference-time
    defaults = {
        "base_model_name": "Wan2.1-I2V-14B-480P",
        "learning_rate": 1e-5,
        "train_architecture": None,
        "lora_rank": 4,
        "lora_alpha": 4,
        "lora_target_modules": "",
        "init_lora_weights": "kaiming",
        "use_gradient_checkpointing": True,
        "tiled": False,
        "tile_size_height": 34,
        "tile_size_width": 34,
        "tile_stride_height": 18,
        "tile_stride_width": 16,
        "output_path": "./",
        "use_dera": False,
        "dera_rank": None,
        "use_dera_spatial": True,
        "use_dera_temporal": True,
        "use_sequence_cond": True,
        "sequence_cond_mode": "sparse",
        "use_channel_cond": False,
        "use_sequence_cond_position_aware_residual": True,
        "use_sequence_cond_loss": False,
        "fast_dev": False,
        "max_num_cond_images": 1,
        "max_num_cond_sketches": 2,
        "random_spaced_cond_frames": False,
        "use_sketch_mask": True,
        "sketch_mask_ratio": 0.2,
        "no_first_sketch": False,
    }
    merged = defaults.copy()
    merged.update(cfg)
    return merged

def initialize_model(resolution="480p", fast_dev=False, device="cuda:0", dtype=torch.bfloat16,
                     wan_model_dir: Optional[str] = None, tooncomposer_dir: Optional[str] = None,
                     hf_token: Optional[str] = None):
    # Initialize model components
    if resolution not in checkpoints_by_resolution:
        raise ValueError(f"Resolution '{resolution}' is not available. Found: {list(checkpoints_by_resolution.keys())}")

    # 1) resolve config and checkpoint from ToonComposer repo (local or HF)
    snapshot_args_path = checkpoints_by_resolution[resolution]["snapshot_args_path"]
    checkpoint_path = checkpoints_by_resolution[resolution]["checkpoint_path"]

    # 2) load model config
    snapshot_args_raw = _load_model_config(snapshot_args_path)
    snapshot_args = _merge_with_defaults(snapshot_args_raw)
    snapshot_args["checkpoint_path"] = checkpoint_path

    # 3) resolve Wan2.1 model root
    snapshot_args["model_root"] = resolve_wan_model_root(preferred_dir=wan_model_dir, hf_token=hf_token)

    # Backward-compat fields
    if "training_max_frame_stride" not in snapshot_args:
        snapshot_args["training_max_frame_stride"] = 4
    snapshot_args["random_spaced_cond_frames"] = False
    args = argparse.Namespace(**snapshot_args)
    if not fast_dev:
        model = ToonComposer(
            base_model_name=args.base_model_name,
            model_root=args.model_root,
            learning_rate=args.learning_rate,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            checkpoint_path=args.checkpoint_path,
            tiled=args.tiled,
            tile_size=(args.tile_size_height, args.tile_size_width),
            tile_stride=(args.tile_stride_height, args.tile_stride_width),
            output_path=args.output_path,
            use_dera=args.use_dera,
            dera_rank=args.dera_rank,
            use_dera_spatial=args.use_dera_spatial,
            use_dera_temporal=args.use_dera_temporal,
            use_sequence_cond=args.use_sequence_cond,
            sequence_cond_mode=args.sequence_cond_mode,
            use_channel_cond=args.use_channel_cond,
            use_sequence_cond_position_aware_residual=args.use_sequence_cond_position_aware_residual,
            use_sequence_cond_loss=args.use_sequence_cond_loss,
            fast_dev=args.fast_dev,
            max_num_cond_images=args.max_num_cond_images,
            max_num_cond_sketches=args.max_num_cond_sketches,
            random_spaced_cond_frames=args.random_spaced_cond_frames,
            use_sketch_mask=args.use_sketch_mask,
            sketch_mask_ratio=args.sketch_mask_ratio,
            no_first_sketch=args.no_first_sketch,
        )
        model = model.to(device, dtype=dtype).eval()
    else:
        print("Fast dev mode. Models will not be loaded.")
        model = None
    print("Models initialized.")
    return model, device, dtype

# -----------------------------------------------------------------------------
# CLI args and global initialization
# -----------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=str, default=os.environ.get("TOONCOMPOSER_RESOLUTION", "480p"), choices=["480p", "608p"], help="Target resolution to load by default.")
    parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "bfloat16"), choices=["bfloat16", "float32"])
    parser.add_argument("--wan_model_dir", type=str, default=os.environ.get("WAN21_I2V_DIR"), help="Local directory containing Wan2.1 model files. If not provided, will try HF cache and download if needed.")
    parser.add_argument("--tooncomposer_dir", type=str, default=os.environ.get("TOONCOMPOSER_DIR"), help="Local directory containing ToonComposer weights with 480p/608p subdirectories. If not provided, will try HF cache and download if needed.")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face token (if needed for gated models).")
    parser.add_argument("--fast_dev", action="store_true", help="Run in fast dev mode without loading heavy models.")
    return parser.parse_args()

_cli_args = _parse_args()

# Resolve ToonComposer repo dir and build resolution mapping
_toon_dir = resolve_tooncomposer_repo_dir(preferred_dir=_cli_args.tooncomposer_dir, hf_token=_cli_args.hf_token)
checkpoints_by_resolution = build_checkpoints_by_resolution(_toon_dir)

_dtype_map = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
fast_dev = bool(_cli_args.fast_dev)
model, device, dtype = initialize_model(
    resolution=_cli_args.resolution,
    fast_dev=fast_dev,
    device=_cli_args.device,
    dtype=_dtype_map[_cli_args.dtype],
    wan_model_dir=_cli_args.wan_model_dir,
    tooncomposer_dir=_cli_args.tooncomposer_dir,
    hf_token=_cli_args.hf_token,
)

def process_conditions(num_items, item_inputs, num_frames, is_sketch=False, target_height=480, target_width=832):
    """Process condition images/sketches into masked video tensor and mask"""
    # Create empty tensors filled with -1
    video = torch.zeros((1, 3, num_frames, target_height, target_width), device=device)
    mask = torch.zeros((1, num_frames), device=device)
    
    for i in range(num_items):
        img, frame_idx = item_inputs[i]
        if img is None or frame_idx is None:
            continue
            
        # Convert PIL image to tensor
        img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 127.5 - 1.0
        if is_sketch:
            img_tensor = -img_tensor
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Resize to model's expected resolution while preserving aspect ratio
        # Get original dimensions
        _, _, h, w = img_tensor.shape
        
        # Resize based on short edge while maintaining aspect ratio
        if h/w < target_height/target_width:
            new_h = target_height
            new_w = int(w * (new_h / h))
        else:  # Width is the short edge
            new_w = target_width
            new_h = int(h * (new_w / w))
            
        # Resize with the calculated dimensions
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode="bilinear")
        
        # Center crop to target resolution if needed
        if new_h > target_height or new_w > target_width:
            # Calculate starting positions for crop
            start_h = max(0, (new_h - target_height) // 2)
            start_w = max(0, (new_w - target_width) // 2)
            # Crop
            img_tensor = img_tensor[:, :, start_h:start_h+target_height, start_w:start_w+target_width]
        
        # Place in video tensor
        frame_idx = min(max(int(frame_idx), 0), num_frames-1)
        if is_sketch:
            video[:, :, frame_idx] = img_tensor[:, :3]  # Handle RGBA sketches
        else:
            video[:, :, frame_idx] = img_tensor
        mask[:, frame_idx] = 1.0
    return video, mask

def process_sketch_masks(num_sketch_masks, sketch_mask_inputs, num_frames, target_height=480, target_width=832):
    """Process sketch masks into a single tensor"""
    # Create empty tensor filled with 1s (1 means no mask, keep original)
    sketch_local_mask = torch.ones((1, 1, num_frames, target_height, target_width), device=device)
    
    for i in range(num_sketch_masks):
        editor_value, frame_idx = sketch_mask_inputs[i]
        if editor_value is None or frame_idx is None:
            continue
            
        # For ImageMask, we need to extract the mask from the editor_value dictionary
        # editor_value is a dict with 'background', 'layers', and 'composite' keys from ImageEditor
        if isinstance(editor_value, dict):
            if "composite" in editor_value and editor_value["composite"] is not None:
                # The 'composite' is the image with mask drawn on it
                # Since we're using ImageMask with fixed black brush, the black areas are the mask
                # Convert the composite to a binary mask (0=masked, 1=not masked)
                # sketch = editor_value["background"]  # This is the sketch
                mask = editor_value["layers"][0] if editor_value["layers"] else None  # This is the mask layer
                if mask is not None:
                    # Convert mask to tensor and normalize
                    mask_array = np.array(mask)
                    mask_array = np.max(mask_array, axis=2)
                    
                    # Convert to tensor, normalize to [0, 1]
                    mask_tensor = torch.from_numpy(mask_array).float()
                    if mask_tensor.max() > 1.0:
                        mask_tensor = mask_tensor / 255.0
                    
                    # Resize to model's expected resolution
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
                    mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(target_height, target_width), mode="nearest")
                    
                    # Invert the mask: black (0) = masked area, white (1) = keep original
                    # We need to invert because in the UI black means "masked"
                    mask_tensor = 1.0 - mask_tensor
                    
                    # Place in sketch_local_mask tensor
                    frame_idx = min(max(int(frame_idx), 0), num_frames-1)
                    sketch_local_mask[:, :, frame_idx] = mask_tensor
                    
    sketch_mask_vis = torch.ones((1, 3, num_frames, target_height, target_width), device=device)
    for t in range(sketch_local_mask.shape[2]):
        for c in range(3):
            sketch_mask_vis[0, c, t, :, :] = torch.where(
                sketch_local_mask[0, 0, t] > 0.5,
                1.0,  # White for unmasked areas
                -1.0  # Black for masked areas
            )
    return sketch_local_mask


def invert_sketch(image):
    """Invert the colors of an image (black to white, white to black)"""
    if image is None:
        return None
    
    # Handle input from ImageMask component (EditorValue dictionary)
    if isinstance(image, dict) and "background" in image:
        # Extract the background image
        bg_image = image["background"]
        
        # Invert the background
        inverted_bg = invert_sketch_internal(bg_image)
        
        # Return updated editor value
        return gr.update(value=inverted_bg)
    
    # Original function for regular images
    return invert_sketch_internal(image)

def invert_sketch_internal(image):
    """Internal function to invert an image"""
    if image is None:
        return None
    
    # Convert to PIL image if needed
    if isinstance(image, str):  # If it's a filepath
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Ensure it's a PIL image now
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.array(image))
        except:
            print(f"Warning: Could not convert image of type {type(image)} to PIL Image")
            return image
    
    # Invert the image
    inverted = Image.fromarray(255 - np.array(image))
    return inverted

def create_blank_mask(canvas_width=832, canvas_height=480):
    """Create a blank white mask image"""
    return Image.new('RGB', (canvas_width, canvas_height), color='white')

def create_mask_with_sketch(sketch, canvas_width=832, canvas_height=480):
    """Create a mask image with sketch as background"""
    if sketch is None:
        return create_blank_mask(canvas_width, canvas_height)
        
    # Convert sketch to PIL if needed
    if not isinstance(sketch, Image.Image):
        sketch = Image.fromarray(np.array(sketch))
    
    # Resize sketch to fit the canvas
    sketch = sketch.resize((canvas_width, canvas_height))
    
    # Create a semi-transparent white layer over the sketch
    overlay = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 128))
    
    # Ensure sketch has alpha channel
    if sketch.mode != 'RGBA':
        sketch = sketch.convert('RGBA')
    
    # Overlay the semi-transparent white layer on the sketch
    result = Image.alpha_composite(sketch, overlay)
    
    # Convert back to RGB for Gradio
    return result.convert('RGB')

def validate_inputs(num_frames, num_cond_images, num_cond_sketches, text_prompt, *args):
    """Validate user inputs and return error messages if any"""
    errors = []
    
    # Check text prompt
    if not text_prompt or text_prompt.strip() == "":
        errors.append("❌ Text prompt is required. Please enter a description for your video.")
    
    # Check condition images
    cond_images_count = 0
    for i in range(int(num_cond_images)):
        img = args[i*2]
        frame_idx = args[i*2+1]
        
        if img is None:
            errors.append(f"❌ Image #{i+1} is missing. Please upload an image or reduce the number of keyframe images.")
        else:
            cond_images_count += 1
            
        if frame_idx is not None and (frame_idx < 0 or frame_idx >= num_frames):
            errors.append(f"❌ Frame index for Image #{i+1} is {frame_idx}, which is out of range. Must be between 0 and {num_frames-1}.")
    
    # Check condition sketches
    num_cond_sketches_index = 8  # Starting index for sketch inputs
    cond_sketches_count = 0
    sketch_frame_indices = []
    
    for i in range(int(num_cond_sketches)):
        sketch_idx = num_cond_sketches_index + i*2
        frame_idx_idx = num_cond_sketches_index + 1 + i*2
        
        if sketch_idx < len(args) and frame_idx_idx < len(args):
            sketch = args[sketch_idx]
            frame_idx = args[frame_idx_idx]
            
            # Check if sketch is provided
            if sketch is None:
                errors.append(f"❌ Sketch #{i+1} is missing. Please upload a sketch or reduce the number of keyframe sketches.")
            else:
                # For ImageMask components, check if background is provided
                if isinstance(sketch, dict):
                    if "background" not in sketch or sketch["background"] is None:
                        errors.append(f"❌ Sketch #{i+1} is missing. Please upload a sketch image.")
                    else:
                        cond_sketches_count += 1
                else:
                    cond_sketches_count += 1
            
            # Check frame index
            if frame_idx is not None and (frame_idx < 0 or frame_idx >= num_frames):
                errors.append(f"❌ Frame index for Sketch #{i+1} is {frame_idx}, which is out of range. Must be between 0 and {num_frames-1}.")
            elif frame_idx is not None:
                sketch_frame_indices.append(frame_idx)
    
    # Check for duplicate frame indices
    image_frame_indices = []
    for i in range(int(num_cond_images)):
        frame_idx = args[i*2+1]
        if frame_idx is not None:
            image_frame_indices.append(frame_idx)
    
    all_frame_indices = image_frame_indices + sketch_frame_indices
    if len(all_frame_indices) != len(set(all_frame_indices)):
        errors.append("❌ Duplicate frame indices detected. Each image and sketch must be placed at a different frame.")
    
    # Check minimum requirements
    if cond_images_count == 0:
        errors.append("❌ At least one input image is required.")
    
    return errors

def tooncomposer_inference(num_frames, num_cond_images, num_cond_sketches, text_prompt, cfg_scale, sequence_cond_residual_scale, resolution, *args):
    # Validate inputs first
    validation_errors = validate_inputs(num_frames, num_cond_images, num_cond_sketches, text_prompt, *args)
    
    if validation_errors:
        error_message = "\n".join(validation_errors)
        return gr.update(value=None), error_message

    try:
        # Parse inputs
        # Get the condition images
        cond_images = []
        for i in range(int(num_cond_images)):
            img = args[i*2]
            frame_idx = args[i*2+1]
            if img is not None and frame_idx is not None:
                cond_images.append((img, frame_idx))
        
        # Get num_cond_sketches
        if num_cond_sketches is None:
            num_cond_sketches = 0
        else:
            num_cond_sketches = int(num_cond_sketches)
        
        # Get condition sketches and masks
        cond_sketches = []
        sketch_masks = []
        num_cond_sketches_index = 8  # Starting index for sketch inputs
        
        for i in range(num_cond_sketches):
            sketch_idx = num_cond_sketches_index + i*2
            frame_idx_idx = num_cond_sketches_index + 1 + i*2
            
            if sketch_idx < len(args) and frame_idx_idx < len(args):
                editor_value = args[sketch_idx]
                frame_idx = args[frame_idx_idx]
                
                if editor_value is not None and frame_idx is not None:
                    # Extract the sketch from the background of the editor value
                    if isinstance(editor_value, dict) and "background" in editor_value:
                        sketch = editor_value["background"]
                        if sketch is not None:
                            cond_sketches.append((sketch, frame_idx))
                            # Also add to sketch_masks for mask processing
                            sketch_masks.append((editor_value, frame_idx))
                    else:
                        # For regular image inputs (first sketch)
                        if editor_value is not None:
                            cond_sketches.append((editor_value, frame_idx))
        
        # Set target resolution based on selection
        target_height, target_width = checkpoints_by_resolution[resolution]["target_height"], checkpoints_by_resolution[resolution]["target_width"]
        
        # Update model resolution
        if not fast_dev:
            model.update_height_width(target_height, target_width)
        
        # Process conditions
        with torch.no_grad():
            # Process image conditions
            masked_cond_video, preserved_cond_mask = process_conditions(
                num_cond_images, cond_images, num_frames, target_height=target_height, target_width=target_width
            )
            
            # Process sketch conditions
            masked_cond_sketch, preserved_sketch_mask = process_conditions(
                len(cond_sketches), cond_sketches, num_frames, is_sketch=True, target_height=target_height, target_width=target_width
            )

            # Process sketch masks (if any)
            sketch_local_mask = None
            if len(sketch_masks) > 0:
                sketch_local_mask = process_sketch_masks(
                    len(sketch_masks), sketch_masks, num_frames, target_height=target_height, target_width=target_width
                )
            else:
                sketch_local_mask = torch.ones((1, 1, num_frames, target_height, target_width), device=device)
             
            if fast_dev:
                print("Fast dev mode, returning dummy video")
                # Create a simple dummy video for testing
                temp_dir = tempfile.mkdtemp()
                video_path = os.path.join(temp_dir, "dummy_video.mp4")
                
                # Create a simple test video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (target_width, target_height))
                
                for i in range(30):  # 30 frames
                    # Create a simple colored frame
                    frame = np.full((target_height, target_width, 3), (i * 8) % 255, dtype=np.uint8)
                    video_writer.write(frame)
                
                video_writer.release()
                return video_path, "✅ Dummy video generated successfully in fast dev mode!"
            
            masked_cond_video = masked_cond_video.to(device=device, dtype=dtype)
            preserved_cond_mask = preserved_cond_mask.to(device=device, dtype=dtype)
            masked_cond_sketch = masked_cond_sketch.to(device=device, dtype=dtype)
            preserved_sketch_mask = preserved_sketch_mask.to(device=device, dtype=dtype)
            
            with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(device).type):
                # Generate video
                model.pipe.device = device
                generated_video = model.pipe(
                    prompt=[text_prompt],
                    negative_prompt=[model.negative_prompt],
                    input_image=None,
                    num_inference_steps=15,
                    num_frames=num_frames,
                    seed=42, tiled=True,
                    input_condition_video=masked_cond_video,
                    input_condition_preserved_mask=preserved_cond_mask,
                    input_condition_video_sketch=masked_cond_sketch,
                    input_condition_preserved_mask_sketch=preserved_sketch_mask,
                    sketch_local_mask=sketch_local_mask,
                    cfg_scale=cfg_scale,
                    sequence_cond_residual_scale=sequence_cond_residual_scale,
                    height=target_height,
                    width=target_width,
                )

            # Convert to PIL images
            video_frames = model.pipe.tensor2video(generated_video[0].cpu())
            
            # Convert PIL images to an MP4 video
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, "generated_video.mp4")
            
            width, height = video_frames[0].size
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))  # 20 fps
            
            for frame in video_frames:
                # Convert PIL image to OpenCV BGR format
                frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            video_writer.release()
            print(f"Generated video saved to {video_path}. Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return video_path, f"✅ Video generated successfully! (with {len(cond_images)} keyframe images, {len(cond_sketches)} keyframe sketches)"
    
    except Exception as e:
        error_msg = f"❌ Error during generation: {str(e)}"
        print(error_msg)
        return gr.update(value=None), error_msg

def create_sample_gallery():
    """Create gallery items for samples"""
    import os
    
    gallery_items = []
    sample_info = [
        {
            "id": 1,
            "title": "Sample 1",
            "description": "Man playing with blue fish underwater (3 sketches)",
            "preview": "samples/1_image1.png"
        },
        {
            "id": 2, 
            "title": "Sample 2",
            "description": "Girl and boy planting a growing flower (2 sketches)",
            "preview": "samples/2_image1.jpg"
        },
        {
            "id": 3,
            "title": "Sample 3", 
            "description": "Ancient Chinese boy giving apple to elder (1 sketch)",
            "preview": "samples/3_image1.png"
        }
    ]
    
    for sample in sample_info:
        if os.path.exists(sample["preview"]):
            gallery_items.append((sample["preview"], f"{sample['title']}: {sample['description']}"))
    
    return gallery_items

def handle_gallery_select(evt: gr.SelectData):
    """Handle gallery selection and load the corresponding sample"""
    sample_id = evt.index + 1  # Gallery index starts from 0, sample IDs start from 1
    return apply_sample_to_ui(sample_id)

def load_sample_data(sample_id):
    """Load sample data based on the selected sample"""
    import os
    
    samples_dir = "samples"
    
    # Sample configurations
    sample_configs = {
        1: {
            "prompt": "Underwater scene: A shirtless man plays with a spiraling blue fish. A whale follows a bag in the man's hand, swimming in circles as the man uses the bag to lure the blue fish forward. Anime. High quality.",
            "num_sketches": 3,
            "image_frame": 0,
            "sketch_frames": [20, 40, 60],
            "num_frames": 61
        },
        2: {
            "prompt": "A girl and a silver-haired boy plant a huge flower. As the camera slowly moves up, the huge flower continues to grow and bloom. Anime. High quality.",
            "num_sketches": 2,
            "image_frame": 0,
            "sketch_frames": [30, 60],
            "num_frames": 61
        },
        3: {
            "prompt": "An ancient Chinese boy holds an apple and smiles as he gives it to an elderly man nearby. Anime. High quality.",
            "num_sketches": 1,
            "image_frame": 0,
            "sketch_frames": [30],
            "num_frames": 33
        }
    }
    
    if sample_id not in sample_configs:
        return None
    
    config = sample_configs[sample_id]
    
    # Load image
    image_path = os.path.join(samples_dir, f"{sample_id}_image1.png")
    if not os.path.exists(image_path):
        image_path = os.path.join(samples_dir, f"{sample_id}_image1.jpg")
    
    # Load sketches
    sketches = []
    for i in range(config["num_sketches"]):
        sketch_path = os.path.join(samples_dir, f"{sample_id}_sketch{i+1}.jpg")
        if os.path.exists(sketch_path):
            sketches.append(sketch_path)
    
    # Load output video
    output_path = os.path.join(samples_dir, f"{sample_id}_out.mp4")
    
    return {
        "prompt": config["prompt"],
        "image": image_path if os.path.exists(image_path) else None,
        "sketches": sketches,
        "image_frame": config["image_frame"],
        "sketch_frames": config["sketch_frames"][:len(sketches)],
        "output_video": output_path if os.path.exists(output_path) else None,
        "num_sketches": len(sketches),
        "num_frames": config["num_frames"]
    }

def apply_sample_to_ui(sample_id):
    """Apply sample data to UI components"""
    sample_data = load_sample_data(sample_id)
    
    if not sample_data:
        return [gr.update() for _ in range(20)]  # Return no updates if sample not found
    
    updates = [gr.update(value=sample_data["num_frames"])]
    
    # Update prompt
    updates.append(gr.update(value=sample_data["prompt"]))
    
    # Update number of sketches
    updates.append(gr.update(value=sample_data["num_sketches"]))
    
    # Update condition image
    updates.append(gr.update(value=sample_data["image"]))
    updates.append(gr.update(value=sample_data["image_frame"]))
    
    # Update sketches (up to 4)
    for i in range(4):
        if i < len(sample_data["sketches"]):
            # Load sketch image
            sketch_img = Image.open(sample_data["sketches"][i])
            # Create ImageMask format
            sketch_dict = {
                "background": sketch_img,
                "layers": [],
                "composite": sketch_img
            }
            updates.append(gr.update(value=sketch_dict))
            updates.append(gr.update(value=sample_data["sketch_frames"][i]))
        else:
            updates.append(gr.update(value=None))
            updates.append(gr.update(value=30))
    
    # Update output video
    updates.append(gr.update(value=sample_data["output_video"]))
    
    # Update status
    updates.append(gr.update(value=f"✅ Loaded Sample {sample_id}: {sample_data['prompt'][:50]}..."))
    
    return updates

if __name__ == "__main__":
    from util.stylesheets import css, pre_js, banner_image
    with gr.Blocks(title="🎨 ToonComposer Demo", css=css, js=pre_js) as iface:
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(banner_image)
            with gr.Column(scale=1):
                gr.Markdown("""
                💡 **Quick Guide**
                1. Set the promopt and number of target frames, input keyframe images/sketches, etc.
                2. Upload keyframe image as the first frame (with index set to 0).
                3. Upload sketches with optional motion masks for controlled generation at specified frame indices.
                4. Click the *Generate* button to create your cartoon video.
                """)
        
        max_num_frames = 61
        cond_images_inputs = []
        cond_sketches_inputs = []
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Video Settings", open=True):
                    num_frames = gr.Slider(
                        minimum=17, maximum=max_num_frames, value=max_num_frames, step=1, label="🎥 Number of Frames",
                        info="Select the total number of frames for the generated video. Should be 4N+"
                    )
                    
                    resolution = gr.Radio(
                        choices=["480p", "608p"],
                        value="480p",
                        label="🎥 Resolution",
                        info="Select the resolution for the generated video."
                    )
                    
                    text_prompt = gr.Textbox(
                        label="📝 Text Prompt",
                        placeholder="Enter a description for the video.",
                        info="Describe what you want to generate in the video.",
                        lines=5
                    )
                    cfg_scale = gr.Slider(
                        minimum=1.0, maximum=15.0, value=7.5, label="⚙️ CFG Scale",
                        info="Adjust the classifier-free guidance scale for generation."
                    )
                    sequence_cond_residual_scale = gr.Slider(
                        minimum=0.0, maximum=1.2, value=1.0, label="⚙️ Pos-aware Residual Scale",
                        info="Adjust the residual scale for the position-aware sequence condition."
                    )
        
            with gr.Column(scale=3):
                with gr.Accordion("Keyframe Image(s)", open=True):
                    num_cond_images = gr.Slider(
                        minimum=1, maximum=4, value=1, step=1, label="🖼️ Number of Keyframe Images",
                        info="Specify how many keyframe color images to use (max 4 images)."
                    )
                    for i in range(4):  # Max 4 condition images
                        with gr.Tab(label=f"Image {i+1}", interactive=i==0) as tab:
                            gr.Markdown("At least one image is required. \n Each image or sketch will be used to control the cartoon geneartion at the given frame index.")
                            image_input = gr.Image(
                                label=f"Image {i+1}", type="pil",
                                placeholder=f"Upload a keyframe image {i+1}..."
                            )
                            frame_index_input = gr.Slider(
                                label=f"Frame Index for Image #{i+1}", minimum=0, maximum=max_num_frames - 1,
                                value=i * (max_num_frames-1) // 3, step=1, 
                                info=f"Frame position for Image {i+1} (0 to {max_num_frames-1})"
                            )
                            cond_images_inputs.append((image_input, frame_index_input, tab))
                            
        
            with gr.Column(scale=3):
                with gr.Accordion("Keyframe Sketch(es)", open=True): 
                    num_cond_sketches = gr.Slider(
                        minimum=1, maximum=4, value=1, step=1, label="✏️ Number of Keyframe Sketch(es)",
                        info="Specify how many keyframe sketches to use (max 4 sketches)."
                    )
                    for i in range(4):  # Max 4 condition sketches
                        with gr.Tab(label=f"Sketch {i + 1}", interactive=i==0) as tab:
                            
                            gr.Markdown("At least one sketch is required. \n You can optionally draw black areas using the brush tool to mark regions where motion can be generated freely.")
                            
                            # Use ImageMask which allows uploading an image and drawing a mask
                            sketch_input = gr.ImageMask(
                                label=f"Sketch {i + 1} with Motion Mask",
                                type="pil",
                                elem_id=f"sketch_mask_{i + 1}"
                            )
                            
                            # All sketches have a frame index input
                            _frame_index_input = gr.Slider(
                                label=f"Frame Index for Sketch #{i + 1}", minimum=0, maximum=max_num_frames - 1,
                                value=max_num_frames-1, step=1,
                                info=f"Frame position for Sketch {i + 1} (0 to {max_num_frames-1})"
                            )
                            
                            cond_sketches_inputs.append((sketch_input, _frame_index_input, tab))
            
        with gr.Row():
            with gr.Column(scale=1):
                # Sample Gallery Section
                with gr.Accordion("🔍 Sample Gallery", open=True):
                    gr.Markdown("Click on any sample image below to load the sample inputs.")
                    sample_gallery = gr.Gallery(
                        value=create_sample_gallery(),
                        label="Sample Examples",
                        show_label=False,
                        elem_id="sample-gallery",
                        columns=3,
                        rows=1,
                        height=200,
                        allow_preview=True,
                        object_fit="contain")
                    
                with gr.Accordion("🛠️ Tools", open=False):
                    tool_input = gr.Image(
                        label=f"Input Image", type="pil",
                        placeholder=f"Upload an image."
                    )
                    invert_btn = gr.Button(f"Invert Colors")
                    invert_btn.click(
                        fn=invert_sketch,
                        inputs=[tool_input],
                        outputs=[tool_input]
                    )
                    
            with gr.Column(scale=1):
                status_text = gr.Textbox(
                    label="📊 Status",
                    value="Ready to generate. Please check your inputs and click Run.",
                    interactive=False,
                    lines=5
                )
                
                with gr.Accordion("🎬 Generated Video", open=True):
                    output_video = gr.Video(
                        label="Video Output",
                        show_label=True
                    )
                    run_button = gr.Button("🚀 Generate Video", variant="primary", size="lg")

        def update_visibility(num_items, num_frames):
            # Update visibility for columns
            updates_images = []
            updates_indices = []
            for i in range(4):
                is_visible = i < num_items
                # is_visible = True
                updates_images.append(gr.update(interactive=is_visible))
                updates_indices.append(gr.update(
                    value=((num_frames - 1) // max(num_items, 1)) * (i + 1),
                    minimum=0, maximum=num_frames-1,
                ))
            return updates_images + updates_indices
        
        def update_visibility_images(num_items, num_frames):
            # Update visibility for columns
            updates_images = []
            updates_indices = []
            for i in range(4):
                is_visible = i < num_items
                updates_images.append(gr.update(interactive=is_visible))
                updates_indices.append(gr.update(
                    value=((num_frames - 1) // max(num_items, 1)) * i,
                    minimum=0, maximum=num_frames-1,
                ))
            return updates_images + updates_indices
        
        def update_frame_ranges(num_items_images, num_items_sketches, num_frames):
            """Update the maximum values for all frame index sliders"""
            updates = []
            for i in range(4):  # Images
                updates.append(gr.update(
                    value=((num_frames - 1) // max(num_items_images, 1)) * i,
                    maximum=num_frames-1
                    ))
            for i in range(4):  # Sketches
                updates.append(gr.update(
                    value=((num_frames - 1) // max(num_items_sketches, 1)) * (i + 1),
                    maximum=num_frames-1))
            return updates
        
        num_cond_images.change(
            fn=update_visibility_images,
            inputs=[num_cond_images, num_frames],
            outputs=[tab for _, _, tab in cond_images_inputs] \
                + [frame_index_input for _, frame_index_input, _ in cond_images_inputs],
        )

        num_cond_sketches.change(
            fn=update_visibility,
            inputs=[num_cond_sketches, num_frames],
            outputs=[tab for _, _, tab in cond_sketches_inputs] \
                + [frame_index_input for _, frame_index_input, _ in cond_sketches_inputs],
        )

        num_frames.change(
            fn=update_frame_ranges,
            inputs=[num_cond_images, num_cond_sketches, num_frames],
            outputs=[frame_index_input for _, frame_index_input, _ in cond_images_inputs] + \
                    [frame_index_input for _, frame_index_input, _ in cond_sketches_inputs]
        )

        def update_resolution(resolution):
            model.update_height_width(checkpoints_by_resolution[resolution]["target_height"], checkpoints_by_resolution[resolution]["target_width"])
            model.load_tooncomposer_checkpoint(checkpoints_by_resolution[resolution]["checkpoint_path"])
            return gr.update(), gr.update()

        resolution.change(
            fn=update_resolution,
            inputs=[resolution],
            outputs=[output_video, run_button]
        )
        
        sample_outputs = [
            num_frames, text_prompt, num_cond_sketches,
            cond_images_inputs[0][0], cond_images_inputs[0][1],  # Image 1
            cond_sketches_inputs[0][0], cond_sketches_inputs[0][1],  # Sketch 1
            cond_sketches_inputs[1][0], cond_sketches_inputs[1][1],  # Sketch 2
            cond_sketches_inputs[2][0], cond_sketches_inputs[2][1],  # Sketch 3
            cond_sketches_inputs[3][0], cond_sketches_inputs[3][1],  # Sketch 4
            output_video, status_text
        ]
        
        sample_gallery.select(
            fn=handle_gallery_select,
            outputs=sample_outputs
        )

        inputs = [num_frames, num_cond_images, num_cond_sketches, text_prompt, cfg_scale, sequence_cond_residual_scale, resolution]
        run_button.click(
            fn=tooncomposer_inference,
            inputs=inputs,
            outputs=[output_video, status_text]
        )
        
        # Add condition image inputs
        for image_input, frame_index_input, _ in cond_images_inputs:
            inputs.append(image_input)
            inputs.append(frame_index_input)
            
        # Add sketch inputs (both regular and ImageMask)
        for sketch_input, frame_index_input, _ in cond_sketches_inputs:
            inputs.append(sketch_input)
            inputs.append(frame_index_input)
            
        iface.launch(server_port=7860, server_name="0.0.0.0",
                     allowed_paths=[os.path.abspath(os.path.join(os.path.dirname(__file__), "gradio_cache")), 
                                   os.path.abspath(os.path.join(os.path.dirname(__file__), "samples"))])
