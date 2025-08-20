import os, torch, lightning, imageio
import numpy as np

from pipeline.i2v_pipeline import WanVideoPipeline


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('medium')


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in frames:
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()
    

def get_base_model_paths(base_model_name, format='dict', model_root="./weights"):
        if base_model_name == "Wan2.1-I2V-14B-480P":
            if format == 'list':
                return [
                    [os.path.join(model_root, f"diffusion_pytorch_model-0000{_idx}-of-00007.safetensors") for _idx in range(1, 8)],
                    os.path.join(model_root, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                    os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.pth"),
                    os.path.join(model_root, "Wan2.1_VAE.pth")
                ]
            elif format == 'dict':
                return {
                    "dit": [os.path.join(model_root, f"diffusion_pytorch_model-0000{_idx}-of-00007.safetensors") for _idx in range(1, 8)],
                    "image_encoder": os.path.join(model_root, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                    "text_encoder": os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.pth"),
                    "vae": os.path.join(model_root, "Wan2.1_VAE.pth")
                }
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            raise ValueError(f"Unsupported base model name: {base_model_name}")


class ToonComposer(lightning.LightningModule):
    def __init__(self, base_model_name="Wan2.1-I2V-14B-480P", model_root=None, learning_rate=1e-5, lora_rank=4, lora_alpha=4, 
                 use_gradient_checkpointing=True, 
                 checkpoint_path=None, video_condition_preservation_mode="first_and_last", 
                 tiled=False, tile_size=(34, 34), tile_stride=(18, 16), output_path=None,
                 use_dera=False, dera_rank=None, use_dera_spatial=True, use_dera_temporal=True,
                 use_sequence_cond=False, sequence_cond_mode="sparse", use_channel_cond=False,
                 use_sequence_cond_position_aware_residual=False,
                 use_sequence_cond_loss=False, fast_dev=False,
                 max_num_cond_images=1, max_num_cond_sketches=2, 
                 random_spaced_cond_frames=False, use_sketch_mask=False, sketch_mask_ratio=0.2, no_first_sketch=False,
                 test_sampling_steps=15, test_sequence_cond_residual_scale=0.5, height=480, width=832, **kwargs):
        super().__init__()
        
        self.pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.bfloat16)
        self.use_dera = use_dera
        self.use_dera_spatial = use_dera_spatial
        self.use_dera_temporal = use_dera_temporal
        self.use_sequence_cond = use_sequence_cond
        self.sequence_cond_mode = sequence_cond_mode
        self.use_channel_cond = use_channel_cond
        self.use_sequence_cond_position_aware_residual = use_sequence_cond_position_aware_residual
        assert not (use_sequence_cond and use_channel_cond), "Cannot use both sequence condition and channel condition."
        self.use_sequence_cond_loss = use_sequence_cond_loss
        
        self.max_num_cond_images = max_num_cond_images
        self.max_num_cond_sketches = max_num_cond_sketches
        
        self.random_spaced_cond_frames = random_spaced_cond_frames
        self.use_sketch_mask = use_sketch_mask
        self.sketch_mask_ratio = sketch_mask_ratio
        self.no_first_sketch = no_first_sketch
        self.test_sampling_steps = test_sampling_steps
        self.test_sequence_cond_residual_scale = test_sequence_cond_residual_scale
        
        self.height = height
        self.width = width
        
        self.current_checkpoint_path = None
        
        paths = get_base_model_paths(base_model_name, format='dict', model_root=model_root)
        if use_sequence_cond:
            assert sequence_cond_mode in ["sparse", "full"], f"Unsupported sequence condition model: {sequence_cond_mode}"
            if sequence_cond_mode == "sparse":
                if use_sketch_mask:
                    sequence_cond_in_dim = 24
                else:
                    sequence_cond_in_dim = 20
            else:
                sequence_cond_in_dim = 20
            use_channel_cond = False
            channel_cond_in_dim = None
        elif use_channel_cond:
            channel_cond_in_dim = 20
            sequence_cond_in_dim = None
            use_sequence_cond = False
        
        dit_config = {
            "use_dera": use_dera,
            "dera_rank": dera_rank,
            "use_dera_spatial": use_dera_spatial,
            "use_dera_temporal": use_dera_temporal,
            "use_sequence_cond": use_sequence_cond,
            "sequence_cond_mode": sequence_cond_mode,
            "sequence_cond_in_dim": sequence_cond_in_dim,
            "use_channel_cond": use_channel_cond,
            "channel_cond_in_dim": channel_cond_in_dim,
            "use_sequence_cond_position_aware_residual": use_sequence_cond_position_aware_residual,
            "use_sequence_cond_loss": use_sequence_cond_loss
        }
        if fast_dev:
            del paths["dit"]
            dit_config.update({
                "model_type": "i2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 36,
                "dim": 512,
                "ffn_dim": 512,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 2,  # 40
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            })
            self.pipe.initialize_dummy_dit(dit_config)
            
        self.pipe.fetch_models_from_checkpoints(
            paths,
            config_dict={
                "dit": dit_config
            })
        
        if use_sequence_cond:
            self.pipe.denoising_model().copy_sequence_cond_patch_embedding_weights()
        elif use_channel_cond:
            self.pipe.denoising_model().copy_patch_embedding_weights_for_channel_cond()
        
        self.freeze_parameters()
            
        if checkpoint_path is not None:
            self.load_tooncomposer_checkpoint(checkpoint_path)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.vae_tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.video_condition_preservation_mode = video_condition_preservation_mode
        self.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"  
        
        if output_path is None:
            output_path = "./"
        self.output_path = output_path
    
    def load_tooncomposer_checkpoint(self, checkpoint_path):
        if checkpoint_path == self.current_checkpoint_path:
            print(f"Skipping loading checkpoint {checkpoint_path} because it is the same as the current checkpoint.")
            return
        self.current_checkpoint_path = checkpoint_path
        self.load_patch_to_model(
            self.pipe.denoising_model(),
            checkpoint_path
        )
        
    def update_height_width(self, height, width):
        self.height = height
        self.width = width
        
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
    def load_patch_to_model(self, model, pretrained_path, state_dict_converter=None):
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            self.loaded_global_step = 0
            self.loaded_current_epoch = 0
            if self.use_sketch_mask:
                seq_cond_embed_weight = state_dict['sequence_cond_patch_embedding.weight']
                current_in_channels = self.pipe.denoising_model().sequence_cond_patch_embedding.in_channels
                if current_in_channels == 24 and seq_cond_embed_weight.shape[1] == 20:
                    new_weight = torch.zeros(
                        seq_cond_embed_weight.shape[0],
                        4,
                        *seq_cond_embed_weight.shape[2:],
                        dtype=seq_cond_embed_weight.dtype
                    )
                    state_dict['sequence_cond_patch_embedding.weight'] = torch.cat([
                        seq_cond_embed_weight, new_weight], dim=1)
            
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"[Checkpoint] {num_updated_keys} parameters are loaded from {pretrained_path}. {num_unexpected_keys} parameters are unexpected.")
