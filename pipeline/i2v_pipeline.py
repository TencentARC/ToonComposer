from diffsynth import ModelManager
from diffsynth.pipelines.base import BasePipeline
from diffsynth.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear

from model.dit import WanModel
from model.text_encoder import WanTextEncoder
from model.vae import WanVideoVAE
from model.image_encoder import WanImageEncoder
from model.prompter import WanPrompter
from scheduler.flow_match import FlowMatchScheduler

import torch, os
from einops import rearrange, repeat
import numpy as np
import PIL.Image
from tqdm import tqdm
from safetensors import safe_open

from model.text_encoder import T5RelativeEmbedding, T5LayerNorm
from model.dit import WanLayerNorm, WanRMSNorm, WanSelfAttention
from model.vae import RMS_norm, CausalConv3d, Upsample


def binary_tensor_to_indices(tensor):
    assert tensor.dim() == 2, "Input tensor must be in [b, t]"
    indices = [(row == 1).nonzero(as_tuple=True)[0] for row in tensor]
    return indices

def propagate_visualize_attention_arg(model, visualize_attention=False):
        """
        Recursively set the visualize_attention parameter to True for all WanSelfAttention modules
        Only for inference/test mode
        """
        for name, module in model.named_modules():
            if isinstance(module, WanSelfAttention):
                if "blocks.0.self_attn" in name or "blocks.19.self_attn" in name or "blocks.39.self_attn" in name:
                    print(f"Set `visualize_attention` to {visualize_attention} for {name}")
                    module.visualize_attention = visualize_attention

class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def fetch_models_from_model_manager(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
    
    def _init_component_from_checkpoint_path(self, model_cls, state_dict_path, strict=True, config_dict=None):
        config = {}
        state_dict = self._load_state_dict(state_dict_path)
        if hasattr(model_cls, "state_dict_converter"):
            state_dict_converter = model_cls.state_dict_converter()
            state_dict = state_dict_converter.from_civitai(state_dict)
            if isinstance(state_dict, tuple):
                state_dict, config = state_dict
        config.update(config_dict or {})
        model = model_cls(**config)
        if "use_local_lora" in config_dict or "use_dera" in config_dict:
            strict = False
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        return model
    
    def _load_state_dict(self, state_dict_paths):
        if isinstance(state_dict_paths, str):
            state_dict_paths = [state_dict_paths]
        state_dict = {}
        for state_dict_path in tqdm(state_dict_paths, desc="Reading file(s) from disk"):
            state_dict.update(self._load_single_file(state_dict_path))
        return state_dict
    
    def _load_single_file(self, file_path):
        if file_path.endswith(".safetensors"):
            return self._load_state_dict_from_safetensors(file_path)
        else:
            return torch.load(file_path, map_location='cpu')
    
    def _load_state_dict_from_safetensors(self, file_path, torch_dtype=None):
        state_dict = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
                if torch_dtype is not None:
                    state_dict[k] = state_dict[k].to(torch_dtype)
        return state_dict
    
    def initialize_dummy_dit(self, config):
        print("Initializing a dummy DIT model.")
        self.dit = WanModel(**config)
        print("Dummy DIT model is initialized.")
    
    def fetch_models_from_checkpoints(self, path_dict, config_dict=None):
        default_config = {"text_encoder": {}, "dit": {}, "vae": {}, "image_encoder": {}}
        config_dict = {**default_config, **(config_dict or {})}
        components = {
            "text_encoder": WanTextEncoder,
            "dit": WanModel,
            "vae": WanVideoVAE,
            "image_encoder": WanImageEncoder
        }
        for name, model_cls in components.items():
            if name not in path_dict:
                print(f"Component {name} is not found in the checkpoint path dict. Skipping.")
                continue
            path = path_dict[name]
            config = config_dict.get(name, {})
            print(f"Loading {name} from {path} with config {config}.")
            setattr(self, name, self._init_component_from_checkpoint_path(model_cls, path, config_dict=config))
            print(f"Initialized {name} from checkpoint.")
        if "text_encoder" in path_dict:
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(path_dict["text_encoder"]), "google/umt5-xxl"))
        print("Initialized prompter from checkpoint.")
        print("All components are initialized from checkpoints.")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models_from_model_manager(model_manager)
        return pipe
    
    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}
    
    def encode_image(self, image, num_frames, height, width):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            image = self.preprocess_image(image.resize((width, height))).to(self.device)
            clip_context = self.image_encoder.encode_image([image])
            msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = self.vae.encode([torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)], device=self.device)[0]
            y = torch.concat([msk, y])
        return {"clip_fea": clip_context, "y": [y]}
    
    def check_and_fix_image_or_video_tensor_input(self, _tensor):
        assert isinstance(_tensor, torch.Tensor), "Input must be a tensor."
        if _tensor.max() <= 255 and _tensor.max() > 1.0:
            _tensor = _tensor.to(self.device) / 127.5 - 1
            print("Input tensor is converted from [0, 255] to [-1, 1].")
        elif _tensor.min() >= 0 and _tensor.max() <= 1:
            _tensor = _tensor.to(self.device) * 2 - 1
            print("Input tensor is converted from [0, 1] to [-1, 1].")
        return _tensor
    
    def encode_video_with_mask(self, video, num_frames, height, width, condition_preserved_mask):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            video = video.to(self.device)
            y = self.vae.encode(video, device=self.device)
            msk = condition_preserved_mask
            assert msk is not None, "The mask must be provided for the masked video input."
            assert msk.dim() == 2, "The mask must be a 2D tensor in [b, t]."
            assert msk.shape[0] == video.shape[0], "The batch size of the mask must be the same as the input video."
            assert msk.shape[1] == num_frames, "The number of frames in the mask must be the same as the input video."
            msk = msk.to(self.device)
            msk = msk.unsqueeze(-1).unsqueeze(-1)
            msk = repeat(msk, 'b t 1 1 -> b t h w', h=height//8, w=width//8)
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(video.shape[0], msk.shape[1] // 4, 4, height//8, width//8)  # b, t, c, h, w
            msk = msk.transpose(1, 2)  # b, c, t, h, w
            y = torch.concat([msk, y], dim=1)
        return y
    
    def encode_video_with_mask_sparse(self, video, height, width, condition_preserved_mask, sketch_local_mask=None):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            batch_size = video.shape[0]
            cond_indices = binary_tensor_to_indices(condition_preserved_mask)
            sequence_cond_compressed_indices = [(cond_index + 3) // 4 for cond_index in cond_indices]
            video = video.to(self.device)
            video_latent = self.vae.encode(video, device=self.device)
            video_latent = video_latent[:, :, sequence_cond_compressed_indices[0], :, :]
            msk = condition_preserved_mask.to(self.device)
            msk = msk.unsqueeze(-1).unsqueeze(-1)  # b, t, 1, 1
            msk = repeat(msk, 'b t 1 1 -> b t h w', h=height//8, w=width//8)
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(batch_size, msk.shape[1] // 4, 4, height//8, width//8)  # b, t, 4, h//8, w//8
            msk = msk.transpose(1, 2)  # b, 4, t, h//8, w//8
            msk = msk[:, :, sequence_cond_compressed_indices[0], :, :]
            
            if sketch_local_mask is not None:
                sketch_local_mask = sketch_local_mask.to(self.device)
                if sketch_local_mask.shape[-2:] != (height//8, width//8):
                    sk_batch_t = sketch_local_mask.shape[0] * sketch_local_mask.shape[2]
                    sketch_local_mask_reshaped = sketch_local_mask.reshape(sk_batch_t, 1, sketch_local_mask.shape[3], sketch_local_mask.shape[4])
                    sketch_local_mask_resized = torch.nn.functional.interpolate(
                        sketch_local_mask_reshaped,
                        size=(height//8, width//8), 
                        mode='nearest'
                    )
                    sketch_local_mask_resized = sketch_local_mask_resized.reshape(
                        sketch_local_mask.shape[0], 
                        sketch_local_mask.shape[1], 
                        sketch_local_mask.shape[2], 
                        height//8, width//8
                    )
                else:
                    sketch_local_mask_resized = sketch_local_mask
                    
                sketch_mask = sketch_local_mask_resized
                sketch_mask = torch.concat([torch.repeat_interleave(sketch_mask[:, :, 0:1], repeats=4, dim=2), sketch_mask[:, :, 1:]], dim=2)
                sketch_mask = sketch_mask.view(batch_size, sketch_mask.shape[1], sketch_mask.shape[2] // 4, 4, height//8, width//8)
                sketch_mask = sketch_mask.permute(0, 1, 3, 2, 4, 5)  # [b, 1, 4, t//4, h//8, w//8]
                sketch_mask = sketch_mask.view(batch_size, 4, sketch_mask.shape[3], height//8, width//8)  # [b, 4, t//4, h//8, w//8]
                sketch_mask = sketch_mask[:, :, sequence_cond_compressed_indices[0], :, :]  # [b, 4, len(indices), h//8, w//8]
                
                combined_latent = torch.cat([msk, video_latent, sketch_mask], dim=1)
            else:
                combined_latent = torch.concat([msk, video_latent], dim=1)
            
        return combined_latent, sequence_cond_compressed_indices  # b, c=(4+16+4=24), t, h, w when sketch_local_mask is provided
    
    def encode_image_or_masked_video(self, image_or_masked_video, num_frames, height, width, condition_preserved_mask=None):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            batch_size = image_or_masked_video.shape[0]
            if isinstance(image_or_masked_video, PIL.Image.Image) or (isinstance(image_or_masked_video, torch.Tensor) and image_or_masked_video.dim() <= 4):
                if isinstance(image_or_masked_video, PIL.Image.Image):
                    image_or_masked_video = self.preprocess_image(image_or_masked_video.resize((width, height))).to(self.device)
                else:
                    if image_or_masked_video.dim() == 3:
                        image_or_masked_video = image_or_masked_video.unsqueeze(0)  # b=1, c, h, w
                    image_or_masked_video = image_or_masked_video.to(self.device)
                y = self.vae.encode([torch.concat([image_or_masked_video.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image_or_masked_video.device)], dim=1)], device=self.device)
                msk_idx_to_be_zero = range(1, num_frames)
                clip_context = self.image_encoder.encode_image(image_or_masked_video.unsqueeze(1))  # need to be [b, 1, c, h, w]
                msk = torch.ones(batch_size, num_frames, height//8, width//8, device=self.device)
                msk[:, msk_idx_to_be_zero] = 0
                msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
                msk = msk.view(batch_size, msk.shape[1] // 4, 4, height//8, width//8)
                msk = msk.transpose(1, 2)
            elif isinstance(image_or_masked_video, torch.Tensor) and image_or_masked_video.dim() == 5:
                image_or_masked_video = image_or_masked_video.to(self.device)
                first_image = image_or_masked_video[:, :, 0, :, :].unsqueeze(1)
                clip_context = self.image_encoder.encode_image(first_image)
                y = self.vae.encode(image_or_masked_video, device=self.device)
                msk = condition_preserved_mask  # b, t
                assert msk is not None, "The mask must be provided for the masked video input."
                assert msk.dim() == 2, "The mask must be a 2D tensor in [b, t]."
                assert msk.shape[0] == batch_size, "The batch size of the mask must be the same as the input video."
                assert msk.shape[1] == num_frames, "The number of frames in the mask must be the same as the input video."
                msk = msk.to(self.device)
                msk = msk.unsqueeze(-1).unsqueeze(-1)  # b, t, 1, 1
                msk = repeat(msk, 'b t 1 1 -> b t h w', h=height//8, w=width//8)
                msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
                msk = msk.view(batch_size, msk.shape[1] // 4, 4, height//8, width//8)  # b, t, 4, h//8, w//8
                msk = msk.transpose(1, 2)  # b, 4, t, h//8, w//8
            else:
                raise ValueError("Input must be an image (PIL/Tensor in [b, c, h, w]) or a masked video (Tensor in [b, c, t, h, w]).")

        y = torch.concat([msk, y], dim=1)
        return {"clip_fea": clip_context, "y": y}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [PIL.Image.fromarray(frame) for frame in frames]
        return frames
    
    def prepare_extra_input(self, latents=None):
        return {"seq_len": latents.shape[2] * latents.shape[3] * latents.shape[4] // 4}
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        # progress_bar_st=None,
        input_condition_video=None,
        input_condition_preserved_mask=None,
        input_condition_video_sketch=None,
        input_condition_preserved_mask_sketch=None,
        sketch_local_mask=None,
        visualize_attention=False,
        output_path=None,
        batch_idx=None,
        sequence_cond_residual_scale=1.0,
    ):
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength, shift=sigma_shift)

        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32).to(self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=noise.dtype, device=noise.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        self.load_models_to_device(["image_encoder", "vae"])
        if input_image is not None and self.image_encoder is not None:
            image_emb = self.encode_image(input_image, num_frames, height, width)
        elif input_condition_video is not None and self.image_encoder is not None:
            assert input_condition_preserved_mask is not None, "`input_condition_preserved_mask` must not be None when `input_condition_video` is given."
            image_emb = self.encode_image_or_masked_video(input_condition_video, num_frames, height, width, input_condition_preserved_mask)
        else:
            image_emb = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        if self.dit.use_sequence_cond:
            assert input_condition_video_sketch is not None, "`input_condition_video_sketch` must not be None when `use_sequence_cond` is True."
            assert input_condition_preserved_mask_sketch is not None, "`input_condition_preserved_mask_sketch` must not be None when `input_condition_video_sketch` is given."
            
            if self.dit.sequence_cond_mode == "sparse":
                sequence_cond, sequence_cond_compressed_indices = self.encode_video_with_mask_sparse(input_condition_video_sketch, height, width, input_condition_preserved_mask_sketch, sketch_local_mask)
                extra_input.update({"sequence_cond": sequence_cond,
                                    "sequence_cond_compressed_indices": sequence_cond_compressed_indices})
            elif self.dit.sequence_cond_mode == "full":
                sequence_cond = self.encode_video_with_mask(input_condition_video_sketch, num_frames, height, width, input_condition_preserved_mask_sketch)
                extra_input.update({"sequence_cond": sequence_cond})
            else:
                raise ValueError(f"Invalid `sequence_cond_model`={self.dit.sequence_cond_mode} in the DIT model.")
            
        elif self.dit.use_channel_cond:
            sequence_cond = self.encode_video_with_mask(input_condition_video_sketch, num_frames, height, width, input_condition_preserved_mask_sketch)
            extra_input.update({"channel_cond": sequence_cond})
            
        self.load_models_to_device([])
        
        if sequence_cond_residual_scale != 1.0:
            extra_input.update({"sequence_cond_residual_scale": sequence_cond_residual_scale})

        # Denoise
        self.load_models_to_device(["dit"])
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)
                _should_visualize_attention = visualize_attention and (progress_id == len(self.scheduler.timesteps) - 1)
                if _should_visualize_attention:
                    print(f"Visualizing attention maps (Step {progress_id + 1}/{len(self.scheduler.timesteps)}).")
                    propagate_visualize_attention_arg(self.dit, True)
        
                # Inference
                noise_pred_posi = self.dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input)
                if isinstance(noise_pred_posi, tuple):
                    noise_pred_posi = noise_pred_posi[0]
                if cfg_scale != 1.0:
                    noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_nega, **image_emb, **extra_input)
                    if isinstance(noise_pred_nega, tuple):
                        noise_pred_nega = noise_pred_nega[0]
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
                
                # If visualization is enabled, save the attention maps
                if _should_visualize_attention:
                    print("Saving attention maps...")
                    from util.model_util import save_attention_maps
                    save_attention_maps(self.dit, output_path, batch_idx, timestep.squeeze().cpu().numpy().item())
                    propagate_visualize_attention_arg(self.dit, False)

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])

        return frames