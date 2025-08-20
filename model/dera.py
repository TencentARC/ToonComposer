import torch
import torch.nn as nn
from einops import rearrange
from .dit import flash_attention
import torch.amp as amp


class DeRAAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 mode="spatial"):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
        if mode == 'spatial':
            self.rope_apply = self.rope_apply_spatial
        else:
            raise ValueError("Invalid mode: {}".format(mode))
    
    @staticmethod
    @amp.autocast(enabled=False, device_type="cuda")
    def rope_apply_spatial(x, grid_size, freqs, sequence_cond_compressed_indices=None):
        batch, _, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        assert len(grid_size) == 2, "grid_size mustbe [h, w]"
        h, w = grid_size[0], grid_size[1]
        seq_len = h * w
        x_i = torch.view_as_complex(x[:, :seq_len].to(torch.float64).reshape(
            batch, seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[1][:h].view(1, h, 1, -1).expand(1, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(1, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1).unsqueeze(0).repeat(batch, 1, 1, 1)
        freqs_i = torch.concat([freqs_i.new_ones(batch, seq_len, 1, c//3), freqs_i], dim=3)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(3)
        return x_i.float()

    def forward(self, x, seq_lens, grid_size, freqs, sequence_cond_compressed_indices):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        def qkv_fn(x):
            q = self.q(x).view(b, s, n, d)
            k = self.k(x).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q_rope = self.rope_apply(q, grid_size, freqs, sequence_cond_compressed_indices)
        k_rope = self.rope_apply(k, grid_size, freqs, sequence_cond_compressed_indices)
        x = flash_attention(
            q=q_rope,
            k=k_rope,
            v=v,
            k_lens=None,
            window_size=self.window_size)
        x = x.flatten(2)
        x = self.o(x)
        return x


class DeRA(nn.Module):
    def __init__(self, dim, rank, use_spatial=True, use_temporal=False):
        super(DeRA, self).__init__()
        self.dim = dim
        self.rank = rank
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal
        
        if not use_spatial and not use_temporal:
            self.attention_mode = "none"
        else:
            assert use_spatial and not use_temporal, "Invalid configuration for ToonComposer."
            self.attention_mode = "spatial"
        
        self.spatial_down_proj = nn.Linear(self.dim, rank, bias=False)
        self.spatial_up_proj = nn.Linear(rank, self.dim, bias=False)
        self.spatial_up_proj.weight.data.zero_()
        self.spatial_attn = DeRAAttention(dim=rank, num_heads=4, window_size=(-1, -1),
                                          mode=self.attention_mode)
                                              
    def forward(self, x, seq_lens, grid_sizes, freqs, sequence_cond_compressed_indices):
        _, actual_seq, _ = x.shape
        if isinstance(grid_sizes, torch.Tensor):
            grid_sizes = tuple(grid_sizes[0].tolist())
            
        if len(grid_sizes) != 3:
            raise ValueError("`grid_sizes` should contain time, spatial height, and width dimensions")
        _, orig_h, orig_w = grid_sizes
        actual_t = actual_seq // (orig_h * orig_w)
        
        x_low = self.spatial_down_proj(x)
        if self.attention_mode == "spatial":
            x_low_spatial = rearrange(x_low, 'b (t h w) r -> (b t) (h w) r', t=actual_t, h=orig_h, w=orig_w)
            x_low_spatial = self.spatial_attn(x_low_spatial, seq_lens, grid_sizes[1:], freqs, sequence_cond_compressed_indices)
            if x_low_spatial.shape[1] != actual_seq:
                x_low = rearrange(x_low_spatial, '(b t) (h w) r -> b (t h w) r', t=actual_t, h=orig_h, w=orig_w)
            else:
                x_low = x_low_spatial
        else:
            raise ValueError("Invalid attention mode: {}".format(self.attention_mode))
        x_out = self.spatial_up_proj(x_low)
        return x_out
    