# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Final, Optional, Callable, Union, Tuple, List, Set, Dict, Type, Literal, Sequence
import math
import warnings
from timm.layers import (
    PatchEmbed, Mlp, DropPath,
    AttentionPoolLatent, PatchDropout, resample_abs_pos_embed, LayerType
)
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from transformers.modeling_utils import is_flash_attn_2_available
# try:
#     from xformers.ops import memory_efficient_attention
# except ImportError:
#     warnings.warn(
#         "xformers not installed, using slow PyTorch implementation of memory_efficient_attention",
#         stacklevel=2,
#     )

def memory_efficient_attention(query, key, value, p):
    """ This code is taken from https://facebookresearch.github.io/xformers/components/ops.html """
    attn_bias = None
    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn = query @ key.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    attn = F.dropout(attn, p)
    attn = attn @ value
    return attn.transpose(1, 2).contiguous()
from functools import partial


if is_flash_attn_2_available():
    from flash_attn import flash_attn_qkvpacked_func


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""The original timm.models.layers.weight_init.trunc_normal_ can not handle bfloat16 yet, here we first
    convert the tensor to float32, apply the trunc_normal_() in float32, and then convert it back to its orignal dtype.
    Fills the input Tensor with values drawn from a truncated normal distribution. The values are effectively drawn
    from the normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """

    with torch.no_grad():
        dtype = tensor.dtype
        tensor_fp32 = tensor.float()
        tensor_fp32 = _no_grad_trunc_normal_(tensor_fp32, mean, std, a, b)
        tensor_dtype = tensor_fp32.to(dtype=dtype)
        tensor.copy_(tensor_dtype)


def init_weights(self):
    if self.pos_embed is not None:
        trunc_normal_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
    trunc_normal_(self.latent, std=self.latent_dim ** -0.5)


def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            deterministic: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm
        self.fused_attn = True
        self.deterministic = deterministic

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if not self.qk_norm:
            if self.head_dim % 32 == 0 and is_flash_attn_2_available():
                # flashattn must have head_dim as a multiple of 32
                x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop.p if self.training else 0.,
                                              deterministic=self.deterministic)
            else:
                q, k, v = qkv.unbind(2)
                x = memory_efficient_attention(q, k, v, p=self.attn_drop.p if self.training else 0.)
            x = x.reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=False):
                # 用上下文的方式强行使用fa
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            deterministic=deterministic,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            ignore_head: bool = False,
            deterministic: bool = False,
            num_recomputing_layers: int = 0
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        # norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        # act_layer = get_act_layer(act_layer) or nn.GELU
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # siglip use PytorchGELUTanh() rather than the vanilla nn.GELU()
        # https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/siglip/configuration_siglip.py#L191
        act_layer = partial(nn.GELU, approximate='tanh')

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.ignore_head = ignore_head
        self.num_recomputing_layers = num_recomputing_layers

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                deterministic=deterministic,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            AttentionPoolLatent.init_weights = init_weights
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode: Literal['jax', 'jax_nlhb', 'moco', ''] = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "is_first_stage", True):
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            skip_last = max(1, len(self.blocks) - self.num_recomputing_layers)
            x = checkpoint_seq(self.blocks, x, skip_last=skip_last)
        else:
            x = self.blocks(x)
        if getattr(self, "is_last_stage", True):
            x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if not getattr(self, "is_last_stage", True):
            return x
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if not self.ignore_head:
            x = self.forward_head(x)
        return x

    def to_pipeline(self, pp_size, pp_rank, pp_splits: Optional[List[int]] = None):
        self.is_first_stage = pp_rank == 0
        self.is_last_stage = pp_rank == pp_size - 1
        if not self.is_first_stage and hasattr(self, "patch_embed"):
            del self.patch_embed, self.cls_token, self.reg_token, self.pos_embed, self.pos_drop, self.patch_drop, self.norm_pre
        if not self.is_last_stage and hasattr(self, "norm"):
            del self.norm, self.attn_pool, self.fc_norm, self.head_drop, self.head
        if pp_splits is not None:
            assert len(self.blocks) == sum(pp_splits)
            splits = np.cumsum([0] + pp_splits)
            self.blocks = self.blocks[splits[pp_rank]:splits[pp_rank + 1]]
        return self


@dataclass
class SigLIPVisionCfg:
    width: int = 1152
    layers: Union[Tuple[int, int, int, int], int] = 27
    heads: int = 16
    patch_size: int = 14
    image_size: Union[Tuple[int, int], int] = 336
    global_pool: str = "map"
    mlp_ratio: float = 3.7362
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False


SigLIP_MODEL_CONFIG = {
    "siglip_so400m_patch14_384": {
        "image_size": 384,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False
    },

    "siglip_so400m_patch14_224": {
        "image_size": 224,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False
    },

    "siglip_large_patch16_384": {
        "image_size": 384,
        "patch_size": 16,
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "mlp_ratio": 4,
        "global_pool": "map",
        "use_checkpoint": False
    }
}


def create_siglip_vit(
        model_name: str = "siglip_so400m_patch14_384",
        image_size: int = 384,
        select_layer: int = -1,
        ckpt_path: str = "",
        **kwargs
):
    assert model_name in SigLIP_MODEL_CONFIG.keys(), f"model name should be in {SigLIP_MODEL_CONFIG.keys()}"

    vision_cfg = SigLIPVisionCfg(**SigLIP_MODEL_CONFIG[model_name])

    if select_layer <= 0:
        layers = min(vision_cfg.layers, vision_cfg.layers + select_layer + 1)
    else:
        layers = min(vision_cfg.layers, select_layer)

    model = VisionTransformer(
        img_size=image_size,
        patch_size=vision_cfg.patch_size,
        embed_dim=vision_cfg.width,
        depth=layers,
        num_heads=vision_cfg.heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        class_token=vision_cfg.class_token,
        global_pool=vision_cfg.global_pool,
        ignore_head=kwargs.get("ignore_head", True),
        weight_init=kwargs.get("weight_init", "skip"),
        num_classes=0,
        deterministic=kwargs.get("deterministic", False),
        num_recomputing_layers=kwargs.get("num_recomputing_layers", 0)
    )

    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location="cpu")

        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        print(f"SigLIP-ViT restores from {ckpt_path},\n"
              f"\tincompatible_keys:', {incompatible_keys}.")

    return model
