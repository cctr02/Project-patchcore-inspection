"""
Backbone extension: registers ConvNeXt V2 FCMAE and DINOv2 variants into
patchcore.backbones._BACKBONES so they can be used with -b <name>.

Importing this module (done automatically by run_patchcore.py) is enough —
no other file needs to be modified.

──────────────────────────────────────────────────────────────────────────────
ConvNeXt V2 — architecture reminder (input 224×224):
    Layer name   Channels   Spatial   Blocks   Notes
    ----------   --------   -------   ------   -------------------------
    stages.0       128       56×56      3       No downsampling (Identity)
    stages.1       256       28×28      3       Includes 2×2 stride-2 down
    stages.2       512       14×14     27       Includes 2×2 stride-2 down
    stages.3      1024        7×7       3       Includes 2×2 stride-2 down

Recommended -le arguments for ConvNeXt (see note.md for full analysis):
    Best single layer  : -le stages.2
    Best multi-scale   : -le stages.1 -le stages.2   (default recommendation)
    Texture-focused    : -le stages.0 -le stages.1
    Three-scale        : -le stages.0 -le stages.1 -le stages.2

Pre-training variants available (timm >= 1.0):
    fcmae                  : pure masked autoencoder, no supervised fine-tuning
    fcmae_ft_in1k          : FCMAE → fine-tuned on ImageNet-1K
    fcmae_ft_in22k_in1k    : FCMAE → fine-tuned on IN-22K → IN-1K  (best)
    fcmae_ft_in22k_in1k_384: same at 384×384 input resolution

──────────────────────────────────────────────────────────────────────────────
DINOv2 — architecture reminder:

    DINOv2 is a Vision Transformer (ViT) with patch_size=14. It produces one
    token per 14×14 pixel region plus a [CLS] token (and register tokens for
    *_reg variants). There is NO spatial hierarchy: all transformer blocks
    output tokens at the same spatial resolution.

    Spatial token grid by input size (must be divisible by 14):
        imagesize   tokens grid   example pretrain_embed_dim
        ---------   -----------   -------------------------
          224×224      16×16      recommended (standard)
          336×336      24×24      better localization
          448×448      32×32      best localization trade-off

    Model dimensions (same at every block):
        dinov2_vits14 / *_reg : 384  channels
        dinov2_vitb14 / *_reg : 768  channels
        dinov2_vitl14 / *_reg : 1024 channels
        dinov2_vitg14 / *_reg : 1536 channels  (use target_embed_dim ≤ 1024)

    Layer naming convention  →  "blocks.N"  (0-based block index):
        ViT-S/B  (12 blocks, 0-11)  : -le blocks.5  -le blocks.11
        ViT-L    (24 blocks, 0-23)  : -le blocks.11 -le blocks.23
        ViT-G    (40 blocks, 0-39)  : -le blocks.19 -le blocks.39

    Recommended PatchCore settings for DINOv2:
        --patchsize 1                (each token already covers 14×14 px)
        --anomaly_scorer_num_nn 3
        --pretrain_embed_dimension <model_channels>
        --target_embed_dimension    <model_channels>
        CosineNN is selected automatically (LayerNorm backbone).

    Example command (ViT-B, 224×224, MVTec):
        python bin/run_patchcore.py --gpu 0 --seed 0 results \\
          patch_core \\
            -b dinov2_vitb14 -le blocks.5 -le blocks.11 \\
            --pretrain_embed_dimension 768 --target_embed_dimension 768 \\
            --anomaly_scorer_num_nn 3 --patchsize 1 \\
          sampler -p 0.1 approx_greedy_coreset \\
          dataset --resize 256 --imagesize 224 -d bottle mvtec /data/mvtec
"""

import patchcore.backbones

# Maps PatchCore backbone key → timm model name used with features_only=True.
# Consumed by run_patchcore.py to auto-enable FeaturesOnlyAggregator + CosineNN.
_CONVNEXTV2_TIMM_NAMES = {
    "convnextv2_tiny_fcmae":           "convnextv2_tiny.fcmae",
    "convnextv2_tiny_fcmae_ft_in1k":   "convnextv2_tiny.fcmae_ft_in1k",
    "convnextv2_tiny_fcmae_ft_in22k":  "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_base_fcmae":           "convnextv2_base.fcmae",
    "convnextv2_base_fcmae_ft_in1k":   "convnextv2_base.fcmae_ft_in1k",
    "convnextv2_base_fcmae_ft_in22k":  "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnextv2_large_fcmae":          "convnextv2_large.fcmae",
    "convnextv2_large_fcmae_ft_in22k": "convnextv2_large.fcmae_ft_in22k_in1k",
}

_CONVNEXTV2_BACKBONES = {
    # --- Tiny (C=[96,192,384,768], depths=[3,3,9,3]) ---
    "convnextv2_tiny_fcmae": (
        'timm.create_model("convnextv2_tiny.fcmae", pretrained=True)'
    ),
    "convnextv2_tiny_fcmae_ft_in1k": (
        'timm.create_model("convnextv2_tiny.fcmae_ft_in1k", pretrained=True)'
    ),
    "convnextv2_tiny_fcmae_ft_in22k": (
        'timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True)'
    ),

    # --- Base (C=[128,256,512,1024], depths=[3,3,27,3]) ---
    "convnextv2_base_fcmae": (
        'timm.create_model("convnextv2_base.fcmae", pretrained=True)'
    ),
    "convnextv2_base_fcmae_ft_in1k": (
        'timm.create_model("convnextv2_base.fcmae_ft_in1k", pretrained=True)'
    ),
    "convnextv2_base_fcmae_ft_in22k": (
        'timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True)'
    ),

    # --- Large (C=[192,384,768,1536], depths=[3,3,27,3]) ---
    "convnextv2_large_fcmae": (
        'timm.create_model("convnextv2_large.fcmae", pretrained=True)'
    ),
    "convnextv2_large_fcmae_ft_in22k": (
        'timm.create_model("convnextv2_large.fcmae_ft_in22k_in1k", pretrained=True)'
    ),
} 

patchcore.backbones._BACKBONES.update(_CONVNEXTV2_BACKBONES)


# ── DINOv2 ────────────────────────────────────────────────────────────────────
# Maps PatchCore backbone key → timm model name.
# Consumed by run_patchcore.py to auto-enable DINOv2Aggregator + CosineNN.
# The *_reg variants include register tokens (better feature quality on dense
# prediction tasks); registers are automatically excluded from spatial output.
_DINOV2_TIMM_NAMES = {
    # Standard variants
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
    # Register variants — 4 register tokens, better for dense tasks (reg4 in timm)
    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitl14_reg": "vit_large_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitg14_reg": "vit_giant_patch14_reg4_dinov2.lvd142m",
}

_DINOV2_BACKBONES = {
    name: f'timm.create_model("{timm_name}", pretrained=True)'
    for name, timm_name in _DINOV2_TIMM_NAMES.items()
}

patchcore.backbones._BACKBONES.update(_DINOV2_BACKBONES)
