"""
Backbone extension: registers ConvNeXt V2 FCMAE variants into
patchcore.backbones._BACKBONES so they can be used with -b <name>.

Importing this module (done automatically by run_patchcore.py) is enough —
no other file needs to be modified.

Architecture reminder for ConvNeXt V2 Base (input 224×224):
    Layer name   Channels   Spatial   Blocks   Notes
    ----------   --------   -------   ------   -------------------------
    stages.0       128       56×56      3       No downsampling (Identity)
    stages.1       256       28×28      3       Includes 2×2 stride-2 down
    stages.2       512       14×14     27       Includes 2×2 stride-2 down
    stages.3      1024        7×7       3       Includes 2×2 stride-2 down

Recommended -le arguments for PatchCore (see note.md for full analysis):
    Best single layer  : -le stages.2
    Best multi-scale   : -le stages.1 -le stages.2   (default recommendation)
    Texture-focused    : -le stages.0 -le stages.1
    Three-scale        : -le stages.0 -le stages.1 -le stages.2

Pre-training variants available (timm >= 1.0):
    fcmae                  : pure masked autoencoder, no supervised fine-tuning
    fcmae_ft_in1k          : FCMAE → fine-tuned on ImageNet-1K
    fcmae_ft_in22k_in1k    : FCMAE → fine-tuned on IN-22K → IN-1K  (best)
    fcmae_ft_in22k_in1k_384: same at 384×384 input resolution
"""

import patchcore.backbones

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
