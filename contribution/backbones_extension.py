"""
Backbone extension: registers ConvNeXt V2 FCMAE and DINOv2 variants into
patchcore.backbones._BACKBONES so they can be used with -b <name>.

Importing this module (done automatically by run_patchcore.py) is enough —
no other file needs to be modified.

════════════════════════════════════════════════════════════════════════════════
 ConvNeXt V2 — complete architecture reference
════════════════════════════════════════════════════════════════════════════════

──── Macro structure (input 224×224) ────────────────────────────────────────

  Component        Output shape     Notes
  ─────────────    ────────────     ──────────────────────────────────────────
  Input            3 × 224 × 224
  Stem             C0 × 56 × 56    4×4 conv stride 4  +  LayerNorm
  Downsample 1→2   C1 × 28 × 28    2×2 conv stride 2  +  LayerNorm
  Downsample 2→3   C2 × 14 × 14    2×2 conv stride 2  +  LayerNorm
  Downsample 3→4   C3 ×  7 × 7    2×2 conv stride 2  +  LayerNorm
  Global avg pool  C3 × 1 × 1
  FC head          1000            linear, for ImageNet classification only
                                   (head is NOT used by PatchCore)

──── Stage channels and depths by variant ────────────────────────────────────

  Variant   stages.0   stages.1   stages.2   stages.3   Depths
  ───────   ────────   ────────   ────────   ────────   ──────────────
  Tiny          96       192        384        768       [3, 3,  9, 3]
  Base         128       256        512       1024       [3, 3, 27, 3]
  Large        192       384        768       1536       [3, 3, 27, 3]

──── Feature maps extracted by PatchCore (Base, 224×224) ────────────────────

  -le arg     Channels   Spatial   Receptive field   What it captures
  ─────────   ────────   ───────   ───────────────   ─────────────────────────
  stages.0      128       56×56    ≈ 7 px (1 block)  Fine texture, edges
  stages.1      256       28×28    ≈ 14–30 px        Mid-level patterns
  stages.2      512       14×14    ≈ 30–90 px        Structural anomalies ★
  stages.3     1024        7×7    ≈ 90–224 px       Global semantics (coarse)

  ★ stages.2 is the single best layer for most anomaly detection tasks.
    For multi-scale: -le stages.1 -le stages.2  (recommended default)

  Spatial dims scale linearly with imagesize:
    imagesize   stages.0    stages.1    stages.2    stages.3
    ─────────   ─────────   ─────────   ─────────   ─────────
       224       56×56       28×28       14×14        7×7
       320       80×80       40×40       20×20       10×10
       448      112×112      56×56       28×28       14×14

──── ConvNeXt V2 block internals ────────────────────────────────────────────

  For each of the N blocks in a stage:

    input (B, C, H, W)
      │
      ├─ depthwise conv 7×7, padding 3, groups=C       ← spatial mixing
      │    no stride, C → C
      │
      ├─ LayerNorm  (channel-last, i.e. on C dimension)
      │    ε = 1e-6, no affine scale for FCMAE variants
      │
      ├─ pointwise Linear C → 4C                       ← channel expansion
      │
      ├─ GELU activation
      │
      ├─ GRN  (Global Response Normalization)           ← new in V2
      │    x_c ← x_c · (‖x_c‖₂ / mean_c(‖x_c‖₂))
      │    prevents feature collapse during MAE pretraining
      │
      ├─ pointwise Linear 4C → C                       ← channel collapse
      │
      └─ residual add  (+ LayerScale γ, init 1e-6)

  NO attention mechanism — purely convolutional.
  Normalization: LayerNorm only (no BatchNorm).
  Activation:    GELU only.

──── Pre-training variants ──────────────────────────────────────────────────

  Key                        Description
  ─────────────────────────  ─────────────────────────────────────────────────
  fcmae                      Pure masked autoencoder, no classification head
                             Good for texture/self-supervised features
  fcmae_ft_in1k              FCMAE → fine-tuned on ImageNet-1K
  fcmae_ft_in22k_in1k        FCMAE → IN-22K (21 841 cls) → IN-1K  ← best
  (384 variant)              Same, native 384×384 input resolution


════════════════════════════════════════════════════════════════════════════════
 DINOv2 — complete architecture reference
════════════════════════════════════════════════════════════════════════════════

──── Macro structure ────────────────────────────────────────────────────────

  DINOv2 is a plain ViT (no hierarchical stages) with patch_size=14.
  Every transformer block outputs tokens at the SAME spatial resolution.

  Component           Output shape (448×448 input, ViT-B example)
  ─────────────────   ─────────────────────────────────────────────────────
  Input               B × 3 × 448 × 448
  Patch embedding     B × (32×32 + 1 + R) × 768   (linear proj of patches)
    patch tokens :    32×32 = 1 024  (one per 14×14 px tile)
    [CLS] token  :    1
    register tok.:    4 (for *_reg variants only)
  Positional emb.     added to patch+CLS tokens; bicubic-resized if needed
  Blocks 0..N-1       B × (N_tokens) × 768   (shape unchanged across blocks)
  Final LayerNorm     B × (N_tokens) × 768
  Head (classif.)     B × 1000                (linear on [CLS]; not used)

  PatchCore uses ONLY the patch tokens from intermediate blocks,
  discarding [CLS] and register tokens, and reshaping:
      (B, H/14 × W/14, C)  →  (B, C, H/14, W/14)

──── Spatial token grid by imagesize ────────────────────────────────────────

  imagesize   patch grid   tokens (excl. CLS+reg)   Effective patch size
  ─────────   ──────────   ────────────────────────  ────────────────────
    224×224     16 × 16         256                     14 × 14 px/token
    336×336     24 × 24         576                     14 × 14 px/token
    448×448     32 × 32        1024                     14 × 14 px/token  ★
    518×518     37 × 37        1369                     14 × 14 px/token  (native)

  ★ 448 px is the best trade-off: dense enough for anomaly localization,
    compatible with ImageNet pre-training statistics.
    imagesize must be divisible by 14.

──── Model dimensions (identical at every block) ────────────────────────────

  Backbone     Embed dim   Heads   Head dim   Blocks   MLP hidden   Params
  ──────────   ─────────   ─────   ────────   ──────   ──────────   ──────
  ViT-S/14       384         6       64         12       1 536       22 M
  ViT-B/14       768        12       64         12       3 072       86 M
  ViT-L/14      1024        16       64         24       4 096      307 M
  ViT-G/14      1536        24       64         40       6 144     1100 M

  All variants: patch_size=14, no CLS in MLP, pre-norm transformer.

──── ViT block internals (same for every block, both standard and reg) ───────

  input tokens  (B, N_tokens, C)
    │
    ├─ LayerNorm  (pre-norm, on C)                  ← normalization #1
    │
    ├─ Multi-Head Self-Attention  (MHSA)            ← attention layer
    │    Q = xW_Q,  K = xW_K,  V = xW_V
    │    Attention = softmax(QKᵀ / √d_head) · V    (d_head = C / n_heads = 64)
    │    Output = concat(heads) · W_O
    │    ALL tokens attend to ALL tokens globally   ← no local window
    │    No relative position bias (absolute pos-embed added at input)
    │
    ├─ residual add
    │
    ├─ LayerNorm  (pre-norm, on C)                  ← normalization #2
    │
    ├─ MLP  (feed-forward network)
    │    Linear  C → 4C
    │    GELU
    │    Linear  4C → C
    │
    └─ residual add

  Normalization: LayerNorm (pre-norm) ONLY — no BatchNorm, no GRN.
  Activation:    GELU in MLP, softmax in attention.
  Dropout:       0.0 at inference (DINOv2 is trained without dropout).

──── Feature layers and what they capture ────────────────────────────────────

  Because every ViT block has GLOBAL self-attention, even early blocks
  have some global context (unlike CNNs where receptive field grows slowly).

  Block range    Feature character
  ────────────   ──────────────────────────────────────────────────────────────
  blocks.0–2     Mostly local: patch projections + minimal attention mixing.
                 Captures low-level texture, color, edges.
  blocks.N/4     Local + emerging global: position-aware local patterns.
                 Good texture + structure complement to a later layer.
  blocks.N/2     Balanced: rich local texture AND semantic context.  ★ mid
                 Best single layer for texture-dominant anomalies.
  blocks.N-1     Fully global: object parts, semantic consistency.   ★ high
                 Best for structural / assembly anomalies.

  Recommended extraction pairs  (-le blocks.A  -le blocks.B):
    ViT-S/B  (12 blocks)  : blocks.5   + blocks.11   (mid + final)
    ViT-L    (24 blocks)  : blocks.11  + blocks.23   (mid + final)
    ViT-G    (40 blocks)  : blocks.19  + blocks.39   (mid + final)

  Alternative: three-layer for maximal coverage
    ViT-B: blocks.3 + blocks.7 + blocks.11

──── Channel and spatial output per layer extracted (ViT-B, 448×448) ────────

  -le arg      Channels   Spatial (H/14, W/14)   Receptive field
  ──────────   ────────   ──────────────────────  ───────────────────────────
  blocks.5       768           32 × 32            Global (full image via attn)
  blocks.11      768           32 × 32            Global (full image via attn)

  Note: channels are ALWAYS equal to embed_dim regardless of block depth.
  The feature content changes (more semantic at higher blocks), not the shape.

──── Register tokens (*_reg variants) ───────────────────────────────────────

  The 4 register tokens act as "memory" for the model to offload high-level
  global information from patch tokens. This reduces patch-token "artifacts"
  (isolated high-norm outlier patches visible in standard ViT attention maps).
  PatchCore discards register tokens automatically (DINOv2Aggregator).
  Use *_reg variants for all dense-prediction tasks including PatchCore.

──── Positional embedding and dynamic resolution ─────────────────────────────

  DINOv2 is pretrained at native 518×518 (37×37 tokens).
  For other resolutions: the absolute pos-embed is bicubically interpolated
  to the new grid size via timm's `resample_abs_pos_embed`.
  `dynamic_img_size=True` is required in timm to enable this on-the-fly.
  PyTorch < 2.0: `antialias` kwarg is monkey-patched away (see common.py).

──── Why CosineNN (not FaissNN) ─────────────────────────────────────────────

  DINOv2 uses LayerNorm everywhere → token L2-norms vary arbitrarily.
  Raw L2 distance in FAISS is dominated by norm differences, not direction.
  L2-normalizing before FAISS (CosineNN) removes the norm effect and makes
  nearest-neighbor search purely direction-based → much better AUROC.

──── Recommended PatchCore settings ─────────────────────────────────────────

  --patchsize 1                  each token IS the patch (14×14 px)
  --pretrain_embed_dimension C   set to embed_dim of chosen model
  --target_embed_dimension C     same (no PCA projection)
  --anomaly_scorer_num_nn 3
  CosineNN is selected automatically.

  Example (ViT-B, 448×448, MVTec bottle):
    python bin/run_patchcore.py --gpu 0 --seed 0 results \\
      patch_core \\
        -b dinov2_vitb14_reg -le blocks.5 -le blocks.11 \\
        --pretrain_embed_dimension 768 --target_embed_dimension 768 \\
        --anomaly_scorer_num_nn 3 --patchsize 1 \\
      sampler -p 0.1 approx_greedy_coreset \\
      dataset --resize 512 --imagesize 448 -d bottle mvtec /data/mvtec

  Example (ViT-L, 448×448, VisA capsules):
    python bin/run_patchcore.py --gpu 0 --seed 0 results \\
      patch_core \\
        -b dinov2_vitl14_reg -le blocks.11 -le blocks.23 \\
        --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \\
        --anomaly_scorer_num_nn 3 --patchsize 1 \\
      sampler -p 0.1 approx_greedy_coreset \\
      dataset --resize 512 --imagesize 448 -d capsules visa /data/VisA_20220922
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
