# Sweep Report — DINOv2B_VisA_Pilot3

Each trial is appended as it finishes (complete, pruned, or failed).

---
## Trial 0 — COMPLETE ✅  `2026-03-08 10:35:37`
**Name**: `IM448_DINOv2B14reg_L5-8_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9649** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9492 |
| capsules | 0.9471 |
| cashew | 0.9291 |
| chewinggum | 0.9940 |
| fryum | 0.9612 |
| macaroni1 | 0.9603 |
| macaroni2 | 0.9341 |
| pcb1 | 0.9654 |
| pcb2 | 0.9652 |
| pcb3 | 0.9959 |
| pcb4 | 0.9816 |
| pipe_fryum | 0.9958 |

---
## Trial 1 — COMPLETE ✅  `2026-03-08 11:19:11`
**Name**: `IM448_DINOv2B14reg_L5_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9732** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9513 |
| capsules | 0.9864 |
| cashew | 0.9326 |
| chewinggum | 0.9753 |
| fryum | 0.9735 |
| macaroni1 | 0.9604 |
| macaroni2 | 0.9900 |
| pcb1 | 0.9707 |
| pcb2 | 0.9653 |
| pcb3 | 0.9951 |
| pcb4 | 0.9810 |
| pipe_fryum | 0.9968 |

---
## Trial 2 — COMPLETE ✅  `2026-03-08 12:02:50`
**Name**: `IM448_DINOv2B14reg_L3-4_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9668** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9480 |
| capsules | 0.9966 |
| cashew | 0.8419 |
| chewinggum | 0.9665 |
| fryum | 0.9658 |
| macaroni1 | 0.9775 |
| macaroni2 | 0.9900 |
| pcb1 | 0.9736 |
| pcb2 | 0.9722 |
| pcb3 | 0.9933 |
| pcb4 | 0.9819 |
| pipe_fryum | 0.9949 |

---
## Trial 3 — COMPLETE ✅  `2026-03-08 12:47:04`
**Name**: `IM448_DINOv2B14reg_L3-8_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9635** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9446 |
| capsules | 0.9525 |
| cashew | 0.9293 |
| chewinggum | 0.9879 |
| fryum | 0.9640 |
| macaroni1 | 0.9591 |
| macaroni2 | 0.9198 |
| pcb1 | 0.9657 |
| pcb2 | 0.9676 |
| pcb3 | 0.9954 |
| pcb4 | 0.9814 |
| pipe_fryum | 0.9953 |

---
## Trial 4 — COMPLETE ✅  `2026-03-08 13:31:02`
**Name**: `IM448_DINOv2B14reg_L7-8_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9670** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9501 |
| capsules | 0.9591 |
| cashew | 0.9341 |
| chewinggum | 0.9868 |
| fryum | 0.9623 |
| macaroni1 | 0.9648 |
| macaroni2 | 0.9414 |
| pcb1 | 0.9684 |
| pcb2 | 0.9660 |
| pcb3 | 0.9953 |
| pcb4 | 0.9806 |
| pipe_fryum | 0.9945 |

---
## Trial 5 — COMPLETE ✅  `2026-03-08 14:15:02`
**Name**: `IM448_DINOv2B14reg_L4-5-7_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9722** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9515 |
| capsules | 0.9673 |
| cashew | 0.9440 |
| chewinggum | 0.9796 |
| fryum | 0.9616 |
| macaroni1 | 0.9741 |
| macaroni2 | 0.9801 |
| pcb1 | 0.9790 |
| pcb2 | 0.9560 |
| pcb3 | 0.9958 |
| pcb4 | 0.9820 |
| pipe_fryum | 0.9950 |

---
## Trial 6 — COMPLETE ✅  `2026-03-08 14:59:04`
**Name**: `IM448_DINOv2B14reg_L3-4-7_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9717** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9569 |
| capsules | 0.9716 |
| cashew | 0.9351 |
| chewinggum | 0.9820 |
| fryum | 0.9649 |
| macaroni1 | 0.9749 |
| macaroni2 | 0.9688 |
| pcb1 | 0.9777 |
| pcb2 | 0.9593 |
| pcb3 | 0.9952 |
| pcb4 | 0.9805 |
| pipe_fryum | 0.9937 |

---
## Trial 7 — PRUNED ✂️   `2026-03-08 15:09:59`
**Name**: `IM448_DINOv2B14reg_L7_P010_D768-768_PS-1_AN-3_S0`

**Stopped after**: 4 class(es) (pruned at `chewinggum`)
**Running mean**: 0.9461

**Pruning criterion** (MedianPruner — running mean vs completed trials at same step):

| Metric | Value |
|--------|------:|
| Running mean (this trial) | 0.9461 |
| Median @ step 4     | 0.9575 |
| Mean @ step 4       | 0.9554 |
| Best @ step 4       | 0.9614 |
| Worst @ step 4      | 0.9382 |
| Δ vs median                | -0.0115 (-1.2%) |
| Reference trials           | 7 |

> Running mean **0.9461** is below the median **0.9575** by 0.0115 (1.2%) → pruned.

**Partial results:**

| Class | AUROC |
|-------|------:|
| candle | 0.9455 |
| capsules | 0.9549 |
| cashew | 0.9078 |
| chewinggum | 0.9760 |

---
## Trial 8 — COMPLETE ✅  `2026-03-08 15:54:21`
**Name**: `IM448_DINOv2B14reg_L4-7-8_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9712** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9593 |
| capsules | 0.9607 |
| cashew | 0.9578 |
| chewinggum | 0.9940 |
| fryum | 0.9626 |
| macaroni1 | 0.9664 |
| macaroni2 | 0.9414 |
| pcb1 | 0.9745 |
| pcb2 | 0.9604 |
| pcb3 | 0.9967 |
| pcb4 | 0.9857 |
| pipe_fryum | 0.9946 |

---
## Trial 9 — PRUNED ✂️   `2026-03-08 16:03:23`
**Name**: `IM448_DINOv2B14reg_L7_P010_D768-768_PS-1_AN-3_S0`

**Stopped after**: 3 class(es) (pruned at `cashew`)
**Running mean**: 0.9361

**Pruning criterion** (MedianPruner — running mean vs completed trials at same step):

| Metric | Value |
|--------|------:|
| Running mean (this trial) | 0.9361 |
| Median @ step 3     | 0.9510 |
| Mean @ step 3       | 0.9482 |
| Best @ step 3       | 0.9593 |
| Worst @ step 3      | 0.9288 |
| Δ vs median                | -0.0149 (-1.6%) |
| Reference trials           | 8 |

> Running mean **0.9361** is below the median **0.9510** by 0.0149 (1.6%) → pruned.

**Partial results:**

| Class | AUROC |
|-------|------:|
| candle | 0.9455 |
| capsules | 0.9549 |
| cashew | 0.9078 |

---
## Trial 10 — COMPLETE ✅  `2026-03-08 16:47:26`
**Name**: `IM448_DINOv2B14reg_L5_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9732** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9513 |
| capsules | 0.9864 |
| cashew | 0.9326 |
| chewinggum | 0.9753 |
| fryum | 0.9735 |
| macaroni1 | 0.9604 |
| macaroni2 | 0.9900 |
| pcb1 | 0.9707 |
| pcb2 | 0.9653 |
| pcb3 | 0.9951 |
| pcb4 | 0.9810 |
| pipe_fryum | 0.9968 |

---
## Trial 11 — COMPLETE ✅  `2026-03-08 17:31:31`
**Name**: `IM448_DINOv2B14reg_L5_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9732** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9513 |
| capsules | 0.9864 |
| cashew | 0.9326 |
| chewinggum | 0.9753 |
| fryum | 0.9735 |
| macaroni1 | 0.9604 |
| macaroni2 | 0.9900 |
| pcb1 | 0.9707 |
| pcb2 | 0.9653 |
| pcb3 | 0.9951 |
| pcb4 | 0.9810 |
| pipe_fryum | 0.9968 |

---
## Trial 12 — COMPLETE ✅  `2026-03-08 18:15:47`
**Name**: `IM448_DINOv2B14reg_L5_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9732** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9513 |
| capsules | 0.9864 |
| cashew | 0.9326 |
| chewinggum | 0.9753 |
| fryum | 0.9735 |
| macaroni1 | 0.9604 |
| macaroni2 | 0.9900 |
| pcb1 | 0.9707 |
| pcb2 | 0.9653 |
| pcb3 | 0.9951 |
| pcb4 | 0.9810 |
| pipe_fryum | 0.9968 |

---
## Trial 13 — COMPLETE ✅  `2026-03-08 18:59:58`
**Name**: `IM448_DINOv2B14reg_L5_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9732** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9513 |
| capsules | 0.9864 |
| cashew | 0.9326 |
| chewinggum | 0.9753 |
| fryum | 0.9735 |
| macaroni1 | 0.9604 |
| macaroni2 | 0.9900 |
| pcb1 | 0.9707 |
| pcb2 | 0.9653 |
| pcb3 | 0.9951 |
| pcb4 | 0.9810 |
| pipe_fryum | 0.9968 |

---
## Trial 14 — PRUNED ✂️   `2026-03-08 19:37:26`
**Name**: `IM448_DINOv2B14reg_L4_P010_D768-768_PS-1_AN-3_S0`

**Stopped after**: 10 class(es) (pruned at `pcb3`)
**Running mean**: 0.9507

**Pruning criterion** (MedianPruner — running mean vs completed trials at same step):

| Metric | Value |
|--------|------:|
| Running mean (this trial) | 0.9507 |
| Median @ step 10     | 0.9688 |
| Mean @ step 10       | 0.9666 |
| Best @ step 10       | 0.9701 |
| Worst @ step 10      | 0.9586 |
| Δ vs median                | -0.0180 (-1.9%) |
| Reference trials           | 12 |

> Running mean **0.9507** is below the median **0.9688** by 0.0180 (1.9%) → pruned.

**Partial results:**

| Class | AUROC |
|-------|------:|
| candle | 0.9397 |
| capsules | 0.9972 |
| cashew | 0.7766 |
| chewinggum | 0.9250 |
| fryum | 0.9723 |
| macaroni1 | 0.9643 |
| macaroni2 | 0.9972 |
| pcb1 | 0.9676 |
| pcb2 | 0.9761 |
| pcb3 | 0.9914 |

---
## Trial 15 — COMPLETE ✅  `2026-03-08 20:21:48`
**Name**: `IM448_DINOv2B14reg_L5_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9732** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9513 |
| capsules | 0.9864 |
| cashew | 0.9326 |
| chewinggum | 0.9753 |
| fryum | 0.9735 |
| macaroni1 | 0.9604 |
| macaroni2 | 0.9900 |
| pcb1 | 0.9707 |
| pcb2 | 0.9653 |
| pcb3 | 0.9951 |
| pcb4 | 0.9810 |
| pipe_fryum | 0.9968 |

---
## Trial 16 — COMPLETE ✅  `2026-03-08 21:06:20`
**Name**: `IM448_DINOv2B14reg_L5_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9732** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9513 |
| capsules | 0.9864 |
| cashew | 0.9326 |
| chewinggum | 0.9753 |
| fryum | 0.9735 |
| macaroni1 | 0.9604 |
| macaroni2 | 0.9900 |
| pcb1 | 0.9707 |
| pcb2 | 0.9653 |
| pcb3 | 0.9951 |
| pcb4 | 0.9810 |
| pipe_fryum | 0.9968 |

---
## Trial 17 — PRUNED ✂️   `2026-03-08 21:15:38`
**Name**: `IM448_DINOv2B14reg_L7_P010_D768-768_PS-1_AN-3_S0`

**Stopped after**: 3 class(es) (pruned at `cashew`)
**Running mean**: 0.9361

**Pruning criterion** (MedianPruner — running mean vs completed trials at same step):

| Metric | Value |
|--------|------:|
| Running mean (this trial) | 0.9361 |
| Median @ step 3     | 0.9568 |
| Mean @ step 3       | 0.9519 |
| Best @ step 3       | 0.9593 |
| Worst @ step 3      | 0.9288 |
| Δ vs median                | -0.0207 (-2.2%) |
| Reference trials           | 14 |

> Running mean **0.9361** is below the median **0.9568** by 0.0207 (2.2%) → pruned.

**Partial results:**

| Class | AUROC |
|-------|------:|
| candle | 0.9455 |
| capsules | 0.9549 |
| cashew | 0.9078 |

---
## Trial 18 — COMPLETE ✅  `2026-03-08 22:00:35`
**Name**: `IM448_DINOv2B14reg_L3_P010_D768-768_PS-1_AN-3_S0`

**Mean AUROC**: **0.9716** (12 classes)

| Class | AUROC |
|-------|------:|
| candle | 0.9609 |
| capsules | 0.9851 |
| cashew | 0.9047 |
| chewinggum | 0.9763 |
| fryum | 0.9735 |
| macaroni1 | 0.9833 |
| macaroni2 | 0.9825 |
| pcb1 | 0.9700 |
| pcb2 | 0.9668 |
| pcb3 | 0.9935 |
| pcb4 | 0.9691 |
| pipe_fryum | 0.9940 |

