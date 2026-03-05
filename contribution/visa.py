"""
VisA (Visual Anomaly) dataset loader for PatchCore.

Expected on-disk structure:
    <source>/                        <- VisA root  (e.g. VisA_20220922/)
    └── <classname>/                 <- e.g. candle/
        ├── Data/
        │   ├── Images/
        │   │   ├── Normal/          <- defect-free images
        │   │   └── Anomaly/         <- defective images
        │   └── Masks/
        │       └── Anomaly/         <- binary anomaly masks
        └── image_anno.csv

Actual CSV format (no split column):
    image                                   label                             mask
    candle/Data/Images/Normal/0000.JPG      normal                            (empty)
    candle/Data/Images/Anomaly/001.JPG      wax melded out of the candle      candle/Data/Masks/Anomaly/001.png

    - Paths are relative to <source> (the VisA root), not to the class folder.
    - A normal image has label == "normal".
    - An anomalous image has a free-text description as label.

Split derivation (no explicit column in the CSV):
    TRAIN : first `train_val_split` fraction of normal images  (default: all)
    VAL   : remaining normal images  (if train_val_split < 1.0)
    TEST  : all anomaly images  +  normal images not used in TRAIN
"""

import csv
import os
from enum import Enum

import numpy as np
import PIL.Image
import torch
from torchvision import transforms

# The 12 categories of the VisA dataset
_CLASSNAMES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

# Standard ImageNet statistics (same as MVTec for consistency)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL   = "val"
    TEST  = "test"


class VisADataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for VisA with the same interface as MVTecDataset.

    __getitem__ returns a dict identical to MVTecDataset:
        {
            "image"      : FloatTensor [3, H, W]  ImageNet-normalised,
            "mask"       : FloatTensor [1, H, W]  (zeros when no mask),
            "classname"  : str,
            "anomaly"    : str  ("good" | "anomaly"),
            "is_anomaly" : int  (0 | 1),
            "image_name" : str,
            "image_path" : str,
        }

    `data_to_iterate` is the flat list consumed by run_patchcore.py:
        [classname, anomaly_label, image_path, mask_path]
    where anomaly_label == "good" for defect-free images (MVTec convention).
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        log_dir=None,
        **kwargs,
    ):
        """
        Args:
            source          : Path to the VisA root folder
                              (the folder that contains candle/, capsules/, ...).
            classname       : Class to load (e.g. "candle"), or None to iterate
                              over all classes.
            resize          : Initial square resize size.
            imagesize       : Final center-crop size.
            split           : DatasetSplit.TRAIN | VAL | TEST.
            train_val_split : Fraction of normal images used for training.
                              The remainder goes to VAL and TEST as "good" samples.
                              Default 1.0 → all normal images in TRAIN,
                              TEST contains only anomalous images.
            log_dir         : If provided, discarded-image entries are appended to
                              <log_dir>/skipped_images.log in addition to stdout.
        """
        super().__init__()
        self.source          = source
        self.split           = split
        self.train_val_split = train_val_split

        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES

        # Stored for crop-aware filtering in _load_from_csv
        self._resize     = resize
        self._imagesize  = imagesize
        self._log_path   = os.path.join(log_dir, "skipped_images.log") if log_dir else None

        self.imgpaths_per_class, self.data_to_iterate = self._load_from_csv()

        # Image transform (identical to MVTec)
        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Mask transform (no normalisation)
        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        # Exposed for run_patchcore.py (save_segmentation_images block)
        self.transform_mean = IMAGENET_MEAN
        self.transform_std  = IMAGENET_STD

        self.imagesize = (3, imagesize, imagesize)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]

        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask > 0).float()  # VisA masks use 0/1 values, not 0/255
        else:
            mask = torch.zeros([1, *image.shape[1:]])

        return {
            "image"      : image,
            "mask"       : mask,
            "classname"  : classname,
            "anomaly"    : anomaly,
            "is_anomaly" : int(anomaly != "good"),
            "image_name" : "/".join(image_path.replace("\\", "/").split("/")[-4:]),
            "image_path" : image_path,
        }

    # ------------------------------------------------------------------
    # CSV parsing and data_to_iterate construction
    # ------------------------------------------------------------------

    def _load_from_csv(self):
        """
        Reads image_anno.csv for each class and builds:
          - imgpaths_per_class : dict[classname][anomaly_key] = [paths]
          - data_to_iterate   : list of [classname, anomaly_key, img_path, mask_path]

        anomaly_key is "good" for normal images (MVTec convention).
        All paths in the CSV are relative to `self.source` (the VisA root).
        """
        imgpaths_per_class = {}
        data_to_iterate    = []

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname)
            csv_path  = os.path.join(classpath, "image_anno.csv")

            if not os.path.isfile(csv_path):
                raise FileNotFoundError(
                    f"Annotation CSV not found: {csv_path}\n"
                    f"Make sure `source` points to the VisA root directory."
                )

            with open(csv_path, newline="") as f:
                all_rows = list(csv.DictReader(f))

            # Separate normal and anomaly rows based on label text
            # (anomaly label is a free-text description, not the word "anomaly")
            normal_rows  = sorted(
                [r for r in all_rows if r["label"].strip().lower() == "normal"],
                key=lambda r: r["image"],
            )
            anomaly_rows = sorted(
                [r for r in all_rows if r["label"].strip().lower() != "normal"],
                key=lambda r: r["image"],
            )

            rows_for_split = self._select_rows(normal_rows, anomaly_rows)

            imgpaths_per_class[classname] = {"good": [], "anomaly": []}

            for row in rows_for_split:
                is_normal   = row["label"].strip().lower() == "normal"
                anomaly_key = "good" if is_normal else "anomaly"

                # Paths in the CSV are relative to the VisA root, not the class folder
                abs_img  = os.path.join(self.source, row["image"].strip())
                rel_mask = row.get("mask", "").strip()
                abs_mask = os.path.join(self.source, rel_mask) if rel_mask else None

                imgpaths_per_class[classname][anomaly_key].append(abs_img)

                # For anomalous TEST images: discard if any anomaly pixel
                # falls outside the CenterCrop region (model never sees it).
                if (self.split == DatasetSplit.TEST
                        and anomaly_key == "anomaly"
                        and abs_mask is not None
                        and not self._anomaly_in_crop(abs_mask)):
                    img_name = os.path.basename(row["image"].strip())
                    msg = f"[DISCARD] {classname}/{img_name} — anomaly pixel outside crop"
                    print(msg)
                    if self._log_path:
                        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
                        with open(self._log_path, "a") as lf:
                            lf.write(msg + "\n")
                    continue

                # Masks are only attached for anomalous images in the test split
                mask_to_store = (
                    abs_mask
                    if (self.split == DatasetSplit.TEST and anomaly_key == "anomaly")
                    else None
                )

                data_to_iterate.append([classname, anomaly_key, abs_img, mask_to_store])

        return imgpaths_per_class, data_to_iterate

    def _anomaly_in_crop(self, mask_path):
        """
        Returns True only if every non-zero pixel in the mask falls inside
        the CenterCrop region (i.e. the model actually sees the anomaly).

        Geometry: Resize(resize) scales the shorter edge to `resize` px,
        then CenterCrop(imagesize) takes a square from the centre.
        """
        mask = PIL.Image.open(mask_path).convert("L")
        w, h = mask.size          # PIL: (width, height)
        arr  = np.array(mask)     # shape (h, w), row=y, col=x

        ys, xs = np.nonzero(arr)
        if len(ys) == 0:
            return True           # mask empty → nothing to discard

        scale    = self._resize / min(w, h)
        sw, sh   = w * scale, h * scale
        x0 = (sw - self._imagesize) / (2 * scale)
        y0 = (sh - self._imagesize) / (2 * scale)
        x1 = x0 + self._imagesize / scale
        y1 = y0 + self._imagesize / scale

        return (xs.min() >= x0 and xs.max() < x1 and
                ys.min() >= y0 and ys.max() < y1)

    def _select_rows(self, normal_rows, anomaly_rows):
        """
        Returns the subset of rows for the requested split.

        A fixed 20 % of normal images is always reserved for TEST evaluation
        so that AUROC can be computed regardless of train_val_split.

        TRAIN : first train_val_split fraction of the 80 % train+val pool
        VAL   : remaining fraction of the 80 % pool  (if train_val_split < 1.0)
        TEST  : the fixed 20 % normal hold-out  +  all anomaly images
        """
        n = len(normal_rows)

        # Fixed 20 % hold-out for TEST (always, irrespective of train_val_split)
        test_start   = int(n * 0.8)
        test_normal  = normal_rows[test_start:]
        trainval     = normal_rows[:test_start]

        # Within the train+val pool, apply train_val_split
        tv_n      = len(trainval)
        split_idx = tv_n if self.train_val_split >= 1.0 else int(tv_n * self.train_val_split)

        if self.split == DatasetSplit.TRAIN:
            return trainval[:split_idx]

        if self.split == DatasetSplit.VAL:
            return trainval[split_idx:]

        # TEST: fixed normal hold-out + all anomaly images
        return test_normal + anomaly_rows
