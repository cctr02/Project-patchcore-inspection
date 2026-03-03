"""
Unit tests for VisADataset.

Run with:
    python -m pytest contribution/test_visa.py --data_path /path/to/VisA -v

The fixture `data_path` is provided by contribution/conftest.py.
A single class is auto-detected from the available subdirectories,
so the tests work regardless of which VisA categories are present.

CSV format reminder (no split column):
    image                                   label                               mask
    candle/Data/Images/Normal/0000.JPG      normal                              (empty)
    candle/Data/Images/Anomaly/001.JPG      wax melded out of the candle        candle/Data/Masks/Anomaly/001.png

Split logic:
    TRAIN : first train_val_split fraction of normal rows  (default 1.0 = all)
    VAL   : remaining normal rows
    TEST  : anomaly rows + held-out normal rows (if train_val_split < 1.0)
"""

import os

import pytest
import torch

from contribution.visa import DatasetSplit, VisADataset, _CLASSNAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESIZE      = 256
IMAGESIZE   = 224
RATIO_80_20 = 0.8   # used for split tests that need both normal and anomaly in TEST


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_available_class(data_path):
    """Return the first _CLASSNAMES entry that actually exists on disk."""
    for name in _CLASSNAMES:
        if os.path.isdir(os.path.join(data_path, name)):
            return name
    pytest.skip(
        f"No VisA class folder found under {data_path}. "
        f"Expected one of: {_CLASSNAMES}"
    )


def _make_dataset(data_path, split, classname=None, train_val_split=1.0):
    if classname is None:
        classname = _first_available_class(data_path)
    return VisADataset(
        source=data_path,
        classname=classname,
        resize=RESIZE,
        imagesize=IMAGESIZE,
        split=split,
        train_val_split=train_val_split,
    )


# ---------------------------------------------------------------------------
# 1. CSV loading and data_to_iterate structure
# ---------------------------------------------------------------------------

class TestCSVLoading:

    def test_train_dataset_is_non_empty(self, data_path):
        ds = _make_dataset(data_path, DatasetSplit.TRAIN)
        assert len(ds) > 0, "Training split returned 0 samples."

    def test_test_dataset_is_non_empty(self, data_path):
        # With default train_val_split=1.0, TEST contains only anomaly images.
        ds = _make_dataset(data_path, DatasetSplit.TEST)
        assert len(ds) > 0, "Test split returned 0 samples (no anomaly images found)."

    def test_data_to_iterate_has_four_fields(self, data_path):
        ds = _make_dataset(data_path, DatasetSplit.TRAIN)
        for entry in ds.data_to_iterate:
            assert len(entry) == 4, (
                f"data_to_iterate entry must have 4 fields "
                f"[classname, anomaly, img_path, mask_path], got {len(entry)}."
            )

    def test_image_files_exist_on_disk(self, data_path):
        ds = _make_dataset(data_path, DatasetSplit.TRAIN)
        for _, _, img_path, _ in ds.data_to_iterate:
            assert os.path.isfile(img_path), f"Image not found on disk: {img_path}"

    def test_test_image_files_exist_on_disk(self, data_path):
        ds = _make_dataset(data_path, DatasetSplit.TEST)
        for _, _, img_path, _ in ds.data_to_iterate:
            assert os.path.isfile(img_path), f"Test image not found on disk: {img_path}"


# ---------------------------------------------------------------------------
# 2. Anomaly label convention (MVTec compatibility)
# ---------------------------------------------------------------------------

class TestAnomalyLabels:

    def test_train_contains_only_good_labels(self, data_path):
        """Training images must all be defect-free (label == 'good')."""
        ds = _make_dataset(data_path, DatasetSplit.TRAIN)
        labels = {entry[1] for entry in ds.data_to_iterate}
        assert labels == {"good"}, (
            f"Training split should only contain 'good' labels, found: {labels}"
        )

    def test_test_contains_anomaly_labels(self, data_path):
        """Test split must always include anomalous samples."""
        ds = _make_dataset(data_path, DatasetSplit.TEST)
        labels = {entry[1] for entry in ds.data_to_iterate}
        assert "anomaly" in labels, "Test split has no 'anomaly' samples."

    def test_test_contains_good_labels_when_split_is_below_one(self, data_path):
        """When train_val_split < 1.0, held-out normal images appear in TEST as 'good'."""
        ds = _make_dataset(data_path, DatasetSplit.TEST, train_val_split=RATIO_80_20)
        labels = {entry[1] for entry in ds.data_to_iterate}
        assert "good" in labels, (
            "With train_val_split=0.8, TEST should include 'good' (normal) samples."
        )

    def test_is_anomaly_flag_matches_label(self, data_path):
        """is_anomaly must be 0 for 'good' and 1 for 'anomaly'."""
        ds = _make_dataset(data_path, DatasetSplit.TEST)
        for i in range(min(len(ds), 20)):
            sample = ds[i]
            expected = int(sample["anomaly"] != "good")
            assert sample["is_anomaly"] == expected, (
                f"is_anomaly mismatch at index {i}: "
                f"anomaly='{sample['anomaly']}' but is_anomaly={sample['is_anomaly']}"
            )

    def test_anomaly_label_is_normalized_to_string_anomaly(self, data_path):
        """Free-text descriptions in the CSV must be mapped to the string 'anomaly'."""
        ds = _make_dataset(data_path, DatasetSplit.TEST)
        anomaly_labels = {entry[1] for entry in ds.data_to_iterate if entry[1] != "good"}
        assert anomaly_labels == {"anomaly"}, (
            f"All non-normal entries should be labelled 'anomaly', got: {anomaly_labels}"
        )


# ---------------------------------------------------------------------------
# 3. Mask handling
# ---------------------------------------------------------------------------

class TestMasks:

    def test_test_anomaly_samples_have_mask_path(self, data_path):
        """Anomalous test entries must carry a non-None mask_path."""
        ds = _make_dataset(data_path, DatasetSplit.TEST)
        for classname, anomaly, _, mask_path in ds.data_to_iterate:
            if anomaly == "anomaly":
                assert mask_path is not None, (
                    f"Anomalous test sample in class '{classname}' has no mask_path."
                )

    def test_train_samples_have_no_mask_path(self, data_path):
        """Training entries must never carry a mask_path."""
        ds = _make_dataset(data_path, DatasetSplit.TRAIN)
        for _, _, _, mask_path in ds.data_to_iterate:
            assert mask_path is None, "Train samples should not have a mask_path."

    def test_normal_getitem_returns_zero_mask(self, data_path):
        """__getitem__ must return an all-zero mask for 'good' test samples."""
        ds = _make_dataset(data_path, DatasetSplit.TEST, train_val_split=RATIO_80_20)
        for i in range(len(ds)):
            if ds.data_to_iterate[i][1] == "good":
                sample = ds[i]
                assert sample["mask"].sum() == 0.0, (
                    "Zero mask expected for a normal test sample."
                )
                break

    def test_anomaly_mask_files_exist_on_disk(self, data_path):
        """Every non-None mask_path must point to an existing file."""
        ds = _make_dataset(data_path, DatasetSplit.TEST)
        for _, anomaly, _, mask_path in ds.data_to_iterate:
            if anomaly == "anomaly":
                assert mask_path is not None
                assert os.path.isfile(mask_path), (
                    f"Mask file not found on disk: {mask_path}"
                )


# ---------------------------------------------------------------------------
# 4. __getitem__ output format (shapes, types, keys)
# ---------------------------------------------------------------------------

class TestGetItem:

    def test_output_keys(self, data_path):
        ds = _make_dataset(data_path, DatasetSplit.TRAIN)
        sample = ds[0]
        expected_keys = {"image", "mask", "classname", "anomaly",
                         "is_anomaly", "image_name", "image_path"}
        assert expected_keys == set(sample.keys()), (
            f"Missing or extra keys.\nExpected: {expected_keys}\nGot: {set(sample.keys())}"
        )

    def test_image_tensor_shape(self, data_path):
        ds = _make_dataset(data_path, DatasetSplit.TRAIN)
        sample = ds[0]
        assert sample["image"].shape == (3, IMAGESIZE, IMAGESIZE), (
            f"Image shape mismatch: {sample['image'].shape}"
        )

    def test_mask_tensor_shape(self, data_path):
        ds = _make_dataset(data_path, DatasetSplit.TRAIN)
        sample = ds[0]
        assert sample["mask"].shape == (1, IMAGESIZE, IMAGESIZE), (
            f"Mask shape mismatch: {sample['mask'].shape}"
        )

    def test_image_is_float_tensor(self, data_path):
        ds = _make_dataset(data_path, DatasetSplit.TRAIN)
        sample = ds[0]
        assert sample["image"].dtype == torch.float32, (
            f"Expected float32 image, got {sample['image'].dtype}"
        )

    def test_anomaly_test_sample_has_non_zero_mask(self, data_path):
        """An anomalous test sample should have a mask with at least one non-zero pixel."""
        ds = _make_dataset(data_path, DatasetSplit.TEST)
        for i in range(len(ds)):
            if ds.data_to_iterate[i][1] == "anomaly":
                sample = ds[i]
                assert sample["mask"].sum() > 0, (
                    f"Expected non-zero mask for anomalous sample at index {i}."
                )
                break


# ---------------------------------------------------------------------------
# 5. Dataset-level attributes consumed by run_patchcore.py
# ---------------------------------------------------------------------------

class TestDatasetAttributes:

    def test_imagesize_attribute(self, data_path):
        ds = _make_dataset(data_path, DatasetSplit.TRAIN)
        assert ds.imagesize == (3, IMAGESIZE, IMAGESIZE), (
            f"imagesize attribute mismatch: {ds.imagesize}"
        )

    def test_transform_mean_std_exposed(self, data_path):
        ds = _make_dataset(data_path, DatasetSplit.TRAIN)
        assert hasattr(ds, "transform_mean"), "Missing attribute: transform_mean"
        assert hasattr(ds, "transform_std"),  "Missing attribute: transform_std"
        assert len(ds.transform_mean) == 3
        assert len(ds.transform_std)  == 3

    def test_imgpaths_per_class_keys(self, data_path):
        classname = _first_available_class(data_path)
        ds = _make_dataset(data_path, DatasetSplit.TRAIN, classname=classname)
        assert classname in ds.imgpaths_per_class
        assert "good"    in ds.imgpaths_per_class[classname]
        assert "anomaly" in ds.imgpaths_per_class[classname]


# ---------------------------------------------------------------------------
# 6. Train / validation / test split consistency
# ---------------------------------------------------------------------------

class TestSplitConsistency:

    def test_train_and_val_are_disjoint(self, data_path):
        ds_train = _make_dataset(data_path, DatasetSplit.TRAIN, train_val_split=RATIO_80_20)
        ds_val   = _make_dataset(data_path, DatasetSplit.VAL,   train_val_split=RATIO_80_20)
        train_paths = {entry[2] for entry in ds_train.data_to_iterate}
        val_paths   = {entry[2] for entry in ds_val.data_to_iterate}
        overlap = train_paths & val_paths
        assert len(overlap) == 0, (
            f"Train and val share {len(overlap)} image(s)."
        )

    def test_train_plus_val_equals_all_normal(self, data_path):
        """train + val must cover exactly all normal images, no more, no less."""
        ds_all   = _make_dataset(data_path, DatasetSplit.TRAIN, train_val_split=1.0)
        ds_train = _make_dataset(data_path, DatasetSplit.TRAIN, train_val_split=RATIO_80_20)
        ds_val   = _make_dataset(data_path, DatasetSplit.VAL,   train_val_split=RATIO_80_20)
        assert len(ds_train) + len(ds_val) == len(ds_all), (
            f"train ({len(ds_train)}) + val ({len(ds_val)}) "
            f"!= all normal ({len(ds_all)})"
        )

    def test_anomaly_images_always_in_test(self, data_path):
        """Anomaly images must never appear in TRAIN or VAL."""
        for split in (DatasetSplit.TRAIN, DatasetSplit.VAL):
            ds = _make_dataset(data_path, split, train_val_split=RATIO_80_20)
            for _, anomaly, _, _ in ds.data_to_iterate:
                assert anomaly == "good", (
                    f"Anomaly image found in {split.value} split."
                )

    def test_default_split_1_gives_no_good_in_test(self, data_path):
        """With train_val_split=1.0 (default), TEST has no 'good' samples."""
        ds = _make_dataset(data_path, DatasetSplit.TEST, train_val_split=1.0)
        good_count = sum(1 for entry in ds.data_to_iterate if entry[1] == "good")
        assert good_count == 0, (
            f"Expected 0 'good' samples in TEST with train_val_split=1.0, "
            f"got {good_count}."
        )
