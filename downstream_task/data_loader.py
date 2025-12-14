# coding=utf-8

from collections import defaultdict

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class DatasetPreparation(Dataset):
    def __init__(self, args, df, iaa_transform=None, transform=None):
        """
        args.concepts should be a list of column names in df, e.g.
        [
            "No_Finding",
            "Mass",
            "Asymmetry",
            "Focal_Asymmetry",
            "Global_Asymmetry",
            "Architectural_Distortion",
            "Suspicious_Calcification",
        ]
        """
        self.args = args
        self.dir_path = args.data_dir / args.img_dir
        self.annotations = df
        self.labels_list = args.concepts
        self.iaa_transform = iaa_transform
        self.transform = transform
        self.mean = args.mean
        self.std = args.std

        # Build mapping from (study_id, image_id) -> list of boxes and labels
        self.image_dict = self._generate_image_dict()
        # Fixed ordering of keys for stable indexing
        self.keys = list(self.image_dict.keys())

    def _generate_image_dict(self):
        image_dict = defaultdict(lambda: {"boxes": [], "labels": []})

        for _, row in self.annotations.iterrows():
            study_id = row["patient_id"]
            image_id = row["image_id"]

            # Box coordinates (already resized)
            boxes = row[
                ["resized_xmin", "resized_ymin", "resized_xmax", "resized_ymax"]
            ].values.tolist()

            # For each concept column, if it's 1, add a box with that class index
            # EXCEPT for "No_Finding" – that should NOT produce any box.
            for class_idx, col_name in enumerate(self.labels_list):
                val = row.get(col_name, 0)
                try:
                    val = int(val)
                except Exception:
                    val = 0

                if col_name == "No_Finding":
                    # We treat No_Finding as an image-level category only.
                    # Do NOT create a bounding box for it.
                    continue

                if val == 1:
                    # Add one box for this class
                    image_dict[(study_id, image_id)]["boxes"].append(
                        boxes + [class_idx]
                    )
                    image_dict[(study_id, image_id)]["labels"].append(class_idx)

        return image_dict

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.get_items(idx)

    def get_items(self, idx):
        study_id, image_id = self.keys[idx]
        boxes = self.image_dict[(study_id, image_id)]["boxes"]
        labels = self.image_dict[(study_id, image_id)]["labels"]
        path = f"{self.dir_path}/{study_id}/{image_id}"

        # Load image (grayscale -> RGB -> numpy)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image).convert("RGB")
        image = np.array(image)

        # Apply imgaug transforms to boxes + image
        if self.iaa_transform and len(boxes) > 0:
            bb_box = []
            for bb in boxes:
                bb_box.append(
                    BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3])
                )

            bbs_on_image = BoundingBoxesOnImage(bb_box, shape=image.shape)
            image, boxes_aug = self.iaa_transform(
                image=image, bounding_boxes=[bbs_on_image]
            )
        else:
            # No augmentation or no boxes; wrap an empty container
            boxes_aug = [BoundingBoxesOnImage([], shape=image.shape)]

        # torchvision transforms (ToTensor)
        if self.transform:
            image = self.transform(image)

        image = image.to(torch.float32)
        image -= image.min()
        max_val = image.max()
        if max_val > 0:
            image /= max_val
        image = torch.tensor((image - self.mean) / self.std, dtype=torch.float32)

        # Build [x1, y1, x2, y2, class] array
        bb_final = []
        for i, bb in enumerate(boxes_aug[0]):
            bb_final.append([bb.x1, bb.y1, bb.x2, bb.y2, labels[i]])

        if len(bb_final) == 0:
            # If no boxes, create a dummy row with -1 label
            bb_final = [[0, 0, 0, 0, -1]]

        target = {
            "boxes": torch.tensor(bb_final, dtype=torch.float32),
            "labels": labels,  # list is fine; unused in your training loop
        }

        return {
            "image": image,
            "target": target,
            "study_id": study_id,
            "image_id": image_id,
            "img_path": path,
        }


def dataset_loader(data):
    image = [s["image"] for s in data]
    res_bbox_tensor = [s["target"]["boxes"] for s in data]
    image_path = [s["img_path"] for s in data]

    max_num_annots = max(annot.shape[0] for annot in res_bbox_tensor)
    if max_num_annots > 0:
        annot_padded = torch.ones(
            (len(res_bbox_tensor), max_num_annots, 5), dtype=torch.float32
        ) * -1

        for idx, annot in enumerate(res_bbox_tensor):
            if annot.shape[0] > 0:
                annot_padded[idx, : annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(res_bbox_tensor), 1, 5), dtype=torch.float32) * -1

    return {
        "image": torch.stack(image),
        "res_bbox_tensor": annot_padded,
        "image_path": image_path,
    }


def get_transforms(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    # No resize → preserve original VinDr dimensions (1520×912)
    train_affine_trans = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(
            rotate=(-10, 10),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            shear=(-5, 5),
        ),
    ])

    # Validation: identity transform (no resize, no crop)
    test_affine_trans = iaa.Noop()

    return transform, train_affine_trans, test_affine_trans


def get_dataloader(args, train=True):
    transform, train_affine_trans, test_affine_trans = get_transforms(args)

    # -------- Validation dataset / loader --------
    valid_dataset = DatasetPreparation(
        args=args,
        df=args.valid_folds,
        iaa_transform=test_affine_trans,
        transform=transform,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset_loader,
    )

    if train:
        # -------- Training dataset --------
        train_dataset = DatasetPreparation(
            args=args,
            df=args.train_folds,
            iaa_transform=train_affine_trans,
            transform=transform,
        )

        # ====== IMBALANCE HANDLING: WeightedRandomSampler (automatic) ======
        num_classes = len(args.concepts)

        # 1) Count how many images contain each class (image-level)
        class_counts = np.zeros(num_classes, dtype=np.float32)
        for key in train_dataset.keys:
            lbls = set(train_dataset.image_dict[key]["labels"])  # unique labels per image
            for lbl in lbls:
                if 0 <= lbl < num_classes:
                    class_counts[lbl] += 1

        eps = 1e-6
        # 2) Inverse-frequency class weights: rarer classes get larger weights
        #    Normalize so that the largest weight is 1.0
        class_weights = 1.0 / (class_counts + eps)
        class_weights = class_weights / class_weights.max()

        # 3) Per-image weight = mean of its class weights
        sample_weights = []
        for key in train_dataset.keys:
            lbls = list(set(train_dataset.image_dict[key]["labels"]))

            if len(lbls) == 0:
                # No labels at all (should be rare): give a small constant weight
                sample_weights.append(0.1)
                continue

            w = 0.0
            valid_cnt = 0
            for lbl in lbls:
                if 0 <= lbl < num_classes:
                    w += class_weights[lbl]
                    valid_cnt += 1

            if valid_cnt > 0:
                w = max(class_weights[lbl] for lbl in lbls if 0 <= lbl < num_classes)  # average over labels
            else:
                w = 0.1

            sample_weights.append(w)

        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),  # one "epoch" worth
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,   # use sampler for imbalance
            shuffle=False,     # IMPORTANT: disable shuffle when using sampler
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset_loader,
        )
        return train_loader, valid_loader, valid_dataset
    else:
        return valid_loader, valid_dataset


def get_dataset(args, train=True):
    return get_dataloader(args, train=train)
