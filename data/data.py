"""
Prepare ALL dataset for cancer image classification.

- Splits dataset into train/val/test (70/15/15)
- Creates Keras ImageDataGenerators (rescaling only by default)
"""

import os
import shutil
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
RAW_DATA_DIR = "data/raw/ALL"        # Original dataset path
OUTPUT_DIR = "data/processed"        # Output path for train/val/test
SEED = 123                           # Random seed
SPLITS = (0.7, 0.15, 0.15)           # Train/val/test ratios

# ----------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------
def make_dirs():
    for split in ["train", "val", "test"]:
        split_path = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_path, exist_ok=True)


def split_dataset():
    """
    Split dataset into train/val/test and copy images into folders.
    """
    classes = os.listdir(RAW_DATA_DIR)
    print(f"Found classes: {classes}")

    for cls in classes:
        cls_path = os.path.join(RAW_DATA_DIR, cls)
        images = np.array(os.listdir(cls_path))

        train_imgs, test_imgs = train_test_split(images, test_size=(1 - SPLITS[0]), random_state=SEED)
        val_size = SPLITS[1] / (SPLITS[1] + SPLITS[2])
        val_imgs, test_imgs = train_test_split(test_imgs, test_size=(1 - val_size), random_state=SEED)

        for split_name, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            split_cls_path = os.path.join(OUTPUT_DIR, split_name, cls)
            os.makedirs(split_cls_path, exist_ok=True)

            for img in split_imgs:
                shutil.copy(os.path.join(cls_path, img), os.path.join(split_cls_path, img))

        print(f"Class {cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")


def create_generators(img_size=(224, 224), batch_size=32, augment=False):
    """
    Return Keras ImageDataGenerators for train/val/test.
    By default: only rescales images.
    If augment=True: adds light online augmentation during training.
    """
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    val_gen = test_val_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    test_gen = test_val_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="Split dataset into train/val/test")
    parser.add_argument("--gen", action="store_true", help="Test creating ImageDataGenerators")
    parser.add_argument("--augment", action="store_true", help="Use light augmentations in train generator")
    args = parser.parse_args()

    if args.split:
        make_dirs()
        split_dataset()

    if args.gen:
        train_gen, val_gen, test_gen = create_generators(augment=args.augment)
        print("Data generators created successfully.")
