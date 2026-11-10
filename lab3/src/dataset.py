import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf
from keras.src.utils import image_dataset_from_directory


def image_dataframe_from_directory(directory: str | Path) -> pd.DataFrame:
    img_files: list[Path] = [img for img in Path(directory).glob("*/*") if img.is_file()]
    df: pd.DataFrame = pd.DataFrame({
        "name": [img.name for img in img_files],
        "class": [img.parent.name for img in img_files]
    })
    return df


def get_data_loaders(
        directory_path: str | Path, seed: int, image_size: tuple, batch_size: int, validation_split: float
) -> tuple[Any, Any]:
    train_ds, val_ds = image_dataset_from_directory(
        directory=directory_path,
        labels="inferred",
        label_mode="binary",
        validation_split=validation_split,
        subset="both",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def clear_dataset(directory: str | Path) -> None:
    corrupted_img_dir = Path("corrupted_images")
    corrupted_img_dir.mkdir(exist_ok=True)

    for path in directory.glob("*/*"):
        if path.suffix.lower() in [".jpg", ".jpeg", "png"]:
            try:
                raw = tf.io.read_file(str(path))
                tf.io.decode_image(raw, channels=3)
            except Exception as e:
                print("Found corrupted image:", e)
                shutil.move(str(path), str(corrupted_img_dir / path.name))
