from pathlib import Path
import shutil
import random

random.seed(42)

source = Path("retina_training_dataset")
destination = Path("retina_split_dataset_v3")

for folder in sorted(source.iterdir()):

    aptos_images = sorted(folder.glob("APTOS_*.png"))

    original_images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith("APTOS_")
    )

    random.shuffle(aptos_images)

    n = len(aptos_images)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    splits = {
        "train": aptos_images[:train_end] + original_images,
        "validation": aptos_images[train_end:val_end],
        "test": aptos_images[val_end:]
    }

    for split, files in splits.items():

        target = destination / split / folder.name
        target.mkdir(parents=True, exist_ok=True)

        for image in files:
            shutil.copy2(image, target / image.name)

        print(folder.name, split, len(files))

print("New dataset splitting completed.")