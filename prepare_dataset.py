from pathlib import Path
import shutil

source = Path(r"E:\PROJECT RELATED MATERIAL ALL THE PROJECT OTHER THEN MENTION IN CV\OFFICE PROJECT\image downloaded from kaggle\archive\1000images")

destination = Path("retina_training_dataset")

categories = {
    "Grade_0": "0.0.Normal",
    "Grade_1": "0.3.DR1",
    "Grade_2": "1.0.DR2",
    "Grade_3": "1.1.DR3"
}

for grade, folder in categories.items():
    target = destination / grade
    target.mkdir(parents=True, exist_ok=True)

    for image in (source / folder).iterdir():
        if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            shutil.copy2(image, target / image.name)

    print(grade, ":", len(list(target.iterdir())), "images")

print("Dataset preparation completed.")
from datasets import load_dataset

ds = load_dataset(
    "bumbledeep/aptos",
    split="train",
    cache_dir="./APTOS_cache"
)

counts = {0: 0, 1: 0, 2: 0, 3: 0}

for i, row in enumerate(ds):
    grade = min(int(row["label_code"]), 3)

    target = Path("retina_training_dataset") / f"Grade_{grade}"
    target.mkdir(parents=True, exist_ok=True)

    image = row["image"].convert("RGB")
    image.save(target / f"APTOS_{i}.png")

    counts[grade] += 1

print("\nAPTOS images added:")

for grade, count in counts.items():
    print(f"Grade_{grade}: {count}")

print("\nCombined dataset:")

for folder in sorted(Path("retina_training_dataset").iterdir()):
    print(folder.name, len(list(folder.iterdir())))