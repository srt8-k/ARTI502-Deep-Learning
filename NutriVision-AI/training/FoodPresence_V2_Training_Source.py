
# NutriVision AI V2
# Protected, exploratory Hard-Negative Training

import hashlib
import random
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import torch
import timm

from PIL import Image, ImageOps
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from google.colab import drive, files

# 1. Protected workspace
drive.mount("/content/drive")

RUN = Path(
    "/content/drive/MyDrive/NutriVision_AI/"
    "experiments_v2/run_20261119_204646"
)
ROOT = RUN.parent.parent
OUT = RUN / "food_presence_head_v2.pth"
ZIP = RUN / "roboflow_food_detection_source.zip"

assert RUN.is_dir(), "V2 workspace missing"
assert not OUT.exists(), "V2 already exists; refusing overwrite"

cnn_file = ROOT / "convnext_stage3_best.pth"
v1_file = ROOT / "food_presence_head_food5k_v1.pth"

assert cnn_file.is_file()
assert v1_file.is_file()
assert torch.cuda.is_available(), "Enable GPU in Colab"

device = "cuda"
print("GPU:", torch.cuda.get_device_name(0))

# 2. Upload Roboflow source once
if not ZIP.exists():
    print("Upload Food Detection.v1i.yolov8.zip")
    uploaded = files.upload()

    assert len(uploaded) == 1, "Upload exactly one ZIP"
    name, content = next(iter(uploaded.items()))
    assert name.lower().endswith(".zip")

    with ZIP.open("xb") as f:
        f.write(content)

    del uploaded, content

print("ZIP:", ZIP)

# 3. Read segmentation class IDs
with ZipFile(ZIP) as z:
    assert z.testzip() is None, "ZIP is damaged"

    yaml_text = z.read("data.yaml").decode("utf-8")
    assert "'empty plate'" in yaml_text
    assert "'plate with food'" in yaml_text

    label_files = {
        PurePosixPath(n).stem: n
        for n in z.namelist()
        if n.startswith("train/labels/")
        and n.endswith(".txt")
    }

    image_files = [
        n for n in z.namelist()
        if n.startswith("train/images/")
        and n.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    grouped = defaultdict(list)

    for image_name in image_files:
        stem = PurePosixPath(image_name).stem
        assert stem in label_files

        lines = z.read(
            label_files[stem]
        ).decode("utf-8").splitlines()

        # YOLO segmentation: first token = class ID
        ids = {
            int(line.split()[0])
            for line in lines if line.strip()
        }
        assert ids <= {0, 1, 2, 3}

        # Remove Roboflow augmentation fingerprint
        
        # Extract original image name, including WEBP sources
        source = re.sub(
            r"_(?:jpg|jpeg|png|webp)\.rf\.[0-9a-f]+$",
            "",
            stem,
            flags=re.IGNORECASE
        )

        assert source != stem, f"Unknown filename: {stem}"

        grouped[source].append((image_name, ids))


# 4. Explicit and safe binary labels
# 0: empty plate; 1: chicken/plate with food/rice
# Images with no annotations are excluded.

empty_sources = []
food_sources = []
unlabelled = 0

for source, variants in grouped.items():
    ids = set.union(
        set(),
        *(classes for _, classes in variants)
    )

    if not ids:
        unlabelled += 1
        continue

    assert not (
        1 in ids and ids & {0, 2, 3}
    ), f"Ambiguous label: {source}"

    variant_labels = {
        tuple(sorted(classes))
        for _, classes in variants
    }
    assert len(variant_labels) == 1

    if ids == {1}:
        empty_sources.append(source)
    else:
        food_sources.append(source)

assert len(empty_sources) == 15
assert len(food_sources) == 346

print("\nRoboflow original scenes:")
print("Empty plates:", len(empty_sources))
print("Food scenes:", len(food_sources))
print("Unannotated excluded:", unlabelled)

# 5. Grouped split: no augmented source in both sets
rng = random.Random(42)
rng.shuffle(empty_sources)
rng.shuffle(food_sources)

empty_val = set(empty_sources[:3])
empty_train = set(empty_sources[3:])

food_val = set(food_sources[:69])
food_train = set(food_sources[69:])

def make_records(sources, label):
    return [
        (image_name, label)
        for source in sorted(sources)
        for image_name, _ in grouped[source]
    ]

rf_train = (
    make_records(empty_train, 0)
    + make_records(food_train, 1)
)
rf_val = (
    make_records(empty_val, 0)
    + make_records(food_val, 1)
)

assert (empty_train | food_train).isdisjoint(
    empty_val | food_val
)

print("Roboflow train:", len(rf_train))
print("Roboflow validation:", len(rf_val))

# 6. Locate Food-5K
food5k = Path(
    "/root/.cache/kagglehub/datasets/"
    "trolukovich/food5k-image-dataset/versions/1"
)

if not food5k.exists():
    import kagglehub
    food5k = Path(
        kagglehub.dataset_download(
            "trolukovich/food5k-image-dataset"
        )
    )

def split_directory(name):
    matches = [
        p for p in food5k.rglob(name)
        if p.is_dir()
        and (p / "food").is_dir()
        and (p / "non_food").is_dir()
    ]
    assert len(matches) == 1
    return matches[0]

# 7. Load existing CNN and V1 normalization
cnn = torch.load(
    cnn_file, map_location="cpu", weights_only=True
)
v1 = torch.load(
    v1_file, map_location="cpu", weights_only=True
)

assert cnn["model_name"] == v1["backbone_model_name"]

backbone = timm.create_model(
    cnn["model_name"],
    pretrained=False,
    num_classes=len(cnn["classes"])
)
backbone.load_state_dict(
    cnn["model_state_dict"],
    strict=True
)
backbone = backbone.to(device).eval()

for p in backbone.parameters():
    p.requires_grad_(False)

transform = timm.data.create_transform(
    **timm.data.resolve_model_data_config(backbone),
    is_training=False
)

class PresenceDataset(ImageFolder):
    def __getitem__(self, index):
        image, class_index = super().__getitem__(index)
        label = int(self.classes[class_index] == "food")
        return image, label

train_ds = PresenceDataset(
    split_directory("training"), transform=transform
)
val_ds = PresenceDataset(
    split_directory("validation"), transform=transform
)
eval_ds = PresenceDataset(
    split_directory("evaluation"), transform=transform
)

# 8. Reproduce V1 duplicate-cleaning rules
def image_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def hash_map(ds):
    return {
        path: image_hash(path)
        for path, _ in ds.samples
    }

train_h = hash_map(train_ds)
val_h = hash_map(val_ds)
eval_h = hash_map(eval_ds)

def clean(ds, hashes, forbidden):
    kept, seen = [], set()

    for path, label in ds.samples:
        h = hashes[path]
        if h in forbidden or h in seen:
            continue
        kept.append((path, label))
        seen.add(h)

    ds.samples = kept
    ds.imgs = kept
    ds.targets = [y for _, y in kept]

clean(val_ds, val_h, set(eval_h.values()))
clean(
    train_ds,
    train_h,
    set(val_h.values()) | set(eval_h.values())
)

assert len(train_ds) == 2977
assert len(val_ds) == 999

# Exclude exact matches with protected Food5K sets
protected = set(val_h.values()) | set(eval_h.values())

with ZipFile(ZIP) as z:
    before = len(rf_train)
    rf_train = [
        (name, label)
        for name, label in rf_train
        if hashlib.sha256(
            z.read(name)
        ).hexdigest() not in protected
    ]

print(
    "Cross-dataset exact matches excluded:",
    before - len(rf_train)
)

# 9. Read Roboflow images without altering ZIP
class RoboflowDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        name, label = self.records[index]

        with ZipFile(ZIP) as z:
            with Image.open(
                BytesIO(z.read(name))
            ) as image:
                image = ImageOps.exif_transpose(
                    image
                ).convert("RGB")

        return transform(image), label

# 10. Extract frozen ConvNeXt features
# num_workers=0 avoids the previous cleanup errors.
def extract_features(ds, title):
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    xs, ys = [], []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=title):
            images = images.to(device)

            with torch.autocast(
                "cuda",
                dtype=(
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            ):
                features = backbone.forward_features(images)
                features = backbone.forward_head(
                    features, pre_logits=True
                )

            xs.append(features.float().cpu().numpy())
            ys.append(labels.numpy())

    return np.concatenate(xs), np.concatenate(ys)

X_train, y_train = extract_features(
    train_ds, "Food5K Training"
)
X_val, y_val = extract_features(
    val_ds, "Food5K Validation"
)
X_rf_train, y_rf_train = extract_features(
    RoboflowDataset(rf_train),
    "Roboflow Training"
)
X_rf_val, y_rf_val = extract_features(
    RoboflowDataset(rf_val),
    "Roboflow Validation"
)

# 11. Train V2 on V1's feature scale
mean = v1["feature_mean"].numpy()
std = v1["feature_std"].numpy()

def scale(X):
    return (X - mean) / std

X_combined = np.concatenate([X_train, X_rf_train])
y_combined = np.concatenate([y_train, y_rf_train])

# Experimental weighting:
# emphasize hard empty plates while retaining food examples
rf_weights = np.where(y_rf_train == 0, 12.0, 0.5)
sample_weights = np.concatenate([
    np.ones(len(y_train)),
    rf_weights
])

head_v2 = LogisticRegression(
    max_iter=1000,
    random_state=42
)
head_v2.fit(
    scale(X_combined),
    y_combined,
    sample_weight=sample_weights
)

# 12. Compare V1 and V2 on development validation
def v1_probability(X):
    weights = v1["linear_weight"].numpy().ravel()
    bias = v1["linear_bias"].item()
    logits = np.clip(
        scale(X) @ weights + bias, -50, 50
    )
    return 1 / (1 + np.exp(-logits))

def compare(title, X, y):
    print("\n===== " + title + " =====")

    predictions = {
        "V1": v1_probability(X),
        "V2": head_v2.predict_proba(scale(X))[:, 1]
    }

    for version, probability in predictions.items():
        pred = (probability >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()

        accuracy = (tp + tn) / len(y)
        food_recall = tp / max(tp + fn, 1)
        false_accept = fp / max(fp + tn, 1)

        print(
            version,
            f"Accuracy={accuracy:.2%}",
            f"Food Recall={food_recall:.2%}",
            f"False Accept={false_accept:.2%}",
            f"CM={[[int(tn), int(fp)], [int(fn), int(tp)]]}"
        )

compare("FOOD5K VALIDATION", X_val, y_val)
compare("ROBOFLOW GROUPED VALIDATION", X_rf_val, y_rf_val)

# 13. Save V2 independently — no overwrites
assert not OUT.exists()

torch.save({
    "backbone_model_name": cnn["model_name"],
    "feature_mean": v1["feature_mean"].cpu(),
    "feature_std": v1["feature_std"].cpu(),
    "linear_weight": torch.as_tensor(
        head_v2.coef_, dtype=torch.float32
    ),
    "linear_bias": torch.as_tensor(
        head_v2.intercept_, dtype=torch.float32
    ),
    "positive_class": "food",
    "negative_class": "non_food",
    "decision_threshold": 0.5,
    "threshold_status": "experimental validation baseline",
    "method": "frozen ConvNeXt + weighted logistic regression",
    "roboflow_mapping": {
        "empty plate": 0,
        "chicken": 1,
        "plate with food": 1,
        "rice": 1
    },
}, OUT)

print("\nV2 EXPERIMENT SAVED:", OUT)
print("V1 checkpoint and original CNN weights unchanged.")
print("Not approved for deployment yet.")
