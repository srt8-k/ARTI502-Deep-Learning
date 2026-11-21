"""Offline Nutrition5k experiment. Run as python -m tools.compare_nutrition --help."""
import argparse
import copy
import csv
import hashlib
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn


def records(root, split):
    ids = (root / "dish_ids/splits" / f"depth_{split}_ids.txt").read_text().split()
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate split IDs")
    labels = {}
    for cafe in (1, 2):
        with (root / "metadata" / f"dish_metadata_cafe{cafe}.csv").open(encoding="latin-1", newline="") as stream:
            for row in csv.reader(stream):
                if row and row[0].startswith("dish_"):
                    if row[0] in labels:
                        raise ValueError(f"Duplicate metadata for {row[0]}")
                    labels[row[0]] = (float(row[1]), float(row[2]), cafe)
    rows = []
    for dish in ids:
        path = root / "imagery/realsense_overhead" / dish / "rgb.png"
        if path.exists():
            kcal, grams, cafe = labels[dish]
            if not torch.isfinite(torch.tensor([kcal, grams])).all() or kcal < 0 or grams <= 0:
                raise ValueError(f"Invalid measured targets for {dish}")
            rows.append((dish, path, kcal, grams, cafe))
    print(f"{split}: {len(rows)} usable overhead RGBs / {len(ids)} official IDs")
    if not rows:
        raise ValueError("No matching Nutrition5k overhead images")
    return rows, set(ids)


def validation_groups(rows):
    """Keep a cafe's whole UTC collection day together, including midnight bursts."""
    groups = []
    previous = None
    for dish, _, _, _, cafe in sorted(rows, key=lambda r: (r[4], r[0])):
        stamp = int(dish.removeprefix("dish_"))
        if (previous is None or cafe != previous[0] or
                (stamp // 86400 != previous[1] // 86400 and stamp - previous[1] > 3600)):
            groups.append(set())
        groups[-1].add(dish)
        previous = cafe, stamp
    return groups


def validation_cohort(rows, manifest):
    groups = validation_groups(rows)
    if manifest:
        chosen = set(manifest.read_text().split())
        if not chosen <= {r[0] for r in rows} or any(g & chosen and g - chosen for g in groups):
            raise ValueError("Validation manifest contains unavailable IDs or splits a collection group")
        return chosen
    random.Random(502).shuffle(groups)
    chosen = set()
    for group in groups[:-1]:
        chosen.update(group)
        if len(chosen) >= len(rows) * 0.15:
            break
    return chosen


@torch.inference_mode()
def extract(service, rows):
    features, scores, baseline = [], [], []
    for start in range(0, len(rows), 8):
        if start % 256 == 0:
            print(f"Extracting {start}/{len(rows)}", flush=True)
        images = []
        for _, path, *_ in rows[start:start + 8]:
            with Image.open(path) as image:
                images.append(service.transform(ImageOps.exif_transpose(image).convert("RGB")))
        batch = torch.stack(images).to(service.device)
        mixed = service.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast("cuda", dtype=torch.bfloat16) if mixed else nullcontext():
            tokens = service.food_model.forward_features(batch)
            visual = service.food_model.forward_head(tokens, pre_logits=True)
            logits = service.food_model.forward_head(tokens)
        visual = visual.float()
        standardized = (visual - service.feature_mean) / service.feature_std.clamp_min(1e-6)
        raw = service.nutrition_model(standardized)
        log_values = (raw * service.target_std + service.target_mean)[:, :2]
        baseline.append(log_values.clamp(0, 9).expm1().cpu())
        features.append(visual.cpu())
        scores.append(logits.float().softmax(1).cpu())
    return torch.cat(features), torch.cat(scores), torch.cat(baseline)


def make_head(width):
    return nn.Sequential(nn.Linear(width, 128), nn.LayerNorm(128), nn.GELU(),
                         nn.Dropout(0.2), nn.Linear(128, 2))


def inputs(kind, visual, probabilities):
    return torch.cat([visual, probabilities], dim=1) if kind == "food-aware" else visual


def decode(raw, bundle):
    estimates = (raw * bundle["y_std"] + bundle["y_mean"]).expm1().clamp_min(0)
    if bundle["kind"] == "density":
        # Training labels for this variant are measured kcal/g and grams.
        return torch.stack([estimates[:, 0] * estimates[:, 1], estimates[:, 1]], dim=1)
    return estimates


def predict(head, bundle, visual, probabilities):
    x = inputs(bundle["kind"], visual, probabilities)
    return decode(head((x - bundle["x_mean"]) / bundle["x_std"]), bundle)


def metrics(pred, actual):
    error = pred - actual
    if not torch.isfinite(error).all():
        raise ValueError("Nonfinite prediction")
    mae = error.abs().mean(0).tolist()
    rmse = error.square().mean(0).sqrt().tolist()
    return dict(kcal_mae=mae[0], mass_mae=mae[1], kcal_rmse=rmse[0], mass_rmse=rmse[1])


def train(kind, visual, probs, labels, fit, valid):
    torch.manual_seed(502)
    x = inputs(kind, visual, probs)
    targets = labels.clone()
    if kind == "density":
        targets[:, 0] /= targets[:, 1]
    y = targets.log1p()
    bundle = dict(kind=kind, x_mean=x[fit].mean(0), x_std=x[fit].std(0, unbiased=False).clamp_min(1e-6),
                  y_mean=y[fit].mean(0), y_std=y[fit].std(0, unbiased=False).clamp_min(1e-6))
    train_x = (x[fit] - bundle["x_mean"]) / bundle["x_std"]
    train_y = (y[fit] - bundle["y_mean"]) / bundle["y_std"]
    head = make_head(x.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
    scales = labels[fit].mean(0).clamp_min(1)
    best, best_epoch = float("inf"), 0
    for epoch in range(1, 81):
        head.train()
        for idx in torch.randperm(len(train_x)).split(64):
            optimizer.zero_grad()
            loss = nn.functional.smooth_l1_loss(head(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            estimate = predict(head, bundle, visual[valid], probs[valid])
            score = ((estimate - labels[valid]).abs().mean(0) / scales).mean().item()
        if score < best:
            best, best_epoch = score, epoch
            bundle["weights"] = copy.deepcopy(head.state_dict())
        if epoch - best_epoch >= 10:
            break
    head.load_state_dict(bundle["weights"])
    bundle["epoch"] = best_epoch
    return head.eval(), bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path, help="New private directory for metrics and candidate")
    parser.add_argument("--validation-ids", type=Path, help="Original group-safe validation IDs if available")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["NUTRIVISION_OPEN_VOCAB"] = "0"
    from backend.ai_service import MODEL_DIR, NutriVisionService
    service = NutriVisionService()
    rows, train_ids = records(args.data, "train")
    test_ids = set((args.data / "dish_ids/splits/depth_test_ids.txt").read_text().split())
    if train_ids & test_ids:
        raise ValueError("Official splits overlap")
    val_ids = validation_cohort(rows, args.validation_ids)
    if not val_ids <= train_ids:
        raise ValueError("Validation IDs must belong to the official training split")
    valid = torch.tensor([r[0] in val_ids for r in rows])
    fit = ~valid
    if not valid.any() or not fit.any():
        raise ValueError("Both fit and validation cohorts are required")
    report = {"fit_ids": [r[0] for r in rows if r[0] not in val_ids],
              "validation_ids": [r[0] for r in rows if r[0] in val_ids],
              "validation": {}, "test": {}}
    report["missing_rgb"] = {"train": sorted(train_ids - {r[0] for r in rows})}
    report["device"] = str(service.device)
    report["torch_version"] = str(torch.__version__)
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["seed"] = 502
    report["checkpoint_sha256"] = {}
    for name in ("dinov2_last4_best.pth", "nutrition_mlp_best.pth"):
        with (MODEL_DIR / name).open("rb") as stream:
            report["checkpoint_sha256"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
    visual, probs, baseline = extract(service, rows)
    labels = torch.tensor([[r[2], r[3]] for r in rows])
    reference = metrics(baseline[valid], labels[valid])
    report["validation"]["release"] = reference
    selected, best = None, float("inf")
    for kind in ("direct", "food-aware", "density"):
        head, bundle = train(kind, visual, probs, labels, fit, valid)
        with torch.no_grad():
            values = metrics(predict(head, bundle, visual[valid], probs[valid]), labels[valid])
        report["validation"][kind] = dict(values, epoch=bundle["epoch"])
        torch.save(bundle, args.output / f"{kind}.pth")
        ratios = [values[k] / max(reference[k], 1e-12) for k in ("mass_mae", "kcal_mae")]
        if sum(ratios) < best:
            best, selected = sum(ratios), (head, bundle)
    # Freeze selection before evaluating any candidate on the official test set.
    report["selected"] = selected[1]["kind"]
    (args.output / "selection.json").write_text(json.dumps(report, indent=2))
    test_rows, test_ids = records(args.data, "test")
    report["test_ids"] = [r[0] for r in test_rows]
    report["missing_rgb"]["test"] = sorted(test_ids - set(report["test_ids"]))
    visual, probs, baseline = extract(service, test_rows)
    labels = torch.tensor([[r[2], r[3]] for r in test_rows])
    report["test"]["release"] = metrics(baseline, labels)
    predictions = {"release": baseline}
    if selected:
        head, bundle = selected
        with torch.no_grad():
            candidate = predict(head, bundle, visual, probs)
            predictions[bundle["kind"]] = candidate
            report["test"][bundle["kind"]] = metrics(candidate, labels)
        report["selected"] = bundle["kind"]
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        keys = ("kcal_mae", "mass_mae", "kcal_rmse", "mass_rmse")
        writer.writerow(["split", "model", "n", *keys])
        for split in ("validation", "test"):
            for name, values in report[split].items():
                writer.writerow([split, name, len(report[f"{split}_ids"]), *(values[k] for k in keys)])
    with (args.output / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dish_id", "cafe", "predicted_food101", "measured_kcal", "measured_g",
                         *(f"{name}_{unit}" for name in predictions for unit in ("kcal", "g"))])
        for i, row in enumerate(test_rows):
            writer.writerow([row[0], row[4], service.classes[probs[i].argmax().item()], *labels[i].tolist(),
                             *(v for pred in predictions.values() for v in pred[i].tolist())])
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("selected", "validation", "test")}, indent=2))


if __name__ == "__main__":
    main()
