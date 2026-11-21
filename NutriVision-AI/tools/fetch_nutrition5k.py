"""Fetch official measured metadata, split lists and their overhead RGB images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from PIL import Image


BASE = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/"


def fetch(root, name, optional=False):
    path = root / name
    if path.is_file():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            with urlopen(BASE + name, timeout=60) as response:
                data = response.read()
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            if optional:
                with Image.open(tmp) as image:
                    image.verify()
            tmp.replace(path)
            return True
        except HTTPError as error:
            if optional and error.code == 404:
                return False
            if error.code != 429 and error.code < 500:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Repeated download failure for {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    ids = []
    for split in ("train", "test"):
        name = f"dish_ids/splits/depth_{split}_ids.txt"
        fetch(args.root, name)
        ids.extend((args.root / name).read_text().split())
    if len(set(ids)) != len(ids) or not all(re.fullmatch(r"dish_\d+", dish) for dish in ids):
        raise ValueError("Invalid or overlapping official IDs")
    for cafe in (1, 2):
        fetch(args.root, f"metadata/dish_metadata_cafe{cafe}.csv")

    def download_rgb(dish):
        return fetch(args.root, f"imagery/realsense_overhead/{dish}/rgb.png", optional=True)

    missing = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, (dish, found) in enumerate(zip(ids, pool.map(download_rgb, ids)), 1):
            if not found:
                missing.append(dish)
            if i % 250 == 0:
                print(f"Checked {i}/{len(ids)}", flush=True)
    print("Official RGBs unavailable (HTTP 404):", missing)
    (args.root / "missing_official_rgb.txt").write_text("\n".join(missing))


if __name__ == "__main__":
    main()
