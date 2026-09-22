"""
SPHAR Dataset Benchmark Loader.
Downloads and caches representative surveillance video clips from AlexanderMelde/SPHAR-Dataset.
"""

import os
import urllib.request
import json
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("sphar_loader")

REPO_API_BASE = "https://api.github.com/repos/AlexanderMelde/SPHAR-Dataset/contents/videos"
RAW_BASE = "https://raw.githubusercontent.com/AlexanderMelde/SPHAR-Dataset/master/videos"

# Recommended benchmark sample set (diverse camera angles, resolutions, and datasets)
BENCHMARK_CLASSES = {
    "hitting": 6,
    "kicking": 6,
    "falling": 6,
    "walking": 6,
    "neutral": 6,
    "sitting": 4,
    "running": 4
}


def fetch_class_file_list(class_name: str) -> List[Dict[str, str]]:
    url = f"{REPO_API_BASE}/{class_name}"
    req = urllib.request.Request(url, headers={"User-Agent": "GuardianMatrix-Benchmark"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [item for item in data if item.get("name", "").endswith(".mp4")]


def download_file(download_url: str, target_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path

    req = urllib.request.Request(download_url, headers={"User-Agent": "GuardianMatrix-Benchmark"})
    with urllib.request.urlopen(req) as resp, open(target_path, "wb") as f:
        f.write(resp.read())
    return target_path


def prepare_sphar_benchmark(
    destination_dir: str = "test_videos/sphar",
    samples_per_class: Dict[str, int] = BENCHMARK_CLASSES
) -> List[Tuple[str, str]]:
    """
    Downloads curated benchmark set from SPHAR repository.
    Returns list of (video_path, ground_truth_label)
    """
    os.makedirs(destination_dir, exist_ok=True)
    manifest: List[Tuple[str, str]] = []

    print("\n[SPHAR Loader] Fetching benchmark video manifest from GitHub...")
    for cls_name, count in samples_per_class.items():
        cls_dir = os.path.join(destination_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        try:
            items = fetch_class_file_list(cls_name)
            # Pick evenly spaced samples across the available videos
            stride = max(1, len(items) // count)
            selected = [items[i] for i in range(0, min(len(items), count * stride), stride)][:count]

            print(f"  Class '{cls_name}': downloading {len(selected)} clips...")
            for item in selected:
                fname = item["name"]
                target = os.path.join(cls_dir, fname)
                durl = item.get("download_url") or f"{RAW_BASE}/{cls_name}/{fname}"
                download_file(durl, target)
                manifest.append((target, cls_name))
        except Exception as e:
            print(f"  Error loading class '{cls_name}': {e}")

    print(f"[SPHAR Loader] Benchmark dataset ready: {len(manifest)} clips loaded into {destination_dir}.\n")
    return manifest


if __name__ == "__main__":
    clips = prepare_sphar_benchmark()
    print("Loaded clips:", len(clips))
