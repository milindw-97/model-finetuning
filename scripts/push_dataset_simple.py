#!/usr/bin/env python3
"""
Simple script to push NeMo dataset to HuggingFace Hub.
Run this directly from your terminal where HF_TOKEN is set.

Usage:
    python scripts/push_dataset_simple.py
"""

import os
import json
from pathlib import Path
from datasets import Dataset, Audio

def main():
    # Configuration
    DATA_DIR = "data"
    REPO_NAME = "milind-plivo/parakeet-training-dataset"

    print("=" * 60)
    print("PUSH HINDI ASR DATASET TO HUGGINGFACE")
    print("=" * 60)

    # Check token - try multiple sources
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    # Try to get stored token from huggingface_hub
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            pass

    # Try reading from cache file directly
    if not token:
        token_paths = [
            os.path.expanduser("~/.cache/huggingface/token"),
            os.path.expanduser("~/.huggingface/token"),
        ]
        for path in token_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    token = f.read().strip()
                    if token:
                        break

    if not token:
        print("ERROR: HuggingFace token not found!")
        print("Options:")
        print("  1. Run: export HF_TOKEN='your_token_here'")
        print("  2. Run: huggingface-cli login")
        return

    print(f"Token found: {token[:10]}...{token[-4:]}")
    print(f"Target repo: {REPO_NAME}")

    # Load manifests and create dataset
    data_path = Path(DATA_DIR)

    all_audio_paths = []
    all_transcriptions = []
    all_durations = []
    all_splits = []

    for split in ["train", "val", "test"]:
        manifest_path = data_path / f"{split}_manifest.json"
        if not manifest_path.exists():
            print(f"Warning: {manifest_path} not found, skipping...")
            continue

        print(f"Loading {split} manifest...")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                all_audio_paths.append(entry["audio_filepath"])
                all_transcriptions.append(entry["text"])
                all_durations.append(entry.get("duration", 0))
                all_splits.append(split)

    print(f"\nTotal samples: {len(all_audio_paths)}")
    print(f"  Train: {all_splits.count('train')}")
    print(f"  Val: {all_splits.count('val')}")
    print(f"  Test: {all_splits.count('test')}")

    # Create dataset
    print("\nCreating HuggingFace dataset...")
    dataset = Dataset.from_dict({
        "audio": all_audio_paths,
        "transcription": all_transcriptions,
        "duration": all_durations,
        "split": all_splits,
    }).cast_column("audio", Audio(sampling_rate=16000))

    print(f"Dataset created: {dataset}")

    # Push to hub
    print(f"\nPushing to {REPO_NAME}...")
    dataset.push_to_hub(
        REPO_NAME,
        token=token,
        private=False,
    )

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"Dataset URL: https://huggingface.co/datasets/{REPO_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
