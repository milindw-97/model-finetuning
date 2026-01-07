#!/usr/bin/env python3
"""
Push Converted Dataset to HuggingFace Hub

This script uploads the NeMo-formatted dataset (audio files + manifests)
to HuggingFace Hub for easy sharing and reproducibility.

Usage:
    python scripts/push_to_huggingface.py \
        --data-dir data \
        --repo-name milind-plivo/parakeet-training-dataset \
        --private

Prerequisites:
    - huggingface-cli login (or set HF_TOKEN environment variable)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import HfApi, create_repo, upload_folder
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DATASET_CARD_TEMPLATE = """---
language:
- hi
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
tags:
- audio
- speech
- asr
- hindi
- nemo
pretty_name: Hindi ASR Dataset (NeMo Format)
size_categories:
- n<1K
---

# Hindi ASR Dataset (NeMo Format)

This dataset contains Hindi audio samples formatted for NVIDIA NeMo ASR training.

## Dataset Description

- **Source**: sivakgp/hindi-books-audio-validation
- **Language**: Hindi (hi)
- **Format**: NeMo Manifest (JSON Lines)
- **Audio Format**: 16kHz mono WAV

## Dataset Structure

```
├── train_manifest.json    # Training set manifest
├── val_manifest.json      # Validation set manifest
├── test_manifest.json     # Test set manifest
├── audio/
│   ├── train/             # Training audio files
│   ├── val/               # Validation audio files
│   └── test/              # Test audio files
└── conversion_info.json   # Conversion metadata
```

## NeMo Manifest Format

Each line in the manifest files is a JSON object:

```json
{{"audio_filepath": "audio/train/train_000000.wav", "text": "हिंदी transcription", "duration": 5.2, "lang": "hi"}}
```

## Usage with NeMo

```python
import nemo.collections.asr as nemo_asr

# Load model
model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-rnnt-1.1b-multilingual")

# Configure training data
model.cfg.train_ds.manifest_filepath = "train_manifest.json"
model.setup_training_data(model.cfg.train_ds)
```

## Statistics

{stats_section}

## License

This dataset is released under the CC-BY-4.0 license.

## Citation

If you use this dataset, please cite the original source and NVIDIA NeMo.
"""


def create_dataset_card(data_dir: str) -> str:
    """
    Create a HuggingFace dataset card.

    Args:
        data_dir: Directory containing the dataset

    Returns:
        Dataset card content as string
    """
    data_path = Path(data_dir)

    # Load conversion info if available
    info_path = data_path / "conversion_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {}

    # Calculate stats
    stats_lines = []

    for split in ["train", "val", "test"]:
        manifest_path = data_path / f"{split}_manifest.json"
        if manifest_path.exists():
            num_samples = 0
            total_duration = 0

            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        num_samples += 1
                        total_duration += entry.get("duration", 0)
                    except json.JSONDecodeError:
                        pass

            stats_lines.append(f"- **{split.capitalize()}**: {num_samples} samples ({total_duration/3600:.2f} hours)")

    stats_section = "\n".join(stats_lines) if stats_lines else "Statistics not available."

    return DATASET_CARD_TEMPLATE.format(stats_section=stats_section)


def update_manifest_paths(manifest_path: str, new_audio_base: str) -> str:
    """
    Update audio paths in manifest to be relative.

    Args:
        manifest_path: Path to manifest file
        new_audio_base: New base path for audio files

    Returns:
        Path to updated manifest
    """
    manifest_path = Path(manifest_path)
    updated_entries = []

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())

                # Get just the filename
                old_path = Path(entry["audio_filepath"])
                # Reconstruct with new base: audio/split/filename.wav
                split_name = old_path.parent.name  # train, val, or test
                filename = old_path.name
                new_path = f"{new_audio_base}/{split_name}/{filename}"

                entry["audio_filepath"] = new_path
                updated_entries.append(entry)

            except (json.JSONDecodeError, KeyError):
                pass

    # Write updated manifest
    output_path = manifest_path.parent / f"{manifest_path.stem}_relative.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in updated_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    return str(output_path)


def push_to_huggingface(
    data_dir: str,
    repo_name: str,
    private: bool = True,
    token: Optional[str] = None
):
    """
    Push dataset to HuggingFace Hub.

    Args:
        data_dir: Directory containing the dataset
        repo_name: HuggingFace repository name (user/repo)
        private: Whether to make the repo private
        token: HuggingFace API token (optional)
    """
    data_path = Path(data_dir)

    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    api = HfApi(token=token)

    # Create repository
    logger.info(f"Creating repository: {repo_name}")
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            private=private,
            token=token,
            exist_ok=True
        )
        logger.info(f"Repository created/exists: {repo_name}")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        raise

    # Create dataset card
    logger.info("Creating dataset card...")
    readme_content = create_dataset_card(data_dir)
    readme_path = data_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    # Update manifest paths to be relative
    logger.info("Updating manifest paths to relative...")
    for split in ["train", "val", "test"]:
        manifest_path = data_path / f"{split}_manifest.json"
        if manifest_path.exists():
            update_manifest_paths(str(manifest_path), "audio")

    # Upload folder
    logger.info(f"Uploading dataset to {repo_name}...")

    try:
        upload_folder(
            folder_path=str(data_path),
            repo_id=repo_name,
            repo_type="dataset",
            token=token,
            commit_message="Upload Hindi ASR dataset in NeMo format",
            ignore_patterns=["*.pyc", "__pycache__", ".git*", "*.DS_Store"]
        )
        logger.info("Upload complete!")

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

    # Print success message
    print("\n" + "=" * 60)
    print("UPLOAD SUCCESSFUL")
    print("=" * 60)
    print(f"\nDataset URL: https://huggingface.co/datasets/{repo_name}")
    print("\nTo use this dataset:")
    print(f'  from datasets import load_dataset')
    print(f'  ds = load_dataset("{repo_name}")')
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Push NeMo-formatted dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing the converted dataset"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="milind-plivo/parakeet-training-dataset",
        help="HuggingFace repository name (user/repo)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or use HF_TOKEN env var)"
    )

    args = parser.parse_args()

    # Validate data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Please run convert_to_nemo.py first.")
        return

    # Check for required files
    required_files = ["train_manifest.json"]
    missing = [f for f in required_files if not (data_path / f).exists()]
    if missing:
        print(f"Error: Missing required files: {missing}")
        print("Please run convert_to_nemo.py first.")
        return

    # Push to HuggingFace
    push_to_huggingface(
        data_dir=args.data_dir,
        repo_name=args.repo_name,
        private=args.private,
        token=args.token
    )


if __name__ == "__main__":
    main()
