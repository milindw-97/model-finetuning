#!/usr/bin/env python3
"""
Download HuggingFace Dataset for Parakeet Fine-Tuning

This script downloads the Hindi audio dataset from HuggingFace
and prepares it for conversion to NeMo format.

Usage:
    python scripts/download_dataset.py --output-dir data/raw

Dataset: sivakgp/hindi-books-audio-validation
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_hindi_dataset(
    output_dir: str = "data/raw",
    dataset_name: str = "sivakgp/hindi-books-audio-validation",
    split: str = "validation",
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Download the Hindi audio dataset from HuggingFace.

    Args:
        output_dir: Directory to save dataset info
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to download
        max_samples: Optional limit on number of samples

    Returns:
        Downloaded Dataset object
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info(f"Split: {split}")

    # Load dataset
    dataset = load_dataset(
        dataset_name,
        split=split,
        trust_remote_code=True
    )

    logger.info(f"Dataset loaded with {len(dataset)} samples")
    logger.info(f"Columns: {dataset.column_names}")

    # Limit samples if specified
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        logger.info(f"Limited to {max_samples} samples")

    # Save dataset info
    info = {
        "dataset_name": dataset_name,
        "split": split,
        "num_samples": len(dataset),
        "columns": dataset.column_names,
        "features": str(dataset.features)
    }

    info_path = output_path / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    logger.info(f"Dataset info saved to: {info_path}")

    # Print sample transcription (text only, avoid audio decoding issues)
    if len(dataset) > 0:
        try:
            # Access raw data without decoding audio
            sample_text = dataset[0]['transcription'] if 'transcription' in dataset.column_names else 'N/A'
            logger.info(f"\nSample transcription: {sample_text[:100]}...")
        except Exception as e:
            logger.warning(f"Could not preview sample: {e}")

    return dataset


def analyze_dataset(dataset: Dataset) -> dict:
    """
    Analyze dataset statistics.

    Args:
        dataset: HuggingFace Dataset

    Returns:
        Dictionary with statistics
    """
    stats = {
        "num_samples": len(dataset),
        "total_duration_seconds": 0,
        "min_duration": float('inf'),
        "max_duration": 0,
        "avg_text_length": 0
    }

    total_text_len = 0

    logger.info("Analyzing dataset...")

    for sample in tqdm(dataset, desc="Analyzing"):
        # Get audio duration
        audio = sample.get("audio")
        if audio and isinstance(audio, dict):
            array = audio.get("array", [])
            sr = audio.get("sampling_rate", 16000)
            duration = len(array) / sr

            stats["total_duration_seconds"] += duration
            stats["min_duration"] = min(stats["min_duration"], duration)
            stats["max_duration"] = max(stats["max_duration"], duration)

        # Get text length
        text = sample.get("transcription", "")
        total_text_len += len(text)

    if stats["min_duration"] == float('inf'):
        stats["min_duration"] = 0

    stats["avg_text_length"] = total_text_len / len(dataset) if len(dataset) > 0 else 0
    stats["total_duration_hours"] = stats["total_duration_seconds"] / 3600

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download Hindi audio dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sivakgp/hindi-books-audio-validation",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to download"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to download"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze dataset statistics"
    )

    args = parser.parse_args()

    # Download dataset
    dataset = download_hindi_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples
    )

    # Analyze if requested
    if args.analyze:
        stats = analyze_dataset(dataset)

        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"Total samples: {stats['num_samples']}")
        print(f"Total duration: {stats['total_duration_hours']:.2f} hours")
        print(f"Duration range: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s")
        print(f"Average text length: {stats['avg_text_length']:.1f} characters")
        print("=" * 50)

        # Save stats
        stats_path = Path(args.output_dir) / "dataset_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_path}")

    print(f"\nDataset ready for conversion.")
    print(f"Next step: python scripts/convert_to_nemo.py --input-dataset {args.dataset}")


if __name__ == "__main__":
    main()
