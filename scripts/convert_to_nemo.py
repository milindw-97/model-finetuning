#!/usr/bin/env python3
"""
Convert HuggingFace Dataset to NeMo Manifest Format

This script converts a HuggingFace audio dataset to NeMo's JSON Lines
manifest format, saving audio files and creating train/val/test splits.

Usage:
    # Standard mode (loads full dataset)
    python scripts/convert_to_nemo.py \
        --input-dataset ai4bharat/IndicVoices_r \
        --subset hindi \
        --output-dir data

    # Streaming mode (low memory, processes one sample at a time)
    python scripts/convert_to_nemo.py \
        --input-dataset ai4bharat/IndicVoices_r \
        --subset hindi \
        --output-dir data \
        --streaming \
        --max-samples 5000

    # Incremental training: skip first 2000 samples, use next 3000
    python scripts/convert_to_nemo.py \
        --input-dataset ai4bharat/IndicVoices_r \
        --subset hindi \
        --output-dir data \
        --streaming \
        --offset 2000 \
        --max-samples 3000

    # With language tags for multilingual models (adds "<hi-IN>" at end of transcript)
    python scripts/convert_to_nemo.py \
        --input-dataset ai4bharat/IndicVoices_r \
        --subset hindi \
        --output-dir data \
        --streaming \
        --max-samples 2000 \
        --lang-tag hi-IN

NeMo Manifest Format (one JSON per line):
    {"audio_filepath": "/path/to/audio.wav", "text": "transcription", "duration": 5.2, "lang": "hi"}
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from datasets import load_dataset, Dataset, Audio, IterableDataset
from tqdm import tqdm

# Disable torchcodec to avoid FFmpeg linking issues
import os

os.environ["HF_AUDIO_DECODER"] = "ffmpeg"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def resample_audio(
    audio_array: np.ndarray, orig_sr: int, target_sr: int = 16000
) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio_array: Audio samples
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio_array

    try:
        import librosa

        return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Fallback: simple decimation/interpolation
        ratio = target_sr / orig_sr
        new_length = int(len(audio_array) * ratio)
        indices = np.linspace(0, len(audio_array) - 1, new_length)
        return np.interp(indices, np.arange(len(audio_array)), audio_array)


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: HuggingFace Dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed

    Returns:
        Tuple of (train, val, test) datasets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )

    # First split: train vs (val + test)
    split1 = dataset.train_test_split(test_size=(val_ratio + test_ratio), seed=seed)
    train = split1["train"]

    # Second split: val vs test
    if test_ratio > 0:
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        split2 = split1["test"].train_test_split(test_size=val_test_ratio, seed=seed)
        val = split2["train"]
        test = split2["test"]
    else:
        val = split1["test"]
        test = Dataset.from_dict({k: [] for k in dataset.column_names})

    return train, val, test


def convert_dataset_to_nemo(
    dataset: Dataset,
    output_dir: str,
    split_name: str,
    audio_column: str = "audio",
    text_column: str = "transcription",
    language: str = "hi",
    target_sr: int = 16000,
    max_duration: float = 30.0,
    min_duration: float = 0.3,
) -> str:
    """
    Convert a HuggingFace dataset split to NeMo manifest format.

    Args:
        dataset: HuggingFace Dataset
        output_dir: Output directory
        split_name: Name of split (train/val/test)
        audio_column: Name of audio column
        text_column: Name of text column
        language: Language code
        target_sr: Target sample rate
        max_duration: Maximum audio duration
        min_duration: Minimum audio duration

    Returns:
        Path to created manifest file
    """
    output_path = Path(output_dir)
    audio_dir = output_path / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / f"{split_name}_manifest.json"

    entries = []
    skipped = 0

    logger.info(f"Converting {split_name} split ({len(dataset)} samples)...")

    for idx, sample in enumerate(tqdm(dataset, desc=f"Converting {split_name}")):
        try:
            # Get audio
            audio = sample.get(audio_column)
            if audio is None:
                skipped += 1
                continue

            if isinstance(audio, dict):
                array = np.array(audio["array"])
                sr = audio.get("sampling_rate", 16000)
            else:
                skipped += 1
                continue

            # Resample if needed
            if sr != target_sr:
                array = resample_audio(array, sr, target_sr)

            # Calculate duration
            duration = len(array) / target_sr

            # Filter by duration
            if duration < min_duration or duration > max_duration:
                skipped += 1
                continue

            # Get transcription
            text = sample.get(text_column, "")
            if not text or not text.strip():
                skipped += 1
                continue

            # Save audio file
            audio_filename = f"{split_name}_{idx:06d}.wav"
            audio_filepath = audio_dir / audio_filename

            # Ensure mono
            if len(array.shape) > 1:
                array = array.mean(axis=-1)

            # Normalize to float32 [-1, 1]
            if array.dtype != np.float32:
                if np.issubdtype(array.dtype, np.integer):
                    max_val = np.iinfo(array.dtype).max
                    array = array.astype(np.float32) / max_val
                else:
                    array = array.astype(np.float32)

            # Save as WAV
            sf.write(str(audio_filepath), array, target_sr)

            # Create manifest entry
            entry = {
                "audio_filepath": str(audio_filepath.absolute()),
                "text": text.strip(),
                "duration": round(duration, 3),
                "lang": language,
            }
            entries.append(entry)

        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            skipped += 1
            continue

    # Write manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Created {split_name} manifest: {manifest_path}")
    logger.info(f"  Samples: {len(entries)}, Skipped: {skipped}")

    return str(manifest_path)


def convert_streaming_to_nemo(
    dataset_name: str,
    subset: Optional[str],
    input_split: str,
    output_dir: str,
    audio_column: str = "audio",
    text_column: str = "transcription",
    language: str = "hi",
    target_sr: int = 16000,
    max_duration: float = 30.0,
    min_duration: float = 0.3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_samples: Optional[int] = None,
    offset: int = 0,
    lang_tag: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Convert a HuggingFace dataset to NeMo format using streaming mode.

    Processes one sample at a time to minimize memory and disk usage.
    Randomly assigns each sample to train/val/test based on ratios.

    Args:
        dataset_name: HuggingFace dataset name
        subset: Dataset subset/config
        input_split: Dataset split to load
        output_dir: Output directory
        audio_column: Name of audio column
        text_column: Name of text column
        language: Language code
        target_sr: Target sample rate
        max_duration: Maximum audio duration
        lang_tag: Language tag to append to transcripts (e.g., "hi-IN" -> "text <hi-IN>")
        min_duration: Minimum audio duration
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        max_samples: Maximum number of samples to process
        offset: Number of samples to skip from the beginning
        seed: Random seed

    Returns:
        Dictionary mapping split names to manifest paths
    """
    random.seed(seed)

    output_path = Path(output_dir)

    # Create directories and open manifest files
    splits = ["train", "val", "test"]
    audio_dirs = {}
    manifest_files = {}
    counters = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    for split in splits:
        audio_dirs[split] = output_path / "audio" / split
        audio_dirs[split].mkdir(parents=True, exist_ok=True)
        manifest_files[split] = open(
            output_path / f"{split}_manifest.json", "w", encoding="utf-8"
        )

    # Load dataset in streaming mode
    logger.info(f"Loading dataset in streaming mode: {dataset_name}")
    if subset:
        logger.info(f"Subset: {subset}")

    try:
        dataset = load_dataset(
            dataset_name,
            subset,
            split=input_split,
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(f"Failed with subset, trying without: {e}")
        dataset = load_dataset(
            dataset_name,
            split=input_split,
            streaming=True,
            trust_remote_code=True,
        )

    # Calculate cumulative thresholds for split assignment
    train_threshold = train_ratio
    val_threshold = train_ratio + val_ratio
    # test is everything above val_threshold

    logger.info(f"Processing samples (streaming mode, max_samples={max_samples}, offset={offset})...")
    logger.info(
        f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}"
    )
    if offset > 0:
        logger.info(f"Skipping first {offset} samples...")
    if lang_tag:
        logger.info(f"Adding language tag at end: text <{lang_tag}>")

    total_processed = 0
    samples_skipped_for_offset = 0

    try:
        for idx, sample in enumerate(
            tqdm(dataset, desc="Converting", total=(max_samples + offset) if max_samples else None)
        ):
            # Skip samples until we reach the offset
            if samples_skipped_for_offset < offset:
                samples_skipped_for_offset += 1
                continue

            if max_samples and total_processed >= max_samples:
                break

            try:
                # Get audio
                audio = sample.get(audio_column)
                if audio is None:
                    skipped += 1
                    continue

                if isinstance(audio, dict):
                    array = np.array(audio["array"])
                    sr = audio.get("sampling_rate", 16000)
                else:
                    skipped += 1
                    continue

                # Resample if needed
                if sr != target_sr:
                    array = resample_audio(array, sr, target_sr)

                # Calculate duration
                duration = len(array) / target_sr

                # Filter by duration
                if duration < min_duration or duration > max_duration:
                    skipped += 1
                    continue

                # Get transcription - try multiple possible column names
                text = None
                for col in [
                    text_column,
                    "transcription",
                    "text",
                    "sentence",
                    "transcript",
                ]:
                    if col in sample and sample.get(col):
                        text = sample.get(col)
                        break

                if not text or not str(text).strip():
                    skipped += 1
                    continue

                text = str(text).strip()

                # Add language tag at end if specified (e.g., "text <hi-IN>")
                if lang_tag:
                    text = f"{text} <{lang_tag}>"

                # Randomly assign to split
                rand_val = random.random()
                if rand_val < train_threshold:
                    split = "train"
                elif rand_val < val_threshold:
                    split = "val"
                else:
                    split = "test"

                # Save audio file
                audio_filename = f"{split}_{counters[split]:06d}.wav"
                audio_filepath = audio_dirs[split] / audio_filename

                # Ensure mono
                if len(array.shape) > 1:
                    array = array.mean(axis=-1)

                # Normalize to float32 [-1, 1]
                if array.dtype != np.float32:
                    if np.issubdtype(array.dtype, np.integer):
                        max_val = np.iinfo(array.dtype).max
                        array = array.astype(np.float32) / max_val
                    else:
                        array = array.astype(np.float32)

                # Save as WAV
                sf.write(str(audio_filepath), array, target_sr)

                # Write manifest entry immediately
                entry = {
                    "audio_filepath": str(audio_filepath.absolute()),
                    "text": text,
                    "duration": round(duration, 3),
                    "lang": language,
                }
                manifest_files[split].write(
                    json.dumps(entry, ensure_ascii=False) + "\n"
                )
                manifest_files[split].flush()  # Flush to disk immediately

                counters[split] += 1
                total_processed += 1

            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                skipped += 1
                continue

    finally:
        # Close all manifest files
        for split in splits:
            manifest_files[split].close()

    # Log results
    logger.info(f"\nConversion complete!")
    logger.info(f"  Total processed: {total_processed}")
    logger.info(f"  Skipped: {skipped}")
    for split in splits:
        logger.info(f"  {split}: {counters[split]} samples")

    return {split: str(output_path / f"{split}_manifest.json") for split in splits}


def validate_manifest(manifest_path: str) -> Dict:
    """
    Validate a NeMo manifest file and return statistics.

    Args:
        manifest_path: Path to manifest file

    Returns:
        Dictionary with validation results
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return {"valid": False, "error": "File not found"}

    stats = {
        "valid": True,
        "num_samples": 0,
        "total_duration_hours": 0,
        "missing_audio_files": 0,
        "languages": {},
    }

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())

                stats["num_samples"] += 1
                stats["total_duration_hours"] += entry.get("duration", 0) / 3600

                lang = entry.get("lang", "unknown")
                stats["languages"][lang] = stats["languages"].get(lang, 0) + 1

                # Check if audio file exists
                audio_path = entry.get("audio_filepath")
                if audio_path and not Path(audio_path).exists():
                    stats["missing_audio_files"] += 1

            except json.JSONDecodeError:
                stats["valid"] = False
                stats["error"] = f"Invalid JSON at line {line_num}"
                break

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace dataset to NeMo manifest format"
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default="ai4bharat/IndicVoices_r",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="Hindi",
        help="Dataset subset/config (e.g., 'hindi' for IndicVoices_r)",
    )
    parser.add_argument(
        "--input-split", type=str, default="train", help="Dataset split to load"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Output directory"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation set ratio"
    )
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument(
        "--audio-column", type=str, default="audio", help="Name of audio column"
    )
    parser.add_argument(
        "--text-column", type=str, default="transcription", help="Name of text column"
    )
    parser.add_argument("--language", type=str, default="hi", help="Language code")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Target sample rate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (low memory, processes one sample at a time)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (recommended with --streaming)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of samples to skip from the beginning (use with --streaming for incremental training)",
    )
    parser.add_argument(
        "--lang-tag",
        type=str,
        default=None,
        help="Language tag to append to transcripts (e.g., 'hi-IN' adds ' <hi-IN>' at end)",
    )

    args = parser.parse_args()

    # Use streaming mode if requested
    if args.streaming:
        logger.info("Using STREAMING mode (low memory)")
        manifests = convert_streaming_to_nemo(
            dataset_name=args.input_dataset,
            subset=args.subset,
            input_split=args.input_split,
            output_dir=args.output_dir,
            audio_column=args.audio_column,
            text_column=args.text_column,
            language=args.language,
            target_sr=args.sample_rate,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            max_samples=args.max_samples,
            offset=args.offset,
            lang_tag=args.lang_tag,
            seed=args.seed,
        )
    else:
        # Standard mode - load full dataset
        logger.info(f"Loading dataset: {args.input_dataset}")
        if args.subset:
            logger.info(f"Subset: {args.subset}")
        try:
            dataset = load_dataset(
                args.input_dataset,
                args.subset,
                split=args.input_split,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed with subset, trying without: {e}")
            dataset = load_dataset(
                args.input_dataset,
                split=args.input_split,
                trust_remote_code=True,
            )

        # Limit samples if specified
        if args.max_samples and len(dataset) > args.max_samples:
            dataset = dataset.select(range(args.max_samples))
            logger.info(f"Limited to {args.max_samples} samples")

        # Cast audio column to use soundfile decoder (avoids torchcodec issues)
        logger.info("Casting audio column to use soundfile decoder...")
        dataset = dataset.cast_column(
            args.audio_column, Audio(sampling_rate=args.sample_rate, decode=True)
        )
        logger.info(f"Loaded {len(dataset)} samples")

        # Split dataset
        logger.info("Splitting dataset...")
        train_ds, val_ds, test_ds = split_dataset(
            dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        logger.info(
            f"Split sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
        )

        # Convert each split
        manifests = {}

        for split_name, split_data in [
            ("train", train_ds),
            ("val", val_ds),
            ("test", test_ds),
        ]:
            if len(split_data) > 0:
                manifest_path = convert_dataset_to_nemo(
                    split_data,
                    args.output_dir,
                    split_name,
                    audio_column=args.audio_column,
                    text_column=args.text_column,
                    language=args.language,
                    target_sr=args.sample_rate,
                )
                manifests[split_name] = manifest_path

    # Validate manifests
    print("\n" + "=" * 60)
    print("MANIFEST VALIDATION")
    print("=" * 60)

    for split_name, manifest_path in manifests.items():
        stats = validate_manifest(manifest_path)
        print(f"\n{split_name.upper()}:")
        print(f"  Samples: {stats['num_samples']}")
        print(f"  Duration: {stats['total_duration_hours']:.2f} hours")
        print(f"  Languages: {stats['languages']}")
        if stats["missing_audio_files"] > 0:
            print(f"  WARNING: {stats['missing_audio_files']} missing audio files")

    print("=" * 60)

    # Save conversion info
    info = {
        "source_dataset": args.input_dataset,
        "source_subset": args.subset,
        "source_split": args.input_split,
        "language": args.language,
        "sample_rate": args.sample_rate,
        "streaming_mode": args.streaming,
        "max_samples": args.max_samples,
        "offset": args.offset,
        "splits": {},
    }

    # Count samples from manifests
    for split_name, manifest_path in manifests.items():
        stats = validate_manifest(manifest_path)
        info["splits"][split_name] = {
            "manifest": manifest_path,
            "samples": stats.get("num_samples", 0),
        }

    info_path = Path(args.output_dir) / "conversion_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nConversion complete!")
    print(f"Manifests saved to: {args.output_dir}/")
    print(f"\nNext step: python scripts/push_to_huggingface.py")


if __name__ == "__main__":
    main()
