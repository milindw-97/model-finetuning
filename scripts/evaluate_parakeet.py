#!/usr/bin/env python3
"""
Evaluation Script for Fine-Tuned Parakeet Model

This script evaluates a fine-tuned Parakeet ASR model on a test set,
computing WER, CER, and generating sample transcriptions.

Usage:
    python scripts/evaluate.py \
        --model outputs/parakeet-hindi-finetuned/final_model.nemo \
        --test-manifest data/test_manifest.json

    Or compare with original model:
    python scripts/evaluate.py \
        --model outputs/parakeet-hindi-finetuned/final_model.nemo \
        --baseline parakeet-rnnt-1.1b-multilingual \
        --test-manifest data/test_manifest.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """
    Load a NeMo ASR model.

    Args:
        model_path: Path to .nemo file or model name

    Returns:
        Loaded model
    """
    import nemo.collections.asr as nemo_asr

    if Path(model_path).exists() and model_path.endswith('.nemo'):
        logger.info(f"Loading model from: {model_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    else:
        logger.info(f"Loading pre-trained model: {model_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_path)

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def load_test_data(manifest_path: str) -> List[Dict]:
    """
    Load test data from manifest file.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        List of test samples
    """
    samples = []

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(samples)} test samples from {manifest_path}")
    return samples


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate.

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        WER as a float
    """
    try:
        from jiwer import wer
        return wer(references, predictions)
    except ImportError:
        # Fallback implementation
        total_errors = 0
        total_words = 0

        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()

            # Simple Levenshtein distance
            m, n = len(ref_words), len(pred_words)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref_words[i - 1] == pred_words[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

            total_errors += dp[m][n]
            total_words += m

        return total_errors / max(total_words, 1)


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Character Error Rate.

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        CER as a float
    """
    try:
        from jiwer import cer
        return cer(references, predictions)
    except ImportError:
        # Fallback: character-level WER
        total_errors = 0
        total_chars = 0

        for pred, ref in zip(predictions, references):
            pred_chars = list(pred.lower())
            ref_chars = list(ref.lower())

            m, n = len(ref_chars), len(pred_chars)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref_chars[i - 1] == pred_chars[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

            total_errors += dp[m][n]
            total_chars += m

        return total_errors / max(total_chars, 1)


def evaluate_model(
    model,
    test_samples: List[Dict],
    batch_size: int = 4,
    max_samples: Optional[int] = None
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Evaluate model on test samples.

    Args:
        model: NeMo ASR model
        test_samples: List of test samples
        batch_size: Batch size for inference
        max_samples: Maximum number of samples to evaluate

    Returns:
        Tuple of (metrics dict, detailed results list)
    """
    if max_samples:
        test_samples = test_samples[:max_samples]

    predictions = []
    references = []
    detailed_results = []

    # Process in batches
    audio_paths = [s['audio_filepath'] for s in test_samples]

    logger.info(f"Transcribing {len(audio_paths)} audio files...")

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Evaluating"):
        batch_paths = audio_paths[i:i + batch_size]
        batch_samples = test_samples[i:i + batch_size]

        try:
            transcriptions = model.transcribe(paths2audio_files=batch_paths, batch_size=batch_size)

            for j, (trans, sample) in enumerate(zip(transcriptions, batch_samples)):
                reference = sample.get('text', '')
                predictions.append(trans)
                references.append(reference)

                detailed_results.append({
                    'audio_filepath': sample['audio_filepath'],
                    'reference': reference,
                    'prediction': trans,
                    'duration': sample.get('duration', 0)
                })

        except Exception as e:
            logger.warning(f"Error processing batch {i}: {e}")
            continue

    # Compute metrics
    metrics = {
        'wer': compute_wer(predictions, references),
        'cer': compute_cer(predictions, references),
        'num_samples': len(predictions)
    }

    return metrics, detailed_results


def print_sample_results(results: List[Dict], num_samples: int = 5):
    """Print sample transcription results."""
    print("\n" + "=" * 80)
    print("SAMPLE TRANSCRIPTIONS")
    print("=" * 80)

    for i, result in enumerate(results[:num_samples]):
        print(f"\nSample {i + 1}:")
        print(f"  Audio: {Path(result['audio_filepath']).name}")
        print(f"  Reference:  {result['reference']}")
        print(f"  Prediction: {result['prediction']}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Parakeet ASR model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned .nemo model or model name"
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        default="data/test_manifest.json",
        help="Path to test manifest file"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Optional: baseline model to compare against"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save detailed results JSON"
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Number of sample transcriptions to display"
    )

    args = parser.parse_args()

    # Check test manifest exists
    if not Path(args.test_manifest).exists():
        logger.error(f"Test manifest not found: {args.test_manifest}")
        return

    # Load test data
    test_samples = load_test_data(args.test_manifest)

    if not test_samples:
        logger.error("No test samples found")
        return

    # Evaluate fine-tuned model
    logger.info("Loading fine-tuned model...")
    model = load_model(args.model)

    logger.info("Evaluating fine-tuned model...")
    metrics, results = evaluate_model(
        model,
        test_samples,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nModel: {args.model}")
    print(f"Test set: {args.test_manifest}")
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"\nWord Error Rate (WER):      {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"Character Error Rate (CER): {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")

    # Evaluate baseline if specified
    if args.baseline:
        logger.info(f"\nLoading baseline model: {args.baseline}")
        baseline_model = load_model(args.baseline)

        logger.info("Evaluating baseline model...")
        baseline_metrics, _ = evaluate_model(
            baseline_model,
            test_samples,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )

        print(f"\n--- BASELINE COMPARISON ---")
        print(f"Baseline: {args.baseline}")
        print(f"Baseline WER: {baseline_metrics['wer']:.4f} ({baseline_metrics['wer']*100:.2f}%)")
        print(f"Baseline CER: {baseline_metrics['cer']:.4f} ({baseline_metrics['cer']*100:.2f}%)")

        wer_improvement = baseline_metrics['wer'] - metrics['wer']
        cer_improvement = baseline_metrics['cer'] - metrics['cer']

        print(f"\nWER Improvement: {wer_improvement:.4f} ({wer_improvement*100:.2f}%)")
        print(f"CER Improvement: {cer_improvement:.4f} ({cer_improvement*100:.2f}%)")

        if wer_improvement > 0:
            print(f"\nFine-tuned model is {wer_improvement/baseline_metrics['wer']*100:.1f}% better (WER)")
        else:
            print(f"\nFine-tuned model is {-wer_improvement/baseline_metrics['wer']*100:.1f}% worse (WER)")

    print("=" * 60)

    # Print sample results
    if args.show_samples > 0:
        print_sample_results(results, args.show_samples)

    # Save detailed results
    if args.output:
        output_data = {
            'model': args.model,
            'test_manifest': args.test_manifest,
            'metrics': metrics,
            'detailed_results': results
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
