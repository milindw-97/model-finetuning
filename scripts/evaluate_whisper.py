#!/usr/bin/env python3
"""
Whisper Model Evaluation Script

Evaluates fine-tuned Whisper models, computes WER/CER metrics,
generates sample transcriptions, and optionally compares with base model.

Usage:
    # Evaluate fine-tuned model
    python scripts/evaluate_whisper.py --model outputs/whisper-hindi-unsloth

    # Compare fine-tuned with base model
    python scripts/evaluate_whisper.py \
        --model outputs/whisper-hindi-unsloth \
        --compare-base

    # Evaluate on specific dataset
    python scripts/evaluate_whisper.py \
        --model outputs/whisper-hindi-unsloth \
        --dataset milind-plivo/parakeet-training-dataset \
        --split test
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from datasets import Audio, load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_processor(
    model_path: str,
    load_in_4bit: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple:
    """
    Load Whisper model and processor.

    Args:
        model_path: Path to fine-tuned model or HuggingFace model name
        load_in_4bit: Whether to load in 4-bit quantization
        device: Device to load model on

    Returns:
        Tuple of (model, processor)
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_path = Path(model_path)

    # Check if this is a fine-tuned model with LoRA weights
    lora_path = model_path / "lora_weights"
    if lora_path.exists():
        logger.info(f"Loading fine-tuned model with LoRA from: {model_path}")

        # Try to load from Unsloth
        try:
            from unsloth import FastModel

            # Load config to get base model name
            config_path = model_path / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                base_model = config.get('model', {}).get('name', 'unsloth/whisper-large-v3-turbo')
                language = config.get('model', {}).get('language', 'Hindi')
                task = config.get('model', {}).get('task', 'transcribe')
            else:
                base_model = 'unsloth/whisper-large-v3-turbo'
                language = 'Hindi'
                task = 'transcribe'

            model, processor = FastModel.from_pretrained(
                model_name=base_model,
                dtype=None,
                load_in_4bit=load_in_4bit,
                auto_model=WhisperForConditionalGeneration,
                whisper_language=language,
                whisper_task=task,
            )

            # Load LoRA weights
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(lora_path))
            model = model.merge_and_unload()
            logger.info("LoRA weights merged successfully")

        except Exception as e:
            logger.warning(f"Could not load with Unsloth: {e}")
            logger.info("Trying standard transformers loading...")

            # Fallback to standard loading
            processor = WhisperProcessor.from_pretrained(str(model_path))
            model = WhisperForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )

    elif model_path.exists():
        # Load from local checkpoint
        logger.info(f"Loading model from: {model_path}")
        processor = WhisperProcessor.from_pretrained(str(model_path))
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    else:
        # Load from HuggingFace
        logger.info(f"Loading model from HuggingFace: {model_path}")
        processor = WhisperProcessor.from_pretrained(str(model_path))
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    model = model.to(device)
    model.eval()

    return model, processor


def transcribe_audio(
    model,
    processor,
    audio_array,
    sampling_rate: int = 16000,
    language: str = "hindi",
    task: str = "transcribe",
    device: str = "cuda"
) -> str:
    """
    Transcribe a single audio sample.

    Args:
        model: Whisper model
        processor: Whisper processor
        audio_array: Audio waveform as numpy array
        sampling_rate: Audio sampling rate
        language: Target language
        task: "transcribe" or "translate"
        device: Device to use

    Returns:
        Transcribed text
    """
    # Prepare input features
    input_features = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    ).input_features.to(device)

    # Force language and task tokens
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task=task
    )

    # Generate
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=225,
        )

    # Decode
    transcription = processor.tokenizer.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription


def evaluate_model(
    model,
    processor,
    dataset,
    audio_column: str = "audio",
    text_column: str = "transcription",
    language: str = "hindi",
    task: str = "transcribe",
    max_samples: Optional[int] = None,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate model on dataset.

    Args:
        model: Whisper model
        processor: Whisper processor
        dataset: HuggingFace dataset
        audio_column: Name of audio column
        text_column: Name of text/transcription column
        language: Target language
        task: "transcribe" or "translate"
        max_samples: Optional limit on samples to evaluate
        device: Device to use

    Returns:
        Dictionary with evaluation results
    """
    import evaluate

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    predictions = []
    references = []
    samples = []

    # Limit samples if specified
    eval_dataset = dataset
    if max_samples and len(dataset) > max_samples:
        eval_dataset = dataset.select(range(max_samples))

    logger.info(f"Evaluating on {len(eval_dataset)} samples...")

    for i, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        audio = sample[audio_column]
        reference = sample[text_column]

        # Transcribe
        prediction = transcribe_audio(
            model=model,
            processor=processor,
            audio_array=audio["array"],
            sampling_rate=audio["sampling_rate"],
            language=language,
            task=task,
            device=device
        )

        predictions.append(prediction)
        references.append(reference)

        # Store samples for display
        if len(samples) < 10:
            samples.append({
                "index": i,
                "reference": reference,
                "prediction": prediction,
                "duration": sample.get("duration", len(audio["array"]) / audio["sampling_rate"])
            })

    # Compute metrics
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    results = {
        "wer": wer,
        "cer": cer,
        "num_samples": len(predictions),
        "samples": samples,
    }

    return results


def print_results(results: Dict, model_name: str = "Model"):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 70)
    print(f"Word Error Rate (WER):       {results['wer']:.4f} ({results['wer']*100:.2f}%)")
    print(f"Character Error Rate (CER):  {results['cer']:.4f} ({results['cer']*100:.2f}%)")
    print(f"Number of samples:           {results['num_samples']}")
    print("=" * 70)

    print("\nSAMPLE TRANSCRIPTIONS:")
    print("-" * 70)
    for sample in results.get("samples", [])[:5]:
        print(f"\n[Sample {sample['index']}] Duration: {sample['duration']:.2f}s")
        print(f"  Reference:  {sample['reference'][:100]}{'...' if len(sample['reference']) > 100 else ''}")
        print(f"  Prediction: {sample['prediction'][:100]}{'...' if len(sample['prediction']) > 100 else ''}")
    print("-" * 70 + "\n")


def compare_models(
    finetuned_results: Dict,
    base_results: Dict
):
    """Print comparison between fine-tuned and base model."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: Fine-tuned vs Base")
    print("=" * 70)

    wer_improvement = base_results['wer'] - finetuned_results['wer']
    cer_improvement = base_results['cer'] - finetuned_results['cer']

    wer_pct = (wer_improvement / base_results['wer']) * 100 if base_results['wer'] > 0 else 0
    cer_pct = (cer_improvement / base_results['cer']) * 100 if base_results['cer'] > 0 else 0

    print(f"\n{'Metric':<25} {'Base':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'WER':<25} {base_results['wer']:.4f}        {finetuned_results['wer']:.4f}         {wer_improvement:+.4f} ({wer_pct:+.1f}%)")
    print(f"{'CER':<25} {base_results['cer']:.4f}        {finetuned_results['cer']:.4f}         {cer_improvement:+.4f} ({cer_pct:+.1f}%)")
    print("=" * 70)

    if wer_improvement > 0:
        print(f"\nFine-tuning IMPROVED WER by {wer_pct:.1f}%")
    elif wer_improvement < 0:
        print(f"\nFine-tuning DEGRADED WER by {-wer_pct:.1f}%")
    else:
        print("\nNo change in WER")

    print("\nSIDE-BY-SIDE SAMPLE COMPARISON:")
    print("-" * 70)

    for i, (ft_sample, base_sample) in enumerate(zip(
        finetuned_results.get('samples', [])[:3],
        base_results.get('samples', [])[:3]
    )):
        print(f"\n[Sample {i}]")
        print(f"  Reference:     {ft_sample['reference'][:80]}...")
        print(f"  Fine-tuned:    {ft_sample['prediction'][:80]}...")
        print(f"  Base model:    {base_sample['prediction'][:80]}...")

    print("-" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model or HuggingFace model name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="milind-plivo/parakeet-training-dataset",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default="audio",
        help="Name of audio column"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="transcription",
        help="Name of text/transcription column"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="hindi",
        help="Target language"
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also evaluate base model for comparison"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Base model for comparison"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--4bit",
        action="store_true",
        dest="load_in_4bit",
        help="Load model in 4-bit quantization"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)

    # Get the right split
    if args.split in dataset:
        eval_data = dataset[args.split]
    elif "split" in dataset.column_names.get("train", []):
        # Filter by split column
        full_data = dataset["train"]
        eval_data = full_data.filter(lambda x: x.get("split") == args.split)
        if len(eval_data) == 0:
            logger.warning(f"No samples found for split '{args.split}', using full dataset")
            eval_data = full_data
    else:
        # Use full dataset
        logger.warning(f"Split '{args.split}' not found, using available data")
        eval_data = dataset[list(dataset.keys())[0]]

    logger.info(f"Evaluation samples: {len(eval_data)}")

    # Cast audio column
    eval_data = eval_data.cast_column(args.audio_column, Audio(sampling_rate=16000))

    # Load and evaluate fine-tuned model
    logger.info(f"Loading fine-tuned model: {args.model}")
    model, processor = load_model_and_processor(
        args.model,
        load_in_4bit=args.load_in_4bit,
        device=device
    )

    finetuned_results = evaluate_model(
        model=model,
        processor=processor,
        dataset=eval_data,
        audio_column=args.audio_column,
        text_column=args.text_column,
        language=args.language,
        max_samples=args.max_samples,
        device=device
    )

    print_results(finetuned_results, "Fine-tuned Model")

    # Optionally compare with base model
    base_results = None
    if args.compare_base:
        logger.info(f"Loading base model for comparison: {args.base_model}")

        # Clear GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        base_model, base_processor = load_model_and_processor(
            args.base_model,
            load_in_4bit=args.load_in_4bit,
            device=device
        )

        base_results = evaluate_model(
            model=base_model,
            processor=base_processor,
            dataset=eval_data,
            audio_column=args.audio_column,
            text_column=args.text_column,
            language=args.language,
            max_samples=args.max_samples,
            device=device
        )

        print_results(base_results, "Base Model")
        compare_models(finetuned_results, base_results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
            "finetuned_results": {
                "wer": finetuned_results["wer"],
                "cer": finetuned_results["cer"],
                "num_samples": finetuned_results["num_samples"],
            }
        }
        if base_results:
            output_data["base_model"] = args.base_model
            output_data["base_results"] = {
                "wer": base_results["wer"],
                "cer": base_results["cer"],
                "num_samples": base_results["num_samples"],
            }
            output_data["improvement"] = {
                "wer": base_results["wer"] - finetuned_results["wer"],
                "cer": base_results["cer"] - finetuned_results["cer"],
            }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
