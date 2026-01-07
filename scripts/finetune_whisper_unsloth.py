#!/usr/bin/env python3
"""
Whisper Large-v3-Turbo Fine-Tuning with Unsloth

This script fine-tunes OpenAI's Whisper model using Unsloth for efficient
LoRA-based training. Uses HuggingFace datasets directly.

Usage:
    python scripts/finetune_whisper_unsloth.py --config configs/whisper_hindi_unsloth.yaml

    Or with command line overrides:
    python scripts/finetune_whisper_unsloth.py \
        --dataset milind-plivo/parakeet-training-dataset \
        --output-dir outputs/whisper-hindi \
        --epochs 10

Prerequisites:
    pip install unsloth transformers datasets accelerate evaluate jiwer
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from datasets import Audio, Dataset, DatasetDict, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    use_unsloth = False

    try:
        from unsloth import FastModel
        logger.info("Unsloth imported successfully")
        use_unsloth = True
    except (ImportError, NotImplementedError, AttributeError, Exception) as e:
        logger.warning(f"Unsloth not available: {e}")
        logger.info("Will use standard transformers + PEFT instead")

    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        logger.info("Transformers imported successfully")
    except ImportError:
        missing.append("transformers")

    try:
        import evaluate
    except ImportError:
        missing.append("evaluate")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) detected")
    else:
        logger.warning("No GPU detected! Training will be slow.")

    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False, use_unsloth

    return True, use_unsloth


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_from_local_manifests(
    data_dir: str = "data",
    sampling_rate: int = 16000,
) -> DatasetDict:
    """
    Load dataset from local NeMo-format manifests.

    Args:
        data_dir: Directory containing manifest files
        sampling_rate: Target sampling rate

    Returns:
        DatasetDict with train/validation/test splits
    """
    import json

    data_path = Path(data_dir)
    splits = {}

    for split_name in ["train", "val", "test"]:
        manifest_path = data_path / f"{split_name}_manifest.json"
        if not manifest_path.exists():
            logger.warning(f"Manifest not found: {manifest_path}")
            continue

        audio_paths = []
        transcriptions = []
        durations = []

        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                audio_paths.append(entry["audio_filepath"])
                transcriptions.append(entry["text"])
                durations.append(entry.get("duration", 0))

        # Create dataset
        split_dataset = Dataset.from_dict({
            "audio": audio_paths,
            "transcription": transcriptions,
            "duration": durations,
        }).cast_column("audio", Audio(sampling_rate=sampling_rate))

        # Map val to validation for compatibility
        key = "validation" if split_name == "val" else split_name
        splits[key] = split_dataset
        logger.info(f"Loaded {len(split_dataset)} samples from {split_name}")

    return DatasetDict(splits)


def load_and_prepare_dataset(
    dataset_name: str,
    audio_column: str = "audio",
    text_column: str = "transcription",
    sampling_rate: int = 16000,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_samples: Optional[int] = None,
    local_data_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Load dataset from HuggingFace or local manifests.

    Args:
        dataset_name: HuggingFace dataset name
        audio_column: Name of audio column
        text_column: Name of text column
        sampling_rate: Target sampling rate
        train_split: Training set ratio
        val_split: Validation set ratio
        test_split: Test set ratio
        max_samples: Optional limit on samples
        local_data_dir: If specified, load from local manifests instead

    Returns:
        DatasetDict with train/validation/test splits
    """
    # Try local data first if specified or if HuggingFace fails
    if local_data_dir:
        logger.info(f"Loading from local data: {local_data_dir}")
        return load_from_local_manifests(local_data_dir, sampling_rate)

    logger.info(f"Loading dataset: {dataset_name}")

    # Try loading from HuggingFace
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        logger.warning(f"Failed to load from HuggingFace: {e}")
        logger.info("Attempting to load from local data directory...")
        if Path("data/train_manifest.json").exists():
            return load_from_local_manifests("data", sampling_rate)
        raise

    # Handle different dataset structures
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            full_dataset = dataset["train"]
        else:
            # Use first available split
            first_split = list(dataset.keys())[0]
            full_dataset = dataset[first_split]
            logger.info(f"Using split: {first_split}")
    else:
        full_dataset = dataset

    logger.info(f"Loaded {len(full_dataset)} samples")
    logger.info(f"Columns: {full_dataset.column_names}")

    # Limit samples if specified
    if max_samples and len(full_dataset) > max_samples:
        full_dataset = full_dataset.select(range(max_samples))
        logger.info(f"Limited to {max_samples} samples")

    # Cast audio column to correct sampling rate
    full_dataset = full_dataset.cast_column(
        audio_column,
        Audio(sampling_rate=sampling_rate)
    )

    # Create splits
    logger.info("Creating train/val/test splits...")

    # First split: train vs (val + test)
    split1 = full_dataset.train_test_split(
        test_size=(val_split + test_split),
        seed=42
    )
    train = split1['train']

    # Second split: val vs test
    if test_split > 0:
        val_test_ratio = test_split / (val_split + test_split)
        split2 = split1['test'].train_test_split(
            test_size=val_test_ratio,
            seed=42
        )
        val = split2['train']
        test = split2['test']
    else:
        val = split1['test']
        test = Dataset.from_dict({k: [] for k in full_dataset.column_names})

    dataset_dict = DatasetDict({
        'train': train,
        'validation': val,
        'test': test
    })

    logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    return dataset_dict


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper training."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present at start
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def finetune_whisper(config: dict, use_unsloth: bool = True):
    """
    Fine-tune Whisper model using Unsloth or standard transformers+PEFT.

    Args:
        config: Configuration dictionary
        use_unsloth: Whether to use Unsloth (requires NVIDIA GPU)
    """
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    import evaluate

    # Extract config sections
    model_config = config.get('model', {})
    lora_config = config.get('lora', {})
    dataset_config = config.get('dataset', {})
    training_config = config.get('training', {})

    # Create output directory
    output_dir = training_config.get('output_dir', './outputs/whisper-hindi-unsloth')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save effective config
    with open(Path(output_dir) / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Model settings
    model_name = model_config.get('name', 'unsloth/whisper-large-v3-turbo')
    load_in_4bit = model_config.get('load_in_4bit', False)
    language = model_config.get('language', 'Hindi')
    task = model_config.get('task', 'transcribe')

    # Use OpenAI model if Unsloth is not available
    if not use_unsloth and model_name.startswith('unsloth/'):
        model_name = model_name.replace('unsloth/', 'openai/')
        logger.info(f"Switched to OpenAI model: {model_name}")

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Language: {language}, Task: {task}")
    logger.info(f"4-bit quantization: {load_in_4bit}")
    logger.info(f"Using Unsloth: {use_unsloth}")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    if use_unsloth:
        from unsloth import FastModel, is_bfloat16_supported

        model, processor = FastModel.from_pretrained(
            model_name=model_name,
            dtype=None,
            load_in_4bit=load_in_4bit,
            auto_model=WhisperForConditionalGeneration,
            whisper_language=language,
            whisper_task=task,
        )
    else:
        # Standard transformers loading
        processor = WhisperProcessor.from_pretrained(model_name)
        processor.tokenizer.set_prefix_tokens(language=language, task=task)

        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        model = model.to(device)

    logger.info("Model loaded successfully")

    # Apply LoRA
    lora_r = lora_config.get('r', 32)
    lora_alpha = lora_config.get('alpha', 32)
    lora_dropout = lora_config.get('dropout', 0)
    target_modules = lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
    use_gradient_checkpointing = lora_config.get('use_gradient_checkpointing', True)

    logger.info(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}")
    logger.info(f"Target modules: {target_modules}")

    if use_unsloth:
        from unsloth import FastModel
        model = FastModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth" if use_gradient_checkpointing else False,
            random_state=config.get('seed', 42),
            use_rslora=False,
            loftq_config=None,
        )
    else:
        # Standard PEFT LoRA
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        lora_config_obj = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config_obj)

    logger.info("LoRA applied successfully")
    model.print_trainable_parameters()

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(
        dataset_name=dataset_config.get('name', 'milind-plivo/parakeet-training-dataset'),
        audio_column=dataset_config.get('audio_column', 'audio'),
        text_column=dataset_config.get('text_column', 'transcription'),
        sampling_rate=dataset_config.get('sampling_rate', 16000),
        train_split=dataset_config.get('train_split', 0.8),
        val_split=dataset_config.get('val_split', 0.1),
        test_split=dataset_config.get('test_split', 0.1),
    )

    # Preprocess dataset for Whisper
    audio_column = dataset_config.get('audio_column', 'audio')
    text_column = dataset_config.get('text_column', 'transcription')

    def prepare_dataset(batch):
        audio = batch[audio_column]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch[text_column]).input_ids
        return batch

    logger.info("Preprocessing dataset...")
    processed_dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset['train'].column_names,
        num_proc=1,  # Audio processing doesn't parallelize well
    )

    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer") if config.get('evaluation', {}).get('compute_cer', True) else None

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute metrics
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        metrics = {"wer": wer}

        if cer_metric:
            cer = cer_metric.compute(predictions=pred_str, references=label_str)
            metrics["cer"] = cer

        return metrics

    # Training arguments
    # Determine precision settings
    use_fp16 = training_config.get('fp16', True) and device == "cuda"
    use_bf16 = False
    if use_unsloth:
        from unsloth import is_bfloat16_supported
        use_bf16 = training_config.get('bf16', False) and is_bfloat16_supported()

    # Use adamw_torch for non-CUDA devices
    optimizer = training_config.get('optim', 'adamw_8bit')
    if device != "cuda":
        optimizer = "adamw_torch"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=training_config.get('learning_rate', 1e-4),
        num_train_epochs=training_config.get('num_train_epochs', 10),
        warmup_steps=training_config.get('warmup_steps', 50),
        weight_decay=training_config.get('weight_decay', 0.01),
        fp16=use_fp16,
        bf16=use_bf16,
        optim=optimizer,
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
        logging_steps=training_config.get('logging_steps', 10),
        eval_strategy="steps",
        eval_steps=training_config.get('eval_steps', 50),
        save_steps=training_config.get('save_steps', 100),
        save_total_limit=training_config.get('save_total_limit', 3),
        predict_with_generate=training_config.get('predict_with_generate', True),
        generation_max_length=training_config.get('generation_max_length', 225),
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",  # Disable external logging
        seed=config.get('seed', 42),
    )

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # Print training info
    print("\n" + "=" * 60)
    print("STARTING WHISPER FINE-TUNING WITH UNSLOTH")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Language: {language}")
    print(f"LoRA rank: {lora_r}")
    print(f"Dataset: {dataset_config.get('name')}")
    print(f"Train samples: {len(processed_dataset['train'])}")
    print(f"Val samples: {len(processed_dataset['validation'])}")
    print(f"Batch size: {training_config.get('per_device_train_batch_size', 4)}")
    print(f"Epochs: {training_config.get('num_train_epochs', 10)}")
    print(f"Output: {output_dir}")
    print("=" * 60 + "\n")

    # Train
    train_result = trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    # Save LoRA weights separately
    lora_path = Path(output_dir) / "lora_weights"
    model.save_pretrained(str(lora_path))
    logger.info(f"LoRA weights saved to: {lora_path}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(processed_dataset['test'])

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Test WER: {test_results.get('eval_wer', 'N/A'):.4f}")
    if 'eval_cer' in test_results:
        print(f"Test CER: {test_results.get('eval_cer', 'N/A'):.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)

    # Save results
    results = {
        'train_results': {
            'train_loss': train_result.training_loss,
            'train_samples': len(processed_dataset['train']),
        },
        'test_results': test_results,
        'config': config,
    }

    import json
    with open(Path(output_dir) / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper with Unsloth"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/whisper_hindi_unsloth.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Override dataset name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate"
    )
    parser.add_argument(
        "--4bit",
        action="store_true",
        dest="load_in_4bit",
        help="Use 4-bit quantization"
    )

    args = parser.parse_args()

    # Check dependencies
    deps_ok, use_unsloth = check_dependencies()
    if not deps_ok:
        return

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        logger.info(f"Loaded config from: {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {}

    # Apply command line overrides
    if args.dataset:
        config.setdefault('dataset', {})['name'] = args.dataset
    if args.output_dir:
        config.setdefault('training', {})['output_dir'] = args.output_dir
    if args.epochs:
        config.setdefault('training', {})['num_train_epochs'] = args.epochs
    if args.batch_size:
        config.setdefault('training', {})['per_device_train_batch_size'] = args.batch_size
    if args.lr:
        config.setdefault('training', {})['learning_rate'] = args.lr
    if args.load_in_4bit:
        config.setdefault('model', {})['load_in_4bit'] = True

    # Run fine-tuning
    finetune_whisper(config, use_unsloth=use_unsloth)


if __name__ == "__main__":
    main()
