#!/usr/bin/env python3
"""
Parakeet RNNT 1.1B Multilingual Fine-Tuning Script

This script provides a complete pipeline for fine-tuning NVIDIA's
parakeet-rnnt-1.1b-multilingual model using the NeMo framework.

Based on: https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/Multilang_ASR.ipynb

Usage:
    python parakeet_rnnt_finetuning.py --config config.yaml

    Or modify the CONFIG section below and run directly.

Requirements:
    pip install nemo_toolkit[asr] pytorch-lightning omegaconf soundfile
"""

import os
import tempfile

# Set temp directory to a location with more space (for .nemo file extraction)
CUSTOM_TMPDIR = "/workspace/tmp"
os.makedirs(CUSTOM_TMPDIR, exist_ok=True)
os.environ["TMPDIR"] = CUSTOM_TMPDIR
os.environ["TEMP"] = CUSTOM_TMPDIR
os.environ["TMP"] = CUSTOM_TMPDIR
tempfile.tempdir = CUSTOM_TMPDIR
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
import shutil

# ==============================================================================
# INSTALLATION CHECK
# ==============================================================================


def check_and_install_dependencies():
    """Check and provide installation instructions for required packages."""
    missing = []

    try:
        import torch

        print(f"✓ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA: {torch.version.cuda}")
    except ImportError:
        missing.append("torch")

    try:
        import nemo
        import nemo.collections.asr as nemo_asr

        print(f"✓ NeMo: {nemo.__version__}")
    except ImportError:
        missing.append("nemo_toolkit[asr]")

    try:
        import pytorch_lightning as pl

        print(f"✓ PyTorch Lightning: {pl.__version__}")
    except ImportError:
        missing.append("pytorch-lightning")

    try:
        from omegaconf import OmegaConf

        print("✓ OmegaConf")
    except ImportError:
        missing.append("omegaconf")

    try:
        import soundfile

        print("✓ SoundFile")
    except ImportError:
        missing.append("soundfile")

    try:
        import librosa

        print("✓ Librosa")
    except ImportError:
        missing.append("librosa")

    if missing:
        print("\n" + "=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        print("Please install the following packages:")
        print(f"\n  pip install {' '.join(missing)}")
        print("\nFor full NeMo installation:")
        print("  pip install 'nemo_toolkit[asr]'")
        print("\nOr from source (recommended for latest features):")
        print(
            "  pip install git+https://github.com/NVIDIA/NeMo.git#egg=nemo_toolkit[asr]"
        )
        print("=" * 60)
        return False

    return True


# ==============================================================================
# CONFIGURATION
# ==============================================================================


@dataclass
class DataConfig:
    """Configuration for dataset handling."""

    # Manifest file paths (NeMo JSON Lines format)
    train_manifest: str = "./data/train_manifest.json"
    val_manifest: str = "./data/val_manifest.json"
    test_manifest: Optional[str] = "./data/test_manifest.json"

    # Audio settings
    sample_rate: int = 16000
    max_duration: float = 20.0  # Max audio duration in seconds
    min_duration: float = 0.1  # Min audio duration in seconds

    # Batch settings
    train_batch_size: int = 16
    val_batch_size: int = 16
    num_workers: int = 4

    # Augmentation
    enable_spec_augment: bool = True
    freq_masks: int = 2
    time_masks: int = 10
    freq_width: int = 27
    time_width: float = 0.05


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model
    pretrained_model: str = "nvidia/parakeet-rnnt-1.1b-multilingual"

    # Output
    output_dir: str = "./outputs/parakeet-rnnt-finetuned"
    exp_name: str = "parakeet_rnnt_finetune"

    # Training hyperparameters
    max_epochs: int = 50
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    warmup_steps: int = 1000
    weight_decay: float = 1e-3

    # Optimizer
    optimizer: str = "adamw"  # adam, adamw, sgd, novograd

    # Scheduler
    scheduler: str = (
        "CosineAnnealing"  # WarmupAnnealing, CosineAnnealing, NoamAnnealing
    )

    # Precision
    precision: str = "16-mixed"  # 32, 16-mixed, bf16-mixed

    # Gradient
    grad_clip: float = 1.0
    accumulate_grad_batches: int = 1

    # Checkpointing
    save_top_k: int = 3
    save_last: bool = True
    checkpoint_every_n_epochs: int = 1

    # Early stopping
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.001

    # Encoder freezing (useful for limited data)
    freeze_encoder: bool = False
    freeze_encoder_epochs: int = 0  # Freeze for N epochs, then unfreeze

    # Resume
    resume_from_checkpoint: Optional[str] = None

    # Device
    devices: int = 1  # Number of GPUs
    accelerator: str = "gpu"  # gpu, cpu, auto

    # Logging
    log_every_n_steps: int = 50


@dataclass
class Config:
    """Master configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Random seed
    seed: int = 42


# ==============================================================================
# MANIFEST UTILITIES
# ==============================================================================


def create_manifest_from_folder(
    audio_dir: str,
    output_manifest: str,
    transcription_file: Optional[str] = None,
    language: str = "en",
    audio_extensions: List[str] = [".wav", ".flac", ".mp3", ".ogg"],
) -> str:
    """
    Create a NeMo manifest file from a folder of audio files.

    Args:
        audio_dir: Directory containing audio files
        output_manifest: Path to output manifest file
        transcription_file: Optional JSON file mapping audio filenames to transcriptions
                           Format: {"audio1.wav": "transcription 1", ...}
        language: Language code for the audio (used for multilingual models)
        audio_extensions: List of audio file extensions to include

    Returns:
        Path to created manifest file

    Manifest format (one JSON per line):
        {"audio_filepath": "/path/to/audio.wav", "text": "transcription", "duration": 2.5, "lang": "en"}
    """
    import soundfile as sf

    audio_dir = Path(audio_dir)
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    # Load transcriptions if provided
    transcriptions = {}
    if transcription_file and Path(transcription_file).exists():
        with open(transcription_file, "r", encoding="utf-8") as f:
            transcriptions = json.load(f)

    entries = []

    for ext in audio_extensions:
        for audio_path in audio_dir.rglob(f"*{ext}"):
            try:
                # Get audio duration
                info = sf.info(str(audio_path))
                duration = info.duration

                # Get transcription
                filename = audio_path.name
                text = transcriptions.get(
                    filename, transcriptions.get(str(audio_path), "")
                )

                if not text:
                    logging.warning(f"No transcription for {filename}, skipping...")
                    continue

                entry = {
                    "audio_filepath": str(audio_path.absolute()),
                    "text": text.strip(),
                    "duration": round(duration, 3),
                    "lang": language,
                }
                entries.append(entry)

            except Exception as e:
                logging.warning(f"Error processing {audio_path}: {e}")
                continue

    # Write manifest
    with open(output_manifest, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logging.info(f"Created manifest with {len(entries)} entries: {output_manifest}")
    return str(output_manifest)


def create_manifest_from_dataset(
    dataset,  # HuggingFace Dataset
    output_manifest: str,
    audio_column: str = "audio",
    text_column: str = "sentence",
    language: str = "en",
    audio_output_dir: Optional[str] = None,
) -> str:
    """
    Create a NeMo manifest from a HuggingFace dataset.

    Args:
        dataset: HuggingFace Dataset object
        output_manifest: Path to output manifest file
        audio_column: Name of the audio column
        text_column: Name of the text/transcription column
        language: Language code
        audio_output_dir: Directory to save audio files (if dataset has in-memory audio)

    Returns:
        Path to created manifest file
    """
    import soundfile as sf

    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    if audio_output_dir:
        audio_output_dir = Path(audio_output_dir)
        audio_output_dir.mkdir(parents=True, exist_ok=True)

    entries = []

    for idx, sample in enumerate(dataset):
        try:
            audio = sample[audio_column]
            text = sample[text_column]

            if isinstance(audio, dict):
                # In-memory audio from HuggingFace
                array = audio["array"]
                sr = audio.get("sampling_rate", 16000)

                if audio_output_dir:
                    # Save audio file
                    audio_path = audio_output_dir / f"audio_{idx:06d}.wav"
                    sf.write(str(audio_path), array, sr)
                    audio_filepath = str(audio_path.absolute())
                else:
                    logging.warning(
                        f"Sample {idx}: No audio_output_dir provided for in-memory audio"
                    )
                    continue

                duration = len(array) / sr
            else:
                # Audio is a file path
                audio_filepath = audio if isinstance(audio, str) else str(audio)
                info = sf.info(audio_filepath)
                duration = info.duration

            entry = {
                "audio_filepath": audio_filepath,
                "text": text.strip(),
                "duration": round(duration, 3),
                "lang": language,
            }
            entries.append(entry)

        except Exception as e:
            logging.warning(f"Error processing sample {idx}: {e}")
            continue

    # Write manifest
    with open(output_manifest, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logging.info(f"Created manifest with {len(entries)} entries: {output_manifest}")
    return str(output_manifest)


def validate_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Validate a manifest file and return statistics.

    Returns:
        Dictionary with statistics about the manifest
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    stats = {
        "total_samples": 0,
        "total_duration_hours": 0,
        "missing_files": [],
        "languages": {},
        "duration_range": {"min": float("inf"), "max": 0},
        "text_length_range": {"min": float("inf"), "max": 0},
    }

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())

                # Check required fields
                audio_path = entry.get("audio_filepath")
                text = entry.get("text", "")
                duration = entry.get("duration", 0)
                lang = entry.get("lang", "unknown")

                stats["total_samples"] += 1
                stats["total_duration_hours"] += duration / 3600

                # Check if audio file exists
                if audio_path and not Path(audio_path).exists():
                    stats["missing_files"].append(audio_path)

                # Language stats
                stats["languages"][lang] = stats["languages"].get(lang, 0) + 1

                # Duration range
                stats["duration_range"]["min"] = min(
                    stats["duration_range"]["min"], duration
                )
                stats["duration_range"]["max"] = max(
                    stats["duration_range"]["max"], duration
                )

                # Text length range
                text_len = len(text)
                stats["text_length_range"]["min"] = min(
                    stats["text_length_range"]["min"], text_len
                )
                stats["text_length_range"]["max"] = max(
                    stats["text_length_range"]["max"], text_len
                )

            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON at line {line_num}")

    # Format output
    if stats["duration_range"]["min"] == float("inf"):
        stats["duration_range"]["min"] = 0
    if stats["text_length_range"]["min"] == float("inf"):
        stats["text_length_range"]["min"] = 0

    return stats


def print_manifest_stats(manifest_path: str):
    """Print manifest statistics."""
    try:
        stats = validate_manifest(manifest_path)
        print(f"\n📊 Manifest Statistics: {manifest_path}")
        print(f"  Total samples: {stats['total_samples']:,}")
        print(f"  Total duration: {stats['total_duration_hours']:.2f} hours")
        print(
            f"  Duration range: {stats['duration_range']['min']:.2f}s - {stats['duration_range']['max']:.2f}s"
        )
        print(
            f"  Text length range: {stats['text_length_range']['min']} - {stats['text_length_range']['max']} chars"
        )
        print(f"  Languages: {stats['languages']}")
        if stats["missing_files"]:
            print(f"  ⚠️  Missing files: {len(stats['missing_files'])}")
    except Exception as e:
        print(f"  Error validating manifest: {e}")


# ==============================================================================
# MODEL FINE-TUNING
# ==============================================================================


class ParakeetRNNTFineTuner:
    """Fine-tuner for Parakeet RNNT 1.1B Multilingual model."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.trainer = None

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory
        Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)

    def load_pretrained_model(self):
        """Load the pre-trained Parakeet RNNT model."""
        import nemo.collections.asr as nemo_asr

        self.logger.info(
            f"Loading pre-trained model: {self.config.training.pretrained_model}"
        )

        model_name = self.config.training.pretrained_model

        # Handle different model name formats
        if model_name.endswith(".nemo"):
            # Local .nemo file
            self.model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_name)
        elif "/" in model_name:
            # HuggingFace style name (nvidia/parakeet-rnnt-1.1b-multilingual)
            # NeMo uses the model name directly
            model_name = model_name.replace("nvidia/", "")
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
        else:
            # Direct NeMo model name
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)

        self.logger.info(f"Model loaded successfully")
        self.logger.info(f"  Encoder: {self.model.encoder.__class__.__name__}")
        self.logger.info(f"  Decoder: {self.model.decoder.__class__.__name__}")
        self.logger.info(f"  Joint: {self.model.joint.__class__.__name__}")

        return self.model

    def configure_model(self):
        """Configure the model for fine-tuning."""
        from omegaconf import OmegaConf, open_dict

        cfg = self.model.cfg
        data_cfg = self.config.data
        train_cfg = self.config.training

        with open_dict(cfg):
            # ==================
            # Training Data
            # ==================
            cfg.train_ds.manifest_filepath = data_cfg.train_manifest
            cfg.train_ds.batch_size = data_cfg.train_batch_size
            cfg.train_ds.num_workers = data_cfg.num_workers
            cfg.train_ds.sample_rate = data_cfg.sample_rate
            cfg.train_ds.max_duration = data_cfg.max_duration
            cfg.train_ds.min_duration = data_cfg.min_duration
            cfg.train_ds.shuffle = True
            cfg.train_ds.pin_memory = True

            # ==================
            # Validation Data
            # ==================
            cfg.validation_ds.manifest_filepath = data_cfg.val_manifest
            cfg.validation_ds.batch_size = data_cfg.val_batch_size
            cfg.validation_ds.num_workers = data_cfg.num_workers
            cfg.validation_ds.sample_rate = data_cfg.sample_rate
            cfg.validation_ds.shuffle = False
            cfg.validation_ds.pin_memory = True

            # ==================
            # Optimizer
            # ==================
            cfg.optim.name = train_cfg.optimizer
            cfg.optim.lr = train_cfg.learning_rate
            cfg.optim.weight_decay = train_cfg.weight_decay

            # Betas for Adam/AdamW
            if train_cfg.optimizer in ["adam", "adamw"]:
                cfg.optim.betas = [0.9, 0.98]

            # ==================
            # Scheduler
            # ==================
            cfg.optim.sched.name = train_cfg.scheduler
            cfg.optim.sched.warmup_steps = train_cfg.warmup_steps
            cfg.optim.sched.min_lr = train_cfg.min_lr

            # ==================
            # Spec Augmentation
            # ==================
            if hasattr(cfg, "spec_augment") and data_cfg.enable_spec_augment:
                cfg.spec_augment.freq_masks = data_cfg.freq_masks
                cfg.spec_augment.time_masks = data_cfg.time_masks
                cfg.spec_augment.freq_width = data_cfg.freq_width
                cfg.spec_augment.time_width = data_cfg.time_width

        self.logger.info("Model configuration updated for fine-tuning")

        # Freeze encoder if specified
        if train_cfg.freeze_encoder:
            self.logger.info("Freezing encoder layers")
            self.model.encoder.freeze()

        return cfg

    def setup_trainer(self):
        """Setup PyTorch Lightning trainer."""
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import (
            ModelCheckpoint,
            EarlyStopping,
            LearningRateMonitor,
            RichProgressBar,
        )
        from nemo.utils.exp_manager import exp_manager

        train_cfg = self.config.training

        # Callbacks
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
        ]

        # Try to use RichProgressBar, fall back to default if not available
        try:
            callbacks.append(RichProgressBar())
        except:
            pass

        # Trainer configuration
        trainer = pl.Trainer(
            devices=train_cfg.devices,
            accelerator=train_cfg.accelerator,
            max_epochs=train_cfg.max_epochs,
            precision=train_cfg.precision,
            accumulate_grad_batches=train_cfg.accumulate_grad_batches,
            gradient_clip_val=train_cfg.grad_clip,
            log_every_n_steps=train_cfg.log_every_n_steps,
            enable_checkpointing=True,
            callbacks=callbacks,
            default_root_dir=train_cfg.output_dir,
        )

        # Experiment manager for NeMo-style logging and checkpointing
        exp_manager_cfg = {
            "exp_dir": train_cfg.output_dir,
            "name": train_cfg.exp_name,
            "checkpoint_callback_params": {
                "monitor": "val_wer",
                "mode": "min",
                "save_top_k": train_cfg.save_top_k,
                "save_last": train_cfg.save_last,
            },
            "create_tensorboard_logger": True,
            "create_wandb_logger": False,
        }

        # Add early stopping if configured
        if train_cfg.early_stop_patience > 0:
            exp_manager_cfg["early_stopping_callback_params"] = {
                "monitor": "val_wer",
                "patience": train_cfg.early_stop_patience,
                "min_delta": train_cfg.early_stop_min_delta,
                "mode": "min",
            }

        exp_manager(trainer, exp_manager_cfg)

        self.trainer = trainer
        self.logger.info("Trainer configured")

        return trainer

    def train(self):
        """Run the fine-tuning process."""
        # Load model if not already loaded
        if self.model is None:
            self.load_pretrained_model()

        # Configure model
        self.configure_model()

        # Setup data loaders
        self.logger.info("Setting up data loaders...")
        self.model.setup_training_data(self.model.cfg.train_ds)
        self.model.setup_validation_data(self.model.cfg.validation_ds)

        # Setup trainer
        if self.trainer is None:
            self.setup_trainer()

        # Print manifest stats
        print_manifest_stats(self.config.data.train_manifest)
        print_manifest_stats(self.config.data.val_manifest)

        # Start training
        self.logger.info("Starting fine-tuning...")
        self.trainer.fit(
            self.model, ckpt_path=self.config.training.resume_from_checkpoint
        )

        # Save final model
        final_model_path = Path(self.config.training.output_dir) / "final_model.nemo"
        self.model.save_to(str(final_model_path))
        self.logger.info(f"Final model saved to: {final_model_path}")

        return self.model

    def evaluate(self, test_manifest: Optional[str] = None):
        """Evaluate the model on test data."""
        from omegaconf import open_dict

        test_manifest = test_manifest or self.config.data.test_manifest

        if not test_manifest or not Path(test_manifest).exists():
            self.logger.warning("No test manifest provided or file not found")
            return None

        self.logger.info(f"Evaluating on: {test_manifest}")

        # Update test config
        with open_dict(self.model.cfg):
            self.model.cfg.test_ds.manifest_filepath = test_manifest
            self.model.cfg.test_ds.batch_size = self.config.data.val_batch_size
            self.model.cfg.test_ds.num_workers = self.config.data.num_workers

        # Setup test data
        self.model.setup_test_data(self.model.cfg.test_ds)

        # Run evaluation
        if self.trainer is None:
            self.setup_trainer()

        results = self.trainer.test(self.model)

        self.logger.info(f"Test Results: {results}")
        return results

    def transcribe(
        self, audio_paths: Union[str, List[str]], batch_size: int = 4
    ) -> List[str]:
        """
        Transcribe audio files using the fine-tuned model.

        Args:
            audio_paths: Single audio path or list of audio paths
            batch_size: Batch size for inference

        Returns:
            List of transcriptions
        """
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]

        self.logger.info(f"Transcribing {len(audio_paths)} audio files...")

        # Use model's transcribe method
        transcriptions = self.model.transcribe(audio=audio_paths, batch_size=batch_size)

        return transcriptions


def load_finetuned_model(model_path: str):
    """
    Load a fine-tuned model for inference.

    Args:
        model_path: Path to .nemo file

    Returns:
        Loaded model
    """
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    model.eval()

    # Move to GPU if available
    import torch

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def transcribe_with_model(
    model, audio_paths: Union[str, List[str]], batch_size: int = 4
):
    """
    Transcribe audio files using a loaded model.

    Args:
        model: Loaded NeMo ASR model
        audio_paths: Single audio path or list of audio paths
        batch_size: Batch size for inference

    Returns:
        List of transcriptions
    """
    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]

    return model.transcribe(audio=audio_paths, batch_size=batch_size)


# ==============================================================================
# EXAMPLE CONFIGURATIONS
# ==============================================================================


def get_example_config_small_dataset():
    """Configuration for fine-tuning on a small dataset (< 10 hours)."""
    return Config(
        data=DataConfig(
            train_manifest="./data/train_manifest.json",
            val_manifest="./data/val_manifest.json",
            test_manifest="./data/test_manifest.json",
            train_batch_size=8,
            val_batch_size=8,
            max_duration=15.0,
        ),
        training=TrainingConfig(
            max_epochs=100,
            learning_rate=5e-5,  # Lower LR for small datasets
            warmup_steps=500,
            freeze_encoder=True,  # Freeze encoder for small datasets
            early_stop_patience=15,
            save_top_k=5,
        ),
    )


def get_example_config_medium_dataset():
    """Configuration for fine-tuning on a medium dataset (10-100 hours)."""
    return Config(
        data=DataConfig(
            train_manifest="./data/train_manifest.json",
            val_manifest="./data/val_manifest.json",
            test_manifest="./data/test_manifest.json",
            train_batch_size=16,
            val_batch_size=16,
            max_duration=20.0,
        ),
        training=TrainingConfig(
            max_epochs=50,
            learning_rate=1e-4,
            warmup_steps=1000,
            freeze_encoder=False,
            freeze_encoder_epochs=5,  # Freeze encoder for first 5 epochs
            early_stop_patience=10,
        ),
    )


def get_example_config_large_dataset():
    """Configuration for fine-tuning on a large dataset (> 100 hours)."""
    return Config(
        data=DataConfig(
            train_manifest="./data/train_manifest.json",
            val_manifest="./data/val_manifest.json",
            test_manifest="./data/test_manifest.json",
            train_batch_size=32,
            val_batch_size=32,
            max_duration=20.0,
            num_workers=8,
        ),
        training=TrainingConfig(
            max_epochs=30,
            learning_rate=3e-4,
            warmup_steps=2000,
            freeze_encoder=False,
            devices=1,  # Set to more if you have multiple GPUs
            precision="bf16-mixed",  # Use bf16 if supported
        ),
    )


def get_example_config_multilingual():
    """Configuration for multilingual fine-tuning."""
    return Config(
        data=DataConfig(
            train_manifest="./data/multilingual_train_manifest.json",
            val_manifest="./data/multilingual_val_manifest.json",
            test_manifest="./data/multilingual_test_manifest.json",
            train_batch_size=16,
            val_batch_size=16,
            max_duration=20.0,
        ),
        training=TrainingConfig(
            max_epochs=50,
            learning_rate=1e-4,
            warmup_steps=2000,
            freeze_encoder=False,
            # For multilingual, ensure manifest has 'lang' field for each sample
        ),
    )


# ==============================================================================
# GPU-SPECIFIC CONFIGURATIONS
# ==============================================================================


def get_config_rtx5090():
    """Optimized config for RTX 5090 (32GB VRAM)."""
    return Config(
        data=DataConfig(
            train_batch_size=20,  # 32GB allows larger batches
            val_batch_size=20,
            num_workers=4,
            max_duration=20.0,
        ),
        training=TrainingConfig(
            precision="16-mixed",  # RTX 5090 uses FP16 (no BF16)
            accumulate_grad_batches=2,  # effective batch = 40
            max_epochs=50,
            learning_rate=1e-4,
            warmup_steps=1000,
        ),
    )


def get_config_a100():
    """Optimized config for A100 (40GB/80GB VRAM)."""
    return Config(
        data=DataConfig(
            train_batch_size=24,  # More VRAM headroom
            val_batch_size=24,
            num_workers=8,  # A100 servers have more CPU
            max_duration=20.0,
        ),
        training=TrainingConfig(
            precision="bf16-mixed",  # A100 supports BF16
            accumulate_grad_batches=2,  # effective batch = 48
            max_epochs=50,
            learning_rate=1e-4,
            warmup_steps=1000,
        ),
    )


def get_config_h100():
    """Optimized config for H100 (80GB VRAM)."""
    return Config(
        data=DataConfig(
            train_batch_size=32,  # H100 80GB allows larger batches
            val_batch_size=32,
            num_workers=8,  # H100 servers have more CPU
            max_duration=20.0,
        ),
        training=TrainingConfig(
            precision="bf16-mixed",  # H100 supports BF16 (and FP8)
            accumulate_grad_batches=2,  # effective batch = 64
            max_epochs=50,
            learning_rate=1e-4,
            warmup_steps=1000,
        ),
    )


def detect_gpu_type():
    """Auto-detect GPU type from CUDA device name."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            if "5090" in name:
                return "rtx5090"
            elif "h100" in name:
                return "h100"
            elif "a100" in name:
                return "a100"
    except Exception:
        pass
    return "rtx5090"  # Default to RTX 5090 settings


# ==============================================================================
# CLI & MAIN
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Parakeet RNNT 1.1B Multilingual model"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "transcribe", "create-manifest", "validate-manifest"],
        default="train",
        help="Operation mode",
    )

    # Training arguments
    parser.add_argument("--train-manifest", type=str, help="Path to training manifest")
    parser.add_argument("--val-manifest", type=str, help="Path to validation manifest")
    parser.add_argument("--test-manifest", type=str, help="Path to test manifest")
    parser.add_argument(
        "--output-dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--freeze-encoder", action="store_true", help="Freeze encoder layers"
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument(
        "--gpu",
        type=str,
        choices=["rtx5090", "a100", "h100", "auto"],
        default="auto",
        help="GPU type for optimized settings (rtx5090, a100, h100, auto)",
    )

    # Model path argument (used for training base model, eval, or transcribe)
    parser.add_argument(
        "--model-path", type=str, help="Path to local .nemo model (base model for training, or model for eval/transcribe)"
    )
    parser.add_argument(
        "--audio", type=str, nargs="+", help="Audio file(s) to transcribe"
    )

    # Manifest creation arguments
    parser.add_argument("--audio-dir", type=str, help="Directory with audio files")
    parser.add_argument(
        "--transcriptions", type=str, help="JSON file with transcriptions"
    )
    parser.add_argument("--language", type=str, default="en", help="Language code")
    parser.add_argument("--manifest-output", type=str, help="Output manifest path")

    return parser.parse_args()


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("PARAKEET RNNT 1.1B MULTILINGUAL FINE-TUNING")
    print("=" * 60 + "\n")

    # Check dependencies
    if not check_and_install_dependencies():
        sys.exit(1)

    args = parse_args()

    if args.mode == "train":
        # Get GPU-specific base config
        gpu_type = args.gpu if args.gpu != "auto" else detect_gpu_type()

        if gpu_type == "h100":
            config = get_config_h100()
            print(f"Using H100 optimized settings (BF16, batch=32)")
        elif gpu_type == "a100":
            config = get_config_a100()
            print(f"Using A100 optimized settings (BF16, batch=24)")
        else:
            config = get_config_rtx5090()
            print(f"Using RTX 5090 optimized settings (FP16, batch=20)")

        # Override with CLI arguments if provided
        config.data.train_manifest = args.train_manifest or "./data/train_manifest.json"
        config.data.val_manifest = args.val_manifest or "./data/val_manifest.json"
        if args.test_manifest:
            config.data.test_manifest = args.test_manifest
        if args.output_dir != "./outputs":
            config.training.output_dir = args.output_dir
        if args.epochs != 50:
            config.training.max_epochs = args.epochs
        if args.batch_size != 16:
            config.data.train_batch_size = args.batch_size
            config.data.val_batch_size = args.batch_size
        if args.lr != 1e-4:
            config.training.learning_rate = args.lr
        if args.freeze_encoder:
            config.training.freeze_encoder = True
        if args.resume:
            config.training.resume_from_checkpoint = args.resume
        if args.model_path:
            config.training.pretrained_model = args.model_path

        # Run training
        finetuner = ParakeetRNNTFineTuner(config)
        model = finetuner.train()

        # Evaluate if test manifest provided
        if config.data.test_manifest:
            finetuner.evaluate()

    elif args.mode == "eval":
        if not args.model_path:
            print("Error: --model-path required for evaluation")
            sys.exit(1)

        config = Config(
            data=DataConfig(
                test_manifest=args.test_manifest or "./data/test_manifest.json",
            )
        )

        finetuner = ParakeetRNNTFineTuner(config)
        finetuner.model = load_finetuned_model(args.model_path)
        finetuner.evaluate()

    elif args.mode == "transcribe":
        if not args.model_path:
            print("Error: --model-path required for transcription")
            sys.exit(1)
        if not args.audio:
            print("Error: --audio required for transcription")
            sys.exit(1)

        model = load_finetuned_model(args.model_path)
        transcriptions = transcribe_with_model(model, args.audio)

        print("\n📝 Transcriptions:")
        for audio, text in zip(args.audio, transcriptions):
            print(f"  {Path(audio).name}: {text}")

    elif args.mode == "create-manifest":
        if not args.audio_dir or not args.manifest_output:
            print("Error: --audio-dir and --manifest-output required")
            sys.exit(1)

        create_manifest_from_folder(
            audio_dir=args.audio_dir,
            output_manifest=args.manifest_output,
            transcription_file=args.transcriptions,
            language=args.language,
        )

    elif args.mode == "validate-manifest":
        manifest_path = args.train_manifest or args.val_manifest or args.test_manifest
        if not manifest_path:
            print(
                "Error: Provide a manifest path (--train-manifest, --val-manifest, or --test-manifest)"
            )
            sys.exit(1)

        print_manifest_stats(manifest_path)


if __name__ == "__main__":
    main()
