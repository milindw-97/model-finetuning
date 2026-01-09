#!/usr/bin/env python3
"""
Hindi Fine-Tuning Script for Parakeet RNNT 1.1B Multilingual

This script fine-tunes NVIDIA's Parakeet RNNT model on Hindi audio data.
Optimized for small datasets with encoder freezing and careful hyperparameters.

Usage:
    python scripts/finetune_hindi.py --config configs/hindi_finetune.yaml

    Or with command line overrides:
    python scripts/finetune_hindi.py \
        --train-manifest data/train_manifest.json \
        --val-manifest data/val_manifest.json \
        --output-dir outputs/hindi-finetuned \
        --epochs 100 \
        --freeze-encoder

Prerequisites:
    - NeMo toolkit installed (pip install nemo_toolkit[asr])
    - GPU with at least 16GB memory (T4 or better)
    - Dataset converted to NeMo manifest format
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

import os

# Use Numba's native CUDA binding to avoid nvJitLink compatibility issues
os.environ["NUMBA_CUDA_USE_NVIDIA_BINDING"] = "0"
os.environ["NUMBA_DISABLE_CUDA"] = "0"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import nemo
        import nemo.collections.asr as nemo_asr

        logger.info(f"NeMo version: {nemo.__version__}")
    except ImportError:
        missing.append("nemo_toolkit[asr]")

    try:
        import lightning.pytorch as pl

        logger.info(f"PyTorch Lightning version: {pl.__version__}")
    except ImportError:
        missing.append("pytorch-lightning")

    try:
        from omegaconf import OmegaConf
    except ImportError:
        missing.append("omegaconf")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        logger.info(f"CUDA: {torch.version.cuda}")
    else:
        logger.warning("No GPU detected! Training will be very slow.")

    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False

    return True


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def validate_manifests(config: dict) -> bool:
    """Validate that manifest files exist."""
    valid = True

    train_manifest = config.get("train_ds", {}).get("manifest_filepath")
    if train_manifest and not Path(train_manifest).exists():
        logger.error(f"Training manifest not found: {train_manifest}")
        valid = False

    val_manifest = config.get("validation_ds", {}).get("manifest_filepath")
    if val_manifest and not Path(val_manifest).exists():
        logger.error(f"Validation manifest not found: {val_manifest}")
        valid = False

    return valid


def finetune_parakeet(
    config: dict,
    train_manifest: Optional[str] = None,
    val_manifest: Optional[str] = None,
    test_manifest: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    freeze_encoder: Optional[bool] = None,
    resume_from: Optional[str] = None,
):
    """
    Fine-tune Parakeet RNNT on Hindi data.

    Args:
        config: Configuration dictionary
        train_manifest: Override training manifest path
        val_manifest: Override validation manifest path
        test_manifest: Override test manifest path
        output_dir: Override output directory
        epochs: Override number of epochs
        batch_size: Override batch size
        learning_rate: Override learning rate
        freeze_encoder: Override encoder freezing
        resume_from: Path to checkpoint to resume from
    """
    import nemo.collections.asr as nemo_asr
    import lightning.pytorch as pl
    from nemo.utils.exp_manager import exp_manager
    from omegaconf import OmegaConf, open_dict

    # Apply command line overrides
    if train_manifest:
        config["train_ds"]["manifest_filepath"] = train_manifest
    if val_manifest:
        config["validation_ds"]["manifest_filepath"] = val_manifest
    if test_manifest:
        config.setdefault("test_ds", {})["manifest_filepath"] = test_manifest
    if output_dir:
        config.setdefault("exp_manager", {})["exp_dir"] = output_dir
        config.setdefault("output", {})["dir"] = output_dir
    if epochs:
        config.setdefault("trainer", {})["max_epochs"] = epochs
    if batch_size:
        config["train_ds"]["batch_size"] = batch_size
        config["validation_ds"]["batch_size"] = batch_size
    if learning_rate:
        config.setdefault("optim", {})["lr"] = learning_rate
    if freeze_encoder is not None:
        config.setdefault("encoder", {})["freeze"] = freeze_encoder

    # Validate manifests
    if not validate_manifests(config):
        logger.error("Manifest validation failed. Run convert_to_nemo.py first.")
        return None

    # Create output directory
    exp_dir = config.get("exp_manager", {}).get(
        "exp_dir", "outputs/parakeet-hindi-finetuned"
    )
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # Save effective config
    config_save_path = Path(exp_dir) / "effective_config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved effective config to: {config_save_path}")

    # Load pre-trained model
    model_name = config.get("model", {}).get("pretrained_name", "parakeet-rnnt-1.1b")
    model_path = config.get("model", {}).get("pretrained_path")

    logger.info(f"Loading pre-trained model: {model_name}")

    if model_path and Path(model_path).exists():
        logger.info(f"Loading from local path: {model_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    else:
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=model_name
        )

    logger.info(f"Model loaded: {model.__class__.__name__}")
    logger.info(f"  Encoder: {model.encoder.__class__.__name__}")
    logger.info(f"  Decoder: {model.decoder.__class__.__name__}")

    # Add this BEFORE with open_dict(cfg):
    train_path = config["train_ds"]["manifest_filepath"]
    val_path = config["validation_ds"]["manifest_filepath"]

    if not Path(train_path).exists():
        raise FileNotFoundError(f"Train manifest missing: {train_path}")
    if not Path(val_path).exists():
        raise FileNotFoundError(f"Val manifest missing: {val_path}")

    logger.info(f"✅ Manifests verified: {train_path}, {val_path}")

    # Update model configuration
    cfg = model.cfg

    with open_dict(cfg):
        # Training data
        cfg.train_ds.manifest_filepath = config["train_ds"]["manifest_filepath"]
        cfg.train_ds.batch_size = config["train_ds"].get("batch_size", 4)
        cfg.train_ds.num_workers = config["train_ds"].get("num_workers", 4)
        cfg.train_ds.sample_rate = config["train_ds"].get("sample_rate", 16000)
        cfg.train_ds.max_duration = config["train_ds"].get("max_duration", 20.0)
        cfg.train_ds.min_duration = config["train_ds"].get("min_duration", 0.3)
        cfg.train_ds.shuffle = config["train_ds"].get("shuffle", True)
        cfg.train_ds.pin_memory = config["train_ds"].get("pin_memory", True)
        cfg.train_ds.use_lhotse = False

        # Validation data
        cfg.validation_ds.manifest_filepath = config["validation_ds"][
            "manifest_filepath"
        ]
        cfg.validation_ds.batch_size = config["validation_ds"].get("batch_size", 4)
        cfg.validation_ds.num_workers = config["validation_ds"].get("num_workers", 4)
        cfg.validation_ds.sample_rate = config["validation_ds"].get(
            "sample_rate", 16000
        )
        cfg.validation_ds.shuffle = False
        cfg.validation_ds.pin_memory = config["validation_ds"].get("pin_memory", True)

        # Optimizer
        optim_config = config.get("optim", {})
        cfg.optim.name = optim_config.get("name", "adamw")
        cfg.optim.lr = optim_config.get("lr", 5e-5)
        cfg.optim.weight_decay = optim_config.get("weight_decay", 0.001)
        if "betas" in optim_config:
            cfg.optim.betas = optim_config["betas"]

        # Scheduler
        sched_config = optim_config.get("sched", {})
        cfg.optim.sched.name = sched_config.get("name", "CosineAnnealing")
        cfg.optim.sched.warmup_steps = sched_config.get("warmup_steps", 100)
        cfg.optim.sched.min_lr = sched_config.get("min_lr", 1e-6)

        # Spec augmentation
        spec_aug_config = config.get("spec_augment", {})
        if hasattr(cfg, "spec_augment") and spec_aug_config:
            cfg.spec_augment.freq_masks = spec_aug_config.get("freq_masks", 2)
            cfg.spec_augment.time_masks = spec_aug_config.get("time_masks", 10)
            cfg.spec_augment.freq_width = spec_aug_config.get("freq_width", 27)
            cfg.spec_augment.time_width = spec_aug_config.get("time_width", 0.05)

        # Use torchaudio RNNT loss instead of Numba (for RTX 5090/Blackwell compatibility)
        # Numba CUDA kernels don't support sm_120 (Blackwell) yet
        if hasattr(cfg, "loss"):
            cfg.loss.loss_name = "torchaudio"
            logger.info("Using torchaudio RNNT loss (RTX 5090/Blackwell compatible)")

    model.cfg = cfg

    # Rebuild the loss module with the new config (required for loss_name change to take effect)
    if hasattr(model, "loss") and hasattr(cfg, "loss"):
        from nemo.collections.asr.losses.rnnt import RNNTLoss
        loss_kwargs = OmegaConf.to_container(cfg.loss.get("loss_kwargs", {})) if cfg.loss.get("loss_kwargs") else {}
        model.loss = RNNTLoss(
            num_classes=model.joint.num_classes_with_blank - 1,
            loss_name=cfg.loss.loss_name,
            loss_kwargs=loss_kwargs,
            reduction=cfg.loss.get("reduction", "mean_batch"),
        )
        logger.info(f"Rebuilt loss module with: {cfg.loss.loss_name}")

    logger.info("Model configuration updated for fine-tuning")

    # Freeze encoder if specified
    encoder_config = config.get("encoder", {})
    if encoder_config.get("freeze", True):
        logger.info("Freezing encoder (recommended for small datasets)")
        model.encoder.freeze()
    else:
        logger.info("Encoder will be trained (not frozen)")

    # Setup data loaders
    logger.info("Setting up data loaders...")
    model.setup_training_data(cfg.train_ds)
    model.setup_validation_data(cfg.validation_ds)

    # Setup trainer
    trainer_config = config.get("trainer", {})

    trainer = pl.Trainer(
        devices=trainer_config.get("devices", 1),
        accelerator=trainer_config.get("accelerator", "gpu"),
        max_epochs=trainer_config.get("max_epochs", 100),
        precision=trainer_config.get("precision", "16-mixed"),
        accumulate_grad_batches=trainer_config.get("accumulate_grad_batches", 4),
        gradient_clip_val=trainer_config.get("gradient_clip_val", 1.0),
        log_every_n_steps=trainer_config.get("log_every_n_steps", 10),
        enable_checkpointing=trainer_config.get("enable_checkpointing", True),
        default_root_dir=exp_dir,
        logger=False,
    )

    # Experiment manager
    # exp_manager_config = config.get("exp_manager", {})
    # exp_cfg = {
    #     "exp_dir": exp_manager_config.get("exp_dir", exp_dir),
    #     "name": exp_manager_config.get("name", "hindi_finetune"),
    #     "checkpoint_callback_params": exp_manager_config.get(
    #         "checkpoint_callback_params",
    #         {
    #             "monitor": "val_wer",
    #             "mode": "min",
    #             "save_top_k": 3,
    #             "save_last": True,
    #         },
    #     ),
    #     "create_tensorboard_logger": exp_manager_config.get(
    #         "create_tensorboard_logger", True
    #     ),
    #     "create_wandb_logger": exp_manager_config.get("create_wandb_logger", False),
    # }

    # Early stopping
    # early_stop = exp_manager_config.get("early_stopping_callback_params")
    # if early_stop:
    #     exp_cfg["early_stopping_callback_params"] = early_stop

    # exp_manager(trainer, exp_cfg)

    # Print training info
    print("\n" + "=" * 60)
    print("STARTING HINDI FINE-TUNING")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Train manifest: {config['train_ds']['manifest_filepath']}")
    print(f"Val manifest: {config['validation_ds']['manifest_filepath']}")
    print(f"Batch size: {config['train_ds'].get('batch_size', 4)}")
    print(f"Learning rate: {config.get('optim', {}).get('lr', 5e-5)}")
    print(f"Max epochs: {trainer_config.get('max_epochs', 100)}")
    print(f"Encoder frozen: {encoder_config.get('freeze', True)}")
    print(f"Output: {exp_dir}")
    print("=" * 60 + "\n")

    # Train
    trainer.fit(model, ckpt_path=resume_from)

    # Save final model
    output_config = config.get("output", {})
    if output_config.get("save_final_model", True):
        final_model_path = Path(exp_dir) / "final_model.nemo"
        model.save_to(str(final_model_path))
        logger.info(f"Final model saved to: {final_model_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {exp_dir}")
    print(f"\nNext steps:")
    print(
        f"  1. Evaluate: python scripts/evaluate.py --model {exp_dir}/final_model.nemo"
    )
    print(
        f"  2. Transcribe: python parakeet_rnnt_finetuning.py --mode transcribe --model-path {exp_dir}/final_model.nemo --audio test.wav"
    )
    print("=" * 60)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Parakeet RNNT 1.1B on Hindi audio"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hindi_finetune.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--train-manifest", type=str, help="Override training manifest path"
    )
    parser.add_argument(
        "--val-manifest", type=str, help="Override validation manifest path"
    )
    parser.add_argument("--test-manifest", type=str, help="Override test manifest path")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument(
        "--freeze-encoder", action="store_true", help="Freeze encoder layers"
    )
    parser.add_argument(
        "--no-freeze-encoder", action="store_true", help="Don't freeze encoder layers"
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        logger.info(f"Loaded config from: {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {}

    # Handle freeze encoder flags
    freeze_encoder = None
    if args.freeze_encoder:
        freeze_encoder = True
    elif args.no_freeze_encoder:
        freeze_encoder = False

    # Run fine-tuning
    finetune_parakeet(
        config=config,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_encoder=freeze_encoder,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
