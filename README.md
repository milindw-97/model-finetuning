# Speech-to-Text Model Fine-Tuning

Fine-tune **OpenAI Whisper** and **NVIDIA Parakeet RNNT** models on custom datasets.

---

## Supported Models

| Model | Framework | Best For |
|-------|-----------|----------|
| **Whisper** (small/medium/large) | HuggingFace Transformers | General use, multi-language |
| **Parakeet RNNT 1.1B** | NVIDIA NeMo | High accuracy, production |

---

## Quick Start

### Whisper Fine-Tuning (Recommended for beginners)

```bash
# 1. Install dependencies
pip install transformers datasets accelerate evaluate jiwer torch torchaudio soundfile

# 2. Open the notebook
jupyter notebook stt_finetuning.ipynb

# 3. Or run from command line (see Whisper CLI section below)
```

### Parakeet Fine-Tuning

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
python scripts/download_dataset.py --output-dir data/raw --analyze
python scripts/convert_to_nemo.py --output-dir data

# 3. Fine-tune
python scripts/finetune_hindi.py --config configs/hindi_finetune.yaml

# 4. Evaluate
python scripts/evaluate_parakeet.py --model outputs/parakeet-hindi-finetuned/final_model.nemo

# 5. Push to HuggingFace
huggingface-cli login
huggingface-cli upload YOUR_USERNAME/my-model outputs/parakeet-hindi-finetuned/final_model.nemo
```

---

## Repository Structure

```
model-finetuning/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
├── Dockerfile                         # Docker container
│
├── stt_finetuning.ipynb              # Whisper & multi-model notebook
├── parakeet_rnnt_finetuning.py       # Parakeet CLI script
├── parakeet_rnnt_finetuning.ipynb    # Parakeet notebook
│
├── configs/
│   └── hindi_finetune.yaml           # Parakeet training config
│
├── scripts/
│   ├── download_dataset.py           # Download HuggingFace datasets
│   ├── convert_to_nemo.py            # Convert to NeMo format
│   ├── finetune_hindi.py             # Parakeet fine-tuning
│   ├── evaluate_parakeet.py          # Model evaluation
│   └── push_to_huggingface.py        # Upload to HuggingFace
│
└── data/                              # Dataset directory
```

---

# WHISPER FINE-TUNING

## Option 1: Using the Notebook (Easiest)

Open `stt_finetuning.ipynb` in Jupyter or Google Colab.

### Step 1: Configure Your Dataset

Edit the configuration cell:

```python
DATASET_CONFIG = DatasetConfig(
    # Your dataset source
    train_datasets=["mozilla-foundation/common_voice_11_0"],

    # For Common Voice, specify language
    dataset_config_name="hi",  # Hindi ("en" for English, "es" for Spanish, etc.)
    dataset_split="train",

    # Column names (adjust for your dataset)
    audio_column="audio",
    text_column="sentence",

    # Limit samples for testing (None for full dataset)
    max_train_samples=1000,
    max_val_samples=100,
    max_test_samples=100,
)
```

### Step 2: Configure the Model

```python
MODELS_TO_TRAIN = [
    ModelConfig(
        model_type="whisper",
        model_name="openai/whisper-small",  # or whisper-medium, whisper-large-v3
        learning_rate=1e-5,
        batch_size=8,          # Reduce if OOM
        num_epochs=3,
        freeze_encoder=False,  # Set True for small datasets
        output_dir="./outputs/whisper-finetuned",
    ),
]
```

### Step 3: Run Training

Execute all cells. The notebook will:
1. Download the dataset
2. Preprocess audio
3. Fine-tune the model
4. Evaluate and save results

### Step 4: Use Your Model

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load fine-tuned model
processor = WhisperProcessor.from_pretrained("./outputs/whisper-finetuned")
model = WhisperForConditionalGeneration.from_pretrained("./outputs/whisper-finetuned")

# Transcribe audio
audio, sr = torchaudio.load("test.wav")
if sr != 16000:
    audio = torchaudio.functional.resample(audio, sr, 16000)

input_features = processor.feature_extractor(
    audio.squeeze().numpy(),
    sampling_rate=16000,
    return_tensors="pt"
).input_features

predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
```

---

## Option 2: Whisper CLI Script

Create a simple training script:

```python
# whisper_finetune.py
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset, Audio
import evaluate
import torch

# Configuration
MODEL_NAME = "openai/whisper-small"
DATASET_NAME = "mozilla-foundation/common_voice_11_0"
LANGUAGE = "hi"  # Hindi
OUTPUT_DIR = "./whisper-hindi-finetuned"

# Load model and processor
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# Configure model
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

# Load dataset
dataset = load_dataset(DATASET_NAME, LANGUAGE, split="train[:1000]")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Split into train/test
dataset = dataset.train_test_split(test_size=0.1)

# Preprocess function
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# Preprocess
dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)

# Data collator
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == model.config.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Metrics
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=3,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

# Train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

trainer.train()

# Save
trainer.save_model()
processor.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")
```

Run it:
```bash
python whisper_finetune.py
```

---

## Whisper Model Sizes

| Model | Parameters | VRAM Required | Speed |
|-------|------------|---------------|-------|
| `openai/whisper-tiny` | 39M | ~1GB | Fastest |
| `openai/whisper-base` | 74M | ~1GB | Fast |
| `openai/whisper-small` | 244M | ~2GB | Good balance |
| `openai/whisper-medium` | 769M | ~5GB | Better accuracy |
| `openai/whisper-large-v3` | 1.5B | ~10GB | Best accuracy |

**Recommendation**: Start with `whisper-small` for testing, use `whisper-large-v3` for production.

---

## Common Whisper Datasets

```python
# Common Voice (many languages)
DatasetConfig(
    train_datasets=["mozilla-foundation/common_voice_11_0"],
    dataset_config_name="hi",  # hi, en, es, fr, de, zh-CN, ja, etc.
    audio_column="audio",
    text_column="sentence",
)

# LibriSpeech (English)
DatasetConfig(
    train_datasets=["librispeech_asr"],
    dataset_config_name=None,
    dataset_split="train.clean.100",
    audio_column="audio",
    text_column="text",
)

# FLEURS (multilingual)
DatasetConfig(
    train_datasets=["google/fleurs"],
    dataset_config_name="hi_in",  # Hindi
    audio_column="audio",
    text_column="transcription",
)

# Your own data (audiofolder format)
DatasetConfig(
    train_datasets=["/path/to/your/audio/folder"],
    audio_column="audio",
    text_column="transcription",
)
```

---

## Option 3: Whisper with Unsloth (Fastest, LoRA-based)

Unsloth provides 2x faster training with LoRA adapters.

### Setup Unsloth

```bash
# Unsloth requires PyTorch 2.2+ with CUDA
# First, upgrade PyTorch if needed:
pip uninstall torch torchaudio torchvision -y
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth
pip install unsloth

# Install other dependencies
pip install peft accelerate evaluate jiwer datasets transformers
```

### Run Training

```bash
# With a HuggingFace dataset
python scripts/finetune_whisper_unsloth.py \
    --dataset ai4bharat/indicvoices_r \
    --output-dir outputs/whisper-hindi-unsloth \
    --epochs 10

# With config file
python scripts/finetune_whisper_unsloth.py --config configs/whisper_hindi_unsloth.yaml

# With 4-bit quantization (less memory)
python scripts/finetune_whisper_unsloth.py --dataset YOUR_DATASET --4bit
```

### Fallback without Unsloth

If Unsloth fails (incompatible GPU, PyTorch version), the script automatically falls back to standard PEFT:

```bash
# Install PEFT for fallback
pip install peft accelerate

# Run - will use PEFT if Unsloth unavailable
python scripts/finetune_whisper_unsloth.py --dataset ai4bharat/indicvoices_r
```

### Unsloth Troubleshooting

**Error: `module 'torch._inductor' has no attribute 'config'`**
```bash
# Upgrade PyTorch to 2.2+
pip uninstall torch torchaudio torchvision -y
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install unsloth --upgrade
```

**Error: CUDA not available**
- Unsloth requires NVIDIA GPU with CUDA
- The script will fallback to PEFT automatically

---

# PARAKEET FINE-TUNING

## Complete Workflow

### Step 1: Setup Environment

```bash
# Option A: pip
pip install -r requirements.txt

# Option B: Conda
conda env create -f environment.yml
conda activate nemo-asr

# Option C: Docker
docker build -t parakeet-finetune .
docker run --gpus all -v $(pwd):/workspace -it parakeet-finetune
```

### Step 2: Download Dataset

```bash
python scripts/download_dataset.py \
    --output-dir data/raw \
    --dataset sivakgp/hindi-books-audio-validation \
    --split validation \
    --analyze
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--output-dir` | Output directory | `data/raw` |
| `--dataset` | HuggingFace dataset | `sivakgp/hindi-books-audio-validation` |
| `--split` | Dataset split | `validation` |
| `--max-samples` | Limit samples | None |
| `--analyze` | Print statistics | False |

### Step 3: Convert to NeMo Format

```bash
python scripts/convert_to_nemo.py \
    --input-dataset sivakgp/hindi-books-audio-validation \
    --output-dir data \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --language hi
```

**Output:**
- `data/train_manifest.json`
- `data/val_manifest.json`
- `data/test_manifest.json`
- `data/audio/train/`, `val/`, `test/`

### Step 4: Fine-Tune

```bash
# Using config file
python scripts/finetune_hindi.py --config configs/hindi_finetune.yaml

# With overrides
python scripts/finetune_hindi.py \
    --config configs/hindi_finetune.yaml \
    --train-manifest data/train_manifest.json \
    --val-manifest data/val_manifest.json \
    --output-dir outputs/my-model \
    --epochs 100 \
    --batch-size 4 \
    --lr 5e-5 \
    --freeze-encoder
```

### Step 5: Evaluate

```bash
python scripts/evaluate_parakeet.py \
    --model outputs/parakeet-hindi-finetuned/final_model.nemo \
    --test-manifest data/test_manifest.json \
    --output results.json
```

### Step 6: Transcribe

```bash
python parakeet_rnnt_finetuning.py \
    --mode transcribe \
    --model-path outputs/parakeet-hindi-finetuned/final_model.nemo \
    --audio test1.wav test2.wav
```

---

## Parakeet Training Config

Edit `configs/hindi_finetune.yaml`:

```yaml
model:
  pretrained_name: "parakeet-rnnt-1.1b-multilingual"

train_ds:
  manifest_filepath: "data/train_manifest.json"
  batch_size: 4

optim:
  lr: 5.0e-5

encoder:
  freeze: true  # Recommended for small datasets

trainer:
  max_epochs: 100
  precision: "16-mixed"
```

---

# PUSH TO HUGGINGFACE

## Upload Whisper Model

```bash
# Login
huggingface-cli login

# Upload model folder
huggingface-cli upload YOUR_USERNAME/whisper-hindi-finetuned ./outputs/whisper-finetuned --repo-type model
```

Or in Python:

```python
from huggingface_hub import HfApi, create_repo

# Create repo
create_repo("YOUR_USERNAME/whisper-hindi-finetuned", repo_type="model", private=True)

# Upload
api = HfApi()
api.upload_folder(
    folder_path="./outputs/whisper-finetuned",
    repo_id="YOUR_USERNAME/whisper-hindi-finetuned",
    repo_type="model"
)
```

## Upload Parakeet Model

```bash
# Login
huggingface-cli login

# Upload .nemo file
huggingface-cli upload YOUR_USERNAME/parakeet-hindi-finetuned \
    outputs/parakeet-hindi-finetuned/final_model.nemo \
    --repo-type model
```

## Upload Dataset

```bash
python scripts/push_to_huggingface.py \
    --data-dir data \
    --repo-name YOUR_USERNAME/hindi-asr-dataset \
    --private
```

---

# TRAINING BEST PRACTICES

## By Dataset Size

| Dataset Size | Model | Learning Rate | Freeze Encoder | Epochs |
|--------------|-------|---------------|----------------|--------|
| < 1 hour | Whisper-small | 5e-6 | Yes | 10-20 |
| 1-10 hours | Whisper-small/medium | 1e-5 | Partial | 5-10 |
| 10-100 hours | Whisper-medium/large | 1e-5 | No | 3-5 |
| > 100 hours | Whisper-large-v3 | 1e-5 | No | 2-3 |

## Memory Optimization

```python
# Reduce batch size
batch_size=4  # or even 2

# Use gradient accumulation for effective larger batch
gradient_accumulation_steps=4  # effective batch = 4 * 4 = 16

# Enable mixed precision
fp16=True  # or bf16=True for newer GPUs
```

## Preventing Overfitting

1. **Freeze encoder** for small datasets
2. **Use lower learning rate** (5e-6 instead of 1e-5)
3. **Enable data augmentation** (SpecAugment)
4. **Early stopping** based on validation WER

---

# SCRIPT REFERENCE

| Script | Purpose | Usage |
|--------|---------|-------|
| `stt_finetuning.ipynb` | Whisper fine-tuning notebook | Open in Jupyter/Colab |
| `scripts/finetune_whisper_unsloth.py` | Whisper + Unsloth/PEFT | `--dataset NAME --epochs 10` |
| `parakeet_rnnt_finetuning.py` | Parakeet CLI | `--mode train/eval/transcribe` |
| `scripts/download_dataset.py` | Download HuggingFace datasets | `--dataset NAME` |
| `scripts/convert_to_nemo.py` | Convert to NeMo format | `--output-dir data` |
| `scripts/finetune_hindi.py` | Parakeet fine-tuning | `--config CONFIG.yaml` |
| `scripts/evaluate_parakeet.py` | Evaluate model | `--model MODEL.nemo` |
| `scripts/push_to_huggingface.py` | Upload dataset | `--repo-name USER/REPO` |

---

# TROUBLESHOOTING

### CUDA Out of Memory
```python
# Reduce batch size
batch_size = 2

# Increase gradient accumulation
gradient_accumulation_steps = 8

# Use smaller model
model_name = "openai/whisper-small"
```

### NumPy 2.0 Incompatibility (NeMo)
```bash
pip install "numpy>=1.22.0,<2.0.0"
```

### Audio Loading Issues
```bash
pip install soundfile librosa torchaudio
```

### Whisper Generation Errors
```python
# Set these in model config
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False
```

---

## References

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [HuggingFace Whisper Fine-Tuning Guide](https://huggingface.co/blog/fine-tune-whisper)
- [NeMo Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- [Parakeet Model Card](https://huggingface.co/nvidia/parakeet-rnnt-1.1b-multilingual)
