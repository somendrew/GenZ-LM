# GenZify 🔥 — Qwen2.5-1.5B Fine-tuned on GenZ Slang

A fine-tuned version of [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) that responds to any question or instruction in **GenZ slang, emojis, and internet culture**.

> no formal language, just pure vibes fr fr no cap 😂🔥

---

## Demo

| Prompt | Response |
|---|---|
| What is photosynthesis? | Plants turning light CO2 water glucose sugar – oxygen free 🌱 food maker or nature's main character 💚😂 |
| Explain black holes | space vacuum cleaners fr 🕳️ gravity so unhinged not even light escapes – einstein said bet 💀😂 |
| Tips to stay productive | 1. no phone first hr 📵 2. pomodoro grind 🍅 3. sleep or brain said lag 😴😂 |

---

## Model Details

| | |
|---|---|
| **Base Model** | Qwen/Qwen2.5-1.5B-Instruct |
| **Fine-tuning Method** | QLoRA (4-bit) |
| **LoRA Rank** | 16 |
| **LoRA Alpha** | 32 |
| **Epochs** | 3 |
| **Training Loss** | 0.877 |
| **Validation Loss** | 1.282 |
| **Training Time** | ~29 mins |
| **Hardware** | Kaggle T4 x2 |
| **Dataset** | Custom GenZ response dataset (10 batches) |

---

## Project Structure

```
GenZify/
├── app.py                  # Gradio demo app
├── requirements.txt        # Dependencies
├── finetune_genz.py        # Training script
├── data_prep.py            # Dataset preparation
└── README.md
```

---

## Quick Start

### Installation
```bash
git clone https://github.com/somendrew/GenZify
cd GenZify
pip install -r requirements.txt
```

### Run Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

MODEL_ID = "somendrew/genz-qwen-2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
)
model.eval()
model.config.use_cache = True

SYSTEM_PROMPT = "You are a GenZ assistant. Reply using GenZ slang and emojis. No formal language, just vibes fr 🔥"

def chat(prompt):
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(chat("What is gravity?"))
```

### Run Gradio Demo Locally
```bash
python app.py
```

---

## Training

The model was fine-tuned using **QLoRA** on Kaggle (T4 x2) using the `trl` library's `SFTTrainer`.

### Dataset Format
```json
{
    "instruction": "What is gravity?",
    "input": "",
    "output": "gravity = earth being clingy af 🌍 pulls everything down no escape – newton got bonked by apple 🍎💀😂"
}
```

### LoRA Config
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

### Training Args
```python
SFTConfig(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)
```

---

## Requirements

```
transformers
gradio
torch
accelerate
bitsandbytes
peft
trl
datasets
```

---

## HuggingFace Links

| Resource | Link |
|---|---|
| 🤗 Merged Model | [somendrew/genz-qwen-2.5-1.5B](https://huggingface.co/somendrew/genz-qwen-2.5-1.5B) |
| 🔌 LoRA Adapter | [somendrew/genz-qwen-2.5-1.5B-adapter](https://huggingface.co/somendrew/genz-qwen-2.5-1.5B-adapter) |
| 🚀 Live Demo | [HuggingFace Space](https://huggingface.co/spaces/somendrew/GenZify) |

---

## License

This project is licensed under the **Apache 2.0 License** — same as the base model.

---

## Acknowledgements

- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) by Alibaba Cloud
- [TRL](https://github.com/huggingface/trl) by HuggingFace
- [PEFT](https://github.com/huggingface/peft) by HuggingFace
- Trained on [Kaggle](https://www.kaggle.com) free GPU tier

---

<p align="center">BUILT WITH 🔥 AND ZERO CHILL</p>
