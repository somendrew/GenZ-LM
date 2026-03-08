# GenZify (a GenZ-LM) 🔥

> A fine-tuned Mistral-7B model that responds to any instruction in Gen-Z slang.

---

## What is this?

GenZify is a QLoRA fine-tuned version of `mistralai/Mistral-7B-Instruct-v0.2`, trained on a custom dataset of 240 instruction-output pairs where every response is written in authentic Gen-Z slang — think *"no cap"*, *"fr fr"*, *"it's giving"*, and *"lowkey slaps"*.

Give it any instruction and it'll answer like your terminally online younger sibling. 💀

---

## Examples

**Input:**
```
Instruction: Generate a sentence with a complex verbal structure using a given vocabulary.
Words: devious, antagonistic, ferocity
```
**Output:**
```
The devious politician went full antagonistic mode, attacking his opponent 
with straight ferocity that left the whole audience shook 😱
```

---

**Input:**
```
Instruction: Explain what gravity is.
```
**Output:**
```
ok so gravity is basically the universe's way of saying 
"you're not leaving bestie" — it's the force that keeps 
everything from just yeeting off into space fr fr 🌍
```

---

## Model Details

| Property | Value |
|---|---|
| Base Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Fine-tuning Method | QLoRA (4-bit quantization + LoRA) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Training Examples | 240 |
| Epochs | 3 |
| Framework | HuggingFace Transformers + TRL |

---

## Project Structure

```
genZify/
├── data/
│   └── data.jsonl               # Training dataset (240 examples)
├── genz-lora-adapter/           # Trained LoRA adapter weights
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── tokenizer files
├── genz-merged-model/           # Full merged model (adapter + base)
│   ├── config.json
│   ├── model-00001-of-00004.safetensors
│   └── ...
└── README.md
```

---

## Quickstart

### Installation

```bash
pip install transformers peft bitsandbytes accelerate torch
```

### Option A: Inference with merged model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = "./genz-merged-model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()
```

### Option B: Inference with adapter only (saves disk space)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base_model, "./genz-lora-adapter")
model.eval()
```

### Generate a response

```python
def generate_genz(instruction, input_text=""):
    user_content = f"{instruction}\n\n{input_text}" if input_text.strip() else instruction

    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# Try it!
print(generate_genz("Explain quantum physics."))
print(generate_genz("Write a haiku about summer."))
print(generate_genz("Give me 3 productivity tips."))
```

---

## Training

The model was fine-tuned using QLoRA on a custom dataset of 240 instruction-output pairs.
Each output is written in Gen-Z slang with emojis, abbreviations, and internet-native phrasing.

### Dataset format

```json
{
  "instruction": "Rearrange the following sentence to make it more interesting.",
  "input": "She left the party early",
  "output": "She yeeted outta that party way too early fr 💀"
}
```

### Reproduce training

```python
# Format dataset
def format_as_messages(example):
    user_content = f"{example['instruction']}\n\n{example['input']}" \
                   if example["input"].strip() else example["instruction"]
    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": example["output"]}
        ]
    }

# Train
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./genz-slang-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    bf16=True,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    eval_strategy="no",
    max_length=512,
    packing=False,
)
```

---

## Hardware Requirements

| Setup | Minimum VRAM | Notes |
|---|---|---|
| 4-bit inference | 6GB | Recommended for most GPUs |
| Full bfloat16 inference | 16GB | Best quality |
| Training (QLoRA) | 15GB | Used T4 GPU on Kaggle |

---

## Limitations

- Trained on only 240 examples — may be inconsistent on rare topics
- Slang style can vary — some outputs more Gen-Z than others
- Not suitable for formal or professional use cases (that's kind of the point tho 💀)
- Based on Mistral-7B-Instruct — inherits its general limitations

---

## License

Apache 2.0 — same as the base Mistral-7B model.

---

## Acknowledgements

- [Mistral AI](https://mistral.ai) for the base model
- [HuggingFace TRL](https://github.com/huggingface/trl) for SFTTrainer
- [Tim Dettmers](https://github.com/TimDettmers/bitsandbytes) for bitsandbytes
- Everyone on Gen-Z Twitter/TikTok for the training data vibes ✨

---

*Built with 🔥 and zero chill.*
