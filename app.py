import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "somendrew/GenZify-adapter"

print("Loading GenZify... no cap this takes a sec ⏳")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.eval()
print("✅ Ready to slay!")

def generate_genz(instruction, input_text, max_new_tokens, temperature, repetition_penalty):
    if not instruction.strip():
        return "bestie u forgot to type something 💀"

    user_content = f"{instruction}\n\n{input_text}" if input_text.strip() else instruction
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=0.9,
            do_sample=True,
            repetition_penalty=float(repetition_penalty),
        )

    response = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

EXAMPLES = [
    ["Generate a sentence with complex vocabulary.", "Words: devious, antagonistic, ferocity", 150, 0.8, 1.1],
    ["Explain what black holes are.", "", 200, 0.8, 1.1],
    ["Give me 3 tips to stay productive.", "", 200, 0.8, 1.1],
    ["Compose a haiku about a summer day.", "", 100, 0.9, 1.1],
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono:wght@400;700&display=swap');
:root{--bg:#0a0a0a;--surface:#111;--border:#2a2a2a;--accent:#e8ff47;--text:#f0f0f0;--text-muted:#666;--radius:12px;}
body,.gradio-container{background:var(--bg)!important;color:var(--text)!important;font-family:'Space Mono',monospace!important;}
.gradio-container{max-width:900px!important;margin:0 auto!important;padding:2rem!important;}
textarea,input[type=text]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important;color:var(--text)!important;}
textarea:focus{border-color:var(--accent)!important;}
.output-box textarea{border-left:4px solid var(--accent)!important;}
button.primary{background:var(--accent)!important;color:#000!important;font-weight:800!important;border-radius:var(--radius)!important;}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:
    gr.HTML("""
        <div style="text-align:center;padding:2rem 0 1rem">
            <h1 style="font-family:Syne,sans-serif;font-size:3.5rem;font-weight:800;color:#e8ff47;letter-spacing:-2px;margin:0">GenZify 🔥</h1>
            <p style="color:#666;font-size:0.85rem;margin-top:0.5rem;letter-spacing:1px">MISTRAL-7B • QLORA FINE-TUNED • GEN-Z SLANG</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            instruction = gr.Textbox(
                label="Instruction",
                placeholder="e.g. Explain quantum physics / Write a haiku...",
                lines=3
            )
            input_text = gr.Textbox(
                label="Input (optional)",
                placeholder="e.g. Words: devious, antagonistic",
                lines=2
            )
        with gr.Column(scale=1):
            max_tokens  = gr.Slider(50,  400, value=150, step=10,   label="Max tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
            rep_penalty = gr.Slider(1.0, 1.5, value=1.1, step=0.05, label="Repetition penalty")

    btn    = gr.Button("🔥 GenZify It", variant="primary")
    output = gr.Textbox(
        label="GenZify Output",
        lines=6,
        interactive=False,
        elem_classes=["output-box"]
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[instruction, input_text, max_tokens, temperature, rep_penalty],
        outputs=output,
        fn=generate_genz,
        cache_examples=False
    )

    btn.click(
        fn=generate_genz,
        inputs=[instruction, input_text, max_tokens, temperature, rep_penalty],
        outputs=output
    )
    instruction.submit(
        fn=generate_genz,
        inputs=[instruction, input_text, max_tokens, temperature, rep_penalty],
        outputs=output
    )

    gr.HTML("<div style='text-align:center;color:#666;font-size:0.75rem;margin-top:2rem;border-top:1px solid #2a2a2a;padding-top:1rem'>BUILT WITH 🔥 AND ZERO CHILL</div>")

demo.launch()