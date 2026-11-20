import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3Guard-Gen-0.6B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

model.eval()


def evaluate(messages):
    # Prepare conversation for the safety judge
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    # Generate safety evaluation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    # Extract labels
    safety = re.search(
        r"Safety:\s*(Safe|Unsafe|Controversial)",
        result
    )
    refusal = re.search(
        r"Refusal:\s*(Yes|No)",
        result
    )

    return {
        "safety": safety.group(1) if safety else None,
        "refusal": refusal.group(1) if refusal else None,
        "raw_output": result
    }
