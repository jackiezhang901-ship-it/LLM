import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./lora-out"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# QLoRA 量化配置（和训练一致）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# 加载 base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)

# 🔥 加载 LoRA adapter
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

print("model device:", next(model.parameters()).device)

# ===== 推理 =====
prompt = """<|system|>
跑得快出牌规则，可出单张，对子，连对最少两对，可出顺子，顺子至少五张，可以出三带二，飞机，四张牌是炸弹，可以四带三,有大牌必须压，每人15张牌
<|user|>
手牌：3,3,3,4,5,5,6,7,8,8,9,10,J,J,Q,K，给出当前轮次如何出牌的最佳策略,不需要后面回合的出法,不需要分析过程,直接给出结果.
<|assistant|>
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
