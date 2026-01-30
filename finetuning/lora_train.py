import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LEN = 512

# 1. load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# 2. load 4bit model（QLoRA）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# 3. LoRA 配置（4GB 安全）
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. dataset handling
def format_messages(example):
    text = ""
    for m in example["messages"]:
        if m["role"] == "system":
            text += f"<|system|>\n{m['content']}\n"
        elif m["role"] == "user":
            text += f"<|user|>\n{m['content']}\n"
        elif m["role"] == "assistant":
            text += f"<|assistant|>\n{m['content']}\n"
    return {"text": text}

dataset = load_dataset("json", data_files="train.json")
dataset = dataset.map(format_messages)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False
    )

dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# 5. training parameter
training_args = TrainingArguments(
    output_dir="./lora-out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=0.3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

# 6. start to train and output
trainer.train()
trainer.save_model("./lora-o")
