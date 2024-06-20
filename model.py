from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch


model_path = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)


def format_prompts(examples):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = f"system:\n{instruction}\n"
        text += f"user:\n{input_text}\n"
        text += f"assistant:\n{output}\n"
        texts.append(f"{text}{tokenizer.eos_token}")
    return {"text": texts}


alpaca_instruct_dataset = load_dataset("yahma/alpaca-cleaned", split="train")
alpaca_instruct_dataset = alpaca_instruct_dataset.map(format_prompts, batched=True)

training_args = SFTConfig(
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    bf16=True,
    max_steps=-1,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=1000,
    output_dir="gpt2-alpaca",
    optim="paged_adamw_32bit",
    dataset_text_field="text",
    max_seq_length=1024,
    lr_scheduler_type="linear"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=alpaca_instruct_dataset,
    formatting_func=format_prompts
)

trainer.train()
trainer.save_model(output_dir="gpt2-alpaca")
