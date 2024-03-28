import torch
from transformers import TrainingArguments, MistralForCausalLM, MistralConfig, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer

# Adjusted configuration for a smaller model with approximately a million parameters
configuration = MistralConfig(
    vocab_size=32000,
    hidden_size=1024, # Reduced from 2048 to 1024 to work on free colab
    intermediate_size=3584, # Reduced from 7168 to 3584 to work on free colab
    num_hidden_layers=12, # Reduced from 24 to 12 to work on free colab
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=4096,
    pad_token_id=2,
    bos_token_id=1,
    eos_token_id=2
)

model = MistralForCausalLM(configuration)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", local_files_only=False) # just using tokenizer of mistral
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset('HuggingFaceTB/cosmopedia-20k', split="train")
dataset = dataset.shuffle(seed=42)
print(f'Number of prompts: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')

def create_prompt_formats(sample):
    output_texts = []
    for i in range(len(sample['text'])):
        formatted_prompt = sample['text'][i]
        output_texts.append(formatted_prompt)
    return output_texts

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    formatting_func=create_prompt_formats,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        max_steps=1000,# Reduced from 10000 to 1000 to work on free colab
        learning_rate=1e-4,
        logging_steps=1,
        output_dir="M_outputs",
        overwrite_output_dir=True,
        save_steps=1000,
        optim="paged_adamw_32bit",
        report_to="none"
    )
)

trainer.train()
trainer.model.save_pretrained("M-final", dtype=torch.float32) # you can use float16
trainer.tokenizer.save_pretrained("M-final")
