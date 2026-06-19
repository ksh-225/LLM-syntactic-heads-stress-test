"""
A.1 — Fine-tune GPT-2 (117M) on WikiText-103 (English).
"""
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir="./gpt2_en",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        fp16=True,
        eval_strategy="steps",
        eval_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model("./gpt2_en_finetuned")


if __name__ == "__main__":
    main()
