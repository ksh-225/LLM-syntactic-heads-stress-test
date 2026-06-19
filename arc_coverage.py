"""
A.2 — Fine-tune mGPT (1.3B) on CC-100 Hindi (500K sentences).
mGPT's SentencePiece tokenizer handles Devanagari script natively.
"""
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset


def main():
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/mGPT")
    model = AutoModelForCausalLM.from_pretrained("sberbank-ai/mGPT")

    raw = load_dataset("cc100", lang="hi", split="train", streaming=True)
    texts = [ex["text"] for i, ex in enumerate(raw) if i < 500_000]
    dataset = Dataset.from_dict({"text": texts}).train_test_split(test_size=0.01)

    def tokenize_hi(ex):
        return tokenizer(ex["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize_hi, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir="./mgpt_hi",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        fp16=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model("./mgpt_hi_finetuned")


if __name__ == "__main__":
    main()
