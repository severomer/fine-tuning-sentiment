import os
import time
import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

# Force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cpu")
print("Device:", device)

# 1. Load dataset
dataset = load_dataset("imdb")
small_train = dataset["train"].shuffle(seed=42).select(range(2000))   # subset for speed
small_test = dataset["test"].shuffle(seed=42).select(range(1000))

# 2. Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = small_train.map(tokenize_fn, batched=True)
test_dataset = small_test.map(tokenize_fn, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 3. Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
).to(device)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=1,         # keep short for CPU demo
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=20,
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 6. Train
print("Starting fine-tuning...")
start = time.time()
trainer.train()
end = time.time()
print(f"✅ CPU training time: {end - start:.2f} seconds")

# 7. Evaluate
print("\nEvaluating fine-tuned model...")
results = trainer.evaluate()
print("Evaluation results:", results)

# 8. Test inference
print("\nTesting inference with fine-tuned model:")
clf = pipeline("sentiment-analysis", model="./results", tokenizer="distilbert-base-uncased")

print(clf("I really loved this movie, it was amazing!"))
print(clf("The film was terrible and boring."))

# 9. Compare with base model
print("\nComparing with base (non-finetuned) model:")
clf_base = pipeline("sentiment-analysis", model="distilbert-base-uncased")
print(clf_base("The film was terrible and boring."))
