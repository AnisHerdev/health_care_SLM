import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Configurations
MODEL_NAME = 't5-small'
INPUT_JSON = 'textbook_chunks.json'
OUTPUT_DIR = './t5_finetuned_on_chunks'
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 2

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

class TextbookChunkDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_input_length=512, max_target_length=512):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        input_text = chunk['text']
        # For demonstration, use the same text as target (autoencoding)
        target_text = chunk['text']
        input_enc = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        target_enc = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_target_length,
            return_tensors='pt'
        )
        return {
            'input_ids': input_enc['input_ids'].squeeze(),
            'attention_mask': input_enc['attention_mask'].squeeze(),
            'labels': target_enc['input_ids'].squeeze()
        }

def main():
    dataset = TextbookChunkDataset(INPUT_JSON, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        report_to=[],  # disables logging to wandb etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model fine-tuned and saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main() 