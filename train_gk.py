import json
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from nltk.tokenize import sent_tokenize
import nltk

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Configuration
INPUT_JSON = "textbook_chunks.json"
OUTPUT_JSON = "training_pairs.json"
MODEL_NAME = "t5-small"  # Replace with your model if specified
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

# Clean text (replace special characters and normalize whitespace)
def clean_text(text):
    text = re.sub(r'', 'o', text)  # Replace special character with 'o'
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Extract first sentence or first 50 words as fallback
def extract_summary(text):
    cleaned_text = clean_text(text)
    try:
        sentences = sent_tokenize(cleaned_text)
        if sentences:
            return sentences[0]
    except Exception:
        pass
    # Fallback: first 50 words
    words = cleaned_text.split()[:50]
    return ' '.join(words)

# Create training pairs
def create_training_pairs():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    training_pairs = []
    for chunk in chunks:
        text = chunk.get('text', '')
        if not text:
            print(f"Warning: Empty text in chunk {chunk['chunk_index']}")
            continue
        cleaned_text = clean_text(text)
        summary = extract_summary(text)
        if not summary:
            print(f"Warning: No summary extracted for chunk {chunk['chunk_index']}")
            continue
        input_text = f"Explain the following: {cleaned_text}"
        training_pairs.append({
            'input': input_text,
            'target': summary,
            'chapter_title': chunk['chapter_title'],
            'chunk_index': chunk['chunk_index']
        })
    
    print(f"Created {len(training_pairs)} training pairs")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(training_pairs, f, ensure_ascii=False, indent=2)
    return training_pairs

# Tokenize dataset
def tokenize_data(pairs, tokenizer):
    inputs = [pair['input'] for pair in pairs]
    targets = [pair['target'] for pair in pairs]
    encodings = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding='max_length')
    target_encodings = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding='max_length')
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

# Main training function
def main():
    print("Loading chunks and creating training pairs...")
    training_pairs = create_training_pairs()
    
    print(f"Loading model and tokenizer: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    print("Tokenizing data...")
    dataset = tokenize_data(training_pairs, tokenizer)
    
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='./model_output',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting training...")
    trainer.train()
    print("Saving model...")
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')
    print("Done.")

if __name__ == "__main__":
    main()
    