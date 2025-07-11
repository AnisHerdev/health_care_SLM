import json
import random
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict

# 1. Load and parse qa_cot_data.json
def load_qa_cot_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    examples = []
    for entry in data:
        qa_cot = entry.get('qa_cot')
        if not qa_cot:
            continue
        # Try to extract question, options, answer, explanation
        try:
            lines = qa_cot.split('\n')
            question = next((l for l in lines if l.strip().lower().startswith('question:')), None)
            options = []
            for l in lines:
                if l.strip().startswith(('A)', 'B)', 'C)', 'D)')):
                    options.append(l.strip())
            answer = next((l for l in lines if l.strip().lower().startswith('answer:')), None)
            explanation = next((l for l in lines if l.strip().lower().startswith('explanation:')), None)
            if question and options and answer and explanation:
                prompt = question + '\n' + '\n'.join(options)
                response = answer + '\n' + explanation
                examples.append({'prompt': prompt, 'response': response})
        except Exception as e:
            continue
    return examples

# 2. Prepare Hugging Face Dataset
def prepare_dataset(examples, test_ratio=0.2, seed=42):
    random.seed(seed)
    random.shuffle(examples)
    n_test = int(len(examples) * test_ratio)
    test = examples[:n_test]
    train = examples[n_test:]
    ds = DatasetDict({
        'train': Dataset.from_list(train),
        'test': Dataset.from_list(test)
    })
    return ds

# 3. Tokenization
def tokenize_function(example, tokenizer, max_input_length=512, max_target_length=256):
    model_inputs = tokenizer(
        example['prompt'], max_length=max_input_length, truncation=True, padding='max_length'
    )
    labels = tokenizer(
        example['response'], max_length=max_target_length, truncation=True, padding='max_length'
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# 4. Main training pipeline
def main():
    data_path = 'qa_cot_data.json'
    model_name = 't5-small'
    output_dir = './t5_qa_cot_model'
    batch_size = 4
    num_train_epochs = 5

    print('Loading data...')
    examples = load_qa_cot_data(data_path)
    print(f'Loaded {len(examples)} usable QA-CoT examples.')
    if len(examples) < 10:
        print('Not enough data to train. Exiting.')
        return

    ds = prepare_dataset(examples)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    def token_fn(ex):
        return tokenize_function(ex, tokenizer)

    print('Tokenizing...')
    tokenized_ds = ds.map(token_fn, batched=False)

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        logging_dir='./logs',
        logging_steps=10,
        report_to='none',
        push_to_hub=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['test'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print('Training...')
    trainer.train()
    print('Evaluating...')
    eval_results = trainer.evaluate()
    print('Eval results:', eval_results)
    print(f'Saving model to {output_dir}...')
    trainer.save_model(output_dir)
    print('Done.')

if __name__ == '__main__':
    main() 