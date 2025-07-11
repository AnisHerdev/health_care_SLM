import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_DIR = './t5_finetuned_on_chunks'

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print("Chatbot is ready! Type your message (or 'exit' to quit):")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    # Encode input
    input_ids = tokenizer.encode(user_input, return_tensors='pt', truncation=True, max_length=256).to(device)
    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Bot:", response)