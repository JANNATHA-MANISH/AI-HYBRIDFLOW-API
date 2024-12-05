import os
import psutil  # For checking RAM usage
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from app.config import MODEL_PATH, HUGGINGFACE_TOKEN, MAX_NEW_TOKENS, TEMPERATURE, TOP_K, TOP_P, NUM_BEAMS, NO_REPEAT_NGRAM_SIZE

# Function to print memory usage
def print_memory_usage():
    process = psutil.Process(os.getpid())  # Get the current process ID
    memory_info = process.memory_info()    # Get memory usage info
    memory_usage_in_gb = memory_info.rss / (1024 ** 3)  # Convert from bytes to GB
    print(f"Current memory usage: {memory_usage_in_gb:.2f} GB")

# Loading the tokenizer and model using the Hugging Face token for authentication
print("Loading the model and tokenizer...")

# Set the device to GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, token=HUGGINGFACE_TOKEN).to(device)

# Check and set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Using EOS token as pad token
    print("Pad token was not defined, using EOS token as pad token.")

# Print model name and initial memory usage after loading
print(f"Model loaded: {MODEL_PATH}")
print_memory_usage()

# Sample function to process input and generate response
def process_with_llama(input_text: str, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, num_beams=NUM_BEAMS, no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE):
    print(f"Processing input: {input_text}")

    # Tokenize the input text and ensure it's on the same device as the model
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

    # Handle attention_mask and padding
    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, dtype=torch.long).to(device)
    
    # Set the pad_token_id if not already defined
    inputs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Generating text using the model with enhanced generation parameters
    outputs = model.generate(
        **inputs,
        #max_new_tokens=max_new_tokens,  # Control output length
        #temperature=temperature,  # For controlling randomness
        top_k=top_k,  # Sampling from the top k tokens
        top_p=top_p,  # Nucleus sampling
        num_beams=num_beams,  # Beam search for better quality
        no_repeat_ngram_size=no_repeat_ngram_size,  # To avoid repetition
        early_stopping=True  # Stop when the model generates an end token
    )

    # Decode the output text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print memory usage after generating output
    print_memory_usage()
    print(f"generated text: {generated_text}")
    
    return generated_text


# Test the function with an example input
if __name__ == "__main__":
    """ input_text = "What is the meaning of life?"
    response = process_with_llama(input_text)
    print("Generated Text: ", response) """
