  
from app.llm_handler import process_with_llama
from app.gemini_api import process_with_gemini

def process_pipeline(input_text: str, use_gemini: bool) -> str:
    """Decides which model (Gemini or Llama) to use based on the 'use_gemini' flag."""
    if use_gemini:
        return process_with_gemini(input_text)  # Process with Gemini API
    else:
        return process_with_llama(input_text)  # Process with local Llama model
