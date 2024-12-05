  
from pydantic import BaseModel

# Schema for input text and model selection
class TextInput(BaseModel):
    input_text: str  # The raw input text
    use_gemini: bool = True  # Flag to choose between Gemini (True) or Llama (False)

# Schema for output (response) from the model (Gemini or Llama)
class TextOutput(BaseModel):
    processed_text: str  # The output generated from the model

# Schema for error responses
class ErrorResponse(BaseModel):
    detail: str  # Description of the error
