  
from fastapi import FastAPI, HTTPException
from app.schemas import TextInput, TextOutput, ErrorResponse
from app.pipeline import process_pipeline

app = FastAPI()

@app.post("/process_text", response_model=TextOutput, responses={400: {"model": ErrorResponse}})
async def process_text(input_data: TextInput):
    try:
        # Process the text using either Llama or Gemini
        result = process_pipeline(input_data.input_text, input_data.use_gemini)
        return TextOutput(processed_text=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
