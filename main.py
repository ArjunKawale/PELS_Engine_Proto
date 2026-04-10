from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pels_engine

# Initialize the FastAPI app
app = FastAPI(title="PELS Prompt Evaluation API")

# Add CORS Middleware so your frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# UPDATE: Define the data structure to include both the task and the prompt
class PromptSubmission(BaseModel):
    task_description: str
    student_prompt: str

# Create the POST endpoint
@app.post("/evaluate")
async def evaluate_prompt_endpoint(submission: PromptSubmission):
    try:
        # UPDATE: Pass both the task description and the prompt to your engine
        result = pels_engine.evaluate_prompt(
            task_description=submission.task_description,
            student_prompt=submission.student_prompt
        )
        
        # Check if the engine caught a JSON parsing error or Gemini error
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        # Catch any unexpected server errors
        raise HTTPException(status_code=500, detail=str(e))

# A simple health check endpoint just to make sure the server is awake
@app.get("/")
async def root():
    return {"status": "PELS-Eval Engine is online and ready."}