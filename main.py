import time
import os
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pels_engine

# Initialize the FastAPI app
app = FastAPI(title="PELS Prompt Evaluation API (Batch Processing)")

# Add CORS Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. API Key Verification
# ==========================================
API_KEY = os.getenv("API_KEY")

def verify_api_key(x_api_key: str = Header(...)):
    """
    Checks for a valid API key in the 'x-api-key' request header.
    Raises 403 if missing or incorrect.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY not set.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key.")

# ==========================================
# 2. Global Rate Limiter Logic
# ==========================================
request_timestamps = []

def rate_limiter():
    global request_timestamps
    current_time = time.time()
    request_timestamps = [t for t in request_timestamps if current_time - t < 60]
    if len(request_timestamps) >= 30:
        raise HTTPException(
            status_code=429, 
            detail="Server rate limit exceeded (30 requests/minute). Please wait 60 seconds and try again."
        )
    request_timestamps.append(current_time)

# ==========================================
# 3. Pydantic Models for Batching
# ==========================================
class QAPair(BaseModel):
    task: str
    prompt: str

class BatchSubmission(BaseModel):
    qa_pairs: List[QAPair]

# ==========================================
# 4. Evaluation Endpoint
# ==========================================
@app.post("/evaluate", dependencies=[Depends(verify_api_key), Depends(rate_limiter)])
def evaluate_batch_endpoint(submission: BatchSubmission):
    try:
        dict_pairs = [{"task": pair.task, "prompt": pair.prompt} for pair in submission.qa_pairs]

        result = pels_engine.evaluate_batch(
            qa_pairs=dict_pairs,
            token_limit=3000
        )
        
        if "error" in result:
            if "SAFETY ABORT" in result["error"]:
                raise HTTPException(status_code=413, detail=result) 
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server fault: {str(e)}")

# ==========================================
# 5. Health Check (no auth needed)
# ==========================================
@app.get("/")
def root():
    return {"status": "PELS-Eval Batch Engine is online and ready."}