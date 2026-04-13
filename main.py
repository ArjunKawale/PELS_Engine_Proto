import time
from fastapi import FastAPI, HTTPException, Depends
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
# 1. Global Rate Limiter Logic
# ==========================================
# This list will store the timestamps of recent requests
request_timestamps = []

def rate_limiter():
    """
    Ensures no more than 30 requests are processed per minute globally.
    If the limit is exceeded, immediately returns a 429 error.
    """
    global request_timestamps
    current_time = time.time()
    
    # 1. Clean up old timestamps (remove anything older than 60 seconds)
    request_timestamps = [t for t in request_timestamps if current_time - t < 60]
    
    # 2. Check if we have hit the 30 request limit
    if len(request_timestamps) >= 30:
        raise HTTPException(
            status_code=429, 
            detail="Server rate limit exceeded (30 requests/minute). Please wait 60 seconds and try again."
        )
    
    # 3. If safe, log this request's timestamp and allow it to pass
    request_timestamps.append(current_time)

# ==========================================
# 2. Pydantic Models for Batching
# ==========================================
class QAPair(BaseModel):
    task: str
    prompt: str

class BatchSubmission(BaseModel):
    qa_pairs: List[QAPair]

# ==========================================
# 3. Evaluation Endpoint
# ==========================================
# We inject the rate_limiter() function as a Dependency here.
# FastAPI will run the rate limiter BEFORE it even looks at the payload.
@app.post("/evaluate", dependencies=[Depends(rate_limiter)])
def evaluate_batch_endpoint(submission: BatchSubmission):
    try:
        # Convert the Pydantic objects into a simple list of dictionaries 
        dict_pairs = [{"task": pair.task, "prompt": pair.prompt} for pair in submission.qa_pairs]

        # Call the batch engine
        result = pels_engine.evaluate_batch(
            qa_pairs=dict_pairs,
            token_limit=3000 # Your safety limit
        )
        
        # Check if the engine caught an error or hit the token safety abort
        if "error" in result:
            if "SAFETY ABORT" in result["error"]:
                # 413 Payload Too Large
                raise HTTPException(status_code=413, detail=result) 
            # 500 Internal Server Error
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions so they don't get caught by the broad except below
        raise
    except Exception as e:
        # Catch any unexpected server/python errors
        raise HTTPException(status_code=500, detail=f"Server fault: {str(e)}")

# ==========================================
# 4. Health Check
# ==========================================
@app.get("/")
def root():
    return {"status": "PELS-Eval Batch Engine is online and ready."}