import os
import json
import logging
from datetime import datetime
import chromadb
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ==========================================
# 0. Logging Configuration
# ==========================================
class SequentialIDFilter(logging.Filter):
    """A custom filter that injects an auto-incrementing ID into every log record."""
    def __init__(self):
        super().__init__()
        self.counter = 1

    def filter(self, record):
        record.log_id = self.counter
        self.counter += 1
        return True

# Initialize Logger
logger = logging.getLogger("PELSEngine")
logger.setLevel(logging.INFO)

# Define Handlers (File and Console)
file_handler = logging.FileHandler("pels_engine.log", encoding="utf-8")
console_handler = logging.StreamHandler()

# Define the format including the dynamic %(log_id)s
formatter = logging.Formatter(
    "[Log ID: %(log_id)04d] %(asctime)s | %(levelname)s | %(message)s", 
    datefmt="%H:%M:%S"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Attach filter and handlers to the logger
seq_filter = SequentialIDFilter()
logger.addFilter(seq_filter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False # Prevents duplicate logs if other libraries use root logger

# ==========================================
# 1. Environment & API Setup
# ==========================================
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    logger.error("ANTHROPIC_API_KEY not found in .env file.")
    raise ValueError("ANTHROPIC_API_KEY not found in .env file.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL = "claude-haiku-4-5-20251001"

# ==========================================
# 2. Vector DB Initialization
# ==========================================
DB_PATH = r"Database\pels_vector_db"

logger.info(f"Loading Vector DB from: {DB_PATH}")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="prompt_examples")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("✅ AI Engine Initialized.")

# ==========================================
# 3. Master System Prompt
# ==========================================
PELS_SYSTEM_PROMPT = """
You are PELS-Eval. Evaluate a batch of student prompts against their tasks.
Score C1-C6 (1-10 integers) using this condensed rubric:
* C1 Foundations (15%): 1-3: Magic box. 4-7: Basic token/temp awareness. 8-10: Explicit limits/temp design.
* C2 Design (20%): 1-3: Generic. 4-7: 1 technique (role/format). 8-10: 2+ combined techniques.
* C3 Output (20%): 1-3: None. 4-7: Format OR length. 8-10: Format+length+structure+constraints.
* C4 Domain (20%): 1-3: None. 4-7: Basic persona. 8-10: Full structure/vocab adaptation.
* C5 Ethics (15%): 1-3: Unaware. 4-7: Avoids obvious harm. 8-10: Proactive ethical constraints.
* C6 Meta (10%): 1-3: Overconfident. 4-7: Flags uncertainty. 8-10: Explicit self-check/limitations.

Evaluate purely on the text and relation to task. Provide scores for each submission and one overall summary.
IMPORTANT:
The provided reference examples are approximate and retrieved via semantic similarity.
They may not exactly match the task or intent.

Use them only as rough indicators of quality levels (weak, developing, strong),
NOT as strict calibration targets.

Prioritize evaluating the student prompt independently based on the rubric.
CRITICAL: Respond with ONLY a valid JSON object. No markdown, no explanation before or after.
Output format:
{
  "evaluations": [
    {"id": <int>, "c1": <int>, "c2": <int>, "c3": <int>, "c4": <int>, "c5": <int>, "c6": <int>}
  ],
  "overall_summary": {
  "strength": "<max 20 words>",
"weakness": "<max 20 words>",
"improvement": "<max 40 words, simple language>"
}
"""

# ==========================================
# 4. Core Functions
# ==========================================
def retrieve_context(student_prompt: str) -> str:
    """Embeds the prompt, queries ChromaDB, returns 1 calibration example."""
    query_embedding = embedding_model.encode([student_prompt]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )
    if results and results['documents'] and len(results['documents'][0]) > 0:
        doc_data = json.loads(results['documents'][0][0])
        return (
            f"Scores: C1:{doc_data.get('c1_score')} | C2:{doc_data.get('c2_score')} | "
            f"C3:{doc_data.get('c3_score')} | C4:{doc_data.get('c4_score')} | "
            f"C5:{doc_data.get('c5_score')} | C6:{doc_data.get('c6_score')} -> "
            f"Justification: {doc_data.get('justification', 'N/A')}"
        )
    return "No calibration data available."


def build_batch_content(qa_pairs: list) -> str:
    """Builds the user message content by sequencing each submission."""
    submissions_text = ""
    for i, pair in enumerate(qa_pairs, 1):
        task   = pair.get('task', '').strip()
        prompt = pair.get('prompt', '').strip()

        if len(prompt) > 1000:
            prompt = prompt[:1000]

        rag_context = retrieve_context(prompt)
        logger.info(f"🔎 [RAG] ID {i} context retrieved: {rag_context}")

        submissions_text += (
            f"--- Submission ID {i} ---\n"
            f"TASK: {task}\n"
            f"STUDENT PROMPT: {prompt}\n"
            f"CALIBRATION DATA: {rag_context}\n\n"
        )
    return submissions_text


def calculate_scores(eval_data: dict) -> dict:
    """Adds weighted final score and skill level to each evaluation in-place."""
    weights = {"c1": 0.15, "c2": 0.20, "c3": 0.20, "c4": 0.20, "c5": 0.15, "c6": 0.10}
    for item in eval_data.get("evaluations", []):
        final_score = round(
            sum(item.get(k, 1) * w for k, w in weights.items()), 2
        )
        item["overall_marks"] = final_score
        item["skill_level"] = (
            "Weak"       if final_score < 4.0 else
            "Developing" if final_score < 7.0 else
            "Strong"
        )
    return eval_data


def log_usage(usage) -> dict:
    """Logs token usage to the file and console, returning the data."""
    input_tokens  = usage.input_tokens
    output_tokens = usage.output_tokens
    cache_write   = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read    = getattr(usage, "cache_read_input_tokens", 0) or 0

    INPUT_PRICE   = 1.00   
    OUTPUT_PRICE  = 5.00   
    CACHE_W_PRICE = 1.25   
    CACHE_R_PRICE = 0.10   

    cost = (
        (input_tokens  / 1_000_000) * INPUT_PRICE  +
        (output_tokens / 1_000_000) * OUTPUT_PRICE +
        (cache_write   / 1_000_000) * CACHE_W_PRICE +
        (cache_read    / 1_000_000) * CACHE_R_PRICE
    )

    logger.info(
        f"📊 Usage | In: {input_tokens:,} | Out: {output_tokens:,} | "
        f"Cache Write: {cache_write:,} | Cache Read: {cache_read:,} | "
        f"Est. Cost: ${cost:.6f}"
    )
    
    return {"input": input_tokens, "output": output_tokens, "cost_usd": round(cost, 6)}


def estimate_input_tokens(system_text: str, user_text: str) -> int:
    """Estimates total input tokens based on character count."""
    total_chars = len(system_text) + len(user_text)
    estimated_tokens = int(total_chars / 3.5)
    return estimated_tokens


# ==========================================
# 5. Main Evaluation Function
# ==========================================
def evaluate_batch(qa_pairs: list, token_limit: int = 3000) -> dict:
    """Single API call evaluates up to 10 task/prompt pairs with a pre-flight token check."""
    
    logger.info(f"Preparing batch of {len(qa_pairs)} submissions...")

    submissions_text = build_batch_content(qa_pairs)

    user_message = (
        "<batch_submissions>\n"
        f"{submissions_text}"
        "</batch_submissions>\n\n"
        "Evaluate all submissions above. Return ONLY the JSON object."
    )

    estimated_tokens = estimate_input_tokens(PELS_SYSTEM_PROMPT, user_message)
    logger.info(f"📏 Pre-flight Check: ~{estimated_tokens:,} estimated input tokens.")

    if estimated_tokens > token_limit:
        logger.error(f"❌ SAFETY ABORT: Request exceeds {token_limit:,} tokens. API call blocked.")
        return {
            "error": "SAFETY ABORT: Input payload too large.",
            "estimated_tokens": estimated_tokens,
            "limit": token_limit
        }

    try:
        logger.info("🚀 Token count safe. Sending batch to Claude Haiku 4.5...")

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": PELS_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        logger.info("✅ Response received successfully.")

        usage_stats = log_usage(response.usage)

        raw_text = response.content[0].text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        eval_data = json.loads(raw_text)
        eval_data = calculate_scores(eval_data)
        eval_data["_usage"] = usage_stats

        logger.info("✅ Batch processing and scoring complete.")
        return eval_data

    except json.JSONDecodeError as e:
        raw = getattr(response, 'content', [{}])
        logger.error(f"JSON parse failed: {e}")
        return {
            "error": f"JSON parse failed: {e}",
            "raw_response": raw[0].text if raw else "No content"
        }
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return {"error": f"Anthropic API error: {e}"}
    except Exception as e:
        logger.error(f"Unexpected engine error: {e}")
        return {"error": f"Unexpected error: {e}"}


# ==========================================
# 6. Execution Test
# ==========================================
if __name__ == "__main__":
    test_batch = [
    {
        "task": "Get advanced guidance for a successful tech job search",
        "prompt": "You are a career counselor. Give me advanced tips for a successful job search in the competitive tech industry. Include strategies for networking, tailoring applications, leveraging referrals, and preparing for technical and behavioral interviews. Also, provide tips on how to follow up after interviews and negotiate job offers."
    },  # STRONG

    {
        "task": "Ask manager for a promotion professionally",
        "prompt": "how to ask for promotion"
    },  # WEAK

    {
        "task": "Improve and optimize resume for job applications",
        "prompt": "You are an experienced resume coach. Polish my resume for a software engineering role. Improve clarity, impact, and ATS optimization. Use strong action verbs and quantify achievements where possible."
    },  # DEVELOPING

    {
        "task": "Prepare thoroughly for job interviews",
        "prompt": "prepare me for an interview"
    },  # WEAK

    {
        "task": "Write a tailored cover letter for a job application",
        "prompt": "You are a hiring manager. Help me write a compelling cover letter for a job application. Tailor it to the role, highlight relevant skills, and keep it concise and professional."
    },  # DEVELOPING

    {
        "task": "Understand what to say during interviews",
        "prompt": "what should I say in an interview"
    },  # WEAK

    {
        "task": "Create a professional bio for online profile",
        "prompt": "Give me a good bio for my profile"
    },  # WEAK

    {
        "task": "Draft a formal professional email",
        "prompt": "You are a corporate communication expert. Write a professional email for workplace communication. Ensure clarity, proper tone, and concise structure."
    },  # DEVELOPING

    {
        "task": "Determine appropriate salary expectations",
        "prompt": "what is a good salary to ask for"
    },  # WEAK

    {
        "task": "Negotiate salary effectively with employer",
        "prompt": "You are a negotiation expert. Help me negotiate salary for a job offer. Provide strategies, sample phrases, and tips to maximize compensation while maintaining professionalism."
    }  # STRONG
]

    result = evaluate_batch(test_batch)

    print("\n" + "="*55)
    print("  FINAL OUTPUT")
    print("="*55)
    print(json.dumps(result, indent=4))