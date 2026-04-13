import os
import json
from datetime import datetime
import chromadb
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ==========================================
# 1. Environment & API Setup
# ==========================================
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL = "claude-haiku-4-5-20251001"

# ==========================================
# 2. Vector DB Initialization
# ==========================================
DB_PATH = r"Database\pels_vector_db"

print(f"Loading Vector DB from: {DB_PATH}")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="prompt_examples")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ AI Engine Initialized.")

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

Evaluate purely on the text. Provide scores for each submission and one overall summary.

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

        # Enforce character cap upstream to save tokens
        if len(prompt) > 1000:
            prompt = prompt[:1000]

        rag_context = retrieve_context(prompt)
        print(f"  🔎 [RAG] ID {i}: {rag_context}")

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
    """Prints token usage and calculates cost for Haiku 4.5 pricing."""
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

    print(f"\n  📊 Token Usage:")
    print(f"     Input:        {input_tokens:,}")
    print(f"     Output:       {output_tokens:,}")
    print(f"     Cache write:  {cache_write:,}  (caching active: {cache_write > 0})")
    print(f"     Cache read:   {cache_read:,}")
    print(f"     💵 Est. cost: ${cost:.6f}  ({cost*100:.4f}¢)")
    return {"input": input_tokens, "output": output_tokens, "cost_usd": round(cost, 6)}


def estimate_input_tokens(system_text: str, user_text: str) -> int:
    """
    Estimates total input tokens based on character count.
    Uses a conservative 3.5 characters per token average.
    """
    total_chars = len(system_text) + len(user_text)
    estimated_tokens = int(total_chars / 3.5)
    return estimated_tokens


# ==========================================
# 5. Main Evaluation Function
# ==========================================
def evaluate_batch(qa_pairs: list, token_limit: int = 3000) -> dict:
    """Single API call evaluates up to 10 task/prompt pairs with a pre-flight token check."""
    
    print("\n" + "="*55)
    print(f"  PREPARING BATCH OF {len(qa_pairs)} SUBMISSIONS")
    print("="*55)

    submissions_text = build_batch_content(qa_pairs)

    user_message = (
        "<batch_submissions>\n"
        f"{submissions_text}"
        "</batch_submissions>\n\n"
        "Evaluate all submissions above. Return ONLY the JSON object."
    )

    # --- PRE-FLIGHT TOKEN CHECK ---
    estimated_tokens = estimate_input_tokens(PELS_SYSTEM_PROMPT, user_message)
    print(f"\n  📏 Pre-flight Check: ~{estimated_tokens:,} estimated input tokens.")

    if estimated_tokens > token_limit:
        print(f"  ❌ SAFETY ABORT: Request exceeds {token_limit:,} tokens. API call blocked.")
        return {
            "error": "SAFETY ABORT: Input payload too large.",
            "estimated_tokens": estimated_tokens,
            "limit": token_limit
        }
    # ------------------------------

    try:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] 🚀 Token count safe. Sending batch to Claude Haiku 4.5...")

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

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] ✅ Response received.")

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

        return eval_data

    except json.JSONDecodeError as e:
        raw = getattr(response, 'content', [{}])
        return {
            "error": f"JSON parse failed: {e}",
            "raw_response": raw[0].text if raw else "No content"
        }
    except anthropic.APIError as e:
        return {"error": f"Anthropic API error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


# ==========================================
# 6. Execution Test
# ==========================================
if __name__ == "__main__":
    test_batch = [
        {
            "task": "Transition from software development to product management with a strong resume",
            "prompt": "You are a senior career coach with 10 years of experience in the IT industry. I am a mid-level software developer looking to transition to a product management role. Write a detailed resume highlighting my technical and leadership skills. Include specific metrics and achievements. Ensure the resume is ATS-friendly and tailored for product management roles."
        },
        {
            "task": "Apply for a project manager job with a professional cover letter",
            "prompt": "You are a recruiter. Write a cover letter for a project manager position."
        },
        {
            "task": "Optimize LinkedIn profile to attract top marketing recruiters",
            "prompt": "You are a LinkedIn optimization expert. I am a digital marketing professional with 5 years of experience. Improve my LinkedIn profile to attract recruiters from top marketing agencies. Highlight my expertise in SEO, content marketing, and social media strategy. Use industry-specific keywords and ensure my profile stands out."
        },
        {
            "task": "Create a resume for a junior software engineer role",
            "prompt": "You are a career coach. Write a resume for a software engineer with 2 years of experience."
        },
        {
            "task": "Prepare effectively for a senior developer behavioral interview",
            "prompt": "You are an interview preparation coach. I have an upcoming behavioral interview for a senior developer role. How should I prepare? Provide a list of common behavioral questions, the STAR method for answering them, and tips for demonstrating leadership and problem-solving skills."
        },
        {
            "task": "Understand what a CV is and how it is used",
            "prompt": "what is a cv"
        },
        {
            "task": "Find a suitable mentor for career growth",
            "prompt": "help me find a mentor"
        },
        {
            "task": "Reach out to a professional connection on LinkedIn",
            "prompt": "write a linkedin message"
        },
        {
            "task": "Improve daily productivity and work efficiency",
            "prompt": "how to be productive at work"
        },
        {
            "task": "Handle failure after an interview and plan next steps",
            "prompt": "i failed an interview what do i do"
        }
    ]

    result = evaluate_batch(test_batch)

    print("\n" + "="*55)
    print("  FINAL OUTPUT")
    print("="*55)
    print(json.dumps(result, indent=4))