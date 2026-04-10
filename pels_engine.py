import os
import json
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ==========================================
# 1. Environment & API Setup
# ==========================================
# Load environment variables from the .env file in the same directory
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please ensure your .env file is set up correctly.")

genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 2. Model Initialization
# ==========================================
# Enforcing JSON output natively
generation_config = {"response_mime_type": "application/json"}
llm_model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)

# ==========================================
# 3. Vector DB Initialization
# ==========================================
# Using your exact absolute path
DB_PATH = r"D:/PROJECTS/PELS2/Database/pels_vector_db"

print(f"Loading Vector DB from: {DB_PATH}")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="prompt_examples")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ AI Engine Initialized.")

# ==========================================
# 4. Master System Prompt
# ==========================================
PELS_SYSTEM_PROMPT = """
You are PELS-Eval, a precise prompt quality assessor for a student assessment platform. Your objective is to score student-written AI prompts across 6 distinct categories based strictly on the provided rubric. You must operate with complete objectivity and adhere strictly to the scoring rules and output formats defined below.

<scoring_rubric>
Evaluate the prompt using the following criteria. Assign an integer score from 1 to 10 for each category.

*   **C1 Foundations (15%):** 
    *   1–3: Treats AI as a magic box. No mention of tokens, context, or temperature. 
    *   4–7: Knows what tokens/temp are but doesn't apply them deliberately. 
    *   8–10: Designs explicitly around context limits, temperature, and model differences. Anticipates failure modes.
*   **C2 Design & Patterns (20%):** 
    *   1–3: "Help me with this." No technique visible. Raw, generic request. 
    *   4–7: Applies one technique (role OR format), but not matched to the specific task. 
    *   8–10: Deliberately composes 2+ techniques. Justifies choices. Knows when NOT to use a technique.
*   **C3 Output Specification (20%):** 
    *   1–3: No output format, length, or structure mentioned. 
    *   4–7: One output dimension only (e.g., format OR length, but not both). 
    *   8–10: Fully specified: format + length + structure + at least one negative constraint or edge case.
*   **C4 Domain Application (20%):** 
    *   1–3: No domain adaptation. Same structure regardless of topic. 
    *   4–7: Domain marker added ("you are a student") but no structural change. 
    *   8–10: Full domain adaptation. Adapts structure, vocabulary, and constraints entirely to the domain.
*   **C5 Ethics & Safety (15%):** 
    *   1–3: No awareness of bias, harm potential, or manipulation risk. 
    *   4–7: Avoids obvious harms, but misses subtle bias or manipulation risks. 
    *   8–10: Proactively maps harm types and builds ethical constraints directly into the prompt structure.
*   **C6 Metacognition (10%):** 
    *   1–3: Equal confidence in all claims. Attributes all failures to the AI. 
    *   4–7: Can explain reasoning after the fact. Sometimes flags uncertainty. 
    *   8–10: Accurately knows what they know vs. don't know. Flags limits. Explicitly requests a self-check or disconfirming evidence.
</scoring_rubric>

<scoring_rules>
1.  **Evaluate Only the Prompt:** You are evaluating the quality of the *prompt itself* in the context of the requested task, not generating a response to the prompt.
2.  **No Extrapolation:** Score based solely on the text provided. Do not assume the user's intent if it is not explicitly written.
3.  **Output Specification:** C3 strictly measures Output Specification, NOT iterative refinement.
</scoring_rules>

<output_format>
Return ONLY a valid JSON object. Do not include markdown formatting like ```json or any conversational text.
{
  "c1": <integer_score>,
  "c2": <integer_score>,
  "c3": <integer_score>,
  "c4": <integer_score>,
  "c5": <integer_score>,
  "c6": <integer_score>,
  "justification": "<string explaining the reasoning in 2-3 sentences>"
}
</output_format>
"""

# ==========================================
# 5. Core Functions
# ==========================================
def retrieve_context(student_prompt: str, n_results: int = 3) -> str:
    """Embeds the prompt, queries ChromaDB, and formats the historical examples."""
    query_embedding = embedding_model.encode([student_prompt]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
    context_str = ""
    if results and results['documents']:
        for i, doc_str in enumerate(results['documents'][0]):
            doc_data = json.loads(doc_str)
            context_str += f"\n--- Historical Example {i+1} ---\n"
            context_str += f"Prompt: {doc_data.get('prompt_text', 'N/A')}\n"
            context_str += f"Scores: C1:{doc_data.get('c1_score')} | C2:{doc_data.get('c2_score')} | C3:{doc_data.get('c3_score')} | C4:{doc_data.get('c4_score')} | C5:{doc_data.get('c5_score')} | C6:{doc_data.get('c6_score')}\n"
            context_str += f"Justification: {doc_data.get('justification', 'N/A')}\n"
            context_str += f"Final Score: {doc_data.get('final_score', 'N/A')} ({doc_data.get('skill_level', 'N/A')})\n"
    
    return context_str

def evaluate_prompt(task_description: str, student_prompt: str) -> dict:
    """The main entry point. Retrieves context, calls Gemini, calculates final score."""
    # 1. Get RAG Context (We still search using the student's prompt to find similar patterns)
    historical_examples = retrieve_context(student_prompt)
    
    # 2. Assemble the payload
    full_prompt = f"""
    {PELS_SYSTEM_PROMPT}
    
    <calibration_examples>
    Here are previously graded prompts from the database to calibrate your scoring:
    {historical_examples}
    </calibration_examples>
    
    <student_submission>
    TARGET TASK: 
    "{task_description}"

    STUDENT PROMPT TO SCORE:
    "{student_prompt}"
    </student_submission>
    """
    
    try:
        # 3. Call Gemini
        response = llm_model.generate_content(full_prompt)
        
        # 4. Parse the JSON response
        eval_data = json.loads(response.text)
        
        # 5. Calculate Final Score using the exact PELS formula
        c1 = eval_data.get('c1', 1)
        c2 = eval_data.get('c2', 1)
        c3 = eval_data.get('c3', 1)
        c4 = eval_data.get('c4', 1)
        c5 = eval_data.get('c5', 1)
        c6 = eval_data.get('c6', 1)
        
        final_score = round(
            (c1 * 0.15) + (c2 * 0.20) + (c3 * 0.20) + 
            (c4 * 0.20) + (c5 * 0.15) + (c6 * 0.10), 
            2
        )
        
        # 6. Determine Skill Level
        if final_score < 4.0:
            skill_level = "Weak"
        elif final_score < 7.0:
            skill_level = "Developing"
        else:
            skill_level = "Strong"
            
        # 7. Append calculated fields to the final payload
        eval_data['overall_marks'] = final_score
        eval_data['skill_level'] = skill_level
        
        return eval_data

    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON from LLM response", "raw_response": getattr(response, 'text', 'No text returned')}
    except Exception as e:
        return {"error": str(e)}

# Optional: Add a simple test block that only runs if you execute this file directly
if __name__ == "__main__":
    print("\nRunning a quick test...\n")
    test_task = "Write a prompt to ask an AI to help you debug a React component."
    test_prompt = "Act as a software engineer. Review my react code."
    
    # Notice we now pass BOTH the task and the prompt
    result = evaluate_prompt(test_task, test_prompt)
    print(json.dumps(result, indent=4))