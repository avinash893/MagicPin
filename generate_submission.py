import os
import json
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"
TEST_PAIRS_FILE = DATASET_DIR / "test_pairs.json"
SUBMISSION_FILE = BASE_DIR / "submission.jsonl"

# Ensure API key is configured
gemini_key = os.environ.get("PS_GEMINIAPIKEY") or os.environ.get("GEMINI_API_KEY")
openai_key = os.environ.get("PS_OPENAIAPIKEY") or os.environ.get("OPENAI_API_KEY")

if not gemini_key and not openai_key:
    print("WARNING: No LLM API key detected in environment. Running composition might fallback to mock.")

# Import composition function from bot.py
# This will initialize the LLM client using env variables automatically
from bot import run_composition_llm

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    if not TEST_PAIRS_FILE.exists():
        print(f"Error: {TEST_PAIRS_FILE} not found. Did you run the generator?")
        return

    test_data = load_json(TEST_PAIRS_FILE)
    pairs = test_data.get("pairs", [])
    print(f"Loaded {len(pairs)} test cases. Generating submissions...")

    submission_lines = []

    for idx, pair in enumerate(pairs):
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair["customer_id"]

        print(f"[{idx+1}/{len(pairs)}] Composing for {test_id} (Merchant: {merchant_id}, Trigger: {trigger_id})...")

        # Load contexts
        merchant_path = DATASET_DIR / "merchants" / f"{merchant_id}.json"
        merchant = load_json(merchant_path)

        category_slug = merchant.get("category_slug")
        category_path = DATASET_DIR / "categories" / f"{category_slug}.json"
        category = load_json(category_path)

        trigger_path = DATASET_DIR / "triggers" / f"{trigger_id}.json"
        trigger = load_json(trigger_path)

        customer = None
        if customer_id:
            customer_path = DATASET_DIR / "customers" / f"{customer_id}.json"
            customer = load_json(customer_path)

        # Generate response using LLM composer
        composed = run_composition_llm(category, merchant, trigger, customer)

        # Construct submission object
        submission_entry = {
            "test_id": test_id,
            "body": composed.get("body", ""),
            "cta": composed.get("cta", "open_ended"),
            "send_as": composed.get("send_as", "vera"),
            "suppression_key": composed.get("suppression_key", ""),
            "rationale": composed.get("rationale", "")
        }

        submission_lines.append(submission_entry)

    # Write to submission.jsonl
    with open(SUBMISSION_FILE, "w", encoding="utf-8") as out_f:
        for entry in submission_lines:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Success! Generated submission file at: {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()
