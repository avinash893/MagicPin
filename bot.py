import os
import time
import json
import re
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Any, List, Dict, Optional

# Load API keys from environment variables
GEMINI_KEY = os.environ.get("PS_GEMINIAPIKEY") or os.environ.get("GEMINI_API_KEY")
OPENAI_KEY = os.environ.get("PS_OPENAIAPIKEY") or os.environ.get("OPENAI_API_KEY")

# Initialize LLM Client
llm_client = None
llm_type = None

if GEMINI_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        llm_client = genai.GenerativeModel('gemini-3.1-flash-lite')
        llm_type = "gemini"
        print("Initialized Gemini model successfully.")
    except Exception as e:
        print(f"Gemini initialization error: {e}")

if not llm_client and OPENAI_KEY:
    try:
        from openai import OpenAI
        llm_client = OpenAI(api_key=OPENAI_KEY)
        llm_type = "openai"
        print("Initialized OpenAI client successfully.")
    except Exception as e:
        print(f"OpenAI initialization error: {e}")

if not llm_client:
    print("WARNING: No LLM API key detected. Bot will use fallback mock rules.")

# Initialize the FastAPI App and Uptime tracker
app = FastAPI()
START_TIME = time.time()

# In-memory database
# We structure this to store contexts by their scope (category, merchant, customer, trigger)
contexts: Dict[str, Dict[str, Any]] = {
    "category": {},
    "merchant": {},
    "customer": {},
    "trigger": {}
}

# Conversation history state: conversation_id -> list of turns
# Each turn will look like: {"role": "vera"|"user", "body": "...", "ts": "..."}
conversations: Dict[str, List[Dict[str, Any]]] = {}

# Keep track of consecutive auto-replies per conversation: conversation_id -> count
auto_reply_tracker: Dict[str, int] = {}


# Pydantic models for incoming request bodies
class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/v1/healthz")
async def healthz():
    # Counts how many contexts are currently stored in memory for each scope
    counts = {scope: len(contexts[scope]) for scope in contexts}
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Antigravity Team",
        "team_members": ["Avinash"],
        "model": "gemini-3.1-flash-lite" if llm_type == "gemini" else "gpt-4o-mini" if llm_type == "openai" else "mock-fallback",
        "approach": "Dynamic context-routing, multi-turn state machine with Hinglish translation support and strict URL-suppression",
        "contact_email": "avinash.1kshah@gmail.com",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    scope = body.scope
    
    # 1. Validate the scope is one of our four valid categories
    if scope not in contexts:
        raise HTTPException(status_code=400, detail=f"Invalid scope: {scope}")
    
    cid = body.context_id
    stored = contexts[scope].get(cid)
    
    # 2. Idempotency Check: Reject strictly older versions
    if stored and stored["version"] > body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": stored["version"]
        }
    
    # 3. Store the new version in memory
    contexts[scope][cid] = {
        "version": body.version,
        "payload": body.payload
    }
    
    return {
        "accepted": True,
        "ack_id": f"ack_{cid}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    
    # Iterate through all triggers the judge flags as active
    for trg_id in body.available_triggers:
        trg_ctx = contexts["trigger"].get(trg_id)
        if not trg_ctx:
            continue
        trg = trg_ctx["payload"]
        
        # 1. Fetch Merchant
        merchant_id = trg.get("merchant_id")
        merchant_ctx = contexts["merchant"].get(merchant_id)
        if not merchant_ctx:
            continue
        merchant = merchant_ctx["payload"]
        
        # 2. Fetch Category
        category_slug = merchant.get("category_slug")
        category_ctx = contexts["category"].get(category_slug)
        if not category_ctx:
            continue
        category = category_ctx["payload"]
        
        # 3. Fetch Customer (if any)
        customer_id = trg.get("customer_id")
        customer = None
        if customer_id:
            customer_ctx = contexts["customer"].get(customer_id)
            if customer_ctx:
                customer = customer_ctx["payload"]
        
        # 4. Run the LLM Composer
        composed = run_composition_llm(category, merchant, trg, customer)
        
        # Generate a unique conversation ID for tracking
        conv_id = f"conv_{merchant_id}_{trg_id}"
        
        # 5. Append the formatted action to send back to the judge
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": "vera_custom_v1",
            "template_params": [composed.get("body", "")],
            "body": composed.get("body", ""),
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": composed.get("suppression_key", trg.get("suppression_key", "")),
            "rationale": composed.get("rationale", "")
        })
        
        # Save outgoing message in conversation memory for follow-ups
        conversations[conv_id] = [{
            "role": "vera",
            "body": composed.get("body", ""),
            "ts": datetime.utcnow().isoformat() + "Z"
        }]
        
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    incoming_msg = body.message
    print(f"DEBUG: incoming_msg={incoming_msg}, conv_id={conv_id}")
    
    # 1. Initialize history if this conversation is new
    if conv_id not in conversations:
        conversations[conv_id] = []
        
    # 2. Check for auto-replies (canned phrases or repeating messages)
    is_auto = is_auto_reply(incoming_msg)
    
    last_turns = [t for t in conversations[conv_id] if t["role"] == "user"]
    if last_turns and last_turns[-1]["body"] == incoming_msg:
        is_auto = True
        
    print(f"DEBUG: is_auto={is_auto}, last_turns={last_turns}")
    if is_auto:
        # Increment counter for auto-replies in this conversation
        count = auto_reply_tracker.get(conv_id, 0) + 1
        auto_reply_tracker[conv_id] = count
        print(f"DEBUG: auto_reply_tracker count={count}")
        
        if count == 1:
            # First auto-reply: Send a friendly warning to the owner
            body_text = "Looks like an auto-reply 😊 When the owner sees this, just reply with 'Yes' to check it out."
            conversations[conv_id].append({"role": "user", "body": incoming_msg, "ts": body.received_at})
            conversations[conv_id].append({"role": "vera", "body": body_text, "ts": datetime.utcnow().isoformat() + "Z"})
            return {
                "action": "send",
                "body": body_text,
                "cta": "binary_yes_no",
                "rationale": "Detected first auto-reply. Sending nudge to the actual owner and waiting."
            }
        elif count == 2:
            # Second auto-reply: Back off 4 hours
            conversations[conv_id].append({"role": "user", "body": incoming_msg, "ts": body.received_at})
            return {
                "action": "wait",
                "wait_seconds": 14400,
                "rationale": "Repeated auto-reply. Backing off 4 hours to avoid spamming."
            }
        else:
            # Third auto-reply: End conversation
            return {
                "action": "end",
                "rationale": "Persistent auto-reply (3x). Closing thread gracefully."
            }

    # 3. Hostility / Opt-out check
    lower_msg = incoming_msg.lower().strip()
    if any(stop_word in lower_msg for stop_word in ["stop", "useless", "spam", "abuse", "don't message", "do not message"]):
        return {
            "action": "end",
            "rationale": "Merchant opted out of messages. Closing conversation."
        }

    # Save user message to history
    conversations[conv_id].append({
        "role": "user",
        "body": incoming_msg,
        "ts": body.received_at
    })
    
    # 4. Load contexts for the LLM
    merchant_ctx = contexts["merchant"].get(body.merchant_id or "")
    merchant = merchant_ctx["payload"] if merchant_ctx else {}
    
    category = {}
    if merchant:
        category_ctx = contexts["category"].get(merchant.get("category_slug", ""))
        category = category_ctx["payload"] if category_ctx else {}
        
    customer = None
    if body.customer_id:
        customer_ctx = contexts["customer"].get(body.customer_id)
        customer = customer_ctx["payload"] if customer_ctx else None

    # 5. Generate reply using LLM
    result = run_reply_llm(conversations[conv_id][:-1], incoming_msg, category, merchant, customer)
    
    action = result.get("action", "send")
    reply_body = result.get("body", "")
    
    if action == "send" and reply_body:
        # Save outgoing message in conversation memory
        conversations[conv_id].append({
            "role": "vera",
            "body": reply_body,
            "ts": datetime.utcnow().isoformat() + "Z"
        })
        return {
            "action": "send",
            "body": reply_body,
            "cta": "open_ended",
            "rationale": result.get("rationale", "")
        }
    elif action == "wait":
        return {
            "action": "wait",
            "wait_seconds": result.get("wait_seconds", 1800),
            "rationale": result.get("rationale", "")
        }
    
    return {
        "action": "end",
        "rationale": result.get("rationale", "Closing conversation.")
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Helper to query the configured LLM API (Gemini or OpenAI) with temperature 0."""
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        if llm_type == "gemini":
            try:
                # We set temperature to 0.0 for strict determinism
                config = genai.types.GenerationConfig(temperature=0.0)
                response = llm_client.generate_content(
                    f"System Instructions:\n{system_prompt}\n\nUser Input:\n{user_prompt}",
                    generation_config=config
                )
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Quota exceeded" in err_str or "ResourceExhausted" in err_str:
                    print(f"Gemini Rate Limit (429) hit. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                print(f"Gemini API call failed: {e}")
                return ""
                
        elif llm_type == "openai":
            try:
                # Call OpenAI chat completion endpoint
                response = llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str or "RateLimitError" in err_str:
                    print(f"OpenAI Rate Limit (429) hit. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                print(f"OpenAI API call failed: {e}")
                return ""
                
    print("Max retries exceeded for LLM call.")
    return ""


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Helper to extract and parse the JSON block from LLM responses."""
    try:
        # LLMs sometimes wrap JSON in markdown block: ```json ... ```
        # We use regex to find the matching brackets {...}
        match = re.search(r'\{[\s\S]*\}', raw_text)
        if match:
            return json.loads(match.group())
        return json.loads(raw_text)
    except Exception as e:
        print(f"Failed to parse JSON: {raw_text}. Error: {e}")
        return {}


def run_composition_llm(category: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict]) -> Dict[str, Any]:
    """Uses LLM to write the initial outbound message matching all constraints."""
    system_prompt = """You are Vera, magicpin's elite merchant-AI assistant on WhatsApp.
Your task is to draft an engagement message based on the business category, merchant stats, and the trigger event.

CRITICAL CONSTRAINTS:
1. STRICT: NEVER include any URL, link, or website in the message body. Meta blocks them in initial WhatsApp messages.
2. SPECIFICITY: Anchor the message on exact metrics (views, reviews, ratings, CTR, offer prices, specific trial days, or compliance drops) from the contexts. Do NOT speak in generic terms. Use numbers to prove points.
3. CATEGORY VOICE:
   - dentists: Speak as a clinical peer. Use technical terms. Address them as "Dr. <Name>". No retail hype/discounts.
   - salons: Warm, professional, beauty-industry focused, practical suggestions.
   - restaurants: Operator-to-operator tone. Fast-paced, focused on volume/orders.
   - gyms: Motivational, fitness coach peer tone.
   - pharmacies: Highly trustworthy, precise, and professional.
4. LANGUAGE:
   - If the merchant/customer language preference has "hi" or "hi-en mix" (Hinglish), mix Hindi words naturally in a professional way (e.g. "Aapka Google profile update kar diya...", "Apke clinic ke liye slots ready hain...").
   - Else, write in professional, friendly English.
5. SINGLE CALL TO ACTION (CTA): The last sentence must be a clear, low-friction, single CTA (e.g. "Reply YES to check it out", "Want me to draft a message for you?"). No multi-choice options unless booking slots.
6. NO FABRICATIONS: Use only facts given in the context. Never invent competitor names, research numbers, or dates.

Your response MUST be a JSON object with these fields:
- body: The WhatsApp message body (string).
- cta: The type of CTA ("binary_yes_no", "open_ended", "none").
- send_as: "vera" (if messaging the merchant) or "merchant_on_behalf" (if customer_id is provided).
- suppression_key: The trigger's suppression_key.
- rationale: A 1-sentence reason why you drafted this message."""

    user_prompt = f"""
=== CONTEXT ===
Category Slug: {category.get('slug', '')}
Voice Config: {json.dumps(category.get('voice', {}))}
Peer Stats: {json.dumps(category.get('peer_stats', {}))}
Digest: {json.dumps(category.get('digest', []))}

Merchant ID: {merchant.get('merchant_id', '')}
Merchant Identity: {json.dumps(merchant.get('identity', {}))}
Merchant Performance: {json.dumps(merchant.get('performance', {}))}
Merchant Active Offers: {[o.get('title') for o in merchant.get('offers', []) if o.get('status') == 'active']}
Merchant Signals: {json.dumps(merchant.get('signals', []))}

Trigger Event: {json.dumps(trigger)}
Customer (Only if send_as should be 'merchant_on_behalf'): {json.dumps(customer) if customer else 'None'}
"""
    
    raw_response = call_llm(system_prompt, user_prompt)
    res = clean_json_response(raw_response)
    
    if not res:
        # Emergency Mock Fallback
        res = {
            "body": f"Hi {merchant.get('identity', {}).get('owner_first_name', 'there')}, noticed your business details are out of date on Google. Want me to help update them?",
            "cta": "binary_yes_no",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", "stale_nudge"),
            "rationale": "Emergency fallback due to JSON parsing failure."
        }
    
    # Safety Check: Hard strip any URLs if generated by mistake
    if "http" in res.get("body", "").lower():
        res["body"] = re.sub(r'https?://\S+', '', res["body"]).strip()
        
    return res


def is_auto_reply(message: str) -> bool:
    """Detect common canned auto-replies or repeating messages."""
    msg = message.lower()
    auto_indicators = [
        "thank you for contacting",
        "canned response",
        "respond shortly",
        "automated assistant",
        "away from my phone",
        "jaankari ke liye bahut-bahut shukriya",
        "hamari team tak pahuncha",
        "auto-reply",
        "is an automated message"
    ]
    for indicator in auto_indicators:
        if indicator in msg:
            return True
    return False


def run_reply_llm(conv_history: List[Dict], user_msg: str, category: Dict, merchant: Dict, customer: Optional[Dict]) -> Dict[str, Any]:
    """Uses LLM to determine next action ('send', 'wait', 'end') and draft reply."""
    system_prompt = """You are Vera, magicpin's merchant assistant.
We have received a reply from the merchant (or customer). Your task is to select the next action:
- "send": If they asked a question or expressed interest, compose the next reply.
- "wait": If they asked to follow up later, wait.
- "end": If they opted out (e.g. "stop", "not interested"), or if conversation is complete.

CONSTRAINTS:
1. STRICT: NEVER generate any URL or website link in the message body.
2. Graceful exit: If the user says stop, return "end".
3. Transition to Action: If they committed (e.g., "lets do it", "update it", "send abstract"), switch to action mode. Send what they asked for directly. No further qualifying questions.
4. Hindi preference: Match the language style of the last user turn.

Output a JSON object:
{
  "action": "send" | "wait" | "end",
  "body": "<composed message if action is send>",
  "wait_seconds": 1800, // if action is wait, else null
  "rationale": "<1-sentence rationale>"
}"""

    user_prompt = f"""
=== HISTORY ===
{json.dumps(conv_history)}

Latest Incoming Message: "{user_msg}"

=== CONTEXT ===
Category: {json.dumps(category)}
Merchant: {json.dumps(merchant)}
Customer: {json.dumps(customer) if customer else 'None'}
"""
    
    raw_response = call_llm(system_prompt, user_prompt)
    res = clean_json_response(raw_response)
    
    if not res:
        res = {
            "action": "end",
            "body": "",
            "wait_seconds": None,
            "rationale": "Fallback response due to JSON parse failure."
        }
        
    if "http" in res.get("body", "").lower():
        res["body"] = re.sub(r'https?://\S+', '', res["body"]).strip()
        
    return res