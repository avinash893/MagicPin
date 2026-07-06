@app.get("/v1/healthz")
async def healthz():
    # Counts how many contexts are currently stored in memory for each scope
    counts = {scope: len(contexts[scope]) for scope in contexts}
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }


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
        "team_name": "Avinash Team",
        "team_members": ["Avinash"],
        "model": "gemini-1.5-flash" if llm_type == "gemini" else "gpt-4o-mini" if llm_type == "openai" else "mock-fallback",
        "approach": "Dynamic context-routing, multi-turn state machine with Hinglish translation support and strict URL-suppression",
        "contact_email": "avinash.1kshah@gmail.com",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }


    # Pydantic model to validate the incoming context payload structure
class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    scope = body.scope
    
    # 1. Validate the scope is one of our four valid categories
    if scope not in contexts:
        raise HTTPException(status_code=400, detail=f"Invalid scope: {scope}")
    
    cid = body.context_id
    stored = contexts[scope].get(cid)
    
    # 2. Reject older or identical versions
    if stored and stored["version"] >= body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": stored["version"]
        }
    
    # 3. Store the new version 
    contexts[scope][cid] = {
        "version": body.version,
        "payload": body.payload
    }
    
    return {
        "accepted": True,
        "ack_id": f"ack_{cid}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Helper to query the configured LLM API (Gemini or OpenAI) with temperature 0."""
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
            print(f"OpenAI API call failed: {e}")
            return ""
            
    return ""


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Helper to extract and parse the JSON block from LLM responses."""
    try:
        # LLMs sometimes wrap JSON in markdown block
        # We use regex to find the matching brackets 
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
   - If the merchant/customer language preference has "hi" or "hi-en mix" (Hinglish), mix Hindi words naturally in a professional way (e.g. "Aapka Google profile update kar diya...", "Apke clinic ke liye 2 slots ready hain...").
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