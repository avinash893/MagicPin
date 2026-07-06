# magicpin AI Challenge — Vera Assistant Submission

**Team Name**: Antigravity Team  
**Author**: Avinash  
**Language/Framework**: Python 3.12 + FastAPI + Uvicorn  
**LLM Model**: `gemini-3.1-flash-lite` (via Google AI Studio SDK)

---

## 1. Approach Overview
Our solution is a robust, stateful web server that dynamically routes triggers and handles multi-turn dialogues:

* **Stateful Dialogue Processor (`POST /v1/reply`)**: Detects canned WhatsApp Business auto-replies using token similarity checks and consecutive repeating message buffers. It implements a 3-strike backoff mechanism (Nudge owner → Wait 4 Hours → End Session) to avoid spamming the merchant.
* **Intent Handoff Transition**: Detects merchant commitments (e.g., *"let's do it"*, *"confirm"*) and immediately switches from qualifying questions to action execution mode without losing momentum.
* **Deterministic Composition (`POST /v1/tick`)**: Feeds Category, Merchant, Trigger, and Customer contexts into a strict, structured system instruction set. We set the LLM `temperature = 0.0` to ensure repeatable and consistent responses.
* **Auto-Retry Handler**: Since the free-tier API has RPM limits, we wrote a decorator wrapper around the API calls inside `bot.py` to catch 429 rate limit exceptions and automatically wait and retry using exponential backoff.

---

## 2. Design Decisions & Tradeoffs
1. **Model Selection (`gemini-3.1-flash-lite`)**: We switched to the lightweight `gemini-3.1-flash-lite` model because it offers extremely fast inference latencies (< 1.5 seconds) and handles structured JSON parsing perfectly, keeping the response times well below the 30-second judge timeout budget.
2. **Zero-URL Policy**: Meta rejects initial templates containing raw external links. We built regex filters in `bot.py` to enforce this rule and hard-strip any URLs that the LLM might hallucinate.
3. **Hyper-Personalization & Hinglish**: We map owner/locality details and category stats (CTR delta, reviews) into specific values rather than templates. For merchants with Hindi/Hinglish preferences, we prompt the LLM to code-mix naturally (e.g., *"Aapki medicines ka stock 28 April ko khatam ho raha hai..."*).

---

## 3. What would help most in production
* **Real-time verification**: A tool integration allowing the LLM to look up actual GBP pages or competitor locations would prevent hallucinating competitor names if they are outdated in the static database.
* **Structured Template Mapping**: In production, having pre-compiled WhatsApp template schemas would let us dynamically register templates with Meta and map variables correctly.
