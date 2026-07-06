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
        llm_client = genai.GenerativeModel('gemini-1.5-flash')
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