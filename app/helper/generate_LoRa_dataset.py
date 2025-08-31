import json
import os
import re

import openai
import google.generativeai as genai
from app.helper.utils import extract_urls, convert_size_to_bytes, logger
import pandas as pd
from dotenv import load_dotenv
from dataset_elicitation_agent.utils import _extract_json, _soft_repair_json

load_dotenv()

# Define API Endpoints for Dataset Generation (Groq, Browserbase, Perplexity)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Check if API Key is set
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running the script.")
if not GEMINI_API_KEY:
    raise ValueError("GEMENI_API_KEY environment variable is not set. Please set it before running the script.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Configure Groq API
groq_client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# GROQ API Request
def prompt_suggestions(query):
    prompt = (
        f"Task: Generate a structured prompt for dataset search for model training.\n"
        f"User Prompt: {query}\n\n"
        f"Model Used: meta-llama/llama-4-maverick-17b-128e-instruct\n"
        f"Rules:\n"
        f"1. ONLY return a valid JSON object as specified below. Do NOT include any explanation or extra text.\n"
        f"2. Analyze key components of the user prompt\n"
        f"3. Identify relevant data types and formats\n"
        f"4. Consider domain-specific requirements\n\n"
        f"5. Include the dataset source and direct URLs for each example.\n"
        f"Output Format:\n"
        f"{{\n"
        f"  \"refined_query\": \"[Refined search query]\",\n"
        f"  \"data_requirements\": [\n"
        f"    \"[Requirement 1]\",\n"
        f"    \"[Requirement 2]\"\n"
        f"  ],\n"
        f"  \"suggested_prompt\": \"[Complete search prompt for model meta-llama/llama-4-maverick-17b-128e-instruct]\"\n"
        f"}}"
    )

    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[
            {"role": "system", "content": "You are a data science expert specializing in dataset discovery. ONLY return valid JSON as specified."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.3
    )

    raw_content = response.choices[0].message.content.strip()
    print("RAW RESPONSE:", raw_content)  # Debug print

    # Try to extract JSON object from the response
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = raw_content

    try:
        result = json.loads(json_str)
        search_query = result["suggested_prompt"]
        logger.info(f"{search_query}")
        return search_query
    except json.JSONDecodeError as e:
        print("Error parsing response, using original query")
        print("JSONDecodeError:", e)
        return query

def summary_topic(query):
    summary_prompt = (
        f"Task: Create a concise, professional summary of the following topic for dataset collection.\n"
        f"Format: 'We will create a dataset focusing on [topic description].'\n"
        f"Requirements:\n"
        f"- Keep it under 20 words\n"
        f"- Be specific and technical\n"
        f"- Focus on data collection purpose\n\n"
        f"Topic: {query}"
    )

    # Call Groq API
    summary_response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=100
    )
    
    summary_text = summary_response.choices[0].message.content.strip()

    return summary_text


def suggest_sources(query):
    logger.info("Planning prompt for generating dataset...")
    prompt = prompt_suggestions(query)  # yours
    logger.info("Prompt for source search: %s", prompt)

    system_msg = {
        "role": "system",
        "content": ("You are a helpful research assistant. "
                    "Return ONLY a valid JSON array. No prose, no code fences. "
                    "Array of objects with keys: source (string), url (string), reason (string).")
    }
    user_msg = {
        "role": "user",
        "content": (
            f"{prompt}\n\n"
            "Respond with JSON ONLY like:\n"
            '[{"source":"Department of Meteorology Malaysia","url":"https://example","reason":"relevance"},'
            '{"source":"Paper XYZ","url":"https://example2","reason":"relevance"}]'
        ),
    }

    logger.info("Searching for relevant sources...")

    def _call_groq(messages, temperature=0.2, max_tokens=2048):
        return groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=temperature,
            top_p=1,
            max_tokens=max_tokens,
            stream=False
        )

    # 1st attempt
    resp = _call_groq([system_msg, user_msg])
    if not resp or not getattr(resp, "choices", None):
        raise Exception("GROQ API Error: No valid response received.")

    raw_output = resp.choices[0].message.content.strip()
    logger.info("Raw output: %s", raw_output)  # ✅ correct logging

    sources_json = _extract_json(raw_output) or _soft_repair_json(raw_output)

    # Retry once with a stricter instruction if parsing failed
    if not isinstance(sources_json, list):
        logger.warning("First parse failed; retrying with stricter instruction.")
        retry_system = {
            "role": "system",
            "content": ("Return ONLY a valid JSON array. No text outside JSON. "
                        "If previous output was partial, re-send the FULL array now.")
        }
        resp2 = _call_groq([retry_system, user_msg], temperature=0.1, max_tokens=3072)
        raw_output2 = resp2.choices[0].message.content.strip() if resp2 and resp2.choices else ""
        logger.info("Retry raw output: %s", raw_output2)
        sources_json = _extract_json(raw_output2) or _soft_repair_json(raw_output2)

    if not isinstance(sources_json, list):
        logger.error("Raw model output (final):\n%s", raw_output)
        raise Exception("GROQ API Error: Invalid JSON format returned.")

    # Normalize
    normalized = []
    for item in sources_json:
        if isinstance(item, dict):
            normalized.append({
                "source": (item.get("source") or "Unknown").strip(),
                "url": (item.get("url") or "").strip(),
                "reason": (item.get("reason") or "").strip(),
            })

    if not normalized:
        logger.error("Parsed JSON but no valid items: %s", sources_json)
        raise Exception("GROQ API Error: Parsed JSON contains no valid items.")

    summary_text = summary_topic(query)

    ranked_sources = "\n".join(f"{i}. {x['source']} - {x['url']}" for i, x in enumerate(normalized, 1))
    logger.info("Here are the most relevant and high-quality sources, ranked by relevance and details:")
    logger.info("%s", ranked_sources)

    return {"summary": summary_text, "sources": normalized}