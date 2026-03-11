# Python Program: Demonstration of Prompt Engineering Techniques
# Techniques Covered: Zero-shot, One-shot, Few-shot Prompting
# Application: Text classification using a language model

import os

from google import genai
from openai import OpenAI


def load_env_file(env_path=".env"):
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def get_client():
    load_env_file()
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None, None
        client = genai.Client(api_key=api_key)
        return "gemini", client

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, None
        return "openai", OpenAI(api_key=api_key)

    return None, None


def get_response(provider, client, prompt):
    if provider == "gemini":
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        fallback_models_raw = os.getenv("GEMINI_MODEL_FALLBACKS", "gemini-2.0-flash-lite,gemini-2.5-flash")
        fallback_models = [model.strip() for model in fallback_models_raw.split(",") if model.strip()]
        candidate_models = [model_name] + [model for model in fallback_models if model != model_name]

        errors = []
        for candidate_model in candidate_models:
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=prompt,
                )
                return f"[{candidate_model}] {response.text}"
            except Exception as error:
                errors.append(f"{candidate_model}: {error}")

        raise RuntimeError("All Gemini model attempts failed -> " + " | ".join(errors))

    if provider == "openai":
        response = client.responses.create(
            model="gpt-4.1",
            input=prompt
        )
        return response.output_text

    raise ValueError("Unsupported provider")


# Function for Zero-Shot Prompting
def zero_shot_prompt():
    print("\n=== ZERO-SHOT PROMPTING ===")

    prompt = """Classify the sentiment of the following sentence as Positive or Negative:
    'The product quality is amazing and I love it.'"""

    response_text = get_response(provider, client, prompt)
    print("Response:", response_text)
    return response_text


# Function for One-Shot Prompting
def one_shot_prompt():
    print("\n=== ONE-SHOT PROMPTING ===")

    prompt = """
Example:
Sentence: I love this phone
Sentiment: Positive

Now classify:
Sentence: This laptop is very slow
Sentiment:
"""

    response_text = get_response(provider, client, prompt)
    print("Response:", response_text)
    return response_text


# Function for Few-Shot Prompting
def few_shot_prompt():
    print("\n=== FEW-SHOT PROMPTING ===")

    prompt = """
Sentence: I love this product
Sentiment: Positive

Sentence: This service is terrible
Sentiment: Negative

Sentence: The experience was wonderful
Sentiment: Positive

Now classify:
Sentence: The food was awful
Sentiment:
"""

    response_text = get_response(provider, client, prompt)
    print("Response:", response_text)
    return response_text


def run_prompt_step(step_function, step_name):
    try:
        response = step_function()
        return {"step": step_name, "status": "success", "response": response}
    except Exception as error:
        print(f"{step_name} failed: {error}")
        return {"step": step_name, "status": "failed", "response": str(error)}


def print_summary(results):
    print("\n=== FINAL SUMMARY ===")
    print(f"{'Technique':<20} {'Status':<10} Response")
    print("-" * 80)
    for item in results:
        preview = item["response"].replace("\n", " ").strip()
        if len(preview) > 60:
            preview = preview[:57] + "..."
        print(f"{item['step']:<20} {item['status']:<10} {preview}")


# Main Program
def main():
    print("PROMPT ENGINEERING APPLICATION USING LLM")

    global provider, client
    provider, client = get_client()
    if provider is None or client is None:
        print("Set LLM_PROVIDER=gemini and GEMINI_API_KEY in your environment or .env file, then run again.")
        return

    print(f"Using provider: {provider}")

    results = [
        run_prompt_step(zero_shot_prompt, "Zero-shot"),
        run_prompt_step(one_shot_prompt, "One-shot"),
        run_prompt_step(few_shot_prompt, "Few-shot"),
    ]
    print_summary(results)


# Execute program
if __name__ == "__main__":
    main()