import ace_lib
import pandas as pd
import json
import openai
import os
import sys
import time
import random

# Default Moonshot Configuration
DEFAULT_MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MOONSHOT_MODEL = "kimi-k2.5"

def get_llm_client(api_key, base_url):
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

def test_llm_connection(api_key, base_url, model):
    print("\nTesting LLM connection...")
    client = get_llm_client(api_key, base_url)
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("LLM connection successful.")
        return True
    except Exception as e:
        print(f"LLM connection failed: {e}")
        return False

def call_llm_with_retry(client, model, system_prompt, operators_df, datafields_df, dataset_id, max_retries=5):
    
    n_ops = len(operators_df)
    n_fields = len(datafields_df)
    
    for attempt in range(max_retries + 1):
        print(f"\nAttempt {attempt + 1}/{max_retries + 1} - Preparing prompt with {n_ops} operators and {n_fields} datafields...")
        
        # Sample rows if needed, otherwise take head
        # Using head for stability, but could be random sample
        ops_subset = operators_df.head(n_ops)
        fields_subset = datafields_df.head(n_fields)
        
        operators_info = ops_subset[['name', 'category', 'description', 'extra_side_note']].to_string()
        datafields_info = fields_subset[['id', 'description', 'subcategory']].to_string()
        
        user_prompt = f"""
Here is the information about available operators (first {n_ops} rows):
{operators_info}

Here is the information about the dataset '{dataset_id}' (first {n_fields} rows):
{datafields_info}

Please come up with several Alpha templates based on this information.
Specify the AI answer in Chinese.
"""
        
        try:
            print("Sending request to LLM...")
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1,
            )
            return completion.choices[0].message.content
            
        except openai.BadRequestError as e:
            error_msg = str(e)
            print(f"LLM Bad Request Error: {error_msg}")
            
            # Check for token limit error
            if "token limit" in error_msg or "context_length_exceeded" in error_msg or "400" in error_msg:
                print("Token limit exceeded. Reducing context size...")
                n_ops = max(1, n_ops // 2)
                n_fields = max(1, n_fields // 2)
                if n_ops == 1 and n_fields == 1:
                    print("Cannot reduce context further.")
                    return f"Failed after retries: {e}"
            else:
                return f"LLM Error (not token related): {e}"
                
        except Exception as e:
            return f"General Error calling LLM: {e}"
            
    return "Max retries exceeded."

def main():
    print("=== BRAIN Alpha Generator Full Version ===\n")

    # 1. Interactive Login
    print("--- Step 1: Login to BRAIN ---")
    email = input("Enter BRAIN Email: ").strip()
    while not email:
        email = input("Email is required. Enter BRAIN Email: ").strip()
    
    import getpass
    password = getpass.getpass("Enter BRAIN Password: ").strip()
    while not password:
        password = getpass.getpass("Password is required. Enter BRAIN Password: ").strip()

    # Monkeypatch ace_lib.get_credentials to use provided inputs
    ace_lib.get_credentials = lambda: (email, password)

    print("Logging in...")
    try:
        s = ace_lib.start_session()
        print("Login successful.")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # 2. LLM Configuration
    print("\n--- Step 2: LLM Configuration ---")
    base_url = input(f"Enter LLM Base URL (default: {DEFAULT_MOONSHOT_BASE_URL}): ").strip()
    if not base_url:
        base_url = DEFAULT_MOONSHOT_BASE_URL
        
    api_key = input("Enter LLM API Key (required): ").strip()
    while not api_key:
        print("API Key is required.")
        api_key = input("Enter LLM API Key: ").strip()
        
    model_name = input(f"Enter LLM Model Name (default: {DEFAULT_MOONSHOT_MODEL}): ").strip()
    if not model_name:
        model_name = DEFAULT_MOONSHOT_MODEL

    if not test_llm_connection(api_key, base_url, model_name):
        print("Aborting due to LLM connection failure.")
        return

    llm_client = get_llm_client(api_key, base_url)

    # 3. Load Operators
    print("\n--- Step 3: Load Operators ---")
    print("Getting operators...")
    try:
        operators_df = ace_lib.get_operators(s)
        operators_df = operators_df[operators_df['scope'] == 'REGULAR']
        print(f"Retrieved {len(operators_df)} operators (REGULAR only).")
        
        print("Fetching documentation for operators...")
        operators_df = operators_df.copy()
        
        def fetch_doc_content(doc_path):
            if pd.isna(doc_path) or not doc_path:
                return None
            url = ace_lib.brain_api_url + doc_path
            try:
                r = s.get(url)
                if r.status_code == 200:
                    return json.dumps(r.json())
                return None
            except Exception:
                return None

        operators_df['extra_side_note'] = operators_df['documentation'].apply(fetch_doc_content)
        operators_df.drop(columns=['documentation', 'level'], inplace=True)
        print("Operators loaded and processed.")
        
    except Exception as e:
        print(f"Failed to get operators: {e}")
        return

    # 4. Dataset Selection
    print("\n--- Step 4: Select Dataset ---")
    region = input("Enter Region (default: USA): ").strip() or "USA"
    delay = input("Enter Delay (default: 1): ").strip() or "1"
    universe = input("Enter Universe (default: TOP3000): ").strip() or "TOP3000"
    
    try:
        delay = int(delay)
    except ValueError:
        print("Invalid delay, using default 1")
        delay = 1

    print(f"Fetching datasets for Region={region}, Delay={delay}, Universe={universe}...")
    try:
        datasets_df = ace_lib.get_datasets(
            s, 
            region=region, 
            delay=delay, 
            universe=universe
        )
        print(f"Retrieved {len(datasets_df)} datasets.")
        # print(datasets_df[['id', 'name', 'category', 'subcategory']].head(10))
        
        # Print all datasets for user selection
        pd.set_option('display.max_rows', None)
        print(datasets_df[['id', 'name', 'category', 'subcategory']])
    except Exception as e:
        print(f"Failed to get datasets: {e}")
        return

    # 5. Dataset Detail
    print("\n--- Step 5: Get Dataset Details ---")
    dataset_id = input("Enter Dataset ID to analyze (e.g., analyst10): ").strip()
    while not dataset_id:
        dataset_id = input("Dataset ID is required: ").strip()

    print(f"Getting datafields for dataset: {dataset_id}...")
    try:
        datafields_df = ace_lib.get_datafields(
            s,
            region=region,
            delay=delay,
            universe=universe,
            data_type="ALL",
            dataset_id=dataset_id
        )
        print(f"Retrieved {len(datafields_df)} datafields.")
    except Exception as e:
        print(f"Failed to get datafields: {e}")
        return

    # 6. Generate Alpha Templates
    print("\n--- Step 6: Generate Alpha Templates ---")
    
    # Load System Prompt
    # Use relative path based on the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    system_prompt_path = os.path.join(script_dir, "what_is_Alpha_template.md")
    
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        print(f"System prompt loaded from {system_prompt_path}")
    except Exception as e:
        print(f"System prompt file not found at {system_prompt_path}, using default. Error: {e}")
        system_prompt = "You are a helpful assistant for generating Alpha templates."

    response = call_llm_with_retry(
        llm_client, 
        model_name, 
        system_prompt, 
        operators_df, 
        datafields_df, 
        dataset_id
    )
    
    print("\n=== LLM Response ===")
    print(response)
    print("====================")

if __name__ == "__main__":
    main()
