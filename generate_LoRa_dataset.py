import json
import os
import re
import openai
import google.generativeai as genai
import yaml
from utils import extract_urls, convert_size_to_bytes
# from automate_feature import automate_feature_engineering
# from scrapper import extract_data_sync
# import kaggle
import pandas as pd
import glob

import time
import requests


# Define API Endpoints for Dataset Generation (Groq, Browserbase, Perplexity)
with open('config/credentials.yml', 'r') as file:
    config = yaml.safe_load(file)

os.environ["GROQ_API_KEY"] = config['groq']['GROQ_API_KEY']
os.environ["GEMINI_API_KEY"] = config['google']['GEMINI_API_KEY']

# Set up the API keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

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
def summary_topic(query="Weather Prediction at Malaysia"):
    summary_prompt = (
        f"Summarize the following prompt into a single sentence that matches this style:\n"
        f"\"We will create dataset by collecting a variety of user queries related to {{topic}}.\"\n\n"
        f"Prompt:\n\"{query}\""
    )

    # Call Groq API
    summary_response = groq_client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=100
    )
    
    summary_text = summary_response.choices[0].message.content.strip()

    return summary_text


def suggest_sources(query="Weather Prediction at Malaysia"):
    # model = genai.GenerativeModel("gemini-1.5-flash")  # Choose the Gemini model

    # prompt = (
    #     f"Suggest one of the best dataset topic for {query}. that are freely available on Kaggle and suitable for AI model training. "
    #     "Ensure the topics align with real-world applications in data science, machine learning, and AI development. "
    #     "Each dataset topic should contain structured data (CSV, JSON, or tabular format) and be commonly used in AI tasks "
    #     "like classification, regression, NLP, computer vision, or time-series forecasting. "
    #     "Only return dataset topics, nothing else—no descriptions or explanations."
    # )


    # response = model.generate_content(prompt)  # Generate content


    prompt = (
        f"You are a data science assistant for an AI crawler that automatically visits and extracts datasets from websites. " 
        f"Respond ONLY with a raw JSON array of dictionaries containing dataset sources. "
        f"Each object must have 'Source Name' and 'URL'. Ensure every URL is valid (no spaces, no broken links). "
        f"Example:\n"
        f"[{{\"Source Name\": \"Kaggle\", \"URL\": \"https://kaggle.com\"}}]\n\n"
        f"For query: {query}\n\n"
        f"No markdown, no explanations — just valid JSON output."
    )


    # Call Groq API
    response = groq_client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    if response and hasattr(response, "choices") and response.choices:
        raw_output = response.choices[0].message.content.strip()

        # Remove markdown formatting (if any) or extra characters
        raw_output = re.sub(r"```(?:json)?", "", raw_output).strip()

        # Ensure valid JSON format - strip trailing commas or broken entries
        raw_output = raw_output.rstrip(",")  # Remove last comma if present

        try:
            summary_text = summary_topic(query)
            parsed = json.loads(raw_output)
           
            return {
                "topic": query,
                "summary": summary_text,
                "sources": parsed
            }
        except json.JSONDecodeError:
            raise Exception("GROQ API Error: Invalid JSON format returned.")
    else:
        raise Exception("GROQ API Error: No valid response received.")

# # Search Kaggle for datasets
# def search_kaggle_datasets(query, min_size=1_000_000, max_size=1_000_000_000, top_n=5):
#     """
#     Search Kaggle for datasets related to a query, filtering by size and votes.
    
#     - min_size: Minimum dataset size in bytes (default: 1MB)
#     - max_size: Maximum dataset size in bytes (default: 1GB)
#     - top_n: Number of top datasets to return
#     """
#     print(f"🔍 Searching Kaggle for datasets related to: {query}...")

#     # Ensure query is well-formatted
#     search_query = query.lower().strip()

#     # Search Kaggle
#     datasets = kaggle.api.dataset_list(search=search_query, sort_by="votes", file_type="csv")

#     dataset_info = []
#     seen_ids = set()  # Avoid duplicates

#     for dataset in datasets:
#         if hasattr(dataset, "language") and dataset.language != "en":
#             continue  # Skip non-English datasets
#         if dataset.ref in seen_ids:
#             continue  # Remove duplicates
        
#         dataset_size = convert_size_to_bytes(dataset.size)  

#         if dataset_size < min_size or dataset_size > max_size:  
#             continue  # Skip datasets that are too small or too large

#         seen_ids.add(dataset.ref)

#         dataset_info.append({
#             "title": dataset.title,
#             "id": dataset.ref,
#             "url": f"https://www.kaggle.com/datasets/{dataset.ref}",
#             "size": dataset_size
#         })

#     # Sort by size (descending) and votes (secondary)
#     dataset_info.sort(key=lambda x: (-x["size"], x["title"]))

#     return dataset_info[:top_n]  # Return only top N datasets

# # Download Kaggle dataset
# def download_kaggle_dataset(dataset_id, download_path="datasets"):
#     """Download and unzip a Kaggle dataset."""
#     os.makedirs(download_path, exist_ok=True)

#     print(f"📥 Downloading dataset: {dataset_id}...")
#     kaggle.api.dataset_download_files(dataset_id, path=download_path, unzip=True)

#     print(f"✅ Dataset downloaded to: {download_path}/{dataset_id}")

# # Choose the best dataset using Groq API
# def dataset_selection_with_groq(datasets, user_prompt):
#     """Choose the best dataset using Groq API"""
#     print("\n📥 Sending dataset list to Groq for selection...")
#     prompt = (
#         "You are an AI assistant. Here are the top Kaggle datasets for a given AI topic:\n\n"
#     )

#     for i, dataset in enumerate(datasets):
#         prompt += f"{i+1}. {dataset['title']} (Size: {dataset['size'] / 1_000_000:.2f} MB)\n"
#         prompt += f"   URL: {dataset['url']}\n\n"

#     prompt += f"Choose the best dataset for {user_prompt} training an AI model based on quality, completeness, and size. Only return the dataset number (1, 2, or 3) with no explanation."
    
#     client = openai.OpenAI(
#         base_url="https://api.groq.com/openai/v1",
#         api_key=GROQ_API_KEY
#     )

#     # Call Groq API
#     response = client.chat.completions.create(
#         model="llama-3.2-3b-preview",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=100
#     )

#     choice = response.choices[0].message.content.strip()

#     try:
#         choice_index = int(choice) - 1  # Convert to index
#         if 0 <= choice_index < len(datasets):
#             best_dataset = datasets[choice_index]
#             print(f"✅ Groq selected: {best_dataset['title']}\n")
#             return best_dataset
#     except ValueError:
#         print("⚠️ Groq API returned an invalid choice.")

#     print("❌ Groq API failed to select a dataset. Defaulting to the first dataset.")
#     return datasets[0]  # Default to first dataset if Groq fails

# Main function to generate dataset
# def generate_lora_dataset(prompt):

    # Step 1: Suggest sources from GROQ API
    # sources_urls = suggest_sources(prompt)


    # dataset_topics = ['Weather Station Data']

    # # Step 2: Extract dataset from Kaggle
    # all_datasets = []

    # for topic in dataset_topics:
    #     kaggle_datasets = search_kaggle_datasets(topic)
    #     all_datasets.extend(kaggle_datasets)  # Merge results into one list

    # # Sort datasets by size (largest first) and return top 5
    # top_datasets = sorted(all_datasets, key=lambda x: x["size"], reverse=True)[:5]

    # # Remove redundant datasets
    # datasets = [dict(t) for t in {tuple(d.items()) for d in top_datasets}]

    # # Print results
    # if datasets:
    #     print("\n✅ Top 5 Available Datasets:")
    #     for dataset in datasets:
    #         print(f"- {dataset['title']} (Size: {dataset['size']} bytes) → {dataset['url']}")
    # else:
    #     print("\n❌ No suitable datasets found. Try refining your queries.")

    
    # # Step 2: Extract additional data using Playwright
    # for line in lora_info.split("\n"):
    #     urls = extract_urls(line)  # Extract clean URLs
    #     print(urls)
    #     if urls:  # Ensure URLs exist
    #         for url in urls:
    #             # extracted_data.append(extract_data_sync(prompt))
    #             try:
    #                 # print(f"Extracting data from: {url}")
    #                 extract_data_sync(prompt)
    #                 # extracted_data.append()
    #             except Exception as e:
    #                 print(f"Error extracting {url}: {e}")
    #                 extracted_data.append(f"Failed to extract content from: {url}")

    # Step 3: Choose the best dataset using Groq API
    # if datasets:
    #     # # let Groq to process the dataset list, choose the best one
    #     # chosen_dataset = dataset_selection_with_groq(datasets, prompt)

    #     # # Download the chosen dataset
    #     # download_kaggle_dataset(chosen_dataset["id"])

    #     # From directory, choose the most related dataset
    #     final_dataset, selected_headers, removed_headers = choose_best_dataset_from_directory("datasets", prompt)

    #     # Perform feature engineering
    #     if final_dataset:
    #         processed_dataset_path = automate_feature_engineering(final_dataset, selected_headers, removed_headers)
    #         print(f"✅ Processed dataset saved to: {processed_dataset_path}")
    #     else:
    #         print("❌ No dataset processed.")

    # else:
    #     print("❌ Merged dataset not found. Run dataset merging first.")

    # return {"dataset": chosen_dataset['title'] + "with" + final_dataset, "sources": datasets}