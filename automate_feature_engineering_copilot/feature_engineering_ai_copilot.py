# AEROML | Automated FEATURE ENGINEERING SELECTION

from langchain_openai import ChatOpenAI

import os
import sys
from pathlib import Path
import yaml
import pandas as pd
from pprint import pprint

from aeroml_data_science_team.agents import make_feature_engineering_agent

# 1.0 SETUP 

# PATHS
PATH_ROOT = "004_automate_feature_engineering_copilot/"

# LLM
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

MODEL    = "gpt-4o-mini"

# LOGGING
LOG      = True
LOG_PATH = os.path.join(os.getcwd(), PATH_ROOT, "ai_functions/")

# Data set
df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

df.info()


# 2.0 CREATE THE AI COPILOT

# Create the AI Copilot

llm = ChatOpenAI(model = MODEL)

feature_engineering_agent = make_feature_engineering_agent(
    model = llm, 
    log=LOG, 
    log_path=LOG_PATH
)

feature_engineering_agent

# Run feature engineer agent on the data

response = feature_engineering_agent.invoke({
    "target_variable": "Churn",
    "data_raw": df.to_dict(),
    "max_retries":3, 
    "retry_count":0
})


# 3.0 RESULTS

# Evaluate the response
list(response.keys())

# Print the feature engineered data
df.info()

pd.DataFrame(response['data_engineered']).info()

# What feature engineering steps were taken?

pprint(response['messages'][0].content)

# What does the data cleaner function look like?

pprint(response['feature_engineer_function'])

# How can I reuse the data cleaning steps as I get new data?

current_dir = Path.cwd() / PATH_ROOT
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from ai_functions.feature_engineer import feature_engineer

df.info()

feature_engineer(df).info()