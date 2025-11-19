# 🚀 AeroML API

A FastAPI application for automated machine learning operations using H2O AutoML.

## 📋 Overview

AeroML API provides a comprehensive REST API for building, training, and deploying machine learning models with minimal configuration. It features real-time model training with H2O AutoML, session management for tracking experiments, and an AI-powered chatbot interface for natural language interaction with trained models.

## ✨ Features

- **🤖 H2O AutoML Integration**: Automated machine learning pipeline with real-time execution logs
- **📤 File Upload**: Upload Excel (.xlsx, .xls) or CSV files directly for model training
- **💬 AI-Powered Chat Interface**: Interact with trained models using natural language (powered by OpenAI)
- **📊 Session Management**: Track and retrieve training sessions with unique session IDs
- **🎯 Model Deployment**: Easy model deployment with AI-generated test prompts
- **📈 Performance Metrics**: Detailed model leaderboards and performance reports
- **🗂️ Dataset Management**: Built-in dataset handling and suggestions
- **🔄 Real-time Streaming**: Live updates during model training
- **💾 Persistent Storage**: Local file-based session data storage
- **🧹 Auto Cleanup**: Automatic cleanup of temporary files after processing

## 🛠️ Tech Stack

- **FastAPI** - Modern, fast web framework
- **H2O AutoML** - Automated machine learning
- **LangChain + OpenAI** - AI-powered chat functionality
- **Pandas** - Data manipulation and analysis
- **Python 3.11+** - Core programming language

## 📦 Prerequisites

- Python 3.11 or higher
- OpenAI API key (for chat functionality)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd aeroml-api
```

2. **Create and activate virtual environment**
```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your-openai-api-key-here
```

## 🏃 Running the Server

**Development mode with auto-reload:**
```bash
uvicorn app.main:app --reload
```

**Or using the direct command:**
```bash
python app/main.py
```

The server will start at `http://127.0.0.1:8000`

## 📚 API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 🔗 API Endpoints

### Model Training

#### Run H2O ML Pipeline
```http
GET /api/model-training/run-h2o-ml-pipeline
```
**Query Parameters:**
- `data_path` (string): Path to training dataset
- `target_variable` (string): Target column name
- `max_runtime_secs` (int): Maximum training time in seconds
- `model_name` (string, optional): Model name for AI agent

**Response:** Server-sent events stream with real-time training progress

#### Run Advanced H2O ML Pipeline (with File Upload)
```http
POST /api/model-training/run-h2o-ml-pipeline-advanced
```
**Content-Type:** `multipart/form-data`

**Form Data Parameters:**
- `file` (required): Dataset file (.xlsx, .xls, or .csv)
- `target_variable` (required): Target column name
- `max_runtime_secs` (optional): Maximum training time in seconds (default: 30)
- `model_name` (optional): Model name - "gpt-4o-mini" or "gpt-oss:20b" (default: "gpt-4o-mini")
- `user_instructions` (optional): Custom instructions for the model
- `exclude_columns` (optional): Comma-separated list of columns to exclude
- `return_predictions` (optional): Return predictions - "true" or "false" (default: "true")
- `return_leaderboard` (optional): Return leaderboard - "true" or "false" (default: "true")
- `return_performance` (optional): Return performance metrics - "true" or "false" (default: "true")

**Example using cURL (Excel):**
```bash
curl -X POST "http://127.0.0.1:8000/api/model-training/run-h2o-ml-pipeline-advanced" \
  -F "file=@/path/to/your/dataset.xlsx" \
  -F "target_variable=Churn" \
  -F "max_runtime_secs=300" \
  -F "model_name=gpt-4o-mini" \
  -F "exclude_columns=customerID,unnecessary_column"
```

**Example using cURL (CSV):**
```bash
curl -X POST "http://127.0.0.1:8000/api/model-training/run-h2o-ml-pipeline-advanced" \
  -F "file=@/path/to/your/dataset.csv" \
  -F "target_variable=Churn" \
  -F "max_runtime_secs=300"
```

**Example using Python:**
```python
import requests

url = "http://127.0.0.1:8000/api/model-training/run-h2o-ml-pipeline-advanced"
# Works with .xlsx, .xls, or .csv files
files = {'file': open('dataset.xlsx', 'rb')}
data = {
    'target_variable': 'Churn',
    'max_runtime_secs': '300',
    'model_name': 'gpt-4o-mini',
    'exclude_columns': 'customerID,unnecessary_column'
}

response = requests.post(url, files=files, data=data, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

**Response:** Server-sent events stream with real-time training progress

### Model Interaction

#### Chat with Model
```http
POST /api/model-training/model-chat/{session_id}
```
**Request Body:**
```json
{
  "message": "Predict churn for a customer with age=35, tenure=24, and monthly charges=75"
}
```

**Response:**
```json
{
  "session_id": "xxx-xxx-xxx",
  "user_input": "...",
  "parsed_request": {...},
  "prediction_result": {...},
  "bot_response": "Based on your input...",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### Get Model Deployment Info
```http
GET /api/model-training/model-deployment/{session_id}
```
Returns model information with AI-generated test prompts.

#### Get Model Info
```http
GET /api/model-training/model-info/{session_id}
```
Returns detailed model information and metadata.

### Session Management

#### List All Sessions
```http
GET /api/model-training-utils/h2o-sessions
```

#### Get Session Data
```http
GET /api/model-training-utils/h2o-session-data/{session_id}
```

#### Delete Session
```http
DELETE /api/model-training-utils/h2o-sessions/{session_id}
```

### Leaderboard & Results

#### Get Model Leaderboard
```http
GET /api/model-training/h2o-leaderboard/{session_id}
```

#### Get ML Recommendations
```http
GET /api/model-training/h2o-ml-recommendations/{session_id}
```

### Utilities

#### Check Pipeline Status
```http
GET /api/model-training-utils/h2o-ml-pipeline-status
```

#### Get H2O Cluster Info
```http
GET /api/model-training-utils/h2o-cluster-info
```

### Dataset Management

#### Suggest Data Sources
```http
POST /api/v1/dataset/suggest-sources
```
**Request Body:**
```json
{
  "modelInput": "customer churn prediction dataset"
}
```

#### Get Dataset
```http
GET /api/v1/dataset/datasets?filename=churn_data.csv&limit=100&offset=0
```

## 💡 Usage Examples

### 1. Train a Model with File Upload

**With Excel file:**
```bash
curl -X POST "http://127.0.0.1:8000/api/model-training/run-h2o-ml-pipeline-advanced" \
  -F "file=@datasets/customer_churn.xlsx" \
  -F "target_variable=Churn" \
  -F "max_runtime_secs=300" \
  -F "model_name=gpt-4o-mini"
```

**With CSV file:**
```bash
curl -X POST "http://127.0.0.1:8000/api/model-training/run-h2o-ml-pipeline-advanced" \
  -F "file=@datasets/customer_churn.csv" \
  -F "target_variable=Churn" \
  -F "max_runtime_secs=300"
```

### 2. Train a Model from Existing Dataset

```bash
curl -X GET "http://127.0.0.1:8000/api/model-training/run-h2o-ml-pipeline?data_path=datasets/churn_data.csv&target_variable=Churn&max_runtime_secs=300"
```

### 3. Chat with a Trained Model

```bash
curl -X POST "http://127.0.0.1:8000/api/model-training/model-chat/your-session-id" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does this model predict?"
  }'
```

### 4. Make a Prediction via Chat

```bash
curl -X POST "http://127.0.0.1:8000/api/model-training/model-chat/your-session-id" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Predict for age=35, income=75000, tenure=24"
  }'
```

### 5. Get Model Performance

```bash
curl -X GET "http://127.0.0.1:8000/api/model-training/h2o-leaderboard/your-session-id"
```

## 📁 Project Structure

```
aeroml-api/
├── app/
│   ├── api/
│   │   ├── model_training_controller.py   # Main ML training endpoints
│   │   ├── h2o_utils_controller.py        # Utility endpoints
│   │   └── dataset_controller_v1.py       # Dataset management
│   ├── helper/
│   │   └── utils.py                       # Helper functions
│   └── main.py                            # FastAPI application entry
├── datasets/                              # Training datasets
├── session_data/                          # Stored session results
│   └── {session_id}/
│       ├── session_data.json
│       ├── leaderboard.json
│       ├── performance.json
│       └── prompt_suggestion.json
├── h2o_machine_learning_agent/           # H2O ML pipeline
├── .env                                   # Environment variables
├── requirements.txt                       # Python dependencies
└── README.md                             # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional
BROWSERBASE_PROJECT_ID=your-project-id
BROWSERBASE_API_KEY=your-api-key
GROQ_API_KEY=your-groq-key
GEMINI_API_KEY=your-gemini-key
```

### Server Configuration

Modify `app/main.py` to change server settings:

```python
if __name__ == '__main__':
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",    # Server host
        port=8000,          # Server port
        reload=True,        # Auto-reload on code changes
        log_level="debug"   # Logging level
    )
```

## 🎯 Key Features Explained

### File Upload

The advanced pipeline endpoint supports direct file upload:
- Upload `.xlsx`, `.xls`, or `.csv` files
- Automatic conversion to CSV for H2O processing (Excel files)
- Validation of target variables and columns
- Dataset shape information in response
- Automatic cleanup of temporary files

### Session-Based Training

Each training run generates a unique session ID that persists:
- Model artifacts
- Performance metrics
- Leaderboard results
- Training configuration
- ML recommendations
- Original filename and dataset shape

This allows you to retrieve results later without keeping H2O sessions active.

### AI Chat Interface

The chatbot can:
- Answer questions about the trained model
- Make predictions from natural language input
- Extract structured data from conversational requests
- Provide detailed prediction explanations
- Filter out unrelated questions

### Real-Time Streaming

Training endpoints use Server-Sent Events (SSE) to stream:
- Training progress updates
- Model performance metrics
- Error messages
- Completion status

## 🐛 Troubleshooting

### OpenAI API Key Error

**Error:** `The api_key client option must be set...`

**Solution:**
1. Ensure `.env` file exists in project root
2. Verify `OPENAI_API_KEY` is set correctly
3. Restart the server after adding the key

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### H2O Initialization Issues

**Error:** H2O cluster connection problems

**Solution:**
- The API automatically manages H2O clusters
- Check `http://127.0.0.1:8000/api/model-training-utils/h2o-cluster-info` for status
- Session data is stored locally and doesn't require active H2O sessions

---

