# H2O Session Data Storage

This document describes the local file storage system for H2O ML pipeline session data.

## Overview

The AeroML API now automatically saves H2O ML pipeline session data to local files, making it easy to retrieve leaderboard metrics, ML recommendations, and performance reports without relying on active H2O sessions.

## Directory Structure

```
session_data/
├── {session_id_1}/
│   ├── leaderboard.json          # Model leaderboard data
│   ├── ml_recommendations.txt    # ML recommendations and steps
│   ├── performance.json          # Model performance metrics
│   └── session_data.json         # Complete session information
├── {session_id_2}/
│   └── ...
└── ...
```

## File Contents

### leaderboard.json
Contains the complete model leaderboard with metrics like:
- Model IDs
- AUC scores
- LogLoss values
- Training times
- Model types

### ml_recommendations.txt
Contains the ML agent's recommended steps and insights:
- Data preprocessing recommendations
- Feature engineering suggestions
- Model selection advice
- Hyperparameter tuning tips

### performance.json
Contains model performance metrics:
- AUC (Area Under Curve)
- LogLoss
- Other relevant metrics

### session_data.json
Contains complete session information:
- Session metadata
- Configuration parameters
- Model paths
- All other session data

## API Endpoints

### List Available Sessions
```
GET /h2o-session-data
```
Returns a list of all available session data directories with information about what data is available for each session.

### Retrieve Session Data
```
GET /h2o-session-data/{session_id}
```
Retrieves all available data for a specific session from local files.

### Get Leaderboard and ML Recommendations
```
GET /h2o-leaderboard/{session_id}
```
Retrieves model leaderboard and ML recommendations from local files. Prioritizes local file storage for fast retrieval.

### Get ML Recommendations Only
```
GET /h2o-ml-recommendations/{session_id}
```
Retrieves only ML recommendations from local files for a specific session.

## Usage Examples

### 1. Run H2O ML Pipeline
```bash
curl -X GET "http://localhost:8000/run-h2o-ml-pipeline?data_path=datasets/churn_data.csv&target_variable=Churn&max_runtime_secs=120&model_name=gpt-4o-mini"
```

### 2. List Available Sessions
```bash
curl -X GET "http://localhost:8000/h2o-session-data"
```

### 3. Retrieve Specific Session Data
```bash
curl -X GET "http://localhost:8000/h2o-session-data/{session_id}"
```

### 4. Get Leaderboard and ML Recommendations
```bash
curl -X GET "http://localhost:8000/h2o-leaderboard/{session_id}"
```

### 5. Get ML Recommendations Only
```bash
curl -X GET "http://localhost:8000/h2o-ml-recommendations/{session_id}"
```

## Benefits

1. **Persistent Storage**: Data is saved locally and persists even after H2O sessions are closed
2. **Fast Retrieval**: No need to wait for H2O session initialization
3. **Complete Data**: All session information is preserved
4. **Easy Access**: Simple REST API endpoints for data retrieval
5. **Offline Access**: Data can be accessed without running H2O

## File Management

- Session data is automatically saved when pipelines complete
- Files are organized by session ID for easy management
- Failed sessions are also saved for debugging purposes
- No automatic cleanup - manual cleanup may be needed for long-term storage

## Error Handling

- If files don't exist, appropriate HTTP 404 errors are returned
- JSON parsing errors are handled gracefully
- File system errors are logged and reported
