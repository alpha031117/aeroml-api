from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
import pandas as pd
import json
import io
from typing import Optional
import os

dataset_validation_router = APIRouter(tags=["dataset-validation"])


@dataset_validation_router.post("/validate-dataset")
async def validate_dataset(
    file: UploadFile = File(..., description="Dataset file (.xlsx, .xls, or .csv)"),
    prompt: str = Form(..., description="Text prompt describing the intended use of the dataset"),
):
    """
    Validates if a dataset is suitable for training based on the provided prompt.
    
    This endpoint:
    1. Accepts an Excel/CSV file containing the dataset
    2. Accepts a text prompt describing the intended model/training goal
    3. Analyzes the dataset structure and content
    4. Uses OpenAI (gpt-4o-mini) to validate if the dataset is appropriate for the described task
    5. Returns detailed validation results and recommendations
    
    Args:
        file: The dataset file (Excel .xlsx, .xls, or CSV .csv)
        prompt: Text description of the intended training task
        
    Returns:
        JSON response containing:
        - is_valid: Boolean indicating if dataset is valid
        - validation_message: Detailed explanation from OpenAI
        - dataset_info: Information about the dataset structure
        - recommendations: Suggestions for improvement (if any)
    """
    try:
        # Validate file type
        filename = file.filename
        if not filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(
                status_code=400,
                detail="Only Excel (.xlsx, .xls) or CSV (.csv) files are supported"
            )
        
        # Validate prompt
        if not prompt or not prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )
        
        # Read the uploaded file
        file_content = await file.read()
        
        # Load the dataset into a DataFrame
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(file_content))
            else:
                df = pd.read_excel(io.BytesIO(file_content))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading file: {str(e)}. Please ensure the file is a valid Excel or CSV file."
            )
        
        # Validate that the dataset is not empty
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded dataset is empty"
            )
        
        # Extract dataset information
        dataset_info = {
            "filename": filename,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
            "sample_data": df.head(5).to_dict(orient="records"),
            "summary_statistics": {}
        }
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            dataset_info["summary_statistics"] = df[numeric_cols].describe().to_dict()
        
        # Prepare the validation prompt for OpenAI
        validation_prompt = f"""You are an expert data scientist tasked with validating whether a dataset is suitable for a specific machine learning task.

        **User's Training Goal:**
        {prompt}

        **Dataset Information:**
        - Filename: {dataset_info['filename']}
        - Number of rows: {dataset_info['num_rows']}
        - Number of columns: {dataset_info['num_columns']}
        - Columns: {', '.join(dataset_info['columns'])}

        **Data Types:**
        {json.dumps(dataset_info['data_types'], indent=2)}

        **Missing Values:**
        {json.dumps(dataset_info['missing_values'], indent=2)}

        **Sample Data (first 5 rows):**
        {json.dumps(dataset_info['sample_data'], indent=2)}

        **Your Task:**
        Please analyze this dataset and determine if it is suitable for the described training goal. Provide your response in the following JSON format:

        {{
            "is_valid": true/false,
            "confidence_score": 0-100,
            "validation_message": "Clear explanation of why the dataset is or isn't suitable",
            "recommendations": [
                "Specific recommendation 1",
                "Specific recommendation 2",
                ...
            ],
            "potential_issues": [
                "Issue 1 (if any)",
                "Issue 2 (if any)",
                ...
            ],
            "suggested_target_column": "name of the column that should be used as target (if applicable)",
            "suggested_preprocessing": [
                "Preprocessing step 1",
                "Preprocessing step 2",
                ...
            ]
        }}

        Focus on:
        1. Whether the dataset contains relevant features for the task
        2. Data quality (missing values, data types, etc.)
        3. Dataset size adequacy
        4. Potential target variable identification
        5. Any data preprocessing needs
        6. Any critical issues that would prevent training

        Respond ONLY with the JSON object, no additional text."""

        # Initialize OpenAI with gpt-4o-mini
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,  # Lower temperature for more consistent validation
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Call OpenAI to validate the dataset
        try:
            response = llm.invoke(validation_prompt)
            response_text = response.content.strip()
            
            # Parse the JSON response
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            validation_result = json.loads(response_text)
            
        except json.JSONDecodeError as e:
            # If OpenAI doesn't return valid JSON, create a fallback response
            validation_result = {
                "is_valid": False,
                "confidence_score": 0,
                "validation_message": f"Error parsing OpenAI response: {str(e)}. Raw response: {response_text[:500]}",
                "recommendations": ["Please try again or contact support"],
                "potential_issues": ["Unable to validate dataset"],
                "suggested_target_column": None,
                "suggested_preprocessing": []
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error calling OpenAI: {str(e)}"
            )
        
        # Prepare the final response
        final_response = {
            "status": "success",
            "dataset_info": {
                "filename": dataset_info["filename"],
                "num_rows": dataset_info["num_rows"],
                "num_columns": dataset_info["num_columns"],
                "columns": dataset_info["columns"],
                "missing_values": dataset_info["missing_values"]
            },
            "validation": validation_result
        }
        
        return JSONResponse(content=final_response, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@dataset_validation_router.get("/health")
async def health_check():
    """
    Simple health check endpoint to verify the dataset validation service is running.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return {
        "status": "healthy",
        "service": "dataset-validation",
        "openai_configured": bool(openai_api_key)
    }

