# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# * Agents: H2O Machine Learning Agent

import os
import json
from typing import TypedDict, Annotated, Sequence, Literal, Optional
import operator

import pandas as pd
from IPython.display import Markdown

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command, Checkpointer
from langgraph.checkpoint.memory import MemorySaver

from aeroml_data_science_team.templates import(
    node_func_execute_agent_code_on_data,
    node_func_human_review,
    node_func_fix_agent_code,
    node_func_report_agent_outputs,
    create_coding_agent_graph,
    BaseAgent,
)
from aeroml_data_science_team.parsers.parsers import PythonOutputParser
from aeroml_data_science_team.utils.regex import (
    relocate_imports_inside_function,
    add_comments_to_top,
    format_agent_name,
    format_recommended_steps,
    get_generic_summary,
)
from aeroml_data_science_team.tools.dataframe import get_dataframe_summary
from aeroml_data_science_team.utils.logging import log_ai_function
from aeroml_data_science_team.tools.h2o import H2O_AUTOML_DOCUMENTATION

AGENT_NAME = "h2o_ml_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


def detect_problem_type(data_raw: dict, target_variable: str) -> dict:
    """
    Automatically detect if the problem is classification or regression.

    Parameters
    ----------
    data_raw : dict
        Dictionary representation of the DataFrame
    target_variable : str
        Name of the target variable column

    Returns
    -------
    dict
        Dictionary with problem type information including:
        - problem_type: 'classification' or 'regression'
        - num_classes: Number of unique classes (for classification)
        - target_dtype: Data type of target column
        - is_binary: Whether it's binary classification
        - is_multiclass: Whether it's multiclass classification
        - is_regression: Whether it's regression
        - recommended_metric: Recommended evaluation metric
        - recommended_stopping_metric: Recommended stopping metric
        - recommended_sort_metric: Recommended sort metric
    """
    df = pd.DataFrame(data_raw)

    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in data columns: {list(df.columns)}")

    target_col = df[target_variable]
    target_dtype = target_col.dtype
    num_unique = target_col.nunique()
    num_samples = len(target_col)

    # Detection logic
    is_categorical = target_dtype == 'object' or target_dtype.name == 'category' or target_dtype.name == 'bool'
    is_low_cardinality = num_unique < 20
    unique_ratio = num_unique / num_samples if num_samples > 0 else 0
    is_ratio_based = unique_ratio > 0.5  # More than 50% unique values suggests continuous/regression

    if is_categorical or (is_low_cardinality and not is_ratio_based):
        # Classification
        is_binary = num_unique == 2
        is_multiclass = num_unique > 2

        return {
            'problem_type': 'classification',
            'num_classes': num_unique,
            'target_dtype': str(target_dtype),
            'is_binary': is_binary,
            'is_multiclass': is_multiclass,
            'is_regression': False,
            'recommended_metric': 'AUC' if is_binary else 'mean_per_class_error',
            'recommended_stopping_metric': 'logloss',
            'recommended_sort_metric': 'AUC' if is_binary else 'mean_per_class_error'
        }
    else:
        # Regression
        return {
            'problem_type': 'regression',
            'num_classes': None,
            'target_dtype': str(target_dtype),
            'is_binary': False,
            'is_multiclass': False,
            'is_regression': True,
            'recommended_metric': 'RMSE',
            'recommended_stopping_metric': 'RMSE',
            'recommended_sort_metric': 'RMSE'
        }


class H2OMLAgent(BaseAgent):
    """
    A Machine Learning agent that uses H2O's AutoML for training,
    allowing the user to specify a model directory for saving the best model.
    If neither model_directory nor log_path is provided, model saving is skipped.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the ML code.
    n_samples : int, optional
        Number of samples used when summarizing the dataset. Defaults to 30.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the Python file for saving the generated code. Defaults to "h2o_automl.py".
    function_name : str, optional
        Name of the function that performs the AutoML training. Defaults to "h2o_automl".
    model_directory : str or None, optional
        Directory to save the H2O Machine Learning model. If None, defaults to log_path (if available).
        If both are None, no model is saved. Defaults to None.
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of the code. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the recommended steps prompt. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the code-explanation step. Defaults to False.
    enable_mlflow : bool, default False
        Whether to enable MLflow logging. If False, skip MLflow entirely.
    mlflow_tracking_uri : str or None
        If provided, sets MLflow tracking URI at runtime.
    mlflow_experiment_name : str
        Name of the MLflow experiment (created if doesn't exist).
    mlflow_run_name : str, default None
        A custom name for the MLflow run.
    checkpointer : langgraph.checkpoint.memory.MemorySaver, optional
        A checkpointer object for saving the agent's state. Defaults to None.
    
    
    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled state graph.
    ainvoke_agent(user_instructions, data_raw, target_variable, ...)
        Asynchronously runs the agent to produce an H2O AutoML model, optionally saving the model to disk.
    invoke_agent(user_instructions, data_raw, target_variable, ...)
        Synchronously runs the agent to produce an H2O AutoML model, optionally saving the model to disk.
    get_leaderboard()
        Retrieves the H2O AutoML leaderboard from the agent's response.
    get_best_model_id()
        Retrieves the best model ID from the agent's response.
    get_model_path()
        Retrieves the saved model path from the agent's response (or None if not saved).
    get_data_raw()
        Retrieves the raw data as a DataFrame from the agent's response.
    get_h2o_train_function(markdown=False)
        Retrieves the H2O AutoML function code generated by the agent.
    get_recommended_ml_steps(markdown=False)
        Retrieves recommended ML steps from the agent's response.
    get_workflow_summary()
        Retrieves a summary of the agent's workflow.
    get_response()
        Returns the entire response dictionary.
    show()
        Visualizes the compiled graph as a Mermaid diagram.
        
    Examples
    --------
    ```python
    from langchain_openai import ChatOpenAI
    import pandas as pd
    from ai_data_science_team.ml_agents import H2OMLAgent

    llm = ChatOpenAI(model="gpt-4o-mini")
    
    df = pd.read_csv("data/churn_data.csv")
    
    ml_agent = H2OMLAgent(
        model=llm, 
        log=True, 
        log_path=LOG_PATH,
        model_directory=MODEL_PATH, 
    )
    
    ml_agent.invoke_agent(
        data_raw=df.drop(columns=["customerID"]),
        user_instructions="Please do classification on 'Churn'. Use a max runtime of 30 seconds.",
        target_variable="Churn"
    )

    # Retrieve and display the leaderboard of models
    ml_agent.get_leaderboard()

    # Get the H2O training function in markdown format
    ml_agent.get_h2o_train_function(markdown=True)

    # Get the recommended machine learning steps in markdown format
    ml_agent.get_recommended_ml_steps(markdown=True)

    # Get a summary of the workflow in markdown format
    ml_agent.get_workflow_summary(markdown=True)

    # Get a summary of the logs in markdown format
    ml_agent.get_log_summary(markdown=True)

    # Get the path to the saved model
    model_path = ml_agent.get_model_path()
    model_path
    ```
    
    Returns
    -------
    H2OMLAgent : langchain.graphs.CompiledStateGraph 
        An instance of the H2O ML agent.
    
    """

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="h2o_automl.py",
        function_name="h2o_automl",
        model_directory=None,  
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        enable_mlflow=False,
        mlflow_tracking_uri=None,
        mlflow_experiment_name="H2O AutoML",
        mlflow_run_name=None,
        checkpointer: Optional[Checkpointer]=None,
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "model_directory": model_directory,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "enable_mlflow": enable_mlflow,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
            "mlflow_run_name": mlflow_run_name,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """
        Creates the compiled graph for the agent.
        """
        self.response = None
        return make_h2o_ml_agent(**self._params)

    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(
        self, 
        data_raw: pd.DataFrame, 
        user_instructions: str=None, 
        target_variable: str=None, 
        max_retries=3, 
        retry_count=0,
        **kwargs
    ):
        """
        Asynchronously trains an H2O AutoML model for the provided dataset,
        saving the best model to disk if model_directory or log_path is available.
        """
        response = await self._compiled_graph.ainvoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "target_variable": target_variable,
            "max_retries": max_retries,
            "retry_count": retry_count
        }, **kwargs)
        self.response = response
        return None

    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str=None,
        target_variable: str=None,
        max_retries=3,
        retry_count=0,
        **kwargs
    ):
        """
        Synchronously trains an H2O AutoML model for the provided dataset,
        saving the best model to disk if model_directory or log_path is available.
        """
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "target_variable": target_variable,
            "max_retries": max_retries,
            "retry_count": retry_count
        }, **kwargs)
        self.response = response
        return None

    def get_leaderboard(self):
        """Returns the H2O AutoML leaderboard as a DataFrame."""
        if self.response and "leaderboard" in self.response:
            return pd.DataFrame(self.response["leaderboard"])
        return None

    def get_best_model_id(self):
        """Returns the best model id from the AutoML run."""
        if self.response and "best_model_id" in self.response:
            return self.response["best_model_id"]
        return None

    def get_model_path(self):
        """Returns the file path to the saved best model, or None if not saved."""
        if self.response and "model_path" in self.response:
            return self.response["model_path"]
        return None

    def get_data_raw(self):
        """Retrieves the raw data as a DataFrame from the response."""
        if self.response and "data_raw" in self.response:
            return pd.DataFrame(self.response["data_raw"])
        return None

    def get_h2o_train_function(self, markdown=False):
        """Retrieves the H2O AutoML function code generated by the agent."""
        if self.response and "h2o_train_function" in self.response:
            code = self.response["h2o_train_function"]
            if markdown:
                return Markdown(f"```python\n{code}\n```")
            return code
        return None

    def get_recommended_ml_steps(self, markdown=False):
        """Retrieves recommended ML steps from the agent's response."""
        if self.response and "recommended_steps" in self.response:
            steps = self.response["recommended_steps"]
            if markdown:
                return Markdown(steps)
            return steps
        return None

    def get_workflow_summary(self, markdown=False):
        """
        Retrieves the agent's workflow summary, if logging is enabled.
        """
        if self.response and self.response.get("messages"):
            summary = get_generic_summary(json.loads(self.response.get("messages")[-1].content))
            if markdown:
                return Markdown(summary)
            else:
                return summary
    
    def get_log_summary(self, markdown=False):
        """
        Logs a summary of the agent's operations, if logging is enabled.
        """
        if self.response:
            if self.response.get('h2o_train_function_path'):
                log_details = f"""
## H2O Machine Learning Agent Log Summary:

Function Path: {self.response.get('h2o_train_function_path')}

Function Name: {self.response.get('h2o_train_function_name')}

Best Model ID: {self.get_best_model_id()}

Model Path: {self.get_model_path()}
                """
                if markdown:
                    return Markdown(log_details) 
                else:
                    return log_details


def make_h2o_ml_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="h2o_automl.py",
    function_name="h2o_automl",
    model_directory=None,
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    enable_mlflow=False,
    mlflow_tracking_uri=None,
    mlflow_experiment_name="H2O AutoML",
    mlflow_run_name=None,
    checkpointer=None,
):
    """
    Creates a machine learning agent that uses H2O for AutoML. 
    The agent will:
      1. Optionally recommend ML steps,
      2. Creates Python code that sets up H2OAutoML,
      3. Executes that code (optionally saving the best model to disk),
      4. Fixes errors if needed,
      5. Optionally explains the code.

    model_directory: Directory to save the model. 
                    If None, defaults to log_path. 
                    If both are None, skip saving.
    """

    llm = model

    # Handle logging directory
    if log:
        if log_path is None:
            log_path = "logs/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    # Check if H2O is installed
    try:
        import h2o
        from h2o.automl import H2OAutoML
    except ImportError as e:
        raise ImportError(
            "The 'h2o' library is not installed. Please install it using pip:\n\n"
            "    pip install h2o\n\n"
            "Visit https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html for details."
        ) from e
        
    if human_in_the_loop:
        if checkpointer is None:
            print("Human in the loop is enabled. A checkpointer is required. Setting to MemorySaver().")
            checkpointer = MemorySaver()
        

    # Define GraphState
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        data_raw: dict
        leaderboard: dict
        best_model_id: str
        model_path: str
        model_results: dict
        target_variable: str
        all_datasets_summary: str
        h2o_train_function: str
        h2o_train_function_path: str
        h2o_train_file_name: str
        h2o_train_function_name: str
        h2o_train_error: str
        max_retries: int
        retry_count: int
        problem_type_info: dict

    # 1) Recommend ML steps (optional)
    def recommend_ml_steps(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND MACHINE LEARNING STEPS")

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        target_variable = state.get("target_variable")

        # Detect problem type
        try:
            problem_info = detect_problem_type(data_raw, target_variable)
        except Exception as e:
            print(f"    Warning: Could not detect problem type: {e}")
            problem_info = {
                'problem_type': 'classification',  # Default fallback
                'num_classes': None,
                'recommended_stopping_metric': 'logloss',
                'recommended_sort_metric': 'AUC'
            }

        all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples)
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        # Add problem type info to summary
        problem_summary = f"""
Problem Type Detection:
- Detected Type: {problem_info['problem_type']}
- Target Variable: {target_variable}
- Number of Classes: {problem_info.get('num_classes', 'N/A')}
- Recommended Stopping Metric: {problem_info['recommended_stopping_metric']}
- Recommended Sort Metric: {problem_info['recommended_sort_metric']}
"""

        recommend_steps_prompt = PromptTemplate(
            template="""
                You are an AutoML Expert using H2O.

                We have the following dataset summary, user instructions, and H2O AutoML documentation:

                Problem Type Information:
                    {problem_summary}

                User instructions:
                    {user_instructions}

                Data Summary:
                    {all_datasets_summary}

                H2O AutoML Documentation:
                    {h2o_automl_documentation}

                Based on the detected problem type ({problem_type}), please recommend a short list of steps or considerations for performing H2OAutoML on this data. Specifically focus on maximizing model accuracy while remaining flexible to user instructions and the dataset.

                IMPORTANT: Adapt your recommendations based on the problem type:
                - For CLASSIFICATION: Recommend logloss/AUC metrics, balance_classes if needed, appropriate nfolds
                - For REGRESSION: Recommend RMSE/MSE metrics, do NOT recommend balance_classes, appropriate nfolds

                - Recommend any parameters and values that might improve performance (predictive accuracy).
                - Recommend the Loss Function, Stopping Criteria, and other advanced parameters based on the problem type.
                - Use the H2O AutoML documentation to your advantage.
                - Exclude deep learning algorithms since these are typically low performance.

                Avoid these:

                - Do not perform data cleaning or feature engineering here. We will handle that separately.
                - Do not limit memory size or CPU usage unless the user specifies it.
                - Do NOT recommend balance_classes for regression problems.
                - Do NOT recommend logloss metric for regression problems.

                Return as a numbered list. You can return short code snippets to demonstrate actions. But do not return a fully coded solution. The H2O AutoML code will be generated separately by a Coding Agent.
            """,
            input_variables=["user_instructions", "all_datasets_summary", "h2o_automl_documentation", "problem_summary", "problem_type"]
        )

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "all_datasets_summary": all_datasets_summary_str,
            "h2o_automl_documentation": H2O_AUTOML_DOCUMENTATION,
            "problem_summary": problem_summary,
            "problem_type": problem_info['problem_type']
        })

        return {
            "recommended_steps": format_recommended_steps(
                recommended_steps.content.strip(),
                heading="# Recommended ML Steps:"
            ),
            "all_datasets_summary": all_datasets_summary_str,
            "problem_type_info": problem_info  # Store for use in code generation
        }

    # 2) Create code
    def create_h2o_code(state: GraphState):
        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))

            data_raw = state.get("data_raw")
            df = pd.DataFrame.from_dict(data_raw)
            all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples)
            all_datasets_summary_str = "\n\n".join(all_datasets_summary)

            # Detect problem type if not already in state
            target_variable = state.get("target_variable")
            try:
                problem_info = detect_problem_type(data_raw, target_variable)
            except Exception as e:
                problem_info = {
                    'problem_type': 'classification',
                    'recommended_stopping_metric': 'logloss',
                    'recommended_sort_metric': 'AUC'
                }
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
            problem_info = state.get("problem_type_info", {
                'problem_type': 'classification',
                'recommended_stopping_metric': 'logloss',
                'recommended_sort_metric': 'AUC'
            })

        print("    * CREATE H2O AUTOML CODE")

        # Extract problem type details
        problem_type = problem_info.get('problem_type', 'classification')
        recommended_stopping_metric = problem_info.get('recommended_stopping_metric', 'logloss')
        recommended_sort_metric = problem_info.get('recommended_sort_metric', 'AUC')
        is_regression = problem_info.get('is_regression', False)

        code_prompt = PromptTemplate(
            template="""
            You are an H2O AutoML agent. Create a Python function named {function_name}(data_raw)
            that runs H2OAutoML on the provided data with a focus on maximizing model accuracy and 
            incorporating user instructions for flexibility.
            
            Do not perform substantial data cleaning or feature engineering here. We will handle that separately.

            We have two variables for deciding where to save the model:
            model_directory = {model_directory} 
            log_path = {log_path}
            
            IMPORTANT: MLflow Parameters if the user wants to enable MLflow with H2O AutoML:
                enable_mlflow: {enable_mlflow}
                mlflow_tracking_uri: {mlflow_tracking_uri}
                mlflow_experiment_name: {mlflow_experiment_name}
                mlflow_run_name: {mlflow_run_name}

            Problem Type Information:
                - Problem Type: {problem_type}
                - Recommended Stopping Metric: {recommended_stopping_metric}
                - Recommended Sort Metric: {recommended_sort_metric}
                - Is Regression: {is_regression}

            CRITICAL: Adapt the generated function based on problem type:
                - If REGRESSION: Use 'RMSE' for stopping_metric and sort_metric, set balance_classes=False
                - If CLASSIFICATION: Use 'logloss' for stopping_metric, 'AUC' (binary) or 'mean_per_class_error' (multiclass) for sort_metric, balance_classes can be True
                - The generated code MUST detect the problem type and adjust parameters automatically

            CRITICAL REQUIREMENT - IMPORTS WITH AUTO-INSTALL:
            ALL import statements MUST be placed INSIDE the function definition, not at the top of the file.
            
            IMPORTANT: For each import, you MUST check if the package is installed, and if not, automatically install it using pip.
            Use this pattern for ALL package imports:
            
            ```python
            # Helper function to install packages if missing
            def _install_package(package_name):
                import subprocess
                import sys
                try:
                    __import__(package_name)
                except ImportError:
                    print(f"Package '{{package_name}}' not found. Installing...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                    print(f"Package '{{package_name}}' installed successfully.")

            # Install and import packages
            _install_package("h2o")
            import h2o
            from h2o.automl import H2OAutoML

            _install_package("pandas")
            import pandas as pd

            import json  # json is built-in, no installation needed

            from contextlib import nullcontext  # contextlib is built-in
            ```
            
            Required imports (place these at the beginning of the function body with auto-install):
                - h2o (with auto-install check)
                - from h2o.automl import H2OAutoML
                - pandas (with auto-install check)
                - json (built-in, no installation needed)
                - from contextlib import nullcontext  (if not using MLflow, built-in)
                - mlflow (only if enable_mlflow is True, with auto-install check)
            
            CRITICAL REQUIREMENT - TYPE HINTS:
            DO NOT use pandas type hints (like pd.DataFrame) in the function signature because pandas is imported INSIDE the function.
            Use simple types or typing module types instead. For example:
                - Use: data_raw (no type hint)
                - OR use: data_raw: dict
                - DO NOT use: data_raw: pd.DataFrame (this will fail!)

            Additional Requirements:
            - ALWAYS include the _install_package() helper function at the start of the function to automatically install missing packages.
            - Convert `data_raw` (pandas DataFrame) into an H2OFrame.
            - Identify the target variable from {target_variable} (if provided).
            - Start H2O if not already started.
            - Use Recommended Steps to guide any advanced parameters (e.g., cross-validation folds, 
            balancing classes, extended training time, stacking) that might improve performance.
            - If the user does not specify anything special, use H2OAutoML defaults (including stacked ensembles).
            - Focus on maximizing accuracy (or the most relevant metric if it's not classification) 
            while remaining flexible to user instructions.
            - Return a dict with keys: leaderboard, best_model_id, model_path, and model_results.
            - If enable_mlfow is True, log the top metrics and save the model as an artifact. (See example function)
            - IMPORTANT: if enable_mlflow is True, make sure to set enable_mlflow to True in the function definition.
            - IMPORTANT: For any additional packages that might be needed (e.g., numpy, scipy), use the _install_package() pattern before importing.
            
            Initial User Instructions (Disregard any instructions that are unrelated to modeling):
                {user_instructions}
            
            Recommended Steps:
                {recommended_steps}

            Data summary for reference:
                {all_datasets_summary}

            Return only code in ```python``` with a single function definition. Use this as an example starting template:
            ```python
            def {function_name}(
                data_raw,  # DO NOT use pd.DataFrame type hint - pandas is imported inside!
                target: str = None,
                max_runtime_secs: int = 300,
                exclude_algos: list = None,
                balance_classes: bool = None,  # Will be auto-determined based on problem type
                nfolds: int = 3,
                seed: int = 42,
                max_models: int = 10,
                stopping_metric: str = None,  # Will be auto-determined based on problem type
                stopping_tolerance: float = 0.001,
                stopping_rounds: int = 3,
                sort_metric: str = None,  # Will be auto-determined based on problem type
                model_directory: str = None,
                log_path: str = None,
                enable_mlflow: bool = False,
                mlflow_tracking_uri: str = None,
                mlflow_experiment_name: str = 'H2O AutoML',
                mlflow_run_name: str = None,
                **kwargs
            ):
                # Helper function to install packages if missing
                def _install_package(package_name):
                    import subprocess
                    import sys
                    try:
                        __import__(package_name)
                    except ImportError:
                        print(f"Package '{{package_name}}' not found. Installing...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name], 
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print(f"Package '{{package_name}}' installed successfully.")
                
                # ALL IMPORTS MUST BE INSIDE THE FUNCTION WITH AUTO-INSTALL
                _install_package("h2o")
                import h2o
                from h2o.automl import H2OAutoML
                
                _install_package("pandas")
                import pandas as pd
                
                import json  # Built-in, no installation needed
                import subprocess  # Needed for package installation
                import sys  # Needed for package installation

                # Optional MLflow usage
                if enable_mlflow:
                    _install_package("mlflow")
                    import mlflow
                    if mlflow_tracking_uri:
                        mlflow.set_tracking_uri(mlflow_tracking_uri)
                    mlflow.set_experiment(mlflow_experiment_name)
                    run_context = mlflow.start_run(run_name=mlflow_run_name)
                else:
                    # Dummy context manager to skip MLflow if not enabled
                    from contextlib import nullcontext
                    run_context = nullcontext()

                exclude_algos = exclude_algos or ["DeepLearning"]  # default if not provided

                # Convert data to DataFrame
                if isinstance(data_raw, dict):
                    df = pd.DataFrame(data_raw)
                else:
                    df = data_raw.copy()

                # Validate target variable
                if target is None:
                    raise ValueError("target parameter is required")
                if target not in df.columns:
                    raise ValueError(f"Target variable '{{target}}' not found in dataset columns: {{list(df.columns)}}")

                # Remove ID columns (columns with all unique values)
                id_columns = [col for col in df.columns if df[col].nunique() == len(df) and col != target]
                if id_columns:
                    df = df.drop(columns=id_columns)

                # Remove columns with all NaN values (prevents dtype errors when converting to H2OFrame)
                nan_columns = [col for col in df.columns if df[col].isna().all()]
                if nan_columns:
                    df = df.drop(columns=nan_columns)

                # Handle missing values in target
                if df[target].isna().any():
                    df = df.dropna(subset=[target])

                # Detect problem type automatically
                target_col = df[target]
                num_unique = target_col.nunique()
                target_dtype = target_col.dtype
                is_categorical = target_dtype == 'object' or target_dtype.name == 'category' or target_dtype.name == 'bool'
                unique_ratio = num_unique / len(df) if len(df) > 0 else 0

                is_classification = is_categorical or (num_unique < 20 and unique_ratio < 0.5)
                is_binary = is_classification and num_unique == 2

                # Auto-determine parameters based on problem type
                if is_classification:
                    # Classification defaults
                    if stopping_metric is None:
                        stopping_metric = 'logloss'
                    if sort_metric is None:
                        sort_metric = 'AUC' if is_binary else 'mean_per_class_error'
                    if balance_classes is None:
                        balance_classes = True  # Default for classification
                else:
                    # Regression defaults
                    balance_classes = False  # Not applicable for regression
                    if stopping_metric is None:
                        stopping_metric = 'RMSE'
                    elif stopping_metric == 'logloss':
                        stopping_metric = 'RMSE'  # Fix invalid metric
                    if sort_metric is None:
                        sort_metric = 'RMSE'
                    elif sort_metric == 'logloss':
                        sort_metric = 'RMSE'  # Fix invalid metric

                with run_context as run:
                    # If using MLflow, track run ID
                    run_id = None
                    if enable_mlflow and run is not None:
                        run_id = run.info.run_id
                        import mlflow
                        

                    # Initialize H2O
                    h2o.init()

                    # Create H2OFrame
                    data_h2o = h2o.H2OFrame(df)

                    # Convert target to factor for classification
                    if is_classification:
                        data_h2o[target] = data_h2o[target].asfactor()

                    # Setup AutoML
                    aml = H2OAutoML(
                        max_runtime_secs=max_runtime_secs,
                        exclude_algos=exclude_algos,
                        balance_classes=balance_classes,
                        nfolds=nfolds,
                        seed=seed,
                        max_models=max_models,
                        stopping_metric=stopping_metric,
                        stopping_tolerance=stopping_tolerance,
                        stopping_rounds=stopping_rounds,
                        sort_metric=sort_metric,
                        **kwargs
                    )

                    # Train
                    x = [col for col in data_h2o.columns if col != target]
                    aml.train(x=x, y=target, training_frame=data_h2o)

                    # Save model if we have a directory/log path
                    if model_directory is None and log_path is None:
                        model_path = None
                    else:
                        path_to_save = model_directory if model_directory else log_path
                        model_path = h2o.save_model(model=aml.leader, path=path_to_save, force=True)

                    # Leaderboard (H2OFrame -> pandas DataFrame -> dict)
                    # IMPORTANT: Use .as_data_frame() to convert H2OFrame to pandas DataFrame
                    # DO NOT use pd.DataFrame() with dtype parameter as it causes "dtype is only supported for one column frames" error
                    leaderboard_df = aml.leaderboard.as_data_frame()
                    leaderboard_dict = leaderboard_df.to_dict()

                    # Gather top-model metrics from the first row
                    top_metrics = leaderboard_df.iloc[0].to_dict()  

                    # Construct model_results
                    model_results = dict(
                        model_flavor= "H2O AutoML",
                        model_path= model_path,
                        best_model_id= aml.leader.model_id,
                        metrics= top_metrics  # all metrics from the top row
                    )

                    # IMPORTANT: Log these to MLflow if enabled
                    if enable_mlflow and run is not None:
                        
                        # Log the top metrics if numeric
                        numeric_metrics = {{k: v for k, v in top_metrics.items() if isinstance(v, (int, float))}}
                        mlflow.log_metrics(numeric_metrics)

                        # Log artifact if we saved the model
                        mlflow.h2o.log_model(aml.leader, artifact_path="model")
                        
                        # Log the leaderboard
                        mlflow.log_table(leaderboard_dict, "leaderboard.json")
                        
                        # Log these parameters (if specified)
                        mlflow.log_params(dict(
                            target= target,
                            max_runtime_secs= max_runtime_secs,
                            exclude_algos= str(exclude_algos),
                            balance_classes= balance_classes,
                            nfolds= nfolds,
                            seed= seed,
                            max_models= max_models,
                            stopping_metric= stopping_metric,
                            stopping_tolerance= stopping_tolerance,
                            stopping_rounds= stopping_rounds,
                            sort_metric= sort_metric,
                            model_directory= model_directory,
                            log_path= log_path
                        ))

                    # Build the output
                    output = dict(
                        leaderboard= leaderboard_dict,
                        best_model_id= aml.leader.model_id,
                        model_path= model_path,
                        model_results= model_results,
                        mlflow_run_id= run_id
                    )

                return output
            ```
            
            Avoid these errors:
            
            - name 'pd' is not defined: This happens when you use pd.DataFrame in function signature before importing pandas. Solution: Do NOT use type hints that reference modules imported inside the function (like pd.DataFrame, h2o.H2OFrame, etc.). Just use data_raw without type hint or use simple types.
            
            - ModuleNotFoundError: No module named 'h2o': Solution: Use the _install_package() helper function before importing. This will automatically install missing packages using pip.
            
            - ModuleNotFoundError: No module named 'pandas': Solution: Use the _install_package() helper function before importing. This will automatically install missing packages using pip.
            
            - ModuleNotFoundError: No module named 'mlflow': Solution: Use the _install_package() helper function before importing mlflow if enable_mlflow is True.
            
            - WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.
            
            - 'list' object has no attribute 'tolist'
            
            - with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):  pandas_df = h2o_df.as_data_frame() # Convert to pandas DataFrame using pd.DataFrame(h2o_df)
            
            - dtype is only supported for one column frames: This error occurs when converting H2OFrame to pandas DataFrame incorrectly or when data has issues.
               Solutions:
               1. ALWAYS use .as_data_frame() method to convert H2OFrame to pandas DataFrame:
                  CORRECT: leaderboard_df = aml.leaderboard.as_data_frame()
                  WRONG: pd.DataFrame(aml.leaderboard) or pd.DataFrame(aml.leaderboard, dtype='float64')
               2. Clean data BEFORE converting pandas DataFrame to H2OFrame:
                  - Remove columns with all NaN values: nan_columns = [col for col in df.columns if df[col].isna().all()]; df = df.drop(columns=nan_columns)
                  - Remove ID columns: id_columns = [col for col in df.columns if df[col].nunique() == len(df) and col != target]; df = df.drop(columns=id_columns)
                  - Reset index: df = df.reset_index(drop=True)
               3. When creating H2OFrame: data_h2o = h2o.H2OFrame(df) - do NOT specify dtype parameter, let H2O handle type inference.
               4. When converting leaderboard: leaderboard_df = aml.leaderboard.as_data_frame() returns pandas DataFrame directly, then convert to dict: leaderboard_dict = leaderboard_df.to_dict()
            
            - h2o.is_running() module 'h2o' has no attribute 'is_running'. Solution: just do h2o.init() and it will check if H2O is running.

            - Stopping metric cannot be logloss for regression: Use 'RMSE' or 'MSE' for regression problems, not 'logloss'. Solution: Detect problem type - if target is numeric with many unique values, use 'RMSE'. Check: target_col.dtype != 'object' and target_col.nunique() > 20 → use 'RMSE'.

            - balance_classes can only be used for classification: This parameter only works for classification. Solution: Set balance_classes=False for regression problems. Auto-detect: if target is numeric/continuous → regression → balance_classes=False.

            - Target column not found: Verify the target variable name exactly matches a column name (case-sensitive). Solution: Check df.columns and ensure exact match.

            - Unknown categorical level: For classification, convert target to factor: data_h2o[target] = data_h2o[target].asfactor()


            """,
            input_variables=[
                "user_instructions",
                "function_name",
                "target_variable",
                "recommended_steps",
                "all_datasets_summary",
                "model_directory",
                "log_path",
                "enable_mlflow",
                "mlflow_tracking_uri",
                "mlflow_experiment_name",
                "mlflow_run_name",
                "problem_type",
                "recommended_stopping_metric",
                "recommended_sort_metric",
                "is_regression",
            ]
        )

        recommended_steps = state.get("recommended_steps", "")
        h2o_code_agent = code_prompt | llm | PythonOutputParser()

        resp = h2o_code_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "function_name": function_name,
            "target_variable": state.get("target_variable"),
            "recommended_steps": recommended_steps,
            "all_datasets_summary": all_datasets_summary_str,
            "model_directory": model_directory,
            "log_path": log_path,
            "enable_mlflow": enable_mlflow,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
            "mlflow_run_name": mlflow_run_name,
            "problem_type": problem_type,
            "recommended_stopping_metric": recommended_stopping_metric,
            "recommended_sort_metric": recommended_sort_metric,
            "is_regression": is_regression,
        })

        resp = relocate_imports_inside_function(resp)
        resp = add_comments_to_top(resp, agent_name=AGENT_NAME)

        # Log the code snippet if requested
        file_path, f_name = log_ai_function(
            response=resp,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite
        )

        return {
            "h2o_train_function": resp,
            "h2o_train_function_path": file_path,
            "h2o_train_file_name": f_name,
            "h2o_train_function_name": function_name,
        }
        
    # Human Review
    prompt_text_human_review = "Are the following Machine Learning instructions correct? (Answer 'yes' or provide modifications)\n{steps}"
    
    if not bypass_explain_code:
        def human_review(state: GraphState) -> Command[Literal["recommend_ml_steps", "explain_h2o_code"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= 'explain_h2o_code',
                no_goto="recommend_ml_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="h2o_train_function",
            )
    else:
        def human_review(state: GraphState) -> Command[Literal["recommend_ml_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= '__end__',
                no_goto="recommend_ml_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="h2o_train_function", 
            )

    # 3) Execute code
    def execute_h2o_code(state):
        result = node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            code_snippet_key="h2o_train_function",
            result_key="h2o_train_result",
            error_key="h2o_train_error",
            agent_function_name=state.get("h2o_train_function_name"),
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            post_processing=lambda x: x,
            error_message_prefix="Error occurred during H2O AutoML: "
        )

        # If no error, extract leaderboard, best_model_id, and model_path
        if not result["h2o_train_error"]:
            if result["h2o_train_result"] and isinstance(result["h2o_train_result"], dict):
                lb = result["h2o_train_result"].get("leaderboard", {})
                best_id = result["h2o_train_result"].get("best_model_id", None)
                mpath = result["h2o_train_result"].get("model_path", None)
                model_results = result["h2o_train_result"].get("model_results", {})

                result["leaderboard"] = lb
                result["best_model_id"] = best_id
                result["model_path"] = mpath
                result["model_results"] = model_results

        return result

    # 4) Fix code if there's an error
    def fix_h2o_code(state: GraphState):
        fix_prompt = """
        You are an H2O AutoML agent. The function {function_name} currently has errors. 
        Please fix it. Return only the corrected function in ```python``` format.
        
        CRITICAL: Make sure to include ALL necessary imports INSIDE the function definition WITH AUTO-INSTALLATION.
        
        REQUIRED: Include a helper function to automatically install missing packages:
        ```python
        def _install_package(package_name):
            import subprocess
            import sys
            try:
                __import__(package_name)
            except ImportError:
                print(f"Package '{{package_name}}' not found. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Package '{{package_name}}' installed successfully.")
        ```
        
        Required imports that MUST be included inside the function with auto-install:
        - _install_package("h2o")
        - import h2o
        - from h2o.automl import H2OAutoML
        - _install_package("pandas")
        - import pandas as pd
        - import json (built-in, no installation needed)
        - import subprocess, sys (for package installation)
        - from contextlib import nullcontext (built-in, if not using MLflow)
        - _install_package("mlflow") and import mlflow (if enable_mlflow is True)
        
        Common errors and solutions:

        1. "name 'h2o' is not defined" or "ModuleNotFoundError: No module named 'h2o'":
           Solution: Add _install_package("h2o") before importing h2o. This will automatically install if missing.

        2. "ModuleNotFoundError: No module named 'pandas'":
           Solution: Add _install_package("pandas") before importing pandas.

        3. "ModuleNotFoundError: No module named 'mlflow'":
           Solution: Add _install_package("mlflow") before importing mlflow (only if enable_mlflow is True).

        4. "Stopping metric cannot be logloss for regression":
           Solution: Detect if target is numeric/continuous and use 'RMSE' instead of 'logloss'.
           Check: target_col.dtype != 'object' and num_unique > 20 → use 'RMSE'

        5. "Target column not found":
           Solution: Verify target variable name matches exactly (case-sensitive).
           List available columns and check spelling.

        6. "balance_classes can only be used for classification":
           Solution: Set balance_classes=False for regression problems.
           Detect problem type: if numeric target with many unique values → regression → balance_classes=False

        7. "Unknown categorical level":
           Solution: For classification, ensure target variable is converted to factor:
           data_h2o[target] = data_h2o[target].asfactor()

        8. High cardinality warnings:
           Solution: Consider excluding high-cardinality categorical columns or use target encoding.

        9. "dtype is only supported for one column frames":
           This error occurs when converting H2OFrame to pandas DataFrame incorrectly or when creating H2OFrame with problematic data.
           
           CRITICAL FIXES:
           
           a) When converting leaderboard H2OFrame to pandas DataFrame:
              WRONG: pd.DataFrame(aml.leaderboard) or pd.DataFrame(aml.leaderboard, dtype='float64')
              CORRECT: leaderboard_df = aml.leaderboard.as_data_frame()  # Returns pandas DataFrame directly
              Then convert to dict: leaderboard_dict = leaderboard_df.to_dict()
           
           b) NEVER use pd.DataFrame() constructor on H2OFrame objects directly:
              - H2OFrame has .as_data_frame() method that handles conversion properly
              - Always use: h2o_frame.as_data_frame() instead of pd.DataFrame(h2o_frame)
           
           c) Data cleaning BEFORE converting pandas DataFrame to H2OFrame:
              - Remove columns with all NaN values: 
                nan_columns = [col for col in df.columns if df[col].isna().all()]
                if nan_columns:
                    df = df.drop(columns=nan_columns)
              - Remove ID columns (all unique values):
                id_columns = [col for col in df.columns if df[col].nunique() == len(df) and col != target]
                if id_columns:
                    df = df.drop(columns=id_columns)
              - Handle missing values in target column:
                if df[target].isna().any():
                    df = df.dropna(subset=[target])
              - Reset index if needed:
                df = df.reset_index(drop=True)
           
           d) When creating H2OFrame from pandas DataFrame:
              - Ensure DataFrame is clean (no all-NaN columns)
              - Use: data_h2o = h2o.H2OFrame(df)  # Simple, let H2O handle type inference
              - Do NOT specify dtype when creating H2OFrame
           
           e) If error persists, check for:
              - Columns with mixed types
              - Columns with all the same value
              - Very large number of columns
              - Memory issues

        IMPORTANT FIXES TO APPLY:
        - Always detect problem type (classification vs regression) from the target variable
        - Use appropriate metrics: 'logloss'/'AUC' for classification, 'RMSE' for regression
        - Set balance_classes=False for regression problems
        - Convert target to factor for classification problems
        - Validate target variable exists before use
        - When converting H2OFrame to pandas: ALWAYS use .as_data_frame() method, NEVER use pd.DataFrame(h2o_frame, dtype=...) with dtype parameter
        - Clean data before converting to H2OFrame: remove all-NaN columns, handle missing values

        Broken code:
        {code_snippet}

        Last Known Error:
        {error}
        
        Make sure your fixed code has this structure:
        def {function_name}(...):
            # Helper function for auto-installation
            def _install_package(package_name):
                import subprocess
                import sys
                try:
                    __import__(package_name)
                except ImportError:
                    print(f"Package '{{package_name}}' not found. Installing...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"Package '{{package_name}}' installed successfully.")

            # Auto-install and import packages
            _install_package("h2o")
            import h2o
            from h2o.automl import H2OAutoML

            _install_package("pandas")
            import pandas as pd

            import json
            import subprocess
            import sys

            # ... MLflow setup ...

            # Convert data
            if isinstance(data_raw, dict):
                df = pd.DataFrame(data_raw)
            else:
                df = data_raw.copy()

            # Validate target
            if target is None:
                raise ValueError("target parameter is required")
            if target not in df.columns:
                raise ValueError(f"Target variable '{{target}}' not found")

            # Data cleaning BEFORE converting to H2OFrame (prevents dtype errors)
            # Remove ID columns (columns with all unique values)
            id_columns = [col for col in df.columns if df[col].nunique() == len(df) and col != target]
            if id_columns:
                df = df.drop(columns=id_columns)

            # Remove columns with all NaN values (prevents dtype errors when converting to H2OFrame)
            nan_columns = [col for col in df.columns if df[col].isna().all()]
            if nan_columns:
                df = df.drop(columns=nan_columns)

            # Handle missing values in target
            if df[target].isna().any():
                df = df.dropna(subset=[target])

            # Reset index to ensure clean DataFrame
            df = df.reset_index(drop=True)

            # Detect problem type
            target_col = df[target]
            num_unique = target_col.nunique()
            is_classification = target_col.dtype == 'object' or target_col.dtype.name == 'category' or (num_unique < 20 and num_unique / len(df) < 0.5)

            # Adjust parameters based on problem type
            if is_classification:
                if stopping_metric is None:
                    stopping_metric = 'logloss'
                if sort_metric is None:
                    sort_metric = 'AUC' if num_unique == 2 else 'mean_per_class_error'
                if balance_classes is None:
                    balance_classes = True
            else:
                balance_classes = False
                if stopping_metric is None or stopping_metric == 'logloss':
                    stopping_metric = 'RMSE'
                if sort_metric is None or sort_metric == 'logloss':
                    sort_metric = 'RMSE'

            # Initialize H2O
            h2o.init()

            # Create H2OFrame - data is already cleaned above
            # DO NOT specify dtype parameter - let H2O handle type inference
            data_h2o = h2o.H2OFrame(df)

            # Convert target to factor for classification
            if is_classification:
                data_h2o[target] = data_h2o[target].asfactor()

            # ... setup AutoML and train ...

            # IMPORTANT: When converting leaderboard H2OFrame to pandas DataFrame:
            # ALWAYS use .as_data_frame() method, NEVER use pd.DataFrame() constructor
            # CORRECT (returns pandas DataFrame directly):
            leaderboard_df = aml.leaderboard.as_data_frame()
            leaderboard_dict = leaderboard_df.to_dict()
            # WRONG (causes "dtype is only supported for one column frames" error):
            # leaderboard_df = pd.DataFrame(aml.leaderboard)
            # leaderboard_df = pd.DataFrame(aml.leaderboard, dtype='float64')
            # leaderboard_df = pd.DataFrame(aml.leaderboard.as_data_frame(), dtype='float64')

            # ... rest of the H2O code ...
        """
        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="h2o_train_function",
            error_key="h2o_train_error",
            llm=llm,
            prompt_template=fix_prompt,
            agent_name=AGENT_NAME,
            file_path=state.get("h2o_train_function_path"),
            function_name=state.get("h2o_train_function_name"),
            log=log
        )

    # 5) Final reporting node
    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "h2o_train_function",
                "h2o_train_function_path",
                "h2o_train_function_name",
                "h2o_train_error",
                "model_path",
                "best_model_id",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="H2O Machine Learning Agent Outputs"
        )

    node_functions = {
        "recommend_ml_steps": recommend_ml_steps,
        "human_review": human_review,
        "create_h2o_code": create_h2o_code,
        "execute_h2o_code": execute_h2o_code,
        "fix_h2o_code": fix_h2o_code,
        "report_agent_outputs": report_agent_outputs,
    }

    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_ml_steps",
        create_code_node_name="create_h2o_code",
        execute_code_node_name="execute_h2o_code",
        fix_code_node_name="fix_h2o_code",
        explain_code_node_name="report_agent_outputs",
        error_key="h2o_train_error",
        max_retries_key="max_retries",
        retry_count_key="retry_count",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",  
        checkpointer=checkpointer,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
        agent_name=AGENT_NAME,
    )

    return app

