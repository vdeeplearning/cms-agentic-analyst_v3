import sys
import io
import json
import traceback
import base64
import threading
import ast
import pandas as pd
import numpy as np
import streamlit as st

# Global lock to make matplotlib plotting thread-safe for concurrent users
EXECUTION_LOCK = threading.Lock()

# Force Agg backend to prevent GUI thread issues in server environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# System prompt explaining the context and the available DataFrames
SYSTEM_PROMPT = """You are an expert healthcare data analyst AI agent.
Your goal is to answer statistical and analytical questions about the CMS Hospital Readmissions Reduction Program (HRRP) dataset.

You have access to the following pre-loaded pandas DataFrames:
1. `merged_df`: The merged dataset containing both HRRP data and Hospital General Information (joined on 'Facility ID'). Use this for county/ownership/hospital-type analysis.
2. `hrrp_df`: The raw Hospital Readmissions Reduction Program dataset.
3. `info_df`: The raw Hospital General Information dataset.

Here are the key columns and descriptions in `merged_df`:
- 'Facility ID': unique 6-digit identifier of the facility (string).
- 'Facility Name': name of the hospital (string).
- 'State': US state of the hospital (string).
- 'Measure Name': readmission measure. The six measures are:
  - 'READM-30-AMI-HRRP': Acute Myocardial Infarction (Heart Attack) 30-Day Readmission
  - 'READM-30-CABG-HRRP': Coronary Artery Bypass Graft 30-Day Readmission
  - 'READM-30-COPD-HRRP': Chronic Obstructive Pulmonary Disease 30-Day Readmission
  - 'READM-30-HF-HRRP': Heart Failure 30-Day Readmission
  - 'READM-30-HIP-KNEE-HRRP': Total Hip/Knee Arthroplasty 30-Day Readmission
  - 'READM-30-PN-HRRP': Pneumonia 30-Day Readmission
- 'Number of Discharges': number of eligible discharges for the measure (float, can be NaN).
- 'Excess Readmission Ratio': ratio of predicted to expected readmissions. > 1 means readmissions are higher than expected (penalty threshold), < 1 means readmissions are lower than expected (float, can be NaN).
- 'Predicted Readmission Rate': predicted readmission rate percentage (float, can be NaN).
- 'Expected Readmission Rate': expected readmission rate percentage (float, can be NaN).
- 'Number of Readmissions': number of readmissions (float, can be NaN).
- 'Start Date' / 'End Date': time period of the measurement (string).
- 'County/Parish': county of the hospital (string, from General Info).
- 'Hospital Type': type of hospital, e.g., 'Acute Care Hospitals', 'Critical Access Hospitals' (string, from General Info).
- 'Hospital Ownership': ownership model, e.g., 'Proprietary', 'Voluntary non-profit', 'Government - Hospital District or Authority' (string, from General Info).
- 'Emergency Services': 'Yes' or 'No' (string, from General Info).
- 'Hospital overall rating': 1 to 5 stars (float, can be NaN, from General Info).
- Additional Quality Group fields in `merged_df`:
  - Demographic directory fields: 'Address', 'City/Town', 'ZIP Code', 'Telephone Number', and 'Meets criteria for birthing friendly designation'.
  - Mortality Group: 'MORT Group Measure Count', 'Count of Facility MORT Measures', 'Count of MORT Measures Better', 'Count of MORT Measures No Different', 'Count of MORT Measures Worse', and 'MORT Group Footnote'.
  - Safety Group: 'Safety Group Measure Count', 'Count of Facility Safety Measures', 'Count of Safety Measures Better', 'Count of Safety Measures No Different', 'Count of Safety Measures Worse', and 'Safety Group Footnote'.
  - General Readmissions Group: 'READM Group Measure Count', 'Count of Facility READM Measures', 'Count of READM Measures Better', 'Count of READM Measures No Different', 'Count of READM Measures Worse', and 'READM Group Footnote'.
  - Patient Experience & Timely Care: 'Pt Exp Group Measure Count', 'Count of Facility Pt Exp Measures', 'Pt Exp Group Footnote', 'TE Group Measure Count', 'Count of Facility TE Measures', and 'TE Group Footnote'.

CRITICAL INSTRUCTIONS:
- You must answer the user's questions by writing Python code that queries these DataFrames.
- Use the tool `execute_pandas_query` to run your Python code.
- Always use `print()` inside your Python code to output the results. Only stdout is captured and returned to you. For example, if you calculate a mean, write `print(df['column'].mean())`.
- If the output contains NaN values, handle them appropriately (e.g. use `dropna()`, ignore them, or print count of valid values).
- When answering questions about counties, remember that counties with the same name can exist in different states (e.g. Washington County). Group by both `County/Parish` and `State` to avoid mixing data across states!
- After receiving the execution output, formulate a clear, precise, and user-friendly explanation of the results. Mention the exact numbers, statistics, and any interesting findings.
- **DIRECT PLOTTING / CHARTS (MATPLOTLIB)**: If the user asks for a chart, visualization, or plot, write standard matplotlib code to draw the chart and call `plt.show()`. The execution environment overrides `plt.show` to automatically capture the active figure, convert it to a base64-encoded PNG, and print it to stdout as an HTML image tag (e.g. `<img src="data:image/png;base64,..." width="100%"/>`).
  - Do NOT copy-paste the base64 code block or image tag into your response. The system will automatically detect the HTML image tag from your code output, extract it, and render the chart for the user. Simply write your natural language analysis.
  - Example:
    ```python
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    # ... plot details ...
    plt.title("Average Readmission Rates")
    plt.show() # Automatically captures and prints the HTML image tag
    ```
- **DATASET TIME LIMITATION**: The CMS HRRP dataset represents a static, aggregated 3-year cohort window (Start Date: `07/01/2021`, End Date: `06/30/2024`) for each hospital/measure. It does **not** contain monthly or annual historical trend intervals. If the user asks for historical trends over months/years, check the unique dates first, explain this limitation, and offer to plot a cross-sectional chart (e.g., readmissions by measure, ownership, or state) instead!
- IMPORTANT: Double-check the user's sorting/filtering request. If they ask for the lowest readmission rates, ensure you query for the lowest values (e.g., sort in ascending order or get the bottom values). Do not assume or copy-paste past answers for different questions.
- IMPORTANT: Every question in the conversation is independent. Do NOT repeat or copy the Python code, queries, or answers from previous turns unless explicitly requested by the user. Always write a fresh Pandas query that specifically targets the columns and constraints requested in the current question.
- Do not write code that performs malicious actions (e.g., trying to write to files, accessing environment variables, or importing OS). Only use standard libraries like pandas, numpy, and python built-ins.
"""

BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "pathlib",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "glob",
    "pickle",
    "importlib",
}

BLOCKED_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
}

BLOCKED_METHODS = {
    "to_csv",
    "to_excel",
    "to_json",
    "to_pickle",
    "to_sql",
    "to_parquet",
    "to_feather",
    "to_hdf",
    "savefig",
}

ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
    "math",
    "statistics",
    "re",
}


def build_schema_context(dataframes: dict) -> str:
    """Build a compact live schema summary for the LLM from loaded DataFrames."""
    sections = [
        "LIVE DATAFRAME SCHEMA CONTEXT:",
        "Use these exact DataFrame names and exact column names. If a needed field is not listed, say the dataset cannot directly answer the question.",
    ]

    for name, df in dataframes.items():
        if not isinstance(df, pd.DataFrame):
            continue
        sections.append(f"\n{name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
        for col, dtype in df.dtypes.astype(str).items():
            sections.append(f"- {col} ({dtype})")

    sections.append(
        "\nKnown limitations: the HRRP data is hospital-level / hospital-measure-level aggregate data, not patient-level claims data. "
        "It has one 3-year measurement window, not monthly or annual time-series rows. "
        "Avoid causal claims unless the analysis truly supports them."
    )
    return "\n".join(sections)


def _literal_strings(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                values.append(elt.value)
            else:
                return []
        return values
    return []


def validate_generated_code(code_str: str, dataframes: dict) -> tuple[bool, str]:
    """
    Validate LLM-generated analysis code before execution.
    This is a guardrail, not a full security sandbox.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Generated code has a syntax error: {e}"

    schema = {
        name: set(df.columns)
        for name, df in dataframes.items()
        if isinstance(df, pd.DataFrame)
    }

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported_names = []
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif node.module:
                imported_names = [node.module]

            for imported in imported_names:
                root = imported.split(".")[0]
                if root in BLOCKED_IMPORTS or imported not in ALLOWED_IMPORTS:
                    return False, f"Import '{imported}' is not allowed in generated analysis code."

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_CALLS:
                return False, f"Call to '{func.id}' is not allowed in generated analysis code."
            if isinstance(func, ast.Attribute):
                if func.attr in BLOCKED_METHODS:
                    return False, f"Method '{func.attr}' is not allowed because generated code must not write files or external outputs."
                if isinstance(func.value, ast.Name) and func.value.id == "st":
                    return False, "Streamlit UI calls are not allowed inside generated analysis code; use print() for output."

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "Dunder attribute access is not allowed in generated analysis code."

        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            df_name = node.value.id
            if df_name in schema:
                requested_cols = _literal_strings(node.slice)
                for col in requested_cols:
                    if col not in schema[df_name]:
                        return False, f"Column '{col}' does not exist in {df_name}. Use the live schema exactly."

    return True, "Code validation passed."

# Tool definition for OpenAI
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_pandas_query",
            "description": "Executes a Python code block to query the pandas DataFrames ('merged_df', 'hrrp_df', 'info_df'). The output printed to stdout is returned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Standard output (stdout) must be used via print() to return results. Do not write file-modifying or OS-level code."
                    }
                },
                "required": ["code"]
            }
        }
    }
]

def execute_pandas_query(code_str: str, dataframes: dict) -> str:
    """
    Executes a string of python code in an environment containing the provided dataframes.
    Captures stdout and returns it. If an exception occurs, returns the exception details.
    """
    is_valid, validation_message = validate_generated_code(code_str, dataframes)
    if not is_valid:
        return (
            "Generated code failed pre-execution validation and was not run.\n"
            f"Reason: {validation_message}\n"
            "Revise the code using only the available DataFrames, exact column names, safe imports, and print() for output."
        )

    # Custom show_plot function to capture matplotlib figure and print HTML image tag
    def show_plot():
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        print(f'\n<img src="data:image/png;base64,{img_str}" width="100%"/>\n')
        plt.close(fig)

    # Bind custom show_plot to plt.show
    plt.show = show_plot

    # Create execution namespace
    local_vars = {
        'pd': pd,
        'np': np,
        'plt': plt,
        'show_plot': show_plot,
        **dataframes
    }
    
    # Redirect stdout
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    try:
        # Compile and execute within a thread-safe lock to prevent matplotlib concurrency issues in multi-user environments
        with EXECUTION_LOCK:
            compiled_code = compile(code_str, '<string>', 'exec')
            exec(compiled_code, globals(), local_vars)
        sys.stdout = old_stdout
        output = redirected_output.getvalue()
        
        if not output.strip():
            return "Execution completed successfully, but there was no stdout output. Did you forget to use print() to output the results?"
        return output
    except Exception as e:
        sys.stdout = old_stdout
        # Clean up traceback to make it readable for the LLM
        tb = traceback.format_exc()
        return f"Error occurred during execution: {e}\nTraceback:\n{tb}"

def run_agent_loop(client, openai_messages, dataframes, model="gpt-4o-mini"):
    """
    Generator function that yields steps of the agent execution.
    Yields dicts with:
      - 'type': 'thought' | 'code_execution' | 'code_output' | 'error' | 'final_answer' | 'new_messages'
      - 'content': the text content, code content, or list of new messages
    """
    # System message configuration
    schema_context = build_schema_context(dataframes)
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{schema_context}"}]
    
    # Add conversation history
    for msg in openai_messages:
        if msg.get("role") == "system":
            continue
        messages.append(msg)
        
    new_messages = []
    max_iterations = 8
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=OPENAI_TOOLS,
                tool_choice="auto"
            )
        except Exception as e:
            yield {"type": "error", "content": f"OpenAI API Error: {e}"}
            return
        
        choice = response.choices[0]
        message = choice.message
        
        # Convert ChatCompletionMessage to a dict for local storage and compatibility
        msg_dict = {
            "role": "assistant",
            "content": message.content
        }
        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in message.tool_calls
            ]
            
        messages.append(msg_dict)
        new_messages.append(msg_dict)
        
        # If there is content, yield it as thought
        if message.content:
            yield {"type": "thought", "content": message.content}
            
        tool_calls = message.tool_calls
        if not tool_calls:
            # If no tool calls, this is the final answer!
            yield {"type": "final_answer", "content": message.content or ""}
            break
            
        # Process tool calls
        for tool_call in tool_calls:
            if tool_call.function.name == "execute_pandas_query":
                # Extract code
                try:
                    args = json.loads(tool_call.function.arguments)
                    code = args.get("code", "")
                except Exception as parse_err:
                    code_err = f"Failed to parse tool call arguments: {parse_err}"
                    yield {"type": "error", "content": code_err}
                    
                    tool_resp = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "execute_pandas_query",
                        "content": code_err
                    }
                    messages.append(tool_resp)
                    new_messages.append(tool_resp)
                    continue
                
                # Yield code execution step
                yield {"type": "code_execution", "content": code}
                
                # Execute query
                output = execute_pandas_query(code, dataframes)
                
                # Yield output step
                yield {"type": "code_output", "content": output}
                
                # Preemptively strip large base64 image strings from the tool response sent to the LLM
                # to prevent context window overload. The host app will extract and display the image.
                import re
                llm_output = re.sub(
                    r'<img src="data:image/png;base64,[^"]+" width="100%"/>',
                    '[Plot successfully rendered and displayed in the user interface]',
                    output
                )
                
                # Append tool response to message history
                tool_resp = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "execute_pandas_query",
                    "content": llm_output
                }
                messages.append(tool_resp)
                new_messages.append(tool_resp)
                
    # Yield the list of new messages so the host app can update its API-compliant history
    yield {"type": "new_messages", "content": new_messages}
