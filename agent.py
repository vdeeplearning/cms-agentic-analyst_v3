import sys
import io
import json
import traceback
import pandas as pd
import numpy as np
import streamlit as st

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

CRITICAL INSTRUCTIONS:
- You must answer the user's questions by writing Python code that queries these DataFrames.
- Use the tool `execute_pandas_query` to run your Python code.
- Always use `print()` inside your Python code to output the results. Only stdout is captured and returned to you. For example, if you calculate a mean, write `print(df['column'].mean())`.
- If the output contains NaN values, handle them appropriately (e.g. use `dropna()`, ignore them, or print count of valid values).
- When answering questions about counties, remember that counties with the same name can exist in different states (e.g. Washington County). Group by both `County/Parish` and `State` to avoid mixing data across states!
- After receiving the execution output, formulate a clear, precise, and user-friendly explanation of the results. Mention the exact numbers, statistics, and any interesting findings.
- **DIRECT PLOTTING / CHARTS**: You can plot charts directly in the Streamlit UI! If the user asks for a chart, visualization, or plot, write Python code that uses Streamlit's native visualization methods like:
  - `st.bar_chart(df)` or `st.line_chart(df)`
  - Matplotlib plots (make sure to call `st.pyplot(fig)`):
    ```python
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    # ... plot details ...
    st.pyplot(fig)
    ```
- **DATASET TIME LIMITATION**: The CMS HRRP dataset represents a static, aggregated 3-year cohort window (Start Date: `07/01/2021`, End Date: `06/30/2024`) for each hospital/measure. It does **not** contain monthly or annual historical trend intervals. If the user asks for historical trends over months/years, check the unique dates first, explain this limitation, and offer to plot a cross-sectional chart (e.g., readmissions by measure, ownership, or state) instead!
- IMPORTANT: Double-check the user's sorting/filtering request. If they ask for the lowest readmission rates, ensure you query for the lowest values (e.g., sort in ascending order or get the bottom values). Do not assume or copy-paste past answers for different questions.
- IMPORTANT: Every question in the conversation is independent. Do NOT repeat or copy the Python code, queries, or answers from previous turns unless explicitly requested by the user. Always write a fresh Pandas query that specifically targets the columns and constraints requested in the current question.
- Do not write code that performs malicious actions (e.g., trying to write to files, accessing environment variables, or importing OS). Only use standard libraries like pandas, numpy, and python built-ins.
"""

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
    # Create execution namespace
    local_vars = {
        'pd': pd,
        'np': np,
        'st': st,
        **dataframes
    }
    
    # Redirect stdout
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    try:
        # Compile and execute
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
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
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
                
                # Append tool response to message history
                tool_resp = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "execute_pandas_query",
                    "content": output
                }
                messages.append(tool_resp)
                new_messages.append(tool_resp)
                
    # Yield the list of new messages so the host app can update its API-compliant history
    yield {"type": "new_messages", "content": new_messages}
