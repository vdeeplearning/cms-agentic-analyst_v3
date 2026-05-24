# Agentic CMS Data Analyst

A web-based AI Data Analyst dashboard built in Python using Streamlit and OpenAI. The application automatically downloads and merges Centers for Medicare & Medicaid Services (CMS) hospital data, cleans the datasets, and uses an AI agent to answer complex statistical questions by writing and executing Python/Pandas code in real-time.

### Deployed Web Application
🔗 **Render URL:** [https://cms-agentic-analyst.onrender.com](https://cms-agentic-analyst.onrender.com) (or [https://cms-agentic-analyst-v3.onrender.com](https://cms-agentic-analyst-v3.onrender.com))

### Screenshots

#### 🤖 AI Data Agent & Chat Interface
![AI Data Agent Chat](assets/agent_chat_screenshot.png)

#### 📊 Key Insights & Visualizations Dashboard
![Insights Dashboard](assets/insights_dashboard_screenshot.png)

---

## What the Agent is Doing (Step-by-Step Workflow)

The agent behaves like a human data analyst in a sandbox environment. Below are the steps executed whenever you ask a question:

```
[ User Input ] 
       │
       ▼
1. System Context Injection (Inject dataset schemas to LLM)
       │
       ▼
2. Reasoning & Planning (LLM plans how to compute the answer)
       │
       ▼
3. Tool Execution Request (LLM requests to execute Python code)
       │
       ▼
4. Local Python Execution (App executes Pandas query on preloaded dfs)
       │
       ▼
5. Output Capture & Loop (Stdout returned to LLM; LLM iterates if needed)
       │
       ▼
6. Final Response Formatting (LLM outputs markdown tables/charts & text)
```

1. **System Context & Schema Loading**: Upon startup, the application downloads and cleans two datasets: the *Hospital Readmissions Reduction Program (HRRP)* data and the *Hospital General Information* data. It joins them into a single enriched DataFrame `merged_df` containing county, state, rating, ownership model, and readmission metrics.
2. **User Question Reception**: When you submit a question (e.g., *"Which 10 counties have the highest readmission rates?"*), the message is appended to the session history.
3. **Agent Reasoning**: The agent (powered by OpenAI's `gpt-4o-mini` or `gpt-4o`) analyzes the question against the pre-loaded DataFrame columns and structures a Pandas query.
4. **Tool Call Execution**: The agent triggers a custom tool named `execute_pandas_query` containing the Python script it wrote.
5. **Local Sandboxed Execution**: The application intercepts the tool call, displays the code in the Streamlit UI, and executes the script locally on the server. The execution environment has isolated access to `merged_df`, `hrrp_df`, and `info_df`.
6. **Stdout Capture**: The execution captures standard output (stdout) and forwards it back to the agent.
7. **Synthesis & Response**: The agent reads the results, formats them into a clean markdown explanation (often with sorted lists, tables, or averages), and presents the final answer to the user.

---

## About the CMS Datasets

The application programmatically downloads, caches, cleans, and merges two distinct public datasets from the Centers for Medicare & Medicaid Services (CMS) Provider Data Catalog:

### 1. Hospital Readmissions Reduction Program (HRRP) Dataset
*   **What it is**: Clinical performance metrics tracking **30-day risk-standardized unplanned readmission rates** for Medicare beneficiaries. The program penalizes hospitals with higher-than-expected readmissions.
*   **Conditions Covered**: Tracks 6 specific conditions and procedures:
    *   **AMI**: Acute Myocardial Infarction (Heart Attack)
    *   **CABG**: Coronary Artery Bypass Graft (Heart Bypass Surgery)
    *   **COPD**: Chronic Obstructive Pulmonary Disease
    *   **HF**: Heart Failure
    *   **THA/TKA (HIP-KNEE)**: Elective Primary Total Hip Arthroplasty and/or Total Knee Arthroplasty (Joint Replacement)
    *   **PN**: Pneumonia
*   **Dataset Size**: **18,330 records** (rows) and **12 columns**. Each hospital has a separate row for each of the 6 measures they are eligible for.
*   **Key Fields**: `Facility ID`, `Number of Discharges`, `Predicted Readmission Rate`, `Expected Readmission Rate`, and `Excess Readmission Ratio` (where values > 1.0 indicate readmissions higher than expected and trigger penalties).

### 2. Hospital General Information Dataset
*   **What it is**: Administrative directory and overall performance metadata for all Medicare-certified hospitals in the United States.
*   **Dataset Size**: **5,432 hospitals** (rows) and **38 columns**.
*   **Data Included (Beyond Readmissions)**:
    *   **Geographic Metadata**: Street Address, City, State, ZIP Code, and **County/Parish**.
    *   **Administrative Classification**: Hospital Type (e.g., *Acute Care*, *Critical Access*) and Hospital Ownership (e.g., *Proprietary*, *Voluntary non-profit*, *Government*).
    *   **Emergency Services**: Indicating whether the hospital offers emergency services (`Yes`/`No`).
    *   **Performance Star Ratings**: The overall rating (1 to 5 stars) summarizing mortality, safety, readmission, patient experience, and timely/effective care metrics.

### 3. Combined Merged Dataset
*   The application merges the HRRP readmissions dataset with selective columns from the Hospital General Information dataset (e.g., County/Parish, Hospital Type, Hospital Ownership, Emergency Services, and Star Rating).
*   This results in a final merged dataset of **18,330 rows and 20 columns** pre-loaded in memory for the agent to query, enabling complex statistical correlations between readmission rates, ownership models, state/county geography, and overall ratings.

---

## Dataset Schema Reference (Available to the Agent)

The AI Data Agent queries the pre-loaded pandas DataFrames using the following unified schema.

### 1. The Merged Dataset (`merged_df`)
This DataFrame is the primary dataset containing the full joint readmission statistics and administrative metadata. It contains **20 fields**:

| # | Field Name | Data Type | Source Dataset | Description / Explanation |
|---|------------|-----------|----------------|---------------------------|
| 1 | `Facility Name` | String | HRRP | The official name of the hospital. |
| 2 | `Facility ID` | String | HRRP | The unique 6-character provider identifier, padded with leading zeros (e.g., `010001`). Used as the join key. |
| 3 | `State` | String | HRRP | The US state abbreviation where the hospital is located. |
| 4 | `Measure Name` | String | HRRP | The specific readmission condition being monitored. The six conditions are: <br>• `READM-30-AMI-HRRP` (Heart Attack)<br>• `READM-30-CABG-HRRP` (Heart Bypass Surgery)<br>• `READM-30-COPD-HRRP` (Pulmonary Disease)<br>• `READM-30-HF-HRRP` (Heart Failure)<br>• `READM-30-HIP-KNEE-HRRP` (Hip/Knee Joint Replacement)<br>• `READM-30-PN-HRRP` (Pneumonia) |
| 5 | `Number of Discharges` | Numeric | HRRP | The cohort size, representing the count of eligible discharges for that measure during the 3-year cohort. Can be `NaN` if too few cases to report. |
| 6 | `Footnote` | String | HRRP | Text codes indicating why data might be missing, excluded, or adjusted for that specific hospital/measure. |
| 7 | `Excess Readmission Ratio` | Numeric | HRRP | The standard penalty metric: the ratio of the hospital's predicted readmissions to its expected readmissions. Ratios **greater than 1.0** indicate higher readmissions than expected and incur Medicare penalties. |
| 8 | `Predicted Readmission Rate` | Numeric | HRRP | The risk-standardized rate percentage of unplanned readmissions within 30 days of discharge. |
| 9 | `Expected Readmission Rate` | Numeric | HRRP | The rate percentage of readmissions expected if the hospital's patient mix (age, comorbidities, sex) were treated at an average US hospital. |
| 10 | `Number of Readmissions` | Numeric | HRRP | The count of actual unplanned readmissions that occurred within 30 days of discharge. Can be `NaN` if count is too low to report. |
| 11 | `Start Date` | String | HRRP | Cohort monitoring start date (`07/01/2021` for the current PY 2026 data release). |
| 12 | `End Date` | String | HRRP | Cohort monitoring end date (`06/30/2024` for the current PY 2026 data release). |
| 13 | `Address` | String | General Info | The physical street address of the hospital. |
| 14 | `City/Town` | String | General Info | The city or town of the hospital. |
| 15 | `ZIP Code` | String | General Info | The 5-digit ZIP code. |
| 16 | `County/Parish` | String | General Info | The county or parish name. **Note:** Group by both `County/Parish` and `State` to avoid merging counties with identical names in different states. |
| 17 | `Hospital Type` | String | General Info | Classification of the hospital (e.g., `Acute Care Hospitals`, `Critical Access Hospitals`, `Children's`). |
| 18 | `Hospital Ownership` | String | General Info | The ownership model (e.g., `Proprietary` (for-profit), `Voluntary non-profit`, `Government - Hospital District or Authority`). |
| 19 | `Emergency Services` | String | General Info | Flag indicating whether the hospital provides 24/7 emergency services (`Yes` or `No`). |
| 20 | `Hospital overall rating` | Numeric | General Info | The overall quality score of the hospital (from 1 to 5 stars), summarizing metrics across safety, mortality, readmissions, and patient experience. |

### 2. The Raw HRRP Dataset (`hrrp_df`)
Contains the original 18,330 rows representing the 30-day readmissions data with the 12 fields (fields 1-12 in the table above).

### 3. The Raw General Information Dataset (`info_df`)
Contains the original 5,432 rows representing all certified hospitals in the US with all 38 columns, including telephone numbers, rating footnotes, and detailed group measure counts (mortality, safety, patient experience, timely care).

---

---

## Local Setup & Run

### Prerequisites
- Python 3.11 or later
- An OpenAI API Key

### Instructions
1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd cms-agentic-analyst_v3
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Data Preprocessing (Optional - App runs it automatically on launch):**
   ```bash
   python data_loader.py
   ```

5. **Start the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser to `http://localhost:8501`.

6. **Input API Key:** Open the left sidebar and input your OpenAI API Key.

---

## Deploying to Render

This repository is pre-configured with a Render blueprint definition (`render.yaml`).

### Setup Steps
1. Push this project to your GitHub repository.
2. Log in to [Render](https://render.com) and go to the **Blueprints** tab.
3. Click **New Blueprint Instance** and connect your GitHub repository.
4. Render will automatically read `render.yaml` and set up the Python environment, download the dependencies, and start the Streamlit server on the assigned port.
5. In Render's environment settings for the service, you can optionally set a default `OPENAI_API_KEY` so users don't have to input it in the GUI.
