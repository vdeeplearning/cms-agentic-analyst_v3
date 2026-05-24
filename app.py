import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from openai import OpenAI
from dotenv import load_dotenv

import data_loader
import agent

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Agentic CMS Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom Google Fonts and custom CSS for high-end aesthetics
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* Typography overrides */
html, body, .stApp, .stMarkdown, .stText, p, li, span, label, button, input, select, textarea {
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}

/* Custom Gradient Title */
.title-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
}

.gradient-title {
    background: linear-gradient(135deg, #4D96FF 0%, #6BCB77 50%, #FF6B6B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    padding: 0;
    letter-spacing: -0.5px;
}

.subtitle {
    color: #94A3B8;
    font-size: 1.1rem;
    margin-top: 8px;
    font-weight: 400;
}

/* Metric Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(77, 150, 255, 0.4);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.metric-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #4D96FF;
    margin: 5px 0;
}

.metric-label {
    font-size: 0.85rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Agent thought container */
.step-container {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid #6BCB77;
    border-radius: 4px 12px 12px 4px;
    padding: 15px;
    margin: 10px 0;
}

.step-header {
    font-weight: 600;
    color: #6BCB77;
    font-size: 0.95rem;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.code-header {
    color: #4D96FF;
    border-left-color: #4D96FF;
}

.output-header {
    color: #E2E8F0;
    border-left-color: #94A3B8;
    background: #0F172A;
    font-family: monospace;
    padding: 10px;
    border-radius: 4px;
    font-size: 0.85rem;
}

/* Banner style */
.banner {
    background: linear-gradient(135deg, rgba(77, 150, 255, 0.1) 0%, rgba(107, 203, 119, 0.1) 100%);
    border: 1px solid rgba(77, 150, 255, 0.2);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ----------------- DATA CACHING -----------------
@st.cache_data(show_spinner="Loading and preprocessing CMS datasets...")
def get_cms_data():
    return data_loader.load_data()

try:
    merged_df, hrrp_df, info_df = get_cms_data()
except Exception as e:
    st.error(f"Error loading CMS data: {e}")
    st.info("Check internet connection or run `python data_loader.py` to fix details.")
    st.stop()

# ----------------- SIDEBAR CONFIG -----------------
st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>⚙️ Configuration</h2>", unsafe_allow_html=True)

# OpenAI API Key setup
api_key_input = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY", "")),
    help="Enter your OpenAI API Key. This will be used to invoke the AI Data Agent. We do not store your key."
)

if api_key_input:
    st.session_state["openai_api_key"] = api_key_input

# Model Selection
model_option = st.sidebar.selectbox(
    "Agent Model",
    ["gpt-4o-mini", "gpt-4o"],
    index=0,
    help="gpt-4o-mini is faster and cost-efficient. gpt-4o is more intelligent for complex reasoning."
)

# Dataset Summary Metrics in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Overview")
st.sidebar.markdown(f"**Hospitals Included:** `{info_df['Facility ID'].nunique():,}`")
st.sidebar.markdown(f"**Readmission Records:** `{hrrp_df.shape[0]:,}`")
st.sidebar.markdown(f"**Preprocessed Rows:** `{merged_df.shape[0]:,}`")

# Pre-baked Questions Helper
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Sample Questions")
pre_baked_questions = [
    "What is the mean national readmission to hospital rate?",
    "Which 10 counties have the highest rates of readmission?",
    "Show a state-by-state comparison of readmissions (average Predicted Readmission Rate).",
    "Which hospital ownership type has the highest average excess readmission ratio?",
    "Compare readmission rates of government-owned hospitals vs proprietary hospitals.",
]

for q in pre_baked_questions:
    if st.sidebar.button(q, key=f"btn_{q}"):
        st.session_state["clicked_question"] = q
        st.rerun()

# ----------------- MAIN PANEL HEADER -----------------
st.markdown("""
<div class="title-container">
    <h1 class="gradient-title">Agentic CMS Data Analyst</h1>
    <div class="subtitle">An AI data agent that downloads Centers for Medicare & Medicaid Services (CMS) readmission metrics and runs local Python queries to answer statistical questions.</div>
</div>
""", unsafe_allow_html=True)

# ----------------- METRICS -----------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">National Avg Readmission Rate</div>
        <div class="metric-val">{merged_df['Predicted Readmission Rate'].mean():.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Hospitals Monitored</div>
        <div class="metric-val">{merged_df['Facility ID'].nunique():,}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Max Excess Ratio</div>
        <div class="metric-val">{merged_df['Excess Readmission Ratio'].max():.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">States Tracked</div>
        <div class="metric-val">{merged_df['State'].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# ----------------- TABS SETUP -----------------
tab_chat, tab_explorer, tab_viz = st.tabs([
    "🤖 AI Data Agent", 
    "🔍 Interactive Data Explorer", 
    "📊 Key Insights & Visualizations"
])

# ----------------- TAB 1: AGENT CHAT -----------------
with tab_chat:
    st.markdown("""
    <div class="banner">
        💡 <b>How it works:</b> Ask any analytical or statistical question about hospital readmissions. The agent will formulate a plan, write Python Pandas code, execute it locally on the server, analyze the output, and present a final response. 
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize message list in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hello! I am your AI CMS Data Analyst. I have preloaded the CMS Hospital Readmissions dataset. You can ask me questions like: \n- *What is the mean national readmission rate?*\n- *Which 10 counties have the highest rates of readmission?*\n- *Is there a correlation between hospital rating and excess readmission ratio?*",
                "steps": []
            }
        ]
        
    if "openai_messages" not in st.session_state:
        st.session_state["openai_messages"] = [
            {
                "role": "assistant",
                "content": st.session_state["messages"][0]["content"]
            }
        ]

    # Render previous conversation
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            # Display any intermediate execution steps
            if "steps" in msg and msg["steps"]:
                with st.expander("🛠️ View Agent Execution Details (" + str(len(msg["steps"]) // 2) + " steps)", expanded=False):
                    for step in msg["steps"]:
                        if step["type"] == "thought":
                            st.markdown(f"**Reasoning:** {step['content']}")
                        elif step["type"] == "code_execution":
                            st.markdown("**💻 Executing Pandas Query:**")
                            st.code(step["content"], language="python")
                        elif step["type"] == "code_output":
                            st.markdown("**📊 Query Output (stdout):**")
                            st.text(step["content"])
                        elif step["type"] == "error":
                            st.error(f"Error: {step['content']}")
            
            # Display final message text
            st.markdown(msg["content"])

    # Add a spacer at the bottom of the conversation to prevent overlap with the fixed chat input widget
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

    # Check for pre-baked clicks
    user_query = ""
    if "clicked_question" in st.session_state:
        user_query = st.session_state.pop("clicked_question")
    
    # Wait for chat input if not pre-baked
    chat_input = st.chat_input("Ask a statistical question about the dataset...")
    if chat_input:
        user_query = chat_input

    # Execute Agent Loop
    if user_query:
        # Display user message in chat immediately
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state["messages"].append({"role": "user", "content": user_query, "steps": []})
        st.session_state["openai_messages"].append({"role": "user", "content": user_query})
        
        # Check API Key
        api_key = st.session_state.get("openai_api_key", "").strip()
        if not api_key:
            with st.chat_message("assistant"):
                st.error("🔑 OpenAI API Key is missing. Please enter your OpenAI API key in the sidebar configuration to run the agent.")
            st.session_state["messages"].append({
                "role": "assistant", 
                "content": "API Key missing error.", 
                "steps": []
            })
        else:
            # Run LLM agent
            with st.chat_message("assistant"):
                # Containers for real-time status updates
                status_box = st.status("🤖 Agent initializing...", expanded=True)
                final_answer_container = st.empty()
                
                # Client initialization
                client = OpenAI(api_key=api_key)
                
                # We'll save steps to append to the message history
                current_steps = []
                final_answer = ""
                
                # Preload context
                df_context = {
                    "merged_df": merged_df,
                    "hrrp_df": hrrp_df,
                    "info_df": info_df
                }
                
                # Execute agent generator loop
                for step in agent.run_agent_loop(
                    client=client, 
                    openai_messages=st.session_state["openai_messages"], 
                    dataframes=df_context,
                    model=model_option
                ):
                    step_type = step["type"]
                    content = step["content"]
                    
                    if step_type == "thought":
                        status_box.update(label="Thinking...", state="running")
                        # Add a visual note to the expander
                        status_box.write(f"💭 **Thought:** {content}")
                        current_steps.append({"type": "thought", "content": content})
                        
                    elif step_type == "code_execution":
                        status_box.update(label="💻 Running local Python/Pandas query...", state="running")
                        status_box.code(content, language="python")
                        current_steps.append({"type": "code_execution", "content": content})
                        
                    elif step_type == "code_output":
                        status_box.write("**Output:**")
                        status_box.text(content)
                        current_steps.append({"type": "code_output", "content": content})
                        
                    elif step_type == "error":
                        status_box.write(f"❌ **Error:** {content}")
                        current_steps.append({"type": "error", "content": content})
                        
                    elif step_type == "new_messages":
                        st.session_state["openai_messages"].extend(content)
                        
                    elif step_type == "final_answer":
                        final_answer = content
                
                # Finish status container
                if final_answer:
                    status_box.update(label="Execution Complete!", state="complete", expanded=False)
                    final_answer_container.markdown(final_answer)
                    
                    # Store response in session state
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": final_answer,
                        "steps": current_steps
                    })
                else:
                    status_box.update(label="Execution failed or timed out.", state="error", expanded=True)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": "I encountered an error during analysis. Please check the logs in the step execution container.",
                        "steps": current_steps
                    })
            
            st.rerun()

# ----------------- TAB 2: DATA EXPLORER -----------------
with tab_explorer:
    st.markdown("### 🔍 Interactive CMS Dataset Explorer")
    st.markdown("Use the filters below to slice and search the preprocessed dataset.")
    
    # Filter Row
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    
    with f_col1:
        states = ["All"] + sorted(merged_df['State'].dropna().unique().tolist())
        sel_state = st.selectbox("State Filter", states)
        
    with f_col2:
        measures = ["All"] + sorted(merged_df['Measure Name'].dropna().unique().tolist())
        sel_measure = st.selectbox("Readmission Measure", measures)
        
    with f_col3:
        ownerships = ["All"] + sorted(merged_df['Hospital Ownership'].dropna().unique().tolist())
        sel_ownership = st.selectbox("Hospital Ownership", ownerships)
        
    with f_col4:
        types = ["All"] + sorted(merged_df['Hospital Type'].dropna().unique().tolist())
        sel_type = st.selectbox("Hospital Type", types)
        
    # Apply filters
    filtered_df = merged_df.copy()
    if sel_state != "All":
        filtered_df = filtered_df[filtered_df['State'] == sel_state]
    if sel_measure != "All":
        filtered_df = filtered_df[filtered_df['Measure Name'] == sel_measure]
    if sel_ownership != "All":
        filtered_df = filtered_df[filtered_df['Hospital Ownership'] == sel_ownership]
    if sel_type != "All":
        filtered_df = filtered_df[filtered_df['Hospital Type'] == sel_type]
        
    # Search input for Hospital Name or County
    search_term = st.text_input("🔍 Search by Hospital Name or County", "")
    if search_term:
        filtered_df = filtered_df[
            filtered_df['Facility Name'].astype(str).str.contains(search_term, case=False) |
            filtered_df['County/Parish'].astype(str).str.contains(search_term, case=False)
        ]
        
    # Pagination/Rows count
    st.markdown(f"Showing **{filtered_df.shape[0]:,}** matching records:")
    
    # Display table (only showing relevant columns to avoid clutter)
    display_cols = [
        'Facility ID', 'Facility Name', 'State', 'County/Parish', 
        'Measure Name', 'Number of Discharges', 'Predicted Readmission Rate', 
        'Expected Readmission Rate', 'Excess Readmission Ratio', 'Hospital overall rating', 'Hospital Ownership'
    ]
    st.dataframe(
        filtered_df[[c for c in display_cols if c in filtered_df.columns]],
        use_container_width=True,
        hide_index=True
    )
    
    # CSV download
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_cms_readmission_data.csv",
        mime="text/csv"
    )

# ----------------- TAB 3: VISUALIZATIONS -----------------
with tab_viz:
    st.markdown("### 📊 Automated Visualizations")
    st.markdown("National aggregates and data trends compiled dynamically from the preprocessed CMS data.")
    
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        # Chart 1: Mean Predicted Readmission Rate by Measure
        st.markdown("#### Mean Predicted Readmission Rate by Measure")
        measure_means = merged_df.groupby('Measure Name')['Predicted Readmission Rate'].mean().reset_index()
        
        chart_measure = alt.Chart(measure_means).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X('Measure Name:N', sort='-y', axis=alt.Axis(labelAngle=-45, title="Readmission Measure")),
            y=alt.Y('Predicted Readmission Rate:Q', title="Avg Readmission Rate (%)"),
            color=alt.Color('Measure Name:N', legend=None, scale=alt.Scale(scheme='bluepurple')),
            tooltip=['Measure Name', alt.Tooltip('Predicted Readmission Rate:Q', format='.2f')]
        ).properties(height=350)
        
        st.altair_chart(chart_measure, use_container_width=True)
        
    with v_col2:
        # Chart 2: Excess Readmission Ratio Distribution
        st.markdown("#### Distribution of Excess Readmission Ratio")
        # We group by hospital to avoid duplicating values since we have multiple measures per hospital
        hospital_ratios = merged_df.groupby('Facility ID')['Excess Readmission Ratio'].mean().reset_index()
        hospital_ratios = hospital_ratios.dropna()
        
        # Create bins for hist
        hist_values, bin_edges = np.histogram(hospital_ratios['Excess Readmission Ratio'], bins=40, range=(0.6, 1.4))
        hist_df = pd.DataFrame({
            'Excess Readmission Ratio Bin': bin_edges[:-1],
            'Count': hist_values
        })
        
        # Standard threshold is 1.0. We want to show a line indicating penalty threshold
        chart_dist = alt.Chart(hist_df).mark_area(
            opacity=0.6,
            color='#FF6B6B'
        ).encode(
            x=alt.X('Excess Readmission Ratio Bin:Q', title="Avg Excess Readmission Ratio (Standardized)"),
            y=alt.Y('Count:Q', title="Count of Hospitals"),
            tooltip=['Excess Readmission Ratio Bin', 'Count']
        ).properties(height=350)
        
        # Add penalty line at 1.0
        rule = alt.Chart(pd.DataFrame({'x': [1.0]})).mark_rule(color='#6BCB77', strokeWidth=2, strokeDash=[4, 4]).encode(x='x:Q')
        
        st.altair_chart(chart_dist + rule, use_container_width=True)
        st.caption("🟢 Green dotted line represents the Neutral Baseline (1.0). Ratio > 1.0 leads to penalties under HRRP.")

    st.markdown("---")
    
    v_col3, v_col4 = st.columns(2)
    
    with v_col3:
        # Chart 3: Mean Predicted Readmission Rate by Hospital Ownership
        st.markdown("#### Mean Excess Readmission Ratio by Hospital Ownership")
        ownership_stats = merged_df.groupby('Hospital Ownership')['Excess Readmission Ratio'].agg(['mean', 'count']).reset_index()
        ownership_stats = ownership_stats.dropna().sort_values(by='mean')
        
        chart_ownership = alt.Chart(ownership_stats).mark_bar().encode(
            y=alt.Y('Hospital Ownership:N', sort='-x', title="Hospital Ownership Type"),
            x=alt.X('mean:Q', title="Mean Excess Readmission Ratio", scale=alt.Scale(domain=[0.9, 1.05])),
            color=alt.Color('mean:Q', scale=alt.Scale(scheme='redyellowgreen', reverse=True), legend=None),
            tooltip=['Hospital Ownership', alt.Tooltip('mean:Q', format='.4f'), 'count']
        ).properties(height=300)
        
        # Add baseline line
        rule2 = alt.Chart(pd.DataFrame({'x': [1.0]})).mark_rule(color='red', strokeWidth=1).encode(x='x:Q')
        
        st.altair_chart(chart_ownership + rule2, use_container_width=True)
        st.caption("A value above 1.0 indicates a higher rate of readmissions than expected, which generally incurs penalties.")
        
    with v_col4:
        # Chart 4: Top 10 States with Highest Readmission Rates
        st.markdown("#### Top 10 States with Highest Mean Predicted Readmission Rate")
        state_means = merged_df.groupby('State')['Predicted Readmission Rate'].mean().reset_index()
        state_means = state_means.sort_values(by='Predicted Readmission Rate', ascending=False).head(10)
        
        chart_states = alt.Chart(state_means).mark_bar(color='#4D96FF').encode(
            x=alt.X('Predicted Readmission Rate:Q', title="Mean Readmission Rate (%)", scale=alt.Scale(domain=[13, 17])),
            y=alt.Y('State:N', sort='-x', title="State"),
            tooltip=['State', alt.Tooltip('Predicted Readmission Rate:Q', format='.2f')]
        ).properties(height=300)
        
        st.altair_chart(chart_states, use_container_width=True)
