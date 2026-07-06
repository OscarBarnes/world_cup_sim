import streamlit as st
import pandas as pd
import arviz as az
import numpy as np

# ---------------------------------------------------------
# 0. PAGE CONFIG & DATA LOADING
# ---------------------------------------------------------
st.set_page_config(page_title="The Multiverse Cup", page_icon="🏆", layout="centered")

@st.cache_resource
def load_model_brain():
    # Using st.cache_resource stops Streamlit from re-loading the heavy trace file 
    # every single time a user clicks a button. 
    return az.from_netcdf("nb_trace.nc")

trace = load_model_brain()
shootouts_df = pd.read_csv("shootouts.csv")

# ---------------------------------------------------------
# 1. TITLE & INTRO (The Hook)
# ---------------------------------------------------------
st.title("🏆 The Multiverse Cup")
st.subheader("Simulating Football Chaos with Bayesian AI")

st.markdown("""
Football is inherently chaotic. We've all seen absolute certainties collapse into shocking 
90th-minute upsets. Standard sports statistics usually look at simple historical averages, 
but they fail to capture the sheer unpredictability of the sport. 

To solve this, I built a predictive system that doesn't just predict *a* single winner. Instead, 
it simulates **1,000 parallel universes** of a tournament to map out every possible reality.
""")

st.write("---")

# ---------------------------------------------------------
# 2. PHASE 1: Data Cleaning & Weighting
# ---------------------------------------------------------
st.header("📈 1. Teaching the AI Football History")
st.markdown("""
A friendly match against a low-ranked team three years ago shouldn't carry the same weight 
as a high-stakes continental knockout match played last month. 

To fix this, the data pipeline dynamically recalculates a **match weight** using an exponential time-decay function. 
Recent matches matter exponentially more than older ones.
""")

# Technical Deep Dive for Dad #1 (Data Analyst)
with st.expander("🛠️ View the Time-Decay Math & Logic"):
    st.markdown("""
    We apply a recency weight using an exponential half-life formula:
    $$W_t = e^{-\lambda \cdot t}$$
    Where $t$ is the days elapsed since the match, and $\lambda$ dictates how fast historical memory decays.
    """)
    # You can display a sample snippet of your cleaning code here if you want!

st.write("---")

# ---------------------------------------------------------
# 3. PHASE 2: Embracing Chaos (The Math Engine)
# ---------------------------------------------------------
st.header("🎲 2. The Engine: Upgrading From Rigid Averages")
st.markdown("""
Most basic sports models use a standard **Poisson distribution** to predict goals. Poisson models assume a 
team scores at a perfectly predictable, steady average rate. 

But football is prone to high variance—wild 0-0 gridlocks or massive 6-1 blowouts. To capture this 'chaos factor,' 
I upgraded the model to a **Negative Binomial distribution**. This introduces an overdispersion parameter 
which allows the AI to respect real-world unpredictability.
""")

with st.expander("🧠 View the Hierarchical PyMC Model Architecture"):
    st.markdown("""
    The model estimates individual attacking and defensive parameters for each team using a non-centered parameterization 
    to prevent geometric funnels during MCMC sampling.
    """)
    # You can paste a code block of your PyMC model block here later

st.write("---")

# ---------------------------------------------------------
# 4. INTERACTIVE ZONE: The Dream Matchup Engine
# ---------------------------------------------------------
st.header("🕹️ 3. Test the Multiverse Machine")
st.markdown("Select any two international sides to run a 1,000-universe simulation of a neutral-ground clash.")

# Mock list of teams extracted from your model trace
# (Later we will replace this with actual unique team names from your dataset)
available_teams = sorted(["England", "France", "Brazil", "Argentina", "DR Congo", "South Africa", "Mexico"])

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Team A", available_teams, index=0)
with col2:
    team_b = st.selectbox("Select Team B", available_teams, index=4)

# Slider for the Chaos Level
chaos_factor = st.slider("Adjust Chaos Level (Overdispersion)", min_value=0.1, max_value=2.0, value=1.0)

if st.button("🚀 Run Match Simulation"):
    st.write(f"### Running 1,000 simulations: {team_a} vs {team_b}...")
    
    # Placeholder for where your match simulation logic outputs its charts/heatmaps
    st.info("Visual scoreline probability grid and win/loss/draw percentages will render here.")

st.write("---")

# ---------------------------------------------------------
# 5. PHASE 4: The Tournament Finale
# ---------------------------------------------------------
st.header("📊 4. The Grand Tournament Simulation")
st.markdown("""
Finally, we put it all together. By taking these team strengths and simulating an entire bracket 
structure 1,000 times, we get an accurate survival matrix showing who is actually built to go all the way.
""")

# Placeholder for your final master dataframe or survival chart
st.caption("Final tournament survival data table will live here.")