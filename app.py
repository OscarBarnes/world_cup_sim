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
    st.markdown(r"""
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

# =========================================================
# HELPER FUNCTION: THE ENGINE
# =========================================================
def simulate_match(team_a, team_b, trace):
    """
    Extracts posterior parameters for both teams and draws 1,000 samples
    from the Negative Binomial distribution to simulate a match.
    """
    # 1. Extract the posterior chains for our variables
    # (Note: Replace "atts", "defs", "intercept" with your exact PyMC variable names!)
    att_A = trace.posterior["atts"].sel(team=team_a).values.flatten()
    def_A = trace.posterior["defs"].sel(team=team_a).values.flatten()
    
    att_B = trace.posterior["atts"].sel(team=team_b).values.flatten()
    def_B = trace.posterior["defs"].sel(team=team_b).values.flatten()
    
    intercept = trace.posterior["intercept"].values.flatten()
    alpha = trace.posterior["alpha"].values.flatten() # Your chaos/overdispersion parameter
    
    # 2. Subsample to exactly 1,000 universes to keep it fast
    num_sims = 1000
    idx = np.random.choice(len(att_A), size=num_sims, replace=False)
    
    # 3. Calculate expected goals (mu) for each universe
    mu_A = np.exp(intercept[idx] + att_A[idx] - def_B[idx])
    mu_B = np.exp(intercept[idx] + att_B[idx] - def_A[idx])
    
    # 4. Simulating the chaos using the Negative Binomial distribution
    # Note: NumPy parameterizes Negative Binomial using (n, p). 
    # We convert our mean (mu) and overdispersion (alpha) to NumPy's format:
    p_A = alpha[idx] / (alpha[idx] + mu_A)
    p_B = alpha[idx] / (alpha[idx] + mu_B)
    
    goals_A = np.random.negative_binomial(alpha[idx], p_A)
    goals_B = np.random.negative_binomial(alpha[idx], p_B)
    
    return goals_A, goals_B

# ---------------------------------------------------------
# 4. INTERACTIVE ZONE: The Dream Matchup Engine
# ---------------------------------------------------------
st.header("🕹️ 3. Test the Multiverse Machine")
st.markdown("Select any two international sides to run a 1,000-universe simulation.")

# Pull the real team names from your model's posterior coordinates
# Dynamically pull the team names from your data sources
try:
    available_teams = sorted(list(trace.posterior.coords["team"].values))
except (AttributeError, KeyError):
    cols = shootouts_df.columns
    if "home_team" in cols and "away_team" in cols:
        all_teams = pd.concat([shootouts_df["home_team"], shootouts_df["away_team"]]).unique()
        available_teams = sorted(list(all_teams))
    elif "team" in cols:
        available_teams = sorted(list(shootouts_df["team"].unique()))
    else:
        available_teams = []

# 📋 THE BOUNCER: Define exactly which teams are allowed in your model (Updated for 2026)
world_cup_teams = [
    "Spain", "Argentina", "France", "Brazil", "Netherlands", "England", 
    "Portugal", "Germany", "Colombia", "Croatia", "Morocco", "Uruguay", 
    "Belgium", "Senegal", "Egypt", "South Korea", "Ecuador", "Mexico", 
    "Norway", "Ivory Coast", "Japan", "Switzerland", "United States", "Turkey", 
    "Australia", "Ghana", "Algeria", "Iran", "Austria", "Canada", 
    "Paraguay", "Saudi Arabia", "Sweden", "Panama", "Scotland", "Tunisia", 
    "South Africa", "Qatar", "Czech Republic", "New Zealand", "Jordan", 
    "Bosnia and Herzegovina", "DR Congo", "Cape Verde", "Uzbekistan", "Iraq", 
    "Curacao", "Haiti"
]

# A quick list comprehension to filter out any team NOT in your World Cup list
available_teams = [team for team in available_teams if team in world_cup_teams]

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Team A", available_teams, index=0)
with col2:
    team_b = st.selectbox("Select Team B", available_teams, index=1 if len(available_teams) > 1 else 0)

# Slider for the Chaos Level
chaos_factor = st.slider("Adjust Chaos Level (Overdispersion)", min_value=0.1, max_value=2.0, value=1.0)

# The Simulation Trigger Button
if st.button("🚀 Run Match Simulation"):
    
    # 1. Run the function we defined above
    with st.spinner("Collapsing quantum probabilities..."):
        goals_A, goals_B = simulate_match(team_a, team_b, trace)
    
    # 2. Math: Calculate outcomes across all 1,000 universes
    wins_A = np.sum(goals_A > goals_B)
    wins_B = np.sum(goals_B > goals_A)
    draws = np.sum(goals_A == goals_B)
    
    pct_A = (wins_A / 1000) * 100
    pct_B = (wins_B / 1000) * 100
    pct_draw = (draws / 1000) * 100
    
    # 3. Display the Headline Results using clean visual columns
    st.write(f"### 📊 Simulation Results: {team_a} vs {team_b}")
    
    m1, m2, m3 = st.columns(3)
    m1.metric(label=f"🏆 {team_a} Win Probability", value=f"{pct_A:.1f}%")
    m2.metric(label="🤝 Draw Probability", value=f"{pct_draw:.1f}%")
    m3.metric(label=f"🏆 {team_b} Win Probability", value=f"{pct_B:.1f}%")
    
    # 4. Contextual summary for the dads
    st.info(f"""
    In 1,000 alternate realities, **{team_a}** won {wins_A} times, 
    **{team_b}** won {wins_B} times, and the match ended in a draw {draws} times. 
    The average expected scoreline across the multiverse was **{goals_A.mean():.1f} - {goals_B.mean():.1f}**.
    """)

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
