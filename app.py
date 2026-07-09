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
    # Load the heavy PyMC trace file once and cache it
    return az.from_netcdf("nb_trace.nc")

trace = load_model_brain()

# ---------------------------------------------------------
# 1. THE HIDDEN TRANSLATION LAYER (284 Teams Map)
# ---------------------------------------------------------
# Load your master results file and replicate your training filter (Post-2018)
results_df = pd.read_csv("results.csv")
results_df['date'] = pd.to_datetime(results_df['date'])
filtered_df = results_df[results_df['date'] >= '2019-01-01']

# Recreate the exact 284 master list in alphabetical order (standard Python mapping)
master_284_teams = sorted(list(pd.concat([filtered_df["home_team"], filtered_df["away_team"]]).unique()))

# 📋 THE BOUNCER: The curated 48 World Cup teams for the UI dropdowns
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

# Only include the World Cup teams that actually exist inside your 284 master dataset
available_teams = [team for team in world_cup_teams if team in master_284_teams]

# ---------------------------------------------------------
# 2. TITLE & INTRO
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

st.header("📈 1. Teaching the AI Football History")
st.markdown("""
A friendly match against a low-ranked team three years ago shouldn't carry the same weight 
as a high-stakes continental knockout match played last month. 

To fix this, the data pipeline dynamically recalculates a **match weight** using an exponential time-decay function. 
Recent matches matter exponentially more than older ones.
""")

with st.expander("🛠️ View the Time-Decay Math & Logic"):
    st.markdown(r"""
    We apply a recency weight using an exponential half-life formula:
    $$W_t = e^{-\lambda \cdot t}$$
    Where $t$ is the days elapsed since the match, and $\lambda$ dictates how fast historical memory decays.
    """)

st.write("---")

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

st.write("---")

# =========================================================
# 3. THE MATCH SIMULATION ENGINE (Using Index Selection)
# =========================================================
def simulate_match(team_a, team_b, trace, chaos_factor, master_list):
    """
    Looks up the numeric ID for both teams from the master list, 
    extracts their parameters, and simulates 1,000 match outcomes.
    """
    # Find the hidden numeric index (0 to 283) for both teams
    idx_A = master_list.index(team_a)
    idx_B = master_list.index(team_b)
    
    # Use .isel() to extract the parameters by their numeric coordinate index
    att_A = trace.posterior["atts"].isel(atts_dim_0=idx_A).values.flatten()
    def_A = trace.posterior["defs"].isel(defs_dim_0=idx_A).values.flatten()
    att_B = trace.posterior["atts"].isel(atts_dim_0=idx_B).values.flatten()
    def_B = trace.posterior["defs"].isel(defs_dim_0=idx_B).values.flatten()
    
    intercept = trace.posterior["intercept"].values.flatten()
    alpha_A = trace.posterior["alpha_home"].values.flatten()
    alpha_B = trace.posterior["alpha_away"].values.flatten()

    num_sims = 1000
    sim_idx = np.random.choice(len(att_A), size=num_sims, replace=False)
    
    # Calculate log-linear expected goals (mu)
    mu_A = np.exp(intercept[sim_idx] + att_A[sim_idx] - def_B[sim_idx])
    mu_B = np.exp(intercept[sim_idx] + att_B[sim_idx] - def_A[sim_idx])
    
    # Apply the UI chaos factor slider adjustment
    adjusted_alpha_A = alpha_A[sim_idx] / chaos_factor
    adjusted_alpha_B = alpha_B[sim_idx] / chaos_factor
    
    p_A = adjusted_alpha_A / (adjusted_alpha_A + mu_A)
    p_B = adjusted_alpha_B / (adjusted_alpha_B + mu_B)
    
    goals_A = np.random.negative_binomial(adjusted_alpha_A, p_A)
    goals_B = np.random.negative_binomial(adjusted_alpha_B, p_B)
    
    return goals_A, goals_B

# ---------------------------------------------------------
# 4. INTERACTIVE ZONE
# ---------------------------------------------------------
st.header("🕹️ 3. Test the Multiverse Machine")
st.markdown("Select any two international sides to run a 1,000-universe simulation.")

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Team A (Home Side)", available_teams, index=0)
with col2:
    team_b = st.selectbox("Select Team B (Away Side)", available_teams, index=1 if len(available_teams) > 1 else 0)

chaos_factor = st.slider("Adjust Chaos Level (Overdispersion)", min_value=0.1, max_value=2.0, value=1.0)

if st.button("🚀 Run Match Simulation"):
    with st.spinner("Collapsing quantum probabilities..."):
        # Pass the teams, the trace, the slider value, and the hidden 284 master list map
        goals_A, goals_B = simulate_match(team_a, team_b, trace, chaos_factor, master_284_teams)
    
    # Calculate outcomes across all universes
    wins_A = np.sum(goals_A > goals_B)
    wins_B = np.sum(goals_B > goals_A)
    draws = np.sum(goals_A == goals_B)
    
    pct_A = (wins_A / 1000) * 100
    pct_B = (wins_B / 1000) * 100
    pct_draw = (draws / 1000) * 100
    
    st.write(f"### 📊 Simulation Results: {team_a} vs {team_b}")
    
    m1, m2, m3 = st.columns(3)
    m1.metric(label=f"🏆 {team_a} Win Probability", value=f"{pct_A:.1f}%")
    m2.metric(label="🤝 Draw Probability", value=f"{pct_draw:.1f}%")
    m3.metric(label=f"🏆 {team_b} Win Probability", value=f"{pct_B:.1f}%")
    
    st.info(f"""
    In 1,000 alternate realities, **{team_a}** won {wins_A} times, 
    **{team_b}** won {wins_B} times, and the match ended in a draw {draws} times. 
    The average expected scoreline across the multiverse was **{goals_A.mean():.1f} - {goals_B.mean():.1f}**.
    """)

# ---------------------------------------------------------
# 5. PHASE 4: Tournament Finale Placeholder
# ---------------------------------------------------------
st.write("---")
st.header("📊 4. The Grand Tournament Simulation")
st.markdown("""
Finally, we put it all together. By taking these team strengths and simulating an entire bracket 
structure 1,000 times, we get an accurate survival matrix showing who is actually built to go all the way.
""")
st.caption("Final tournament survival data table will live here.")
