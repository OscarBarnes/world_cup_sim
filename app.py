import arviz as az
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import nbinom, poisson
import streamlit as st


# ---------------------------------------------------------
# 0. PAGE CONFIG & DATA LOADING
# ---------------------------------------------------------
st.set_page_config(page_title="2026 FIFA World Cup Simulation", layout="centered")

@st.cache_resource
def load_model_brain():
    import os
    
    # Failsafe 1: Check if the file actually exists in the directory
    if not os.path.exists("nb_trace.nc"):
        st.error("🚨 **CRITICAL ERROR:** The model brain file `nb_trace.nc` is completely missing from your GitHub repository root folder! Please check your file names.")
        st.stop()
        
    # Failsafe 2: Check if the file was corrupted or uploaded as an empty file
    if os.path.getsize("nb_trace.nc") == 0:
        st.error("🚨 **CRITICAL ERROR:** `nb_trace.nc` was found, but it is completely empty (0 bytes)! Your Git commit may have cut off. Please re-upload the original full trace file.")
        st.stop()
        
    # If it passes both checks, load it safely
    return az.from_netcdf("nb_trace.nc")

trace = load_model_brain()

# ---------------------------------------------------------
# 1. THE HIDDEN TRANSLATION LAYER (284 Teams Map)
# ---------------------------------------------------------
# The exact, immovable index map exported directly from the training notebook
master_284_teams = ['Abkhazia', 'Afghanistan', 'Albania', 'Alderney', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Artsakh', 'Aruba', 'Australia', 'Austria', 'Aymara', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barawa', 'Barbados', 'Basque Country', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Biafra', 'Bolivia', 'Bonaire', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'British Virgin Islands', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cascadia', 'Catalonia', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chagos Islands', 'Chameria', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic', 'DR Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'East Turkestan', 'Ecuador', 'Egypt', 'El Salvador', 'Elba Island', 'Ellan Vannin', 'England', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Falkland Islands', 'Faroe Islands', 'Fiji', 'Finland', 'France', 'Franconia', 'French Guiana', 'Frøya', 'Gabon', 'Galicia', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Gozo', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Hitra', 'Hmong', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Isle of Man', 'Isle of Wight', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kabylia', 'Kazakhstan', 'Kenya', 'Kernow', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Kárpátalja', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luhansk PR', 'Luxembourg', 'Macau', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Mapuche', 'Marshall Islands', 'Martinique', 'Matabeleland', 'Maule Sur', 'Mauritania', 'Mauritius', 'Mayotte', 'Menorca', 'Mexico', 'Moldova', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia', 'Northern Cyprus', 'Northern Ireland', 'Northern Mariana Islands', 'Norway', 'Oman', 'Orkney', 'Padania', 'Pakistan', 'Palestine', 'Panama', 'Panjab', 'Papua New Guinea', 'Paraguay', 'Parishes of Jersey', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Raetia', 'Republic of Ireland', 'Romania', 'Russia', 'Rwanda', 'Réunion', 'Saint Barthélemy', 'Saint Helena', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Saudi Arabia', 'Scotland', 'Senegal', 'Serbia', 'Seychelles', 'Shetland', 'Sierra Leone', 'Singapore', 'Sint Maarten', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'Somaliland', 'South Africa', 'South Korea', 'South Ossetia', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Surrey', 'Sweden', 'Switzerland', 'Syria', 'Székely Land', 'Sápmi', 'São Tomé and Príncipe', 'Tahiti', 'Taiwan', 'Tajikistan', 'Tamil Eelam', 'Tanzania', 'Thailand', 'Tibet', 'Ticino', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu', 'Two Sicilies', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Koreans in Japan', 'United States', 'United States Virgin Islands', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela', 'Vietnam', 'Wales', 'West Papua', 'Western Armenia', 'Western Isles', 'Yemen', 'Ynys Môn', 'Yorkshire', 'Yoruba Nation', 'Zambia', 'Zanzibar', 'Zimbabwe', 'Åland Islands']

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

tournament_groups_fixed = {
    'Group A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'Group B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'Group C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'Group D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'Group E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'Group F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'Group G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'Group H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'Group I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'Group J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'Group K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'Group L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

# Check for any team name mismatches between World Cup list and master list
missing_teams = [team for team in world_cup_teams if team not in master_284_teams]
if missing_teams:
    st.warning(f"⚠️ **Name Mismatch Alert:** {len(missing_teams)} team(s) couldn't be found in `master_284_teams`: {missing_teams}. Please check spelling/formatting.")

# Only include the World Cup teams that actually exist inside your 284 master dataset
available_teams = [team for team in world_cup_teams if team in master_284_teams]

# Convert team name strings in the fixed groups into numeric indices (0 to 47)
fixed_groups_ids = [
    [available_teams.index(team) for team in team_list]
    for group_name, team_list in tournament_groups_fixed.items()
]

# ---------------------------------------------------------
# 2. TITLE & INTRO
# ---------------------------------------------------------
st.title("2026 FIFA World Cup Monte Carlo Simulation")
st.caption("Simulating 1,000 Parallel World Cup Realities using Bayesian Statistics")

st.markdown("""
Football is chaotic. We’ve all seen huge shocking upsets completed in the last few minutes, like Paraguay stunning Germany to progress to the round of 16. Because hitting the post or 90th minute red cards can alter the score, predicting the scoreline or picking the tournament winner is almost impossible. Instead of looking at a single guess, building a model to map out the possibilities and unexpected shocks is much more interesting, we can use probabilities to study the chaos.

To capture the randomness, I built a Bayesian Negative Binomial model trained on weighted historic international match data. The model estimates two traits of every country, their expected goals scored and expected goals conceded. By matching up one team’s attack directly against another’s defence, it calculates goal probabilities and samples realistic match scorelines. Simulate these fixtures thousands of times across the World Cup bracket, we can then look at how the tournament progresses.
""")

st.divider()

# ---------------------------------------------------------
# 3. MODEL METHODOLOGY
# ---------------------------------------------------------
st.header("How the Statistical Engine Works")

st.markdown("""
To simulate matches realistically, I built a **Hierarchical Bayesian Negative Binomial Model** trained on historical international match data. 

Here is the step-by-step statistical pipeline that turns raw scorelines into tournament probabilities:
""")

# ---------------------------------------------------------
# DATA PREPROCESSING & WEIGHTING SECTION
# ---------------------------------------------------------
with st.expander("1. Data Preprocessing & Weighting", expanded=False):
    st.markdown(r"""
    The raw historical match dataset was obtained from [Kaggle's International Football Results (1872–Present)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017). 
    
    To ensure team strength estimates reflect modern squad quality rather than outdated tactical eras or retired generations of players, the dataset was filtered to matches played from **2018 onwards**. The data was reduced to key features: match date, home team, away team, home/away scores, tournament type, and neutral venue indicators. Numeric team IDs were created across all 284 international sides for efficient array indexing inside PyMC.

    To prevent friendly results or minor tournaments from distorting team ratings, two distinct weighting multipliers were applied to every match:

    #### 1. Tournament Stature Weighting ($W_{\text{tournament}}$)
    High-stakes competitive ties (e.g., FIFA World Cup fixtures or continental championships like the UEFA Euro and Copa América) receive weights close to $1.0$. Conversely, friendlies and minor exhibition cups, where managers frequently experiment with secondary lineups, are weighted significantly lower (down to $0.30$). This ensures a $3\text{--}0$ victory in a World Cup tie provides far more influence during parameter estimation than a $3\text{--}0$ win in a friendly.

    #### 2. Exponential Recency Weighting ($W_{\text{recency}}$)
    Team form naturally evolves over time. To give recent performances greater weight without applying an arbitrary cutoff date, an **exponential time-decay function** was applied:

    $$W_{\text{recency}} = \exp\left(-\frac{\Delta t}{730}\right)$$

    Where $\Delta t$ represents the number of days elapsed between the match date and the most recent match in the dataset. A denominator of $730$ days ($2\text{ years}$) was chosen because national teams play relatively infrequently (roughly 10–12 matches per year). Dividing by 730 flattens the decay curve smoothly, preserving sufficient sample size while ensuring modern form dominates.

    #### Combined Match Weight ($\bar{W}$)
    Both weights are multiplied together and normalized relative to the maximum weight:

    $$W_{\text{final}} = W_{\text{tournament}} \times W_{\text{recency}}$$

    $$\bar{W} = \frac{W_{\text{final}}}{\max(W_{\text{final}})}$$

    This normalized weight ($\bar{W}$) enters the PyMC likelihood function, prioritizing recent, high-stakes competitive matches during Bayesian parameter estimation.
    """)

# ---------------------------------------------------------
# ESTIMATING TEAM TRAITS & EXPECTED GOALS
# ---------------------------------------------------------
with st.expander("2. Estimating Team Traits & Expected Goals", expanded=False):
    st.markdown(r"""
    Before predicting match outcomes, how good each team is at attacking and defending must be quantified. Because a nation's football capability cannot be directly measured, the model treats team strength as latent (hidden) parameters estimated from historical match results.

    #### 1. Dual Team Traits: Attack ($\text{att}_i$) & Defense ($\text{def}_i$)
    For every nation $i$ across all 284 international sides, the model assigns two distinct traits:
    * **Attacking Strength ($\text{att}_i$):** A team's ability to create and finish scoring chances.
    * **Defensive Solidity ($\text{def}_i$):** A team's ability to suppress and prevent opponent scoring chances.

    ##### The Sum-to-Zero Constraint
    To ensure mathematical identifiability (so the model doesn't endlessly shift all ratings up or down together), both trait sets are forced to sum to zero across all 284 teams:

    $$\sum_{i=1}^{284} \text{att}_i = 0 \quad \text{and} \quad \sum_{i=1}^{284} \text{def}_i = 0$$

    This means that a rating of $0.0$ represents a completely average international side. A positive attack rating indicates an above-average attack, while a positive defence rating means an above-average defence that reduces opponent scoring.

    #### 2. Non-Centered Parameterisation Architecture
    Estimating parameters for 284 teams simultaneously can cause MCMC samplers to get stuck in mathematical traps. To avoid this, the model uses a non-centered architecture:

    $$\text{atts\_raw} = \text{atts\_offset} \times \sigma_{\text{att}}, \quad \text{atts\_offset} \sim \text{Normal}(0, 1)$$

    $$\text{defs\_raw} = \text{defs\_offset} \times \sigma_{\text{def}}, \quad \text{defs\_offset} \sim \text{Normal}(0, 1)$$

    * **$\sigma_{\text{att}}$ & $\sigma_{\text{def}}$:** Global hyper-priors (drawn from an Exponential distribution) that learn how widely team quality varies across the entire world.
    * **Standard Offsets:** The sampler samples from standard normal distributions ($\text{Normal}(0, 1)$) and scales them afterwards, keeping the mathematical space smooth and collision-free.

    #### 3. The Expected Goals Equation ($\theta$)
    When Home Team $H$ plays Away Team $A$, their traits are pitted directly against each other to calculate their expected average scoring rates ($\theta_H$ and $\theta_A$). Because expected goal rates must always be positive, the parameters are combined inside a linear predictor and passed through an exponential link function ($\exp$):

    $$\theta_H = \exp\left(\beta_0 + (\gamma \cdot \text{neutral\_mask}) + \text{att}_H - \text{def}_A\right)$$

    $$\theta_A = \exp\left(\beta_0 + \text{att}_A - \text{def}_H\right)$$

    ##### Breakdown of Equation Components:
    * **$\beta_0$ (intercept):** The global baseline goal-scoring rate for a neutral match between two average sides.
    * **$\gamma$ (home_adv):** The goal boost gained from playing at home. This is multiplied by `neutral_mask` ($0$ for neutral World Cup venues, $1$ for true home matches).
    * **$\text{att}_H - \text{def}_A$:** The direct matchup advantage. If Team H's attack is stronger than Team A's defense, this term is positive and increases Team H's expected goals.

    Expected goals represent the theoretical average scoreline if two teams played each other hundreds of times (e.g., England expected goals $\theta_H = 1.82$, Scotland expected goals $\theta_A = 0.65$).

    Notice that $\theta$ is a continuous decimal, it does not tell us the probability of exact scorelines like $2\text{--}0$ or $1\text{--}1$ yet. To convert this expected average into discrete integer goals, we pass $\theta$ into a goal probability distribution in the next step.
    """)

# ---------------------------------------------------------
# POISSON VS NEGATIVE BINOMIAL
# ---------------------------------------------------------
with st.expander(
    "3. Goal Distributions: Upgrading Poisson to Negative Binomial",
    expanded=False,
):
    st.markdown(r"""
    Now that the expected goal rate ($\theta$) for each team is calculated, a probability distribution is used to translate this continuous decimal (e.g., $\theta = 1.64$ goals) into exact, discrete scoreline probabilities ($0, 1, 2, 3, \dots$ goals).

    #### 1. The Baseline: Poisson Distribution
    Initially, the model uses a standard **Poisson distribution** as the goal translator. Poisson is widely used in sports modeling for rare discrete events and relies on a single parameter, $\theta$.

    The defining mathematical property of the Poisson distribution is **equidispersion**, where the variance equals the mean:

    $$\text{Variance} = \text{Mean} = \theta$$

    This variance means that if a team has an expected goal rate of $\theta = 1.64$, a Poisson model strictly fixes the goal spread around $1.64$ (a standard deviation of $\sqrt{1.64} \approx 1.28$ goals). Under this rigid formula:
    * The probability of $1$ goal is locked at $\approx 31.8\%$.
    * High-scoring blowouts ($4+$ goals) are capped at $\approx 8.4\%$.

    ##### The Overdispersion Problem ($\text{Variance} > \text{Mean}$)
    In reality, World Cup matches feature much higher variance (**overdispersion**). Events like early red cards, tactical collapses, high-pressure shootouts, or stubborn $0\text{--}0$ stalemates occur far more often in international football, especially when underdogs like Cape Verde or Curaçao take on tactical powerhouses like Germany or Spain. 

    Because Poisson assumes goals happen at a completely constant, independent rate across 90 minutes, it fails to account for these chaotic match dynamics.

    #### 2. The Solution: Negative Binomial Distribution
    To fix this issue, the model upgrades to a **Negative Binomial distribution**. It introduces two global overdispersion parameters ($\alpha_{\text{home}}$ and $\alpha_{\text{away}}$), expanding the variance formula to:

    $$\text{Variance} = \theta + \frac{\theta^2}{\alpha}$$

    Here, $\alpha$ acts as a **chaos regulator**:
    * **As $\alpha$ gets larger:** The extra variance term ($\frac{\theta^2}{\alpha}$) vanishes, and the model behaves like a standard Poisson distribution.
    * **As $\alpha$ gets smaller:** It inflates the variance relative to the mean, giving the model the flexibility to assign realistic probabilities to both stubborn $0\text{--}0$ clean sheets and wild, unexpected blowouts.
    """)

    st.subheader("Visualising the Distribution Shift")

    # Generate distribution probabilities for theta = 1.64
    goals = np.arange(0, 8)
    theta_val = 1.64
    alpha_val = 2.0  # High overdispersion for demonstration

    # Poisson probabilities
    poisson_pmf = poisson.pmf(goals, theta_val)

    # Negative Binomial conversion (scipy uses n and p)
    # n = alpha, p = alpha / (alpha + mu)
    p_param = alpha_val / (alpha_val + theta_val)
    nb_pmf = nbinom.pmf(goals, alpha_val, p_param)

    # Create Plotly Comparison Figure
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=goals,
            y=poisson_pmf,
            name="Poisson (Variance = 1.64)",
            marker_color="#1f77b4",
            opacity=0.75,
        )
    )
    fig.add_trace(
        go.Bar(
            x=goals,
            y=nb_pmf,
            name="Negative Binomial (Variance = 2.98, α=2.0)",
            marker_color="#ff7f0e",
            opacity=0.75,
        )
    )

    fig.update_layout(
        title=f"Goal Probability Comparison for Expected Goals (θ) = {theta_val}",
        xaxis_title="Goals Scored in Match",
        yaxis_title="Probability",
        barmode="group",
        legend=dict(x=0.55, y=0.98),
        template="plotly_white",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(r"""
    **Notice the shift in the graph above:** The Negative Binomial distribution (orange) lowers the peak around $1$ and $2$ goals and shifts probability mass into the **tails**—increasing the likelihood of both **$0$ goals** (stalemates) and **$4+$ goals** (blowouts).

    #### 3. MCMC Sampling & Posterior Uncertainty (`nb_trace.nc`)
    Rather than picking single fixed estimates for team ratings ($\text{att}_i, \text{def}_i$) or overdispersion ($\alpha$), the model uses **Markov Chain Monte Carlo (MCMC)** sampling in PyMC. 

    MCMC samples thousands of plausible parameter combinations from the data, saving them into a trace file (`nb_trace.nc`). When simulating matches in Section 5, the model doesn't just rely on single point averages, it samples directly from this full parameter collection, preserving true statistical uncertainty across every simulated World Cup fixture.
    """)

# ---------------------------------------------------------
# MODEL CONVERGENCE & DIAGNOSTICS
# ---------------------------------------------------------
with st.expander("4. Model Convergence & Diagnostic Validation", expanded=False):
    st.markdown(r"""
    Before deploying the MCMC trace (`nb_trace.nc`) into tournament simulations, rigorous diagnostic checks were conducted using **ArviZ** (`az.summary`) to ensure the Markov chains successfully converged and accurately fitted the underlying data.

    #### 1. Convergence Diagnostics ($\hat{R}$ / Gelman-Rubin Statistic)
    The $\hat{R}$ statistic compares variance between independent MCMC chains to variance within each chain. 
    * **Target:** $\hat{R} \le 1.05$ (ideally $1.00$).
    * **Result:** Across all 284 team attacking parameters ($\text{att}_i$), defensive parameters ($\text{def}_i$), global intercepts, and overdispersion terms ($\alpha$), $\hat{R}$ reached **1.00**. This confirms that all chains successfully mixed and converged on the exact same target posterior distribution.

    #### 2. Effective Sample Size ($\text{ESS}_{\text{bulk}}$ & $\text{ESS}_{\text{tail}}$)
    Because MCMC draws are sequentially correlated, the **Effective Sample Size (ESS)** measures the number of truly independent samples obtained from the trace.
    * **Target:** $\text{ESS}_{\text{bulk}} > 400$ per chain.
    * **Result:** Non-centered parameterisation prevented the sampler from encountering geometric funnel traps, producing high ESS values across latent team traits and allowing reliable estimation of posterior credible intervals.

    #### 3. Posterior Predictive Checks (Scoreline Validation)
    To validate model fit against real-world football dynamics, synthetic match datasets were generated from the posterior distribution using `pm.sample_posterior_predictive`. 
    
    Comparing simulated scorelines against historical ground-truth results confirmed that the Negative Binomial model accurately reproduced observed clean-sheet rates, draw frequencies, and blowout probabilities—validating its readiness for tournament simulation.
    """)

# ---------------------------------------------------------
# TOURNAMENT SIMULATION ENGINE
# ---------------------------------------------------------
with st.expander(
    "5. Tournament Monte Carlo Simulation", expanded=False
):
    st.markdown(r"""
    With team traits estimated, goal distributions established, and model convergence validated, the final step is linking individual match predictions into a full **2026 FIFA World Cup Tournament Engine**.

    #### 1. Sampling Posterior Parameters
    Rather than assuming fixed, static team ratings, every single tournament run draws a fresh parameter combination from the MCMC posterior trace (`nb_trace.nc`). 

    This ensures that one simulation might feature a slightly underperforming favourite, while another features a peak underdog run, fully preserving Bayesian uncertainty across all fixtures.

    #### 2. Executing the Official 2026 FIFA Structure (104 Matches)
    Each tournament run simulates the complete 48-team World Cup structure step-by-step:

    * **Group Stage (72 Matches):** Teams in Groups A through L play a 3-match round-robin. Standings are ranked by **Points $\rightarrow$ Goal Difference $\rightarrow$ Goals Scored**.
    * **Wildcard Qualification:** The **12 group winners**, **12 runners-up**, and the **top 8 third-place teams** advance to the Round of 32.
    * **Knockout Bracket (32 Matches):** Teams follow the official FIFA knockout path. Any match ending in a draw is resolved via simulated penalty shootouts.

    #### 3. Simulating 1,000 Parallel Universes
    By repeating this process **1,000 times**, we create 1,000 parallel World Cup realities. Aggregating the outcomes across all runs converts unpredictable single-match chaos into reliable, probabilistic forecasts for every country's chances of lifting the trophy.
    """)

st.divider()

# ---------------------------------------------------------
# 4. INTERACTIVE MATCH SIMULATION
# ---------------------------------------------------------
def simulate_match(team_a, team_b, trace, chaos_factor, master_list):
    """
    Looks up the numeric ID for both teams from the master list, 
    extracts their parameters, and simulates 1,000 match outcomes.
    """
    idx_A = master_list.index(team_a)
    idx_B = master_list.index(team_b)
    
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
# MATCH SIMULATOR SECTION
# ---------------------------------------------------------
st.header("Test the Match Simulation")

st.markdown(r"""
Before simulating the entire 2026 World Cup tournament, use this single-match sandbox to test how two nations match up head-to-head across **1,000 parallel match realities**.

#### Understanding the Chaos Level Slider ($\alpha$)
The **Chaos Level** slider directly controls the **overdispersion parameter ($\alpha$)** in the Negative Binomial model:

* **High Chaos ($\text{Smaller } \alpha$):** Increases match variance. This simulates high-entropy tournament conditions where red cards, early goals, and tactical collapses lead to shock upsets, high-scoring thrillers, or $0\text{--}0$ stalemates.
* **Low Chaos ($\text{Larger } \alpha$):** Suppresses match variance toward standard Poisson behavior ($\text{Variance} \approx \text{Mean}$). Games play out strictly according to baseline team quality, making favorites significantly more dominant.

Select your teams, adjust the overdispersion, and launch the simulation below!
""")

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Team A (Home Side)", available_teams, index=0)
with col2:
    team_b = st.selectbox("Select Team B (Away Side)", available_teams, index=1 if len(available_teams) > 1 else 0)

chaos_factor = st.slider("Adjust Chaos Level (Overdispersion)", min_value=0.1, max_value=2.0, value=1.0)

if st.button("🚀 Run Match Simulation"):
    with st.spinner("Collapsing quantum probabilities..."):
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

st.write("---")

# ---------------------------------------------------------
# GRAND TOURNAMENT ENGINE SECTION
# ---------------------------------------------------------
st.header("The Grand Tournament Engine")

st.markdown(r"""
Scaling up from single-match predictions, this engine simulates **1,000 complete 2026 World Cup tournaments** (104,000 total matches) across the official 48-team structure. 

#### 📖 How to Interpret the Output:
* **1. The World Cup Survival Matrix:** Displays each nation's statistical probability (%) of reaching each progressive milestone—from surviving the group stage (Round of 32) all the way to lifting the trophy.
* **2. Multiverse Shockers:** Highlights notable "giant-killing" upsets where lower-ranked underdogs knocked out elite powerhouses in elimination ties across the simulated realities.
* **3. High-Scoring Thrillers:** Showcases extreme, high-entropy "tail events"—the wildest, highest-scoring goal-fests recorded across all 100,000+ matches.

Adjust the chaos level below and launch the 1,000-universe simulation!
""")

# Setup simulation parameters
st.subheader("⚙️ Simulation Settings")
c1, c2 = st.columns(2)
with c1:
    tourney_chaos = st.slider("Tournament Chaos Level (Overdispersion)", min_value=0.1, max_value=2.0, value=1.0, key="grand_chaos")
with c2:
    st.info("⚡ **Fixed Multiverse Scale:** Set to exactly **1,000 Full Tournaments** for statistically robust survival probabilities.")

def run_1000_tournaments(chaos_factor):
    num_sims = 1000
    num_samples = len(trace.posterior["intercept"].values.flatten())
    
    # 1. Initialize tracking structures
    stats = {team: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "Final": 0, "Champ": 0} for team in available_teams}
    thrillers = []  # To store wild, high-scoring matches
    upsets = []     # To store major knockout giant-killings
    
    # Group letter mapping corresponding to index 0 to 11
    group_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    
    # Extract parameter matrices once for max speed
    team_indices = [master_284_teams.index(t) for t in available_teams]
    atts_all = trace.posterior["atts"].values.reshape(-1, 284)[:, team_indices]
    defs_all = trace.posterior["defs"].values.reshape(-1, 284)[:, team_indices]
    intercept_all = trace.posterior["intercept"].values.flatten()
    alpha_home_all = trace.posterior["alpha_home"].values.flatten()
    alpha_away_all = trace.posterior["alpha_away"].values.flatten()
    
    id_to_team = {i: team for i, team in enumerate(available_teams)}
    
    # Pre-calculate overall team power ratings (Att - Def) for upset detection
    avg_atts = atts_all.mean(axis=0)
    avg_defs = defs_all.mean(axis=0)
    power_ratings = avg_atts - avg_defs
    
    for sim_id in range(1, num_sims + 1):
        s = np.random.randint(0, num_samples)
        
        groups = fixed_groups_ids
        
        winners = {}
        runners_up = {}
        third_place_pool = []
        
        # --- GROUP STAGE ---
        for g_idx, group in enumerate(groups):
            g_letter = group_letters[g_idx]
            g_stats = {t_id: {"pts": 0, "gd": 0, "gs": 0} for t_id in group}
            group_size = len(group)
            
            for i in range(group_size):
                for j in range(i + 1, group_size):
                    tA, tB = group[i], group[j]
                    
                    mu_A = np.exp(intercept_all[s] + atts_all[s, tA] - defs_all[s, tB])
                    mu_B = np.exp(intercept_all[s] + atts_all[s, tB] - defs_all[s, tA])
                    
                    aA = alpha_home_all[s] / chaos_factor
                    aB = alpha_away_all[s] / chaos_factor
                    
                    gA = np.random.negative_binomial(aA, aA / (aA + mu_A))
                    gB = np.random.negative_binomial(aB, aB / (aB + mu_B))
                    
                    g_stats[tA]["gs"] += gA
                    g_stats[tB]["gs"] += gB
                    g_stats[tA]["gd"] += (gA - gB)
                    g_stats[tB]["gd"] += (gB - gA)
                    
                    if gA > gB: g_stats[tA]["pts"] += 3
                    elif gB > gA: g_stats[tB]["pts"] += 3
                    else:
                        g_stats[tA]["pts"] += 1
                        g_stats[tB]["pts"] += 1
                    
                    # Track thrillers (7+ total goals)
                    if (gA + gB) >= 7 and len(thrillers) < 15:
                        thrillers.append({
                            "Universe": f"#{sim_id}",
                            "Stage": "Group Stage",
                            "Matchup": f"{id_to_team[tA]} {gA} - {gB} {id_to_team[tB]}",
                            "Total Goals": gA + gB
                        })
                        
            ranked = sorted(group, key=lambda x: (g_stats[x]["pts"], g_stats[x]["gd"], g_stats[x]["gs"]), reverse=True)
            winners[g_letter] = ranked[0]
            runners_up[g_letter] = ranked[1]
            third_place_pool.append((ranked[2], g_stats[ranked[2]]))
            
        # Top 8 3rd-place teams advance as wildcards
        ranked_thirds = sorted(third_place_pool, key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gs"]), reverse=True)
        top8_thirds = [ranked_thirds[k][0] for k in range(8)]
        
        # Map top 8 third-place teams to wildcard match slots (74, 77, 79, 80, 81, 82, 85, 87)
        assigned_wildcards = {
            74: top8_thirds[0],
            77: top8_thirds[1],
            79: top8_thirds[2],
            80: top8_thirds[3],
            81: top8_thirds[4],
            82: top8_thirds[5],
            85: top8_thirds[6],
            87: top8_thirds[7],
        }
        
        # Record all 32 qualified teams
        for t in list(winners.values()) + list(runners_up.values()) + top8_thirds:
            stats[id_to_team[t]]["R32"] += 1
        
        # --- KNOCKOUT STAGE HELPER ---
        def play_ko(tA, tB, stage_name):
            mu_A = np.exp(intercept_all[s] + atts_all[s, tA] - defs_all[s, tB])
            mu_B = np.exp(intercept_all[s] + atts_all[s, tB] - defs_all[s, tA])
            aA = alpha_home_all[s] / chaos_factor
            aB = alpha_away_all[s] / chaos_factor
            
            gA = np.random.negative_binomial(aA, aA / (aA + mu_A))
            gB = np.random.negative_binomial(aB, aB / (aB + mu_B))
            
            # Thriller check for knockouts
            if (gA + gB) >= 6 and len(thrillers) < 15:
                thrillers.append({
                    "Universe": f"#{sim_id}",
                    "Stage": stage_name,
                    "Matchup": f"{id_to_team[tA]} {gA} - {gB} {id_to_team[tB]}",
                    "Total Goals": gA + gB
                })
                
            winner = None
            if gA > gB: winner = tA
            elif gB > gA: winner = tB
            else: winner = tA if np.random.rand() > 0.5 else tB # Penalty Shootout
            
            loser = tB if winner == tA else tA
            
            # Upset check (if underdog rating is significantly lower but wins)
            rating_gap = power_ratings[loser] - power_ratings[winner]
            if rating_gap > 0.45 and len(upsets) < 15:
                upsets.append({
                    "Universe": f"#{sim_id}",
                    "Stage": stage_name,
                    "Giant Killer": id_to_team[winner],
                    "Fallen Heavyweight": id_to_team[loser],
                    "Score": f"{gA}-{gB}" if winner == tA else f"{gB}-{gA}",
                    "Rating Gap": round(rating_gap, 2)
                })
                
            return winner

        # --- OFFICIAL 2026 KNOCKOUT BRACKET ---
        # Round of 32 (Matches 73 to 88)
        r32 = {}
        r32[73] = play_ko(runners_up['A'], runners_up['B'], "Round of 32")
        r32[74] = play_ko(winners['E'], assigned_wildcards[74], "Round of 32")
        r32[75] = play_ko(winners['F'], runners_up['C'], "Round of 32")
        r32[76] = play_ko(winners['C'], runners_up['F'], "Round of 32")
        r32[77] = play_ko(winners['I'], assigned_wildcards[77], "Round of 32")
        r32[78] = play_ko(runners_up['E'], runners_up['I'], "Round of 32")
        r32[79] = play_ko(winners['A'], assigned_wildcards[79], "Round of 32")
        r32[80] = play_ko(winners['L'], assigned_wildcards[80], "Round of 32")
        r32[81] = play_ko(winners['D'], assigned_wildcards[81], "Round of 32")
        r32[82] = play_ko(winners['G'], assigned_wildcards[82], "Round of 32")
        r32[83] = play_ko(runners_up['K'], runners_up['L'], "Round of 32")
        r32[84] = play_ko(winners['H'], runners_up['J'], "Round of 32")
        r32[85] = play_ko(winners['B'], assigned_wildcards[85], "Round of 32")
        r32[86] = play_ko(winners['J'], runners_up['H'], "Round of 32")
        r32[87] = play_ko(winners['K'], assigned_wildcards[87], "Round of 32")
        r32[88] = play_ko(runners_up['D'], runners_up['G'], "Round of 32")

        # Round of 16 (Matches 89 to 96)
        r16 = {}
        r16[89] = play_ko(r32[73], r32[74], "Round of 16")
        r16[90] = play_ko(r32[75], r32[77], "Round of 16")
        r16[91] = play_ko(r32[76], r32[78], "Round of 16")
        r16[92] = play_ko(r32[79], r32[80], "Round of 16")
        r16[93] = play_ko(r32[83], r32[84], "Round of 16")
        r16[94] = play_ko(r32[81], r32[82], "Round of 16")
        r16[95] = play_ko(r32[86], r32[88], "Round of 16")
        r16[96] = play_ko(r32[85], r32[87], "Round of 16")

        for t in r16.values(): stats[id_to_team[t]]["R16"] += 1

        # Quarterfinals (Matches 97 to 100)
        qf = {}
        qf[97] = play_ko(r16[89], r16[90], "Quarterfinal")
        qf[98] = play_ko(r16[93], r16[94], "Quarterfinal")
        qf[99] = play_ko(r16[91], r16[92], "Quarterfinal")
        qf[100] = play_ko(r16[95], r16[96], "Quarterfinal")

        for t in qf.values(): stats[id_to_team[t]]["QF"] += 1

        # Semifinals (Matches 101 & 102)
        sf = {}
        sf[101] = play_ko(qf[97], qf[98], "Semifinal")
        sf[102] = play_ko(qf[99], qf[100], "Semifinal")

        for t in sf.values(): stats[id_to_team[t]]["SF"] += 1

        # Finalists & Champion
        stats[id_to_team[sf[101]]]["Final"] += 1
        stats[id_to_team[sf[102]]]["Final"] += 1

        champion = play_ko(sf[101], sf[102], "Final")
        stats[id_to_team[champion]]["Champ"] += 1

    # Format Survival Matrix DataFrame
    matrix_records = []
    for team, steps in stats.items():
        matrix_records.append({
            "Team": team,
            "Round of 32 %": (steps["R32"] / num_sims) * 100,
            "Round of 16 %": (steps["R16"] / num_sims) * 100,
            "Quarterfinals %": (steps["QF"] / num_sims) * 100,
            "Semifinals %": (steps["SF"] / num_sims) * 100,
            "Final %": (steps["Final"] / num_sims) * 100,
            "Champion %": (steps["Champ"] / num_sims) * 100
        })
        
    df_matrix = pd.DataFrame(matrix_records).sort_values(by="Champion %", ascending=False)
    df_upsets = pd.DataFrame(upsets)
    df_thrillers = pd.DataFrame(thrillers).sort_values(by="Total Goals", ascending=False) if thrillers else pd.DataFrame()
    
    return df_matrix, df_upsets, df_thrillers

# --- TRIGGER BUTTON & DISPLAY ---
if st.button("🚀 Run 1,000 Multiverse Simulations"):
    with st.spinner("Simulating 1,000 full tournament brackets across parallel dimensions..."):
        df_matrix, df_upsets, df_thrillers = run_1000_tournaments(tourney_chaos)
        
    st.write("---")
    
    # 1. SURVIVAL MATRIX DISPLAY
    st.subheader("🏆 1. The World Cup Survival Matrix")
    st.markdown("Percentage chance of each country reaching each progressive milestone across 1,000 realities:")
    st.dataframe(
        df_matrix.style.format({
            "Round of 32 %": "{:.1f}%",
            "Round of 16 %": "{:.1f}%",
            "Quarterfinals %": "{:.1f}%",
            "Semifinals %": "{:.1f}%",
            "Final %": "{:.1f}%",
            "Champion %": "{:.1f}%"
        }),
        use_container_width=True,
        height=500
    )

    # 2. DRAMATIC UPSETS LOG DISPLAY
    st.write("---")
    st.subheader("⚡ 2. Multiverse Shockers: Major Knockout Upsets")
    st.markdown("Notable instances where a lower-ranked team defeated an elite favorite in a knockout tie:")
    if not df_upsets.empty:
        st.dataframe(df_upsets, use_container_width=True)
    else:
        st.info("No giant-killings met the threshold in this run. Try increasing the Chaos Level!")

    # 3. HIGH SCORING THRILLERS LOG DISPLAY
    st.write("---")
    st.subheader("🔥 3. High-Scoring Multiverse Thrillers")
    st.markdown("A sample of the highest scoring, wild goalfests recorded during the 100,000+ total matches:")
    if not df_thrillers.empty:
        st.dataframe(df_thrillers, use_container_width=True)
    else:
        st.info("No extreme thrillers met the score threshold in this run.")
