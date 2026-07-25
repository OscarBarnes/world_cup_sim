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

# ---------------------------------------------------------
# 3. INTERACTIVE MATCH SIMULATION
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
# 4. THE GRAND TOURNAMENT SIMULATION & MULTIVERSE LOGS
# ---------------------------------------------------------
st.header("📊 4. The Grand Tournament Multiverse Engine")

st.markdown("""
Welcome to the main event. While single-match predictions are useful, tournament football introduces cumulative fatigue, 
bracket pathways, and knockout chaos. 

Below, the engine simulates **1,000 complete 2026 World Cup tournaments** (104,000 total matches). It maps each team's 
path through the 12-group phase, the 32-team knockout bracket, and tracks the most dramatic realities in the multiverse.
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
