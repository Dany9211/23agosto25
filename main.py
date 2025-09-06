import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import json

# =========================================================================
# IMPOSTAZIONI PAGINA E TITOLO
# =========================================================================
st.set_page_config(page_title="Analisi Dati Campionati", layout="wide")
st.title("Filtro DB CGMBET (aggiornato per campionati_uniti_puliti.csv)")
st.markdown("---")

# =========================================================================
# CARICAMENTO DATI
# =========================================================================
@st.cache_data
def load_data(file_path):
    """
    Carica i dati da un file CSV con un delimitatore specifico.
    """
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', header=0)
        st.success(f"File CSV caricato con successo da {file_path}. Colonne: {len(df.columns)}")
        return df
    except Exception as e:
        st.error(f"Errore durante la lettura del file CSV: {e}")
        return pd.DataFrame()

df = load_data("campionati_uniti_puliti.csv")

# =========================================================================
# PULIZIA E PRE-ELABORAZIONE DEI DATI
# =========================================================================
if not df.empty:
    # Mappatura dei nomi delle colonne dal file all'applicazione
    column_map = {
        'date_GMT': 'Data',
        'league': 'League',
        'home_team_name': 'Home_Team',
        'away_team_name': 'Away_Team',
        'home_team_goal_count': 'Gol_Home_FT',
        'away_team_goal_count': 'Gol_Away_FT',
        'home_team_goal_count_half_time': 'Gol_Home_HT',
        'away_team_goal_count_half_time': 'Gol_Away_HT',
        'Game Week': 'Giornata',
        'odds_ft_home_team_win': 'Odd_Home',
        'odds_ft_draw': 'Odd_Draw',
        'odds_ft_away_team_win': 'Odd_Away',
        'odds_ft_over15': 'Odd_Over_1.5',
        'odds_ft_over25': 'Odd_Over_2.5',
        'odds_ft_over35': 'Odd_Over_3.5',
        'odds_ft_over45': 'Odd_Over_4.5',
        'odds_ft_under15': 'Odd_Under_1.5',
        'odds_ft_under25': 'Odd_Under_2.5',
        'odds_ft_under35': 'Odd_Under_3.5',
        'odds_ft_under45': 'Odd_Under_4.5',
        'odds_btts_yes': 'BTTS_SI',
        'odds_btts_no': 'BTTS_NO',
        'Risultato_FT': 'Risultato_FT_Mio',
        'Risultato_HT': 'Risultato_HT_Mio',
        'timestamp': 'Timestamp',
        'home_team_goal_timings': 'Minutaggio_Gol_Home',
        'away_team_goal_timings': 'Minutaggio_Gol_Away',
        'home_team_corner_count': 'Corner_Home',
        'away_team_corner_count': 'Corner_Away',
        'home_team_shots': 'Tiri_Home',
        'away_team_shots': 'Tiri_Away',
        'home_team_shots_on_target': 'Tiri_in_Porta_Home',
        'away_team_shots_on_target': 'Tiri_in_Porta_Away',
        'home_team_fouls': 'Falli_Home',
        'away_team_fouls': 'Falli_Away',
        'home_team_yellow_cards': 'Gialli_Home',
        'away_team_yellow_cards': 'Gialli_Away',
        'home_team_red_cards': 'Rossi_Home',
        'away_team_red_cards': 'Rossi_Away',
        'home_team_possession': 'Possesso_Home',
        'away_team_possession': 'Possesso_Away',
        'Home Team Pre-Match xG': 'xG_Home',
        'Away Team Pre-Match xG': 'xG_Away',
        'team_a_xg': 'xG_Finale_Home',
        'team_b_xg': 'xG_Finale_Away',
    }
    df.rename(columns=column_map, inplace=True)

    # Sostituisci i valori NaN e vuoti con un valore predefinito
    df.fillna('', inplace=True)

    # Gestione delle colonne numeriche con virgola come separatore decimale
    cols_to_convert_float = [
        'Odd_Home', 'Odd_Draw', 'Odd_Away', 'Odd_Over_1.5', 'Odd_Over_2.5', 'Odd_Over_3.5', 'Odd_Over_4.5', 
        'Odd_Under_1.5', 'Odd_Under_2.5', 'Odd_Under_3.5', 'Odd_Under_4.5',
        'BTTS_SI', 'BTTS_NO', 'xG_Home', 'xG_Away', 'xG_Finale_Home', 'xG_Finale_Away'
    ]
    for col in cols_to_convert_float:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=True).replace('', np.nan).astype(float)

    # Conversione della colonna 'Data' e 'Anno'
    if 'Data' in df.columns:
        df['Data'] = pd.to_datetime(df['Data'], format='%b %d %Y - %I:%M%p', errors='coerce')
        df['Anno'] = df['Data'].dt.year
    
    # Conversione a int delle colonne goal/statistiche
    cols_to_convert_int = [
        'Gol_Home_FT', 'Gol_Away_FT', 'Gol_Home_HT', 'Gol_Away_HT', 'Giornata', 
        'Corner_Home', 'Corner_Away', 'Tiri_Home', 'Tiri_Away', 'Tiri_in_Porta_Home', 
        'Tiri_in_Porta_Away', 'Falli_Home', 'Falli_Away', 'Gialli_Home', 'Gialli_Away', 
        'Rossi_Home', 'Rossi_Away', 'Possesso_Home', 'Possesso_Away'
    ]
    for col in cols_to_convert_int:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Creazione di colonne aggiuntive per analisi
    df['Tot_Gol_FT'] = df['Gol_Home_FT'] + df['Gol_Away_FT']
    df['Tot_Gol_HT'] = df['Gol_Home_HT'] + df['Gol_Away_HT']
    df['Tot_Gol_2T'] = df['Tot_Gol_FT'] - df['Tot_Gol_HT']
    df['Gol_Casa_2T'] = df['Gol_Home_FT'] - df['Gol_Home_HT']
    df['Gol_Trasferta_2T'] = df['Gol_Away_FT'] - df['Gol_Away_HT']
    
    # Creazione delle colonne di risultato in formato testuale/numerico
    df['Risultato_FT_Numerico'] = np.where(df['Gol_Home_FT'] > df['Gol_Away_FT'], 1, np.where(df['Gol_Home_FT'] < df['Gol_Away_FT'], 2, 'X'))
    df['Risultato_HT_Numerico'] = np.where(df['Gol_Home_HT'] > df['Gol_Away_HT'], 1, np.where(df['Gol_Home_HT'] < df['Gol_Away_HT'], 2, 'X'))
else:
    st.stop()

# =========================================================================
# FUNZIONI DI BACKTEST
# =========================================================================

def esegui_backtest(df, market, strategy, stake, lay_commission=0.05):
    """
    Esegue un backtest su un DataFrame filtrato per un mercato e una strategia specifici.
    """
    profit_loss = 0
    numero_scommesse = 0
    vincite = 0
    perdite = 0
    
    odd_column_map = {
        "1 (Casa)": "Odd_Home", "X (Pareggio)": "Odd_Draw", "2 (Trasferta)": "Odd_Away",
        "BTTS SI FT": "BTTS_SI", "BTTS NO FT": "BTTS_NO",
        "Over 0.5 FT": "Odd_Over_0.5", "Over 1.5 FT": "Odd_Over_1.5", "Over 2.5 FT": "Odd_Over_2.5", "Over 3.5 FT": "Odd_Over_3.5", "Over 4.5 FT": "Odd_Over_4.5",
        "Under 0.5 FT": "Odd_Under_0.5", "Under 1.5 FT": "Odd_Under_1.5", "Under 2.5 FT": "Odd_Under_2.5", "Under 3.5 FT": "Odd_Under_3.5", "Under 4.5 FT": "Odd_Under_4.5",
        "Over 0.5 HT": "Odd_Over_0.5_HT", "Over 1.5 HT": "Odd_Over_1.5_HT", "Over 2.5 HT": "Odd_Over_2.5_HT",
    }
    odd_col = odd_column_map.get(market)
    
    if odd_col not in df.columns or df[odd_col].isnull().all():
        return None
    
    for _, row in df.iterrows():
        odd = row[odd_col]
        
        if pd.isna(odd) or odd <= 1.0:
            continue
            
        numero_scommesse += 1
        
        is_winning_bet = False
        
        # Logica per il mercato e la vittoria
        if market == "1 (Casa)":
            is_winning_bet = (row['Gol_Home_FT'] > row['Gol_Away_FT'])
        elif market == "X (Pareggio)":
            is_winning_bet = (row['Gol_Home_FT'] == row['Gol_Away_FT'])
        elif market == "2 (Trasferta)":
            is_winning_bet = (row['Gol_Home_FT'] < row['Gol_Away_FT'])
        elif market == "BTTS SI FT":
            is_winning_bet = (row['Gol_Home_FT'] > 0 and row['Gol_Away_FT'] > 0)
        elif market == "BTTS NO FT":
            is_winning_bet = (row['Gol_Home_FT'] == 0 or row['Gol_Away_FT'] == 0)
        elif market.startswith("Over"):
            value = float(market.split()[1])
            is_winning_bet = (row['Tot_Gol_FT'] > value)
        elif market.startswith("Under"):
            value = float(market.split()[1])
            is_winning_bet = (row['Tot_Gol_FT'] < value)
        elif market.startswith("Over") and "HT" in market:
            value = float(market.split()[1])
            is_winning_bet = (row['Tot_Gol_HT'] > value)
        
        # Applica la strategia (Back o Lay)
        if strategy == "Back":
            if is_winning_bet:
                profit_loss += (stake * odd) - stake
                vincite += 1
            else:
                profit_loss -= stake
                perdite += 1
        elif strategy == "Lay":
            if is_winning_bet:
                profit_loss -= stake * (odd - 1) * (1 + lay_commission)
                perdite += 1
            else:
                profit_loss += stake * (1 - lay_commission)
                vincite += 1

    roi = (profit_loss / (numero_scommesse * stake) * 100) if numero_scommesse > 0 else 0
    win_rate = (vincite / numero_scommesse * 100) if numero_scommesse > 0 else 0
    odd_minima = df[odd_col].min() if not df[odd_col].empty else None

    return {
        "Mercato": market,
        "Strategia": strategy,
        "Scommesse": numero_scommesse,
        "Vincite": vincite,
        "Perdite": perdite,
        "Profitto (€)": round(profit_loss, 2),
        "ROI %": round(roi, 2),
        "WinRate %": round(win_rate, 2),
        "Odd Minima": round(odd_minima, 2) if odd_minima else "-",
    }

def display_backtest_results(results):
    if not results:
        st.warning("Nessun dato di backtest da mostrare.")
        return
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="ROI %", ascending=False).reset_index(drop=True)
    st.dataframe(df_results)

# =========================================================================
# FILTRI PER L'UTENTE
# =========================================================================
st.sidebar.header("Filtri Dati")

# Filtro per League
if 'League' in df.columns:
    unique_leagues = sorted(df['League'].unique())
    selected_leagues = st.sidebar.multiselect('Seleziona Campionato', unique_leagues)
    if selected_leagues:
        df = df[df['League'].isin(selected_leagues)]

# Filtro per anno
if 'Anno' in df.columns:
    unique_years = sorted(df['Anno'].unique(), reverse=True)
    selected_years = st.sidebar.multiselect('Seleziona Anno', unique_years, default=unique_years)
    if selected_years:
        df = df[df['Anno'].isin(selected_years)]

# Filtri per squadre
unique_teams = sorted(pd.concat([df['Home_Team'], df['Away_Team']]).unique())
selected_home_team = st.sidebar.multiselect('Seleziona Squadra di Casa', unique_teams)
selected_away_team = st.sidebar.multiselect('Seleziona Squadra in Trasferta', unique_teams)

if selected_home_team:
    df = df[df['Home_Team'].isin(selected_home_team)]
if selected_away_team:
    df = df[df['Away_Team'].isin(selected_away_team)]

# Filtri avanzati per gol e quote
st.sidebar.subheader("Filtri per Risultato e Goal")
selected_ft_result = st.sidebar.multiselect('Filtro Risultato Finale', ['1', 'X', '2'])
if selected_ft_result:
    df = df[df['Risultato_FT_Numerico'].isin(selected_ft_result)]

selected_ht_result = st.sidebar.multiselect('Filtro Risultato Primo Tempo', ['1', 'X', '2'])
if selected_ht_result:
    df = df[df['Risultato_HT_Numerico'].isin(selected_ht_result)]

# Filtri Over/Under FT
st.sidebar.subheader("Filtri Over/Under FT")
over_options = [f"Over {x}.5" for x in range(5)]
under_options = [f"Under {x}.5" for x in range(5)]
selected_overs = st.sidebar.multiselect("Seleziona Over", over_options)
selected_unders = st.sidebar.multiselect("Seleziona Under", under_options)

for opt in selected_overs:
    val = float(opt.split()[1])
    df = df[df['Tot_Gol_FT'] > val]
for opt in selected_unders:
    val = float(opt.split()[1])
    df = df[df['Tot_Gol_FT'] < val]
    
# Filtri per quote
st.sidebar.subheader("Filtri per Quote")
odd_filters = {
    'Odd_Home': st.sidebar.slider('Odd Casa', min_value=1.0, max_value=20.0, value=(1.0, 20.0), step=0.1),
    'Odd_Draw': st.sidebar.slider('Odd Pareggio', min_value=1.0, max_value=20.0, value=(1.0, 20.0), step=0.1),
    'Odd_Away': st.sidebar.slider('Odd Trasferta', min_value=1.0, max_value=20.0, value=(1.0, 20.0), step=0.1),
    'Odd_Over_2.5': st.sidebar.slider('Odd Over 2.5', min_value=1.0, max_value=10.0, value=(1.0, 10.0), step=0.1),
    'BTTS_SI': st.sidebar.slider('Odd BTTS SI', min_value=1.0, max_value=10.0, value=(1.0, 10.0), step=0.1),
}
for col, val_range in odd_filters.items():
    if col in df.columns:
        df = df[(df[col] >= val_range[0]) & (df[col] <= val_range[1])]

# Filtro per goal nel primo tempo
st.sidebar.subheader("Filtri Goal nel 1° Tempo")
min_gol_ht = st.sidebar.slider('Min. Gol nel 1° Tempo', min_value=0, max_value=10, value=0, step=1)
max_gol_ht = st.sidebar.slider('Max. Gol nel 1° Tempo', min_value=0, max_value=10, value=10, step=1)
df = df[(df['Tot_Gol_HT'] >= min_gol_ht) & (df['Tot_Gol_HT'] <= max_gol_ht)]

# Filtro per goal nel secondo tempo
st.sidebar.subheader("Filtri Goal nel 2° Tempo")
min_gol_2t = st.sidebar.slider('Min. Gol nel 2° Tempo', min_value=0, max_value=10, value=0, step=1)
max_gol_2t = st.sidebar.slider('Max. Gol nel 2° Tempo', min_value=0, max_value=10, value=10, step=1)
df = df[(df['Tot_Gol_2T'] >= min_gol_2t) & (df['Tot_Gol_2T'] <= max_gol_2t)]

# =========================================================================
# MAIN DASHBOARD E ANALISI
# =========================================================================
if not df.empty:
    st.write(f"### Dati Filtrati ({len(df)} partite)")
    st.dataframe(df)
    
    st.download_button(
        label="Scarica i dati filtrati",
        data=df.to_csv(index=False),
        file_name='dati_filtrati.csv',
        mime='text/csv',
    )
    
    st.markdown("---")
    
    # === Analisi Statistiche ===
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Analisi Risultato Finale")
        if not df.empty:
            ft_results = df['Risultato_FT_Numerico'].value_counts(normalize=True).mul(100).rename('Percentuale %').reset_index()
            ft_results.rename(columns={'index': 'Risultato'}, inplace=True)
            st.dataframe(ft_results)
        else:
            st.warning("Nessun dato filtrato.")

    with col2:
        st.subheader("Analisi Risultato Primo Tempo")
        if not df.empty:
            ht_results = df['Risultato_HT_Numerico'].value_counts(normalize=True).mul(100).rename('Percentuale %').reset_index()
            ht_results.rename(columns={'index': 'Risultato'}, inplace=True)
            st.dataframe(ht_results)
        else:
            st.warning("Nessun dato filtrato.")
            
    st.markdown("---")

    # === Backtest Automatico ===
    st.subheader("Backtest Automatico su Tutti i Mercati")
    stake = st.number_input(
        "Stake per scommessa", 
        min_value=1.0, 
        value=1.0, 
        step=0.5, 
        key="auto_bt_stake"
    )

    all_markets = [
        "1 (Casa)", "X (Pareggio)", "2 (Trasferta)",
        "BTTS SI FT", "BTTS NO FT",
        "Over 1.5 FT", "Over 2.5 FT", "Over 3.5 FT", "Over 4.5 FT",
        "Under 1.5 FT", "Under 2.5 FT", "Under 3.5 FT", "Under 4.5 FT",
    ]
    strategies = ["Back", "Lay"]
    
    results = []
    if not df.empty:
        for mkt in all_markets:
            for strat in strategies:
                res = esegui_backtest(df, mkt, strat, stake)
                if res:
                    results.append(res)
    
    display_backtest_results(results)

    st.markdown("---")

    # === Creazione di Pattern ===
    st.subheader("Crea e Salva il Tuo Pattern")
    pattern_name = st.text_input("Nome del tuo pattern:", "Pattern Personalizzato")
    
    if st.button("Salva Pattern"):
        pattern_details = {
            "Nome": pattern_name,
            "Filtri Applicati": {
                "Campionati": selected_leagues if 'selected_leagues' in locals() else 'Tutti',
                "Anni": selected_years if 'selected_years' in locals() else 'Tutti',
                "Squadre Casa": selected_home_team if 'selected_home_team' in locals() else 'Tutte',
                "Squadre Trasferta": selected_away_team if 'selected_away_team' in locals() else 'Tutte',
                "Risultato FT": selected_ft_result,
                "Risultato HT": selected_ht_result,
                "Over FT": selected_overs,
                "Under FT": selected_unders,
                "Odd Casa": odd_filters['Odd_Home'],
                "Odd Pareggio": odd_filters['Odd_Draw'],
                "Odd Trasferta": odd_filters['Odd_Away'],
                "Odd Over 2.5": odd_filters['Odd_Over_2.5'],
                "Odd BTTS SI": odd_filters['BTTS_SI'],
                "Min Gol HT": min_gol_ht,
                "Max Gol HT": max_gol_ht,
                "Min Gol 2T": min_gol_2t,
                "Max Gol 2T": max_gol_2t,
            },
            "Risultati Backtest": results
        }
        
        # Salva il pattern in un file
        pattern_file_path = f"{pattern_name.replace(' ', '_').replace('.', '')}.json"
        with open(pattern_file_path, "w") as f:
            json.dump(pattern_details, f, indent=4)
        
        st.success(f"Pattern '{pattern_name}' salvato con successo in {pattern_file_path}!")

