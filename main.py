import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

st.set_page_config(page_title="Filtri Dati Calcio", layout="wide")
st.title("⚽ Dashboard Filtri partite squadre Calcio")
st.write("Carica il tuo file CSV per iniziare l'analisi.")

# ---------- Utility ----------
def odd_min_from_percent(p: float):
    if p and p > 0:
        return round(100.0 / p, 2)
    return None

def style_table(df: pd.DataFrame, percent_cols):
    if isinstance(percent_cols, str):
        percent_cols = [percent_cols]
    fmt = {col: "{:.2f}%" for col in percent_cols}
    fmt.update({
        "Odd Minima": lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        "Odd Minima >= 2 Gol": lambda x: "-" if pd.isna(x) else f"{x:.2f}",
        "Conteggio": "{:,.0f}",
        "Partite con Gol": "{:,.0f}",
    })
    styler = (df.style
                .format(fmt)
                .background_gradient(subset=percent_cols, cmap="RdYlGn")
                .set_properties(**{"text-align": "center"})
                .set_table_styles([{ 'selector': 'th', 'props': 'text-align: center;' }])
            )
    return styler

def gen_buckets(step:int):
    if step == 15:
        return ['0-15','16-30','31-45','45+','46-60','61-75','76-90','90+']
    labels = ['0-5','6-10','11-15','16-20','21-25','26-30','31-35','36-40','41-45','45+',
              '46-50','51-55','56-60','61-65','66-70','71-75','76-80','81-85','86-90','90+']
    return labels

def bucket_label_5min(minute:int):
    if minute <= 5: return '0-5'
    if minute <= 10: return '6-10'
    if minute <= 15: return '11-15'
    if minute <= 20: return '16-20'
    if minute <= 25: return '21-25'
    if minute <= 30: return '26-30'
    if minute <= 35: return '31-35'
    if minute <= 40: return '36-40'
    if minute <= 45: return '41-45'
    if minute <= 50: return '46-50'
    if minute <= 55: return '51-55'
    if minute <= 60: return '56-60'
    if minute <= 65: return '61-65'
    if minute <= 70: return '66-70'
    if minute <= 75: return '71-75'
    if minute <= 80: return '76-80'
    if minute <= 85: return '81-85'
    if minute <= 90: return '86-90'
    return '90+'

def bucket_label_15min(minute:int):
    if minute <= 15: return '0-15'
    if minute <= 30: return '16-30'
    if minute <= 45: return '31-45'
    if minute <= 60: return '46-60'
    if minute <= 75: return '61-75'
    if minute <= 90: return '76-90'
    return '90+'

def buckets_from_tokens_step(cell, step:int):
    result = []
    if pd.isna(cell) or str(cell).strip()=='': return result
    for token in [t.strip() for t in str(cell).split(',') if t.strip()!='']:
        m = re.fullmatch(r"(\d+)'(\d+)", token)
        if m:
            base = int(m.group(1))
            if base == 45:
                result.append('45+'); continue
            elif base >= 90:
                result.append('90+'); continue
            minute = base + int(m.group(2))
        else:
            p_clean = re.sub(r"[^0-9]", "", token)
            if p_clean == "": continue
            minute = int(p_clean)
        result.append(bucket_label_5min(minute) if step==5 else bucket_label_15min(minute))
    return result

def timeframes_table(df_subset: pd.DataFrame, step:int):
    buckets = gen_buckets(step)
    counts_matches_with_goal = {b: 0 for b in buckets}
    counts_matches_with_2plus = {b: 0 for b in buckets}
    total_matches = len(df_subset)

    for _, row in df_subset.iterrows():
        b_list = buckets_from_tokens_step(row.get('home_team_goal_timings', np.nan), step) +                  buckets_from_tokens_step(row.get('away_team_goal_timings', np.nan), step)
        if not b_list:
            continue
        per_bucket = {b:0 for b in buckets}
        for b in b_list:
            if b in per_bucket:
                per_bucket[b] += 1
        for b, c in per_bucket.items():
            if c >= 1:
                counts_matches_with_goal[b] += 1
            if c >= 2:
                counts_matches_with_2plus[b] += 1

    rows = []
    for b in buckets:
        with_goal = counts_matches_with_goal[b]
        pct = round((with_goal / total_matches) * 100, 2) if total_matches else 0.0
        odd_min = odd_min_from_percent(pct)
        g2 = counts_matches_with_2plus[b]
        pct2 = round((g2 / total_matches) * 100, 2) if total_matches else 0.0
        odd_min2 = odd_min_from_percent(pct2) if pct2 > 0 else None
        rows.append([b, with_goal, pct, odd_min, pct2, odd_min2])

    tf_df = pd.DataFrame(rows, columns=['Timeframe','Partite con Gol','Percentuale %','Odd Minima','>= 2 Gol %','Odd Minima >= 2 Gol'])
    return tf_df

def parse_minutes_numeric(cell):
    if pd.isna(cell) or str(cell).strip()=='': return []
    out = []
    for token in [t.strip() for t in str(cell).split(',') if t.strip()!='']:
        m = re.fullmatch(r"(\d+)'(\d+)", token)
        if m:
            base = int(m.group(1)); extra = int(m.group(2))
            out.append(base + extra)
        else:
            p_clean = re.sub(r"[^0-9]", "", token)
            if p_clean != "":
                out.append(int(p_clean))
    return out

def earliest_first_half_min(cell):
    if pd.isna(cell) or str(cell).strip()=='':
        return None
    earliest = None
    for token in [t.strip() for t in str(cell).split(',') if t.strip()!='']:
        m = re.fullmatch(r"(\d+)'(\d+)", token)
        if m:
            base = int(m.group(1)); extra = int(m.group(2))
            if base < 45:
                minute = base + extra
            elif base == 45:
                minute = 45 + extra
            else:
                continue
        else:
            digits = re.sub(r"[^0-9]", "", token)
            if digits == "":
                continue
            minute = int(digits)
            if minute > 45:
                continue
        if earliest is None or minute < earliest:
            earliest = minute
    return earliest

def compute_first_to_score_ht(df):
    total_matches = len(df)
    home_first = away_first = no_goal = simultaneous = 0
    if {'home_team_goal_timings','away_team_goal_timings'}.issubset(df.columns):
        for _, row in df.iterrows():
            h_min = earliest_first_half_min(row.get('home_team_goal_timings', np.nan))
            a_min = earliest_first_half_min(row.get('away_team_goal_timings', np.nan))
            if h_min is None and a_min is None:
                no_goal += 1
            elif h_min is not None and (a_min is None or h_min < a_min):
                home_first += 1
            elif a_min is not None and (h_min is None or a_min < h_min):
                away_first += 1
            else:
                simultaneous += 1
    fts_df = pd.DataFrame({
        'Esito': ['Home First (HT)', 'Away First (HT)', 'No Goal (HT)', 'Stesso minuto (HT)'],
        'Conteggio': [home_first, away_first, no_goal, simultaneous]
    })
    fts_df['Percentuale %'] = (fts_df['Conteggio'] / total_matches * 100).round(2) if total_matches else 0
    fts_df['Odd Minima'] = fts_df['Percentuale %'].apply(odd_min_from_percent)
    return fts_df

def minutes_second_half(cell):
    """Return list of minutes in second half (>=46), including 90'+."""
    if pd.isna(cell) or str(cell).strip()=='': return []
    out = []
    for token in [t.strip() for t in str(cell).split(',') if t.strip()!='']:
        m = re.fullmatch(r"(\d+)'(\d+)", token)
        if m:
            base = int(m.group(1)); extra = int(m.group(2))
            if base >= 46:
                out.append(base + extra)
        else:
            digits = re.sub(r"[^0-9]", "", token)
            if digits != "" and int(digits) >= 46:
                out.append(int(digits))
    return out

def earliest_second_half_min(cell):
    mins = minutes_second_half(cell)
    return min(mins) if mins else None

def sh_cs_label(h, a):
    if 0 <= h <= 3 and 0 <= a <= 3:
        return f"{h} - {a}"
    if h == a:
        return "Any Other Draw"
    elif h > a:
        return "Any Other Home Win"
    else:
        return "Any Other Away Win"

def ft_cs_label(h, a):
    if 0 <= h <= 3 and 0 <= a <= 3:
        return f"{h} - {a}"
    if h == a:
        return "Any Other Draw"
    elif h > a:
        return "Any Other Home Win"
    else:
        return "Any Other Away Win"

def earliest_full_time_min(cell):
    if pd.isna(cell) or str(cell).strip() == '':
        return None
    
    earliest = None
    for token in [t.strip() for t in str(cell).split(',') if t.strip() != '']:
        m = re.fullmatch(r"(\d+)'(\d+)", token)
        if m:
            base = int(m.group(1)); extra = int(m.group(2))
            minute = base + extra
        else:
            digits = re.sub(r"[^0-9]", "", token)
            if digits == "":
                continue
            minute = int(digits)
        
        if earliest is None or minute < earliest:
            earliest = minute
            
    return earliest

def compute_first_to_score_ft(df):
    total_matches = len(df)
    home_first = away_first = no_goal = simultaneous = 0
    if {'home_team_goal_timings', 'away_team_goal_timings'}.issubset(df.columns):
        for _, row in df.iterrows():
            h_min = earliest_full_time_min(row.get('home_team_goal_timings', np.nan))
            a_min = earliest_full_time_min(row.get('away_team_goal_timings', np.nan))
            if h_min is None and a_min is None:
                no_goal += 1
            elif h_min is not None and (a_min is None or h_min < a_min):
                home_first += 1
            elif a_min is not None and (h_min is None or a_min < h_min):
                away_first += 1
            else:
                simultaneous += 1
    
    fts_df = pd.DataFrame({
        'Esito': ['Home First (FT)', 'Away First (FT)', 'No Goal (FT)', 'Stesso minuto (FT)'],
        'Conteggio': [home_first, away_first, no_goal, simultaneous]
    })
    fts_df['Percentuale %'] = (fts_df['Conteggio'] / total_matches * 100).round(2) if total_matches else 0
    fts_df['Odd Minima'] = fts_df['Percentuale %'].apply(odd_min_from_percent)
    return fts_df

def create_conversion_rate_table(df):
    required_cols = {'home_team_goal_count', 'away_team_goal_count', 'home_team_shots_on_target', 'away_team_shots_on_target'}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()
    
    df_temp = df.copy()
    df_temp.fillna(0, inplace=True)

    df_temp['total_goals'] = df_temp['home_team_goal_count'] + df_temp['away_team_goal_count']
    df_temp['total_shots_on_target'] = df_temp['home_team_shots_on_target'] + df_temp['away_team_shots_on_target']
    
    df_temp = df_temp[df_temp['total_shots_on_target'] > 0].copy()

    data = []
    for shots in range(1, 11):
        subset = df_temp[df_temp['total_shots_on_target'] == shots]
        if not subset.empty:
            avg_goals = subset['total_goals'].mean()
            avg_goals_rounded = round(avg_goals, 2)
            data.append([shots, avg_goals_rounded])
            
    return pd.DataFrame(data, columns=['Tiri in Porta', 'Media Gol'])

def create_avg_goals_summary_table(df):
    required_cols_ht = {'total_goals_at_half_time'}
    required_cols_ft = {'total_goals_at_full_time'}
    
    data = {}
    
    if required_cols_ht.issubset(df.columns):
        avg_goals_ht = df['total_goals_at_half_time'].mean()
        data['Media Gol HT'] = round(avg_goals_ht, 2)
    else:
        data['Media Gol HT'] = 'N/A'
    
    if required_cols_ft.issubset(df.columns) and required_cols_ht.issubset(df.columns):
        sh_goals = df['total_goals_at_full_time'] - df['total_goals_at_half_time']
        avg_goals_sh = sh_goals.mean()
        data['Media Gol SH'] = round(avg_goals_sh, 2)
    else:
        data['Media Gol SH'] = 'N/A'
    
    if required_cols_ft.issubset(df.columns):
        avg_goals_ft = df['total_goals_at_full_time'].mean()
        data['Media Gol FT'] = round(avg_goals_ft, 2)
    else:
        data['Media Gol FT'] = 'N/A'
        
    summary_df = pd.DataFrame([data])
    summary_df = summary_df.transpose().rename(columns={0: 'Valore'})
    
    return summary_df
    
def create_year_summary_table(base_df, odds_df):
    if 'anno' not in base_df.columns or 'anno' not in odds_df.columns:
        return pd.DataFrame()
    
    base_counts = base_df.groupby('anno').size().rename('Conteggio Totale')
    odds_counts = odds_df.groupby('anno').size().rename('Conteggio Filtrato Quote')
    
    if 'date' in base_df.columns:
        base_recent_date = base_df.groupby('anno')['date'].max().rename('Data Recente Totale').dt.strftime('%d-%m-%Y')
        odds_recent_date = odds_df.groupby('anno')['date'].max().rename('Data Recente Filtrato Quote').dt.strftime('%d-%m-%Y')
        
        summary_df = pd.concat([base_counts, base_recent_date, odds_counts, odds_recent_date], axis=1)
    else:
        summary_df = pd.concat([base_counts, odds_counts], axis=1)
        
    summary_df = summary_df.fillna('N/A')
    
    for col in summary_df.columns:
        if 'Conteggio' in col:
            summary_df[col] = summary_df[col].astype(int)
    
    summary_df = summary_df.reset_index()
    summary_df = summary_df.rename(columns={'anno': 'Anno'})
    
    return summary_df
    
def get_last_matches_info(df, home_team, away_team):
    """
    Trova l'ultima data e l'avversario per l'ultima partita in casa e fuori casa
    delle squadre selezionate.
    """
    info = {
        'home_team_home_match': None,
        'home_team_away_match': None,
        'away_team_home_match': None,
        'away_team_away_match': None
    }

    if 'date' in df.columns and 'home_team_name' in df.columns and 'away_team_name' in df.columns:
        df_sorted = df.sort_values(by='date', ascending=False)
        
        # Ultima partita in casa della squadra di casa selezionata
        last_home_match_home_team = df_sorted[df_sorted['home_team_name'] == home_team].iloc[0] if not df_sorted[df_sorted['home_team_name'] == home_team].empty else None
        if last_home_match_home_team is not None:
            info['home_team_home_match'] = {
                'date': last_home_match_home_team['date'].strftime('%d-%m-%Y'),
                'opponent': last_home_match_home_team['away_team_name']
            }
        
        # Ultima partita fuori casa della squadra di casa selezionata
        last_away_match_home_team = df_sorted[df_sorted['away_team_name'] == home_team].iloc[0] if not df_sorted[df_sorted['away_team_name'] == home_team].empty else None
        if last_away_match_home_team is not None:
            info['home_team_away_match'] = {
                'date': last_away_match_home_team['date'].strftime('%d-%m-%Y'),
                'opponent': last_away_match_home_team['home_team_name']
            }
            
        # Ultima partita in casa della squadra in trasferta selezionata
        last_home_match_away_team = df_sorted[df_sorted['home_team_name'] == away_team].iloc[0] if not df_sorted[df_sorted['home_team_name'] == away_team].empty else None
        if last_home_match_away_team is not None:
            info['away_team_home_match'] = {
                'date': last_home_match_away_team['date'].strftime('%d-%m-%Y'),
                'opponent': last_home_match_away_team['away_team_name']
            }

        # Ultima partita fuori casa della squadra in trasferta selezionata
        last_away_match_away_team = df_sorted[df_sorted['away_team_name'] == away_team].iloc[0] if not df_sorted[df_sorted['away_team_name'] == away_team].empty else None
        if last_away_match_away_team is not None:
            info['away_team_away_match'] = {
                'date': last_away_match_away_team['date'].strftime('%d-%m-%Y'),
                'opponent': last_away_match_away_team['home_team_name']
            }
            
    return info


# ---------- Load ----------
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=';', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Errore durante la lettura del file. Assicurati che sia un file CSV con separatore ';'. Errore: {e}")
        return pd.DataFrame()

    num_cols = [
        'home_team_goal_count_half_time','away_team_goal_count_half_time',
        'home_team_goal_count','away_team_goal_count',
        'home_team_shots_on_target','away_team_shots_on_target',
        'odds_ft_home_team_win','odds_ft_draw','odds_ft_away_team_win',
        'odds_ft_over25','anno','Game Week'
    ]
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col].replace({',':'.'}, regex=True, inplace=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'status' in df.columns:
        df = df[df['status'].str.lower() != 'incomplete']

    if {'giorno', 'mese', 'anno'}.issubset(df.columns) and 'date' not in df.columns:
        df['date'] = pd.to_datetime(df[['giorno', 'mese', 'anno']].astype(str).agg('-'.join, axis=1), format='%d-%m-%Y', errors='coerce')

    if {'home_team_goal_count_half_time','away_team_goal_count_half_time'}.issubset(df.columns):
        def get_ht_result(row):
            if row['home_team_goal_count_half_time'] > row['away_team_goal_count_half_time']:
                return 'Vittoria Casa'
            elif row['home_team_goal_count_half_time'] < row['away_team_goal_count_half_time']:
                return 'Vittoria Trasferta'
            else:
                return 'Pareggio'
        df['Risultato HT'] = df.apply(get_ht_result, axis=1)
        df['HT Score'] = df['home_team_goal_count_half_time'].astype('Int64').astype(str) + ' - ' + df['away_team_goal_count_half_time'].astype('Int64').astype(str)
        def map_to_betfair_ht_cs(row):
            h = int(row['home_team_goal_count_half_time']) if pd.notnull(row['home_team_goal_count_half_time']) else None
            a = int(row['away_team_goal_count_half_time']) if pd.notnull(row['away_team_goal_count_half_time']) else None
            if h is None or a is None: return None
            if 0 <= h <= 3 and 0 <= a <= 3: return f"{h} - {a}"
            if h == a: return 'Any Other Draw'
            elif h > a: return 'Any Other Home Win'
            else: return 'Any Other Away Win'
        df['HT CS (Betfair)'] = df.apply(map_to_betfair_ht_cs, axis=1)
        df['total_goals_at_half_time'] = df['home_team_goal_count_half_time'] + df['away_team_goal_count_half_time']

    if {'home_team_goal_count','away_team_goal_count'}.issubset(df.columns):
        df['total_goals_at_full_time'] = df['home_team_goal_count'] + df['away_team_goal_count']
    
    if 'date' in df.columns and not df['date'].isnull().all():
        df.sort_values(by='date', ascending=True, inplace=True)
    else:
        st.sidebar.warning("Colonna 'date' non valida. Le partite non verranno ordinate cronologicamente.")

    return df

# ---------- UI Upload ----------
uploaded_file = st.file_uploader("Scegli un file CSV", type=["csv"], key="uploader1")

if uploaded_file is None:
    st.info("In attesa di caricamento del file CSV.")
    st.stop()

with st.spinner('Caricamento dati in corso...'):
    df = load_data(uploaded_file)

if df.empty:
    st.error("Il file caricato è vuoto o non può essere processato.")
    st.stop()

st.success("File caricato con successo!")

latest_date_full_dataset = "N/A"
if 'date' in df.columns and not df['date'].isnull().all():
    latest_date_full_dataset = df['date'].max().strftime('%d-%m-%Y')

st.markdown(f"**Ultima data del dataset: {latest_date_full_dataset}**")

# ---------- Sidebar Filters ----------
st.sidebar.header("Opzioni di Filtraggio")
if 'league' in df.columns:
    leagues = sorted(df['league'].dropna().unique().tolist())
    selected_leagues = st.sidebar.multiselect("Seleziona Campionato(i)", leagues, default=[])
else:
    selected_leagues = []
    st.sidebar.warning("Colonna 'league' non trovata.")

selected_years = []
if 'anno' in df.columns and not df['anno'].isnull().all():
    years_series = df['anno'].dropna().astype(int)
    max_year = int(years_series.max())
    year_options = {
        'Ultimo anno': [max_year],
        'Ultimi 2 anni': list(range(max_year-1, max_year+1)),
        'Ultimi 3 anni': list(range(max_year-2, max_year+1)),
        'Ultimi 4 anni': list(range(max_year-3, max_year+1)),
        'Ultimi 5 anni': list(range(max_year-4, max_year+1)),
        'Ultimi 6 anni': list(range(max_year-5, max_year+1)),
        'Ultimi 7 anni': list(range(max_year-6, max_year+1)),
        'Ultimi 8 anni': list(range(max_year-7, max_year+1)),
        'Ultimi 9 anni': list(range(max_year-8, max_year+1)),
        'Ultimi 10 anni': list(range(max_year-9, max_year+1)),
        'Tutti': sorted(years_series.unique().tolist())
    }
    year_choice = st.sidebar.selectbox("Seleziona Anno/i", options=list(year_options.keys()), index=len(year_options) - 1)
    selected_years = year_options[year_choice]
else:
    st.sidebar.warning("Colonna 'anno' non trovata.")

if 'Game Week' in df.columns and not df['Game Week'].isnull().all():
    gws = sorted(df['Game Week'].dropna().astype(int).unique().tolist())
    gw_min, gw_max = min(gws), max(gws)
    selected_gw = st.sidebar.slider("Range Giornata", min_value=gw_min, max_value=gw_max, value=(gw_min, gw_max))
else:
    selected_gw = None
    st.sidebar.warning("Colonna 'Game Week' non trovata.")

# Filtri Squadra
teams = []
if 'home_team_name' in df.columns and 'away_team_name' in df.columns:
    all_teams = pd.unique(df[['home_team_name', 'away_team_name']].values.ravel('K'))
    teams = sorted(all_teams.tolist())
    selected_home_team = st.sidebar.selectbox("Seleziona Squadra di Casa", ['Tutte'] + teams, index=0)
    selected_away_team = st.sidebar.selectbox("Seleziona Squadra in Trasferta", ['Tutte'] + teams, index=0)
else:
    selected_home_team = 'Tutte'
    selected_away_team = 'Tutte'
    st.sidebar.warning("Colonne 'home_team_name' e/o 'away_team_name' non trovate.")

last_matches_count = 'Tutte'
total_analysis_toggle = False
if (selected_home_team != 'Tutte' or selected_away_team != 'Tutte') and ('date' in df.columns or ('anno' in df.columns and 'Game Week' in df.columns)):
    num_matches_options = ['Tutte'] + [3, 5, 10, 15, 20, 25, 30, 50, 60, 75, 90, 100]
    last_matches_count = st.sidebar.selectbox("Analizza le ultime N partite", options=num_matches_options)
    total_analysis_toggle = st.sidebar.checkbox("Analizza Partite Totali (Home/Away)", value=False)
else:
    st.sidebar.warning("Colonne necessarie non presenti per l'analisi squadra per squadra.")

selected_ht_results = []
if 'HT Score' in df.columns:
    ht_scores = sorted(df['HT Score'].dropna().unique().tolist())
    selected_ht_results = st.sidebar.multiselect("Filtra Risultato HT", ht_scores, default=[])
else:
    st.sidebar.warning("Colonna 'HT Score' non trovata.")

st.sidebar.subheader("Filtri Quote FT (opzionali)")
odds_filters = {}
for label, col in {'Casa':'odds_ft_home_team_win','X':'odds_ft_draw','Trasferta':'odds_ft_away_team_win'}.items():
    if col in df.columns:
        mn = st.sidebar.number_input(f"Min {label}", min_value=1.01, step=0.01, key=f"min_{col}")
        mx = st.sidebar.number_input(f"Max {label}", min_value=1.01, step=0.01, value=100.00, key=f"max_{col}")
        odds_filters[col] = (mn, mx)

# ---------- Apply filters (split: no-odds vs with-odds) ----------
base_filtered = df.copy()
if selected_leagues: base_filtered = base_filtered[base_filtered['league'].isin(selected_leagues)]
if selected_years: base_filtered = base_filtered[base_filtered['anno'].isin(selected_years)]
if selected_gw: base_filtered = base_filtered[(base_filtered['Game Week'] >= selected_gw[0]) & (base_filtered['Game Week'] <= selected_gw[1])]
if selected_ht_results and 'HT Score' in base_filtered.columns:
    base_filtered = base_filtered[base_filtered['HT Score'].isin(selected_ht_results)]

# Apply Team & Last N matches filter
team_filtered_df = pd.DataFrame()
analysis_title = "Dati Analizzati"

if selected_home_team != 'Tutte' or selected_away_team != 'Tutte':
    if 'home_team_name' in base_filtered.columns and 'away_team_name' in base_filtered.columns:
        
        # New logic based on the toggle
        if total_analysis_toggle and last_matches_count != 'Tutte':
            team_df_home = base_filtered[
                (base_filtered['home_team_name'] == selected_home_team) |
                (base_filtered['away_team_name'] == selected_home_team)
            ].copy()
            team_df_away = base_filtered[
                (base_filtered['home_team_name'] == selected_away_team) |
                (base_filtered['away_team_name'] == selected_away_team)
            ].copy()
            
            sort_cols = ['anno', 'Game Week']
            if 'date' in team_df_home.columns and not team_df_home['date'].isnull().all():
                sort_cols = ['date']
            
            home_last_n = team_df_home.sort_values(by=sort_cols, ascending=False).head(last_matches_count)
            away_last_n = team_df_away.sort_values(by=sort_cols, ascending=False).head(last_matches_count)
            
            team_filtered_df = pd.concat([home_last_n, away_last_n]).drop_duplicates().sort_values(by=sort_cols, ascending=False)
            analysis_title = f"Analisi Combinata: ultime {len(home_last_n)} partite totali di {selected_home_team} e ultime {len(away_last_n)} partite totali di {selected_away_team}"
        
        else: # Original logic for specific home/away matches
            if selected_home_team != 'Tutte' and selected_away_team != 'Tutte':
                home_df = base_filtered[base_filtered['home_team_name'] == selected_home_team].copy()
                away_df = base_filtered[base_filtered['away_team_name'] == selected_away_team].copy()
                
                if last_matches_count != 'Tutte':
                    sort_cols = ['anno', 'Game Week']
                    if 'date' in home_df.columns and not home_df['date'].isnull().all():
                        sort_cols = ['date']
                    
                    home_df = home_df.sort_values(by=sort_cols, ascending=False).head(last_matches_count)
                    away_df = away_df.sort_values(by=sort_cols, ascending=False).head(last_matches_count)

                team_filtered_df = pd.concat([home_df, away_df])
                analysis_title = f"Analisi Combinata: ultime {len(home_df)} partite casalinghe di {selected_home_team} e ultime {len(away_df)} in trasferta di {selected_away_team}"

            elif selected_home_team != 'Tutte':
                team_filtered_df = base_filtered[base_filtered['home_team_name'] == selected_home_team]
                if last_matches_count != 'Tutte':
                    if 'date' in team_filtered_df.columns and not team_filtered_df['date'].isnull().all():
                        team_filtered_df = team_filtered_df.sort_values(by='date', ascending=False).head(last_matches_count)
                    else:
                        team_filtered_df = team_filtered_df.sort_values(by=['anno', 'Game Week'], ascending=False).head(last_matches_count)
                analysis_title = f"Analisi Partite Casalinghe di {selected_home_team}"

            elif selected_away_team != 'Tutte':
                team_filtered_df = base_filtered[base_filtered['away_team_name'] == selected_away_team]
                if last_matches_count != 'Tutte':
                    if 'date' in team_filtered_df.columns and not team_filtered_df['date'].isnull().all():
                        team_filtered_df = team_filtered_df.sort_values(by='date', ascending=False).head(last_matches_count)
                    else:
                        team_filtered_df = team_filtered_df.sort_values(by=['anno', 'Game Week'], ascending=False).head(last_matches_count)
                analysis_title = f"Analisi Partite in Trasferta di {selected_away_team}"
    else:
        team_filtered_df = base_filtered.copy()
else:
    team_filtered_df = base_filtered.copy()

odds_filtered = team_filtered_df.copy()
for col, (mn, mx) in odds_filters.items():
    if col in odds_filtered.columns:
        odds_filtered = odds_filtered[(odds_filtered[col] >= mn) & (odds_filtered[col] <= mx)]

# Dettagli ultime partite per le squadre selezionate
if selected_home_team != 'Tutte' or selected_away_team != 'Tutte':
    st.markdown("---")
    st.markdown("### Dettagli Ultime Partite Selezionate")
    
    last_matches_info = get_last_matches_info(df, selected_home_team, selected_away_team)
    
    st.markdown(f"**Ultima data del dataset per la squadra di casa selezionata ({selected_home_team}):**")
    
    if last_matches_info['home_team_home_match']:
        info = last_matches_info['home_team_home_match']
        st.markdown(f"- **Ultima partita in casa:** {info['date']} vs {info['opponent']}")
    else:
        st.markdown("- Nessuna partita in casa trovata per la squadra di casa selezionata.")

    if last_matches_info['home_team_away_match']:
        info = last_matches_info['home_team_away_match']
        st.markdown(f"- **Ultima partita fuori casa:** {info['date']} vs {info['opponent']}")
    else:
        st.markdown("- Nessuna partita fuori casa trovata per la squadra di casa selezionata.")

    st.markdown(f"**Ultima data del dataset per la squadra in trasferta selezionata ({selected_away_team}):**")

    if last_matches_info['away_team_home_match']:
        info = last_matches_info['away_team_home_match']
        st.markdown(f"- **Ultima partita in casa:** {info['date']} vs {info['opponent']}")
    else:
        st.markdown("- Nessuna partita in casa trovata per la squadra in trasferta selezionata.")
        
    if last_matches_info['away_team_away_match']:
        info = last_matches_info['away_team_away_match']
        st.markdown(f"- **Ultima partita fuori casa:** {info['date']} vs {info['opponent']}")
    else:
        st.markdown("- Nessuna partita fuori casa trovata per la squadra in trasferta selezionata.")
    st.markdown("---")

st.subheader("Riepilogo Partite per Anno")
year_summary_df = create_year_summary_table(team_filtered_df, odds_filtered)
if not year_summary_df.empty:
    st.dataframe(year_summary_df, use_container_width=True)
else:
    st.info("Colonna 'anno' non presente nel dataset o dati insufficienti per il riepilogo.")

st.subheader(f"Anteprima {analysis_title} (con filtri quota)")
st.write(f"Righe filtrate: **{len(odds_filtered)}**")

latest_date_filtered_dataset = "N/A"
if 'date' in odds_filtered.columns and not odds_filtered['date'].isnull().all():
    latest_date_filtered_dataset = odds_filtered['date'].max().strftime('%d-%m-%Y')

st.write(f"Ultima data del campione analizzato: **{latest_date_filtered_dataset}**")
st.dataframe(odds_filtered, use_container_width=True)

# ---------- Capitolo 1: Distribuzione Gol per Timeframe — Totale (senza filtri quota) ----------
st.markdown(f"## Capitolo 1: Distribuzione Gol per Timeframe — Totale ({len(team_filtered_df)} partite)")
if 'date' in team_filtered_df.columns and not team_filtered_df['date'].isnull().all():
    st.write(f"Data più recente del campione: **{team_filtered_df['date'].max().strftime('%d-%m-%Y')}**")
req_cols = {'home_team_goal_timings','away_team_goal_timings'}
if not req_cols.issubset(team_filtered_df.columns):
    st.info("Colonne 'home_team_goal_timings' e/o 'away_team_goal_timings' non presenti nel dataset.")
else:
    col15, col5 = st.columns(2)
    with col15:
        st.markdown("**Ogni 15 minuti**")
        tf_total_15 = timeframes_table(team_filtered_df, step=15)
        st.dataframe(style_table(tf_total_15, ['Percentuale %','>= 2 Gol %']), use_container_width=True)
    with col5:
        st.markdown("**Ogni 5 minuti (con 45+)**")
        tf_total_5 = timeframes_table(team_filtered_df, step=5)
        st.dataframe(style_table(tf_total_5, ['Percentuale %','>= 2 Gol %']), use_container_width=True)

    st.markdown("---")
    st.markdown("### Tasso di Conversione (Gol per Tiri in porta)")
    conversion_rate_table = create_conversion_rate_table(team_filtered_df)
    if not conversion_rate_table.empty:
        st.dataframe(conversion_rate_table, use_container_width=True)
    else:
        st.info("Dati insufficienti o colonne 'home_team_shots_on_target' e/o 'away_team_shots_on_target' non presenti per il calcolo.")

    st.markdown("---")
    st.markdown("### Riepilogo Media Gol")
    avg_goals_summary_table = create_avg_goals_summary_table(team_filtered_df)
    if not avg_goals_summary_table.empty:
        st.dataframe(avg_goals_summary_table, use_container_width=True)
    else:
        st.info("Colonne 'total_goals_at_half_time' e/o 'total_goals_at_full_time' non presenti per il calcolo.")


# ---------- Capitolo 2: Distribuzione Gol per Timeframe — Con filtri quota ----------
st.markdown(f"## Capitolo 2: Distribuzione Gol per Timeframe — Con filtri quota ({len(odds_filtered)} partite)")
if 'date' in odds_filtered.columns and not odds_filtered['date'].isnull().all():
    st.write(f"Data più recente del campione: **{odds_filtered['date'].max().strftime('%d-%m-%Y')}**")
if not req_cols.issubset(odds_filtered.columns):
    st.info("Colonne 'home_team_goal_timings' e/o 'away_team_goal_timings' non presenti nel dataset.")
else:
    col15b, col5b = st.columns(2)
    with col15b:
        st.markdown("**Ogni 15 minuti**")
        tf_odds_15 = timeframes_table(odds_filtered, step=15)
        st.dataframe(style_table(tf_odds_15, ['Percentuale %','>= 2 Gol %']), use_container_width=True)
    with col5b:
        st.markdown("**Ogni 5 minuti (con 45+)**")
        tf_odds_5 = timeframes_table(odds_filtered, step=5)
        st.dataframe(style_table(tf_odds_5, ['Percentuale %','>= 2 Gol %']), use_container_width=True)

    st.markdown("---")
    st.markdown("### Tasso di Conversione (Gol per Tiri in porta)")
    conversion_rate_table_odds = create_conversion_rate_table(odds_filtered)
    if not conversion_rate_table_odds.empty:
        st.dataframe(conversion_rate_table_odds, use_container_width=True)
    else:
        st.info("Dati insufficienti o colonne 'home_team_shots_on_target' e/o 'away_team_shots_on_target' non presenti per il calcolo.")

    st.markdown("---")
    st.markdown("### Riepilogo Media Gol")
    avg_goals_summary_table_odds = create_avg_goals_summary_table(odds_filtered)
    if not avg_goals_summary_table_odds.empty:
        st.dataframe(avg_goals_summary_table_odds, use_container_width=True)
    else:
        st.info("Colonne 'total_goals_at_half_time' e/o 'total_goals_at_full_time' non presenti per il calcolo.")


# ---------- 3) Statistiche HT ----------
with st.expander(f"Statistiche HT ({len(odds_filtered)} partite)"):
    if odds_filtered.empty or not {'home_team_goal_count_half_time','away_team_goal_count_half_time'}.issubset(odds_filtered.columns):
        st.info("Dati insufficienti per statistiche HT.")
    else:
        total_matches = len(odds_filtered)
        st.markdown(f"### Risultati Esatti HT ({total_matches})")
        betfair_order = ["0 - 0","0 - 1","0 - 2","0 - 3",
                         "1 - 0","1 - 1","1 - 2","1 - 3",
                         "2 - 0","2 - 1","2 - 2","2 - 3",
                         "3 - 0","3 - 1","3 - 2","3 - 3",
                         "Any Other Home Win","Any Other Away Win","Any Other Draw"]
        dist = odds_filtered['HT CS (Betfair)'].value_counts(dropna=False).reindex(betfair_order, fill_value=0)
        df_cs = pd.DataFrame({'HT (Betfair)': dist.index, 'Conteggio': dist.values})
        df_cs['Percentuale %'] = (df_cs['Conteggio'] / total_matches * 100).round(2)
        df_cs['Odd Minima'] = df_cs['Percentuale %'].apply(odd_min_from_percent)
        df_cs['order'] = df_cs['HT (Betfair)'].apply(lambda x: betfair_order.index(x))
        df_cs = df_cs.sort_values('order').drop(columns=['order'])
        st.dataframe(style_table(df_cs, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### WinRate HT ({total_matches})")
        ht_home = odds_filtered['home_team_goal_count_half_time']
        ht_away = odds_filtered['away_team_goal_count_half_time']
        home_w = (ht_home > ht_away).sum()
        draws = (ht_home == ht_away).sum()
        away_w = (ht_home < ht_away).sum()
        df_wr = pd.DataFrame({
            'Esito': ['1 (Casa)','X (Pareggio)','2 (Trasferta)'],
            'Conteggio': [home_w, draws, away_w]
        })
        df_wr['WinRate %'] = (df_wr['Conteggio'] / total_matches * 100).round(2)
        df_wr['Odd Minima'] = df_wr['WinRate %'].apply(odd_min_from_percent)
        st.dataframe(style_table(df_wr, ['WinRate %']), use_container_width=True)
        st.markdown(f"### Over Goals HT ({total_matches})")
        goal_lines = [0.5,1.5,2.5,3.5,4.5]
        tg = odds_filtered['total_goals_at_half_time']
        over_rows = []
        for gl in goal_lines:
            over_count = int((tg > (gl - 0.5)).sum())
            over_pct = round(over_count / total_matches * 100, 2)
            over_rows.append([f"Over {gl} HT", over_count, over_pct, odd_min_from_percent(over_pct)])
        df_over = pd.DataFrame(over_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
        st.dataframe(style_table(df_over, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### Under Goals HT ({total_matches})")
        under_rows = []
        for gl in goal_lines:
            over_count = int((tg > (gl - 0.5)).sum())
            under_count = int(total_matches - over_count)
            under_pct = round(under_count / total_matches * 100, 2)
            under_rows.append([f"Under {gl} HT", under_count, under_pct, odd_min_from_percent(under_pct)])
        df_under = pd.DataFrame(under_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
        st.dataframe(style_table(df_under, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### Doppia Chance HT ({total_matches})")
        count_1X = int(((ht_home > ht_away) | (ht_home == ht_away)).sum())
        count_X2 = int(((ht_home < ht_away) | (ht_home == ht_away)).sum())
        count_12 = int((ht_home != ht_away).sum())
        dc_df = pd.DataFrame({
            'Mercato': ['1X','X2','12'],
            'Conteggio': [count_1X, count_X2, count_12]
        })
        dc_df['Percentuale %'] = (dc_df['Conteggio'] / total_matches * 100).round(2)
        dc_df['Odd Minima'] = dc_df['Percentuale %'].apply(odd_min_from_percent)
        st.dataframe(style_table(dc_df, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### To Score HT ({total_matches})")
        ts_home_count = int((ht_home >= 1).sum())
        ts_away_count = int((ht_away >= 1).sum())
        ts_df = pd.DataFrame({
            'Squadra': ['Home segna HT', 'Away segna HT'],
            'Conteggio': [ts_home_count, ts_away_count]
        })
        ts_df['Percentuale %'] = (ts_df['Conteggio'] / total_matches * 100).round(2)
        ts_df['Odd Minima'] = ts_df['Percentuale %'].apply(odd_min_from_percent)
        st.dataframe(style_table(ts_df, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### BTTS HT ({total_matches})")
        btts_yes_count = int(((ht_home >= 1) & (ht_away >= 1)).sum())
        btts_no_count = int(total_matches - btts_yes_count)
        btts_df = pd.DataFrame({
            'Mercato': ['BTTS SI (HT)','BTTS NO (HT)'],
            'Conteggio': [btts_yes_count, btts_no_count]
        })
        btts_df['Percentuale %'] = (btts_df['Conteggio'] / total_matches * 100).round(2)
        btts_df['Odd Minima'] = btts_df['Percentuale %'].apply(odd_min_from_percent)
        st.dataframe(style_table(btts_df, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### First to Score (HT) ({total_matches})")
        home_first = away_first = no_goal = simultaneous = 0
        if {'home_team_goal_timings','away_team_goal_timings'}.issubset(odds_filtered.columns):
            for _, row in odds_filtered.iterrows():
                h_min = earliest_second_half_min(row.get('home_team_goal_timings', np.nan))
                a_min = earliest_second_half_min(row.get('away_team_goal_timings', np.nan))
                if h_min is None and a_min is None:
                    no_goal += 1
                elif h_min is not None and (a_min is None or h_min < a_min):
                    home_first += 1
                elif a_min is not None and (h_min is None or a_min < h_min):
                    away_first += 1
                else:
                    simultaneous += 1
            fts_sh_df = pd.DataFrame({
                'Esito': ['Home First (SH)', 'Away First (SH)', 'No Goal (SH)', 'Stesso minuto (SH)'],
                'Conteggio': [home_first, away_first, no_goal, simultaneous]
            })
            fts_sh_df['Percentuale %'] = (fts_sh_df['Conteggio'] / total_matches * 100).round(2)
            fts_sh_df['Odd Minima'] = fts_sh_df['Percentuale %'].apply(odd_min_from_percent)
            st.dataframe(style_table(fts_sh_df, ['Percentuale %']), use_container_width=True)
        else:
            st.info("Colonne minuti gol non presenti: impossibile calcolare First to Score.")

# ---------- Statistiche SH (Secondo Tempo) ----------
with st.expander(f"Statistiche SH (Secondo Tempo) ({len(odds_filtered)} partite)"):
    if odds_filtered.empty or not {'home_team_goal_timings','away_team_goal_timings'}.issubset(odds_filtered.columns):
        st.info("Per le statistiche SH servono le colonne minuti gol (home_team_goal_timings/away_team_goal_timings).")
    else:
        total_matches = len(odds_filtered)
        sh_home_counts = []
        sh_away_counts = []
        for _, row in odds_filtered.iterrows():
            hmins = minutes_second_half(row.get('home_team_goal_timings', np.nan))
            amins = minutes_second_half(row.get('away_team_goal_timings', np.nan))
            sh_home_counts.append(len(hmins))
            sh_away_counts.append(len(amins))
        sh_home = pd.Series(sh_home_counts, index=odds_filtered.index)
        sh_away = pd.Series(sh_away_counts, index=odds_filtered.index)
        sh_total_goals = sh_home + sh_away
        st.markdown(f"### Risultati Esatti SH ({total_matches})")
        betfair_order = ["0 - 0","0 - 1","0 - 2","0 - 3",
                         "1 - 0","1 - 1","1 - 2","1 - 3",
                         "2 - 0","2 - 1","2 - 2","2 - 3",
                         "3 - 0","3 - 1","3 - 2","3 - 3",
                         "Any Other Home Win","Any Other Away Win","Any Other Draw"]
        labels = [sh_cs_label(h, a) for h, a in zip(sh_home, sh_away)]
        dist = pd.Series(labels).value_counts(dropna=False).reindex(betfair_order, fill_value=0)
        df_cs_sh = pd.DataFrame({'SH (Betfair)': dist.index, 'Conteggio': dist.values})
        df_cs_sh['Percentuale %'] = (df_cs_sh['Conteggio'] / total_matches * 100).round(2)
        df_cs_sh['Odd Minima'] = df_cs_sh['Percentuale %'].apply(odd_min_from_percent)
        df_cs_sh['order'] = df_cs_sh['SH (Betfair)'].apply(lambda x: betfair_order.index(x))
        df_cs_sh = df_cs_sh.sort_values('order').drop(columns=['order'])
        st.dataframe(style_table(df_cs_sh, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### WinRate SH ({total_matches})")
        home_w = int((sh_home > sh_away).sum())
        draws = int((sh_home == sh_away).sum())
        away_w = int((sh_home < sh_away).sum())
        df_wr_sh = pd.DataFrame({
            'Esito': ['1 (Casa)','X (Pareggio)','2 (Trasferta)'],
            'Conteggio': [home_w, draws, away_w]
        })
        df_wr_sh['WinRate %'] = (df_wr_sh['Conteggio'] / total_matches * 100).round(2)
        df_wr_sh['Odd Minima'] = df_wr_sh['WinRate %'].apply(odd_min_from_percent)
        st.dataframe(style_table(df_wr_sh, ['WinRate %']), use_container_width=True)
        st.markdown(f"### Over Goals SH ({total_matches})")
        goal_lines = [0.5,1.5,2.5,3.5,4.5]
        over_rows = []
        for gl in goal_lines:
            over_count = int((sh_total_goals > (gl - 0.5)).sum())
            over_pct = round(over_count / total_matches * 100, 2)
            over_rows.append([f"Over {gl} SH", over_count, over_pct, odd_min_from_percent(over_pct)])
        df_over_sh = pd.DataFrame(over_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
        st.dataframe(style_table(df_over_sh, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### Under Goals SH ({total_matches})")
        under_rows = []
        for gl in goal_lines:
            over_count = int((sh_total_goals > (gl - 0.5)).sum())
            under_count = int(total_matches - over_count)
            under_pct = round(under_count / total_matches * 100, 2)
            under_rows.append([f"Under {gl} SH", under_count, under_pct, odd_min_from_percent(under_pct)])
        df_under_sh = pd.DataFrame(under_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
        st.dataframe(style_table(df_under_sh, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### Doppia Chance SH ({total_matches})")
        count_1X = int(((sh_home > sh_away) | (sh_home == sh_away)).sum())
        count_X2 = int(((sh_home < sh_away) | (sh_home == sh_away)).sum())
        count_12 = int((sh_home != sh_away).sum())
        dc_sh_df = pd.DataFrame({
            'Mercato': ['1X','X2','12'],
            'Conteggio': [count_1X, count_X2, count_12]
        })
        dc_sh_df['Percentuale %'] = (dc_sh_df['Conteggio'] / total_matches * 100).round(2)
        dc_sh_df['Odd Minima'] = dc_sh_df['Percentuale %'].apply(odd_min_from_percent)
        st.dataframe(style_table(dc_sh_df, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### To Score SH ({total_matches})")
        ts_home = int((sh_home >= 1).sum())
        ts_away = int((sh_away >= 1).sum())
        ts_sh_df = pd.DataFrame({
            'Squadra': ['Home segna SH', 'Away segna SH'],
            'Conteggio': [ts_home, ts_away]
        })
        ts_sh_df['Percentuale %'] = (ts_sh_df['Conteggio'] / total_matches * 100).round(2)
        ts_sh_df['Odd Minima'] = ts_sh_df['Percentuale %'].apply(odd_min_from_percent)
        st.dataframe(style_table(ts_sh_df, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### BTTS SH ({total_matches})")
        btts_yes = int(((sh_home >= 1) & (sh_away >= 1)).sum())
        btts_no = int(total_matches - btts_yes)
        btts_sh_df = pd.DataFrame({
            'Mercato': ['BTTS SI (SH)','BTTS NO (SH)'],
            'Conteggio': [btts_yes, btts_no]
        })
        btts_sh_df['Percentuale %'] = (btts_sh_df['Conteggio'] / total_matches * 100).round(2)
        btts_sh_df['Odd Minima'] = btts_sh_df['Percentuale %'].apply(odd_min_from_percent)
        st.dataframe(style_table(btts_sh_df, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### First to Score (SH) ({total_matches})")
        home_first = away_first = no_goal = simultaneous = 0
        if {'home_team_goal_timings','away_team_goal_timings'}.issubset(odds_filtered.columns):
            for _, row in odds_filtered.iterrows():
                h_min = earliest_second_half_min(row.get('home_team_goal_timings', np.nan))
                a_min = earliest_second_half_min(row.get('away_team_goal_timings', np.nan))
                if h_min is None and a_min is None:
                    no_goal += 1
                elif h_min is not None and (a_min is None or h_min < a_min):
                    home_first += 1
                elif a_min is not None and (h_min is None or a_min < h_min):
                    away_first += 1
                else:
                    simultaneous += 1
            fts_sh_df = pd.DataFrame({
                'Esito': ['Home First (SH)', 'Away First (SH)', 'No Goal (SH)', 'Stesso minuto (SH)'],
                'Conteggio': [home_first, away_first, no_goal, simultaneous]
            })
            fts_sh_df['Percentuale %'] = (fts_sh_df['Conteggio'] / total_matches * 100).round(2)
            fts_sh_df['Odd Minima'] = fts_sh_df['Percentuale %'].apply(odd_min_from_percent)
            st.dataframe(style_table(fts_sh_df, ['Percentuale %']), use_container_width=True)
        else:
            st.info("Colonne minuti gol non presenti: impossibile calcolare First to Score.")


# ---------- Statistiche FT (Full Time) ----------
with st.expander(f"Statistiche FT (Full Time) ({len(odds_filtered)} partite)"):
    if odds_filtered.empty or not {'home_team_goal_count','away_team_goal_count'}.issubset(odds_filtered.columns):
        st.info("Per le statistiche FT servono le colonne 'home_team_goal_count' e 'away_team_goal_count'.")
    else:
        total_matches = len(odds_filtered)
        ft_home = odds_filtered['home_team_goal_count']
        ft_away = odds_filtered['away_team_goal_count']
        ft_total_goals = odds_filtered['total_goals_at_full_time']

        st.markdown(f"### Risultati Esatti FT ({total_matches})")
        betfair_order = ["0 - 0","0 - 1","0 - 2","0 - 3",
                         "1 - 0","1 - 1","1 - 2","1 - 3",
                         "2 - 0","2 - 1","2 - 2","2 - 3",
                         "3 - 0","3 - 1","3 - 2","3 - 3",
                         "Any Other Home Win","Any Other Away Win","Any Other Draw"]
        labels = [ft_cs_label(h, a) for h, a in zip(ft_home, ft_away)]
        dist = pd.Series(labels).value_counts(dropna=False).reindex(betfair_order, fill_value=0)
        df_cs_ft = pd.DataFrame({'FT (Betfair)': dist.index, 'Conteggio': dist.values})
        df_cs_ft['Percentuale %'] = (df_cs_ft['Conteggio'] / total_matches * 100).round(2)
        df_cs_ft['Odd Minima'] = df_cs_ft['Percentuale %'].apply(odd_min_from_percent)
        df_cs_ft['order'] = df_cs_ft['FT (Betfair)'].apply(lambda x: betfair_order.index(x))
        df_cs_ft = df_cs_ft.sort_values('order').drop(columns=['order'])
        st.dataframe(style_table(df_cs_ft, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### WinRate FT ({total_matches})")
        home_w = int((ft_home > ft_away).sum())
        draws = int((ft_home == ft_away).sum())
        away_w = int((ft_home < ft_away).sum())
        df_wr_ft = pd.DataFrame({
            'Esito': ['1 (Casa)','X (Pareggio)','2 (Trasferta)'],
            'Conteggio': [home_w, draws, away_w]
        })
        df_wr_ft['WinRate %'] = (df_wr_ft['Conteggio'] / total_matches * 100).round(2)
        df_wr_ft['Odd Minima'] = df_wr_ft['WinRate %'].apply(odd_min_from_percent)
        st.dataframe(style_table(df_wr_ft, ['WinRate %']), use_container_width=True)
        st.markdown(f"### Over Goals FT ({total_matches})")
        goal_lines = [0.5,1.5,2.5,3.5,4.5]
        over_rows = []
        for gl in goal_lines:
            over_count = int((ft_total_goals > (gl - 0.5)).sum())
            over_pct = round(over_count / total_matches * 100, 2)
            over_rows.append([f"Over {gl} FT", over_count, over_pct, odd_min_from_percent(over_pct)])
        df_over_ft = pd.DataFrame(over_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
        st.dataframe(style_table(df_over_ft, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### Under Goals FT ({total_matches})")
        under_rows = []
        for gl in goal_lines:
            over_count = int((ft_total_goals > (gl - 0.5)).sum())
            under_count = int(total_matches - over_count)
            under_pct = round(under_count / total_matches * 100, 2)
            under_rows.append([f"Under {gl} FT", under_count, under_pct, odd_min_from_percent(under_pct)])
        df_under_ft = pd.DataFrame(under_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
        st.dataframe(style_table(df_under_ft, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### Doppia Chance FT ({total_matches})")
        count_1X = int(((ft_home > ft_away) | (ft_home == ft_away)).sum())
        count_X2 = int(((ft_home < ft_away) | (ft_home == ft_away)).sum())
        count_12 = int((ft_home != ft_away).sum())
        dc_ft_df = pd.DataFrame({
            'Mercato': ['1X','X2','12'],
            'Conteggio': [count_1X, count_X2, count_12]
        })
        dc_ft_df['Percentuale %'] = (dc_ft_df['Conteggio'] / total_matches * 100).round(2)
        dc_ft_df['Odd Minima'] = dc_ft_df['Percentuale %'].apply(odd_min_from_percent)
        st.dataframe(style_table(dc_ft_df, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### BTTS FT ({total_matches})")
        btts_yes = int(((ft_home >= 1) & (ft_away >= 1)).sum())
        btts_no = int(total_matches - btts_yes)
        btts_ft_df = pd.DataFrame({
            'Mercato': ['BTTS SI (FT)','BTTS NO (FT)'],
            'Conteggio': [btts_yes, btts_no]
        })
        btts_ft_df['Percentuale %'] = (btts_ft_df['Conteggio'] / total_matches * 100).round(2)
        btts_ft_df['Odd Minima'] = btts_ft_df['Percentuale %'].apply(odd_min_from_percent)
        st.dataframe(style_table(btts_ft_df, ['Percentuale %']), use_container_width=True)
        st.markdown(f"### First to Score (FT) ({total_matches})")
        if {'home_team_goal_timings','away_team_goal_timings'}.issubset(odds_filtered.columns):
            fts_df = compute_first_to_score_ft(odds_filtered)
            st.dataframe(style_table(fts_df, ['Percentuale %']), use_container_width=True)
        else:
            st.info("Colonne minuti gol non presenti: impossibile calcolare First to Score.")

# ---------- Capitolo 3: H2H (Scontri Diretti) ----------
st.markdown("---")
st.markdown("## Capitolo 3: H2H (Scontri Diretti)")

if selected_home_team == 'Tutte' or selected_away_team == 'Tutte':
    st.info("Seleziona sia una squadra di casa che una squadra in trasferta per attivare l'analisi H2H.")
else:
    h2h_toggle = st.checkbox("Analizza Partite Totali H2H (Home/Away)")
    
    h2h_df = df.copy()
    
    # Applica i filtri di base (campionato, anno, giornata)
    if selected_leagues: h2h_df = h2h_df[h2h_df['league'].isin(selected_leagues)]
    if selected_years: h2h_df = h2h_df[h2h_df['anno'].isin(selected_years)]
    if selected_gw: h2h_df = h2h_df[(h2h_df['Game Week'] >= selected_gw[0]) & (h2h_df['Game Week'] <= selected_gw[1])]

    if h2h_toggle:
        # H2H totale: casa contro trasferta e viceversa
        h2h_df = h2h_df[
            ((h2h_df['home_team_name'] == selected_home_team) & (h2h_df['away_team_name'] == selected_away_team)) |
            ((h2h_df['home_team_name'] == selected_away_team) & (h2h_df['away_team_name'] == selected_home_team))
        ]
        h2h_df['is_home_for_h2h'] = h2h_df['home_team_name'] == selected_home_team
        
    else:
        # H2H specifico: solo quando la squadra A gioca in casa contro la squadra B
        h2h_df = h2h_df[
            (h2h_df['home_team_name'] == selected_home_team) & 
            (h2h_df['away_team_name'] == selected_away_team)
        ]

    # Applica filtri quota agli scontri diretti
    h2h_odds_filtered = h2h_df.copy()
    for col, (mn, mx) in odds_filters.items():
        if col in h2h_odds_filtered.columns:
            h2h_odds_filtered = h2h_odds_filtered[(h2h_odds_filtered[col] >= mn) & (h2h_odds_filtered[col] <= mx)]

    if h2h_odds_filtered.empty:
        st.info("Nessuno scontro diretto trovato per le squadre selezionate con i filtri applicati.")
    else:
        st.write(f"Scontri diretti analizzati: **{len(h2h_odds_filtered)}** partite.")
        st.dataframe(h2h_odds_filtered)

        st.markdown("### Riepilogo Media Gol H2H")
        h2h_avg_goals_summary = create_avg_goals_summary_table(h2h_odds_filtered)
        if not h2h_avg_goals_summary.empty:
            st.dataframe(h2h_avg_goals_summary, use_container_width=True)
        else:
            st.info("Dati insufficienti per il riepilogo media gol H2H.")

        st.markdown("---")
        st.markdown("### Distribuzione Gol per Timeframe H2H")
        req_cols_h2h = {'home_team_goal_timings','away_team_goal_timings'}
        if not req_cols_h2h.issubset(h2h_odds_filtered.columns):
            st.info("Colonne 'home_team_goal_timings' e/o 'away_team_goal_timings' non presenti per l'analisi del timeframe H2H.")
        else:
            col15_h2h, col5_h2h = st.columns(2)
            with col15_h2h:
                st.markdown("**Ogni 15 minuti**")
                tf_h2h_15 = timeframes_table(h2h_odds_filtered, step=15)
                st.dataframe(style_table(tf_h2h_15, ['Percentuale %','>= 2 Gol %']), use_container_width=True)
            with col5_h2h:
                st.markdown("**Ogni 5 minuti (con 45+)**")
                tf_h2h_5 = timeframes_table(h2h_odds_filtered, step=5)
                st.dataframe(style_table(tf_h2h_5, ['Percentuale %','>= 2 Gol %']), use_container_width=True)
        
        # Statistiche HT H2H
        with st.expander(f"Statistiche HT H2H ({len(h2h_odds_filtered)} partite)"):
            if h2h_odds_filtered.empty or not {'home_team_goal_count_half_time','away_team_goal_count_half_time'}.issubset(h2h_odds_filtered.columns):
                st.info("Dati insufficienti per le statistiche HT H2H.")
            else:
                total_matches = len(h2h_odds_filtered)
                st.markdown(f"### Risultati Esatti HT H2H ({total_matches})")
                betfair_order = ["0 - 0","0 - 1","0 - 2","0 - 3",
                                 "1 - 0","1 - 1","1 - 2","1 - 3",
                                 "2 - 0","2 - 1","2 - 2","2 - 3",
                                 "3 - 0","3 - 1","3 - 2","3 - 3",
                                 "Any Other Home Win","Any Other Away Win","Any Other Draw"]
                dist = h2h_odds_filtered['HT CS (Betfair)'].value_counts(dropna=False).reindex(betfair_order, fill_value=0)
                df_cs = pd.DataFrame({'HT (Betfair)': dist.index, 'Conteggio': dist.values})
                df_cs['Percentuale %'] = (df_cs['Conteggio'] / total_matches * 100).round(2)
                df_cs['Odd Minima'] = df_cs['Percentuale %'].apply(odd_min_from_percent)
                df_cs['order'] = df_cs['HT (Betfair)'].apply(lambda x: betfair_order.index(x))
                df_cs = df_cs.sort_values('order').drop(columns=['order'])
                st.dataframe(style_table(df_cs, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### WinRate HT H2H ({total_matches})")
                ht_home = h2h_odds_filtered['home_team_goal_count_half_time']
                ht_away = h2h_odds_filtered['away_team_goal_count_half_time']
                home_w = (ht_home > ht_away).sum()
                draws = (ht_home == ht_away).sum()
                away_w = (ht_home < ht_away).sum()
                df_wr = pd.DataFrame({
                    'Esito': ['1 (Casa)','X (Pareggio)','2 (Trasferta)'],
                    'Conteggio': [home_w, draws, away_w]
                })
                df_wr['WinRate %'] = (df_wr['Conteggio'] / total_matches * 100).round(2)
                df_wr['Odd Minima'] = df_wr['WinRate %'].apply(odd_min_from_percent)
                st.dataframe(style_table(df_wr, ['WinRate %']), use_container_width=True)
                st.markdown(f"### Over Goals HT H2H ({total_matches})")
                goal_lines = [0.5,1.5,2.5,3.5,4.5]
                tg = h2h_odds_filtered['total_goals_at_half_time']
                over_rows = []
                for gl in goal_lines:
                    over_count = int((tg > (gl - 0.5)).sum())
                    over_pct = round(over_count / total_matches * 100, 2)
                    over_rows.append([f"Over {gl} HT", over_count, over_pct, odd_min_from_percent(over_pct)])
                df_over = pd.DataFrame(over_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
                st.dataframe(style_table(df_over, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### Under Goals HT H2H ({total_matches})")
                under_rows = []
                for gl in goal_lines:
                    over_count = int((tg > (gl - 0.5)).sum())
                    under_count = int(total_matches - over_count)
                    under_pct = round(under_count / total_matches * 100, 2)
                    under_rows.append([f"Under {gl} HT", under_count, under_pct, odd_min_from_percent(under_pct)])
                df_under = pd.DataFrame(under_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
                st.dataframe(style_table(df_under, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### Doppia Chance HT H2H ({total_matches})")
                count_1X = int(((ht_home > ht_away) | (ht_home == ht_away)).sum())
                count_X2 = int(((ht_home < ht_away) | (ht_home == ht_away)).sum())
                count_12 = int((ht_home != ht_away).sum())
                dc_df = pd.DataFrame({
                    'Mercato': ['1X','X2','12'],
                    'Conteggio': [count_1X, count_X2, count_12]
                })
                dc_df['Percentuale %'] = (dc_df['Conteggio'] / total_matches * 100).round(2)
                dc_df['Odd Minima'] = dc_df['Percentuale %'].apply(odd_min_from_percent)
                st.dataframe(style_table(dc_df, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### To Score HT H2H ({total_matches})")
                ts_home_count = int((ht_home >= 1).sum())
                ts_away_count = int((ht_away >= 1).sum())
                ts_df = pd.DataFrame({
                    'Squadra': ['Home segna HT', 'Away segna HT'],
                    'Conteggio': [ts_home_count, ts_away_count]
                })
                ts_df['Percentuale %'] = (ts_df['Conteggio'] / total_matches * 100).round(2)
                ts_df['Odd Minima'] = ts_df['Percentuale %'].apply(odd_min_from_percent)
                st.dataframe(style_table(ts_df, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### BTTS HT H2H ({total_matches})")
                btts_yes_count = int(((ht_home >= 1) & (ht_away >= 1)).sum())
                btts_no_count = int(total_matches - btts_yes_count)
                btts_df = pd.DataFrame({
                    'Mercato': ['BTTS SI (HT)','BTTS NO (HT)'],
                    'Conteggio': [btts_yes_count, btts_no_count]
                })
                btts_df['Percentuale %'] = (btts_df['Conteggio'] / total_matches * 100).round(2)
                btts_df['Odd Minima'] = btts_df['Percentuale %'].apply(odd_min_from_percent)
                st.dataframe(style_table(btts_df, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### First to Score (HT) H2H ({total_matches})")
                if {'home_team_goal_timings','away_team_goal_timings'}.issubset(h2h_odds_filtered.columns):
                    fts_df = compute_first_to_score_ht(h2h_odds_filtered)
                    st.dataframe(style_table(fts_df, ['Percentuale %']), use_container_width=True)
                else:
                    st.info("Colonne minuti gol non presenti: impossibile calcolare First to Score.")

        # Statistiche SH H2H
        with st.expander(f"Statistiche SH H2H ({len(h2h_odds_filtered)} partite)"):
            if h2h_odds_filtered.empty or not {'home_team_goal_timings','away_team_goal_timings'}.issubset(h2h_odds_filtered.columns):
                st.info("Per le statistiche SH servono le colonne minuti gol (home_team_goal_timings/away_team_goal_timings).")
            else:
                total_matches = len(h2h_odds_filtered)
                sh_home_counts = []
                sh_away_counts = []
                for _, row in h2h_odds_filtered.iterrows():
                    hmins = minutes_second_half(row.get('home_team_goal_timings', np.nan))
                    amins = minutes_second_half(row.get('away_team_goal_timings', np.nan))
                    sh_home_counts.append(len(hmins))
                    sh_away_counts.append(len(amins))
                sh_home = pd.Series(sh_home_counts, index=h2h_odds_filtered.index)
                sh_away = pd.Series(sh_away_counts, index=h2h_odds_filtered.index)
                sh_total_goals = sh_home + sh_away
                st.markdown(f"### Risultati Esatti SH H2H ({total_matches})")
                betfair_order = ["0 - 0","0 - 1","0 - 2","0 - 3",
                                 "1 - 0","1 - 1","1 - 2","1 - 3",
                                 "2 - 0","2 - 1","2 - 2","2 - 3",
                                 "3 - 0","3 - 1","3 - 2","3 - 3",
                                 "Any Other Home Win","Any Other Away Win","Any Other Draw"]
                labels = [sh_cs_label(h, a) for h, a in zip(sh_home, sh_away)]
                dist = pd.Series(labels).value_counts(dropna=False).reindex(betfair_order, fill_value=0)
                df_cs_sh = pd.DataFrame({'SH (Betfair)': dist.index, 'Conteggio': dist.values})
                df_cs_sh['Percentuale %'] = (df_cs_sh['Conteggio'] / total_matches * 100).round(2)
                df_cs_sh['Odd Minima'] = df_cs_sh['Percentuale %'].apply(odd_min_from_percent)
                df_cs_sh['order'] = df_cs_sh['SH (Betfair)'].apply(lambda x: betfair_order.index(x))
                df_cs_sh = df_cs_sh.sort_values('order').drop(columns=['order'])
                st.dataframe(style_table(df_cs_sh, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### WinRate SH H2H ({total_matches})")
                home_w = int((sh_home > sh_away).sum())
                draws = int((sh_home == sh_away).sum())
                away_w = int((sh_home < sh_away).sum())
                df_wr_sh = pd.DataFrame({
                    'Esito': ['1 (Casa)','X (Pareggio)','2 (Trasferta)'],
                    'Conteggio': [home_w, draws, away_w]
                })
                df_wr_sh['WinRate %'] = (df_wr_sh['Conteggio'] / total_matches * 100).round(2)
                df_wr_sh['Odd Minima'] = df_wr_sh['WinRate %'].apply(odd_min_from_percent)
                st.dataframe(style_table(df_wr_sh, ['WinRate %']), use_container_width=True)
                st.markdown(f"### Over Goals SH H2H ({total_matches})")
                goal_lines = [0.5,1.5,2.5,3.5,4.5]
                over_rows = []
                for gl in goal_lines:
                    over_count = int((sh_total_goals > (gl - 0.5)).sum())
                    over_pct = round(over_count / total_matches * 100, 2)
                    over_rows.append([f"Over {gl} SH", over_count, over_pct, odd_min_from_percent(over_pct)])
                df_over_sh = pd.DataFrame(over_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
                st.dataframe(style_table(df_over_sh, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### Under Goals SH H2H ({total_matches})")
                under_rows = []
                for gl in goal_lines:
                    over_count = int((sh_total_goals > (gl - 0.5)).sum())
                    under_count = int(total_matches - over_count)
                    under_pct = round(under_count / total_matches * 100, 2)
                    under_rows.append([f"Under {gl} SH", under_count, under_pct, odd_min_from_percent(under_pct)])
                df_under_sh = pd.DataFrame(under_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
                st.dataframe(style_table(df_under_sh, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### Doppia Chance SH H2H ({total_matches})")
                count_1X = int(((sh_home > sh_away) | (sh_home == sh_away)).sum())
                count_X2 = int(((sh_home < sh_away) | (sh_home == sh_away)).sum())
                count_12 = int((sh_home != sh_away).sum())
                dc_sh_df = pd.DataFrame({
                    'Mercato': ['1X','X2','12'],
                    'Conteggio': [count_1X, count_X2, count_12]
                })
                dc_sh_df['Percentuale %'] = (dc_sh_df['Conteggio'] / total_matches * 100).round(2)
                dc_sh_df['Odd Minima'] = dc_sh_df['Percentuale %'].apply(odd_min_from_percent)
                st.dataframe(style_table(dc_sh_df, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### To Score SH H2H ({total_matches})")
                ts_home = int((sh_home >= 1).sum())
                ts_away = int((sh_away >= 1).sum())
                ts_sh_df = pd.DataFrame({
                    'Squadra': ['Home segna SH', 'Away segna SH'],
                    'Conteggio': [ts_home, ts_away]
                })
                ts_sh_df['Percentuale %'] = (ts_sh_df['Conteggio'] / total_matches * 100).round(2)
                ts_sh_df['Odd Minima'] = ts_sh_df['Percentuale %'].apply(odd_min_from_percent)
                st.dataframe(style_table(ts_sh_df, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### BTTS SH H2H ({total_matches})")
                btts_yes = int(((sh_home >= 1) & (sh_away >= 1)).sum())
                btts_no = int(total_matches - btts_yes)
                btts_sh_df = pd.DataFrame({
                    'Mercato': ['BTTS SI (SH)','BTTS NO (SH)'],
                    'Conteggio': [btts_yes, btts_no]
                })
                btts_sh_df['Percentuale %'] = (btts_sh_df['Conteggio'] / total_matches * 100).round(2)
                btts_sh_df['Odd Minima'] = btts_sh_df['Percentuale %'].apply(odd_min_from_percent)
                st.dataframe(style_table(btts_sh_df, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### First to Score (SH) H2H ({total_matches})")
                if {'home_team_goal_timings','away_team_goal_timings'}.issubset(h2h_odds_filtered.columns):
                    home_first = away_first = no_goal = simultaneous = 0
                    for _, row in h2h_odds_filtered.iterrows():
                        h_min = earliest_second_half_min(row.get('home_team_goal_timings', np.nan))
                        a_min = earliest_second_half_min(row.get('away_team_goal_timings', np.nan))
                        if h_min is None and a_min is None:
                            no_goal += 1
                        elif h_min is not None and (a_min is None or h_min < a_min):
                            home_first += 1
                        elif a_min is not None and (h_min is None or a_min < h_min):
                            away_first += 1
                        else:
                            simultaneous += 1
                    fts_sh_df = pd.DataFrame({
                        'Esito': ['Home First (SH)', 'Away First (SH)', 'No Goal (SH)', 'Stesso minuto (SH)'],
                        'Conteggio': [home_first, away_first, no_goal, simultaneous]
                    })
                    fts_sh_df['Percentuale %'] = (fts_sh_df['Conteggio'] / total_matches * 100).round(2)
                    fts_sh_df['Odd Minima'] = fts_sh_df['Percentuale %'].apply(odd_min_from_percent)
                    st.dataframe(style_table(fts_sh_df, ['Percentuale %']), use_container_width=True)
                else:
                    st.info("Colonne minuti gol non presenti: impossibile calcolare First to Score.")

        # Statistiche FT H2H
        with st.expander(f"Statistiche FT H2H ({len(h2h_odds_filtered)} partite)"):
            if h2h_odds_filtered.empty or not {'home_team_goal_count','away_team_goal_count'}.issubset(h2h_odds_filtered.columns):
                st.info("Per le statistiche FT H2H servono le colonne 'home_team_goal_count' e 'away_team_goal_count'.")
            else:
                total_matches = len(h2h_odds_filtered)
                ft_home = h2h_odds_filtered['home_team_goal_count']
                ft_away = h2h_odds_filtered['away_team_goal_count']
                ft_total_goals = h2h_odds_filtered['total_goals_at_full_time']

                st.markdown(f"### Risultati Esatti FT H2H ({total_matches})")
                betfair_order = ["0 - 0","0 - 1","0 - 2","0 - 3",
                                 "1 - 0","1 - 1","1 - 2","1 - 3",
                                 "2 - 0","2 - 1","2 - 2","2 - 3",
                                 "3 - 0","3 - 1","3 - 2","3 - 3",
                                 "Any Other Home Win","Any Other Away Win","Any Other Draw"]
                labels = [ft_cs_label(h, a) for h, a in zip(ft_home, ft_away)]
                dist = pd.Series(labels).value_counts(dropna=False).reindex(betfair_order, fill_value=0)
                df_cs_ft = pd.DataFrame({'FT (Betfair)': dist.index, 'Conteggio': dist.values})
                df_cs_ft['Percentuale %'] = (df_cs_ft['Conteggio'] / total_matches * 100).round(2)
                df_cs_ft['Odd Minima'] = df_cs_ft['Percentuale %'].apply(odd_min_from_percent)
                df_cs_ft['order'] = df_cs_ft['FT (Betfair)'].apply(lambda x: betfair_order.index(x))
                df_cs_ft = df_cs_ft.sort_values('order').drop(columns=['order'])
                st.dataframe(style_table(df_cs_ft, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### WinRate FT H2H ({total_matches})")
                home_w = int((ft_home > ft_away).sum())
                draws = int((ft_home == ft_away).sum())
                away_w = int((ft_home < ft_away).sum())
                df_wr_ft = pd.DataFrame({
                    'Esito': ['1 (Casa)','X (Pareggio)','2 (Trasferta)'],
                    'Conteggio': [home_w, draws, away_w]
                })
                df_wr_ft['WinRate %'] = (df_wr_ft['Conteggio'] / total_matches * 100).round(2)
                df_wr_ft['Odd Minima'] = df_wr_ft['WinRate %'].apply(odd_min_from_percent)
                st.dataframe(style_table(df_wr_ft, ['WinRate %']), use_container_width=True)
                st.markdown(f"### Over Goals FT H2H ({total_matches})")
                goal_lines = [0.5,1.5,2.5,3.5,4.5]
                over_rows = []
                for gl in goal_lines:
                    over_count = int((ft_total_goals > (gl - 0.5)).sum())
                    over_pct = round(over_count / total_matches * 100, 2)
                    over_rows.append([f"Over {gl} FT", over_count, over_pct, odd_min_from_percent(over_pct)])
                df_over_ft = pd.DataFrame(over_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
                st.dataframe(style_table(df_over_ft, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### Under Goals FT H2H ({total_matches})")
                under_rows = []
                for gl in goal_lines:
                    over_count = int((ft_total_goals > (gl - 0.5)).sum())
                    under_count = int(total_matches - over_count)
                    under_pct = round(under_count / total_matches * 100, 2)
                    under_rows.append([f"Under {gl} FT", under_count, under_pct, odd_min_from_percent(under_pct)])
                df_under_ft = pd.DataFrame(under_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
                st.dataframe(style_table(df_under_ft, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### Doppia Chance FT H2H ({total_matches})")
                count_1X = int(((ft_home > ft_away) | (ft_home == ft_away)).sum())
                count_X2 = int(((ft_home < ft_away) | (ft_home == ft_away)).sum())
                count_12 = int((ft_home != ft_away).sum())
                dc_ft_df = pd.DataFrame({
                    'Mercato': ['1X','X2','12'],
                    'Conteggio': [count_1X, count_X2, count_12]
                })
                dc_ft_df['Percentuale %'] = (dc_ft_df['Conteggio'] / total_matches * 100).round(2)
                dc_ft_df['Odd Minima'] = dc_ft_df['Percentuale %'].apply(odd_min_from_percent)
                st.dataframe(style_table(dc_ft_df, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### BTTS FT H2H ({total_matches})")
                btts_yes = int(((ft_home >= 1) & (ft_away >= 1)).sum())
                btts_no = int(total_matches - btts_yes)
                btts_ft_df = pd.DataFrame({
                    'Mercato': ['BTTS SI (FT)','BTTS NO (FT)'],
                    'Conteggio': [btts_yes, btts_no]
                })
                btts_ft_df['Percentuale %'] = (btts_ft_df['Conteggio'] / total_matches * 100).round(2)
                btts_ft_df['Odd Minima'] = btts_ft_df['Percentuale %'].apply(odd_min_from_percent)
                st.dataframe(style_table(btts_ft_df, ['Percentuale %']), use_container_width=True)
                st.markdown(f"### First to Score (FT) H2H ({total_matches})")
                if {'home_team_goal_timings','away_team_goal_timings'}.issubset(h2h_odds_filtered.columns):
                    fts_df = compute_first_to_score_ft(h2h_odds_filtered)
                    st.dataframe(style_table(fts_df, ['Percentuale %']), use_container_width=True)
                else:
                    st.info("Colonne minuti gol non presenti: impossibile calcolare First to Score.")
