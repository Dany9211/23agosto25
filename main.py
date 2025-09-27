import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

st.set_page_config(page_title="Filtri Dati Calcio", layout="wide")
st.title("⚽ Dashboard Filtri partite squadre Calcio⚽")
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
    home_gf_goals = {b: 0 for b in buckets}
    home_ga_goals = {b: 0 for b in buckets}
    away_gf_goals = {b: 0 for b in buckets}
    away_ga_goals = {b: 0 for b in buckets}
    total_matches = len(df_subset)

    sel_home = globals().get('selected_home_team', None)
    sel_away = globals().get('selected_away_team', None)

    for _, row in df_subset.iterrows():
        home_tokens = row.get('home_team_goal_timings', np.nan)
        away_tokens = row.get('away_team_goal_timings', np.nan)
        hb_list = buckets_from_tokens_step(home_tokens, step) or []
        ab_list = buckets_from_tokens_step(away_tokens, step) or []

        # original presence logic
        both_list = hb_list + ab_list
        if not both_list:
            continue
        per_bucket = {b: 0 for b in buckets}
        for b in both_list:
            if b in per_bucket:
                per_bucket[b] += 1
        for b, c in per_bucket.items():
            if c >= 1:
                counts_matches_with_goal[b] += 1
            if c >= 2:
                counts_matches_with_2plus[b] += 1

        # Row team names (if available)
        home_name = row.get('home_team_name') if 'home_team_name' in row else None
        away_name = row.get('away_team_name') if 'away_team_name' in row else None

        # HOME side logic
        if sel_home and sel_home != 'Tutte':
            if home_name == sel_home:
                for b in hb_list:
                    if b in home_gf_goals: home_gf_goals[b] += 1
                for b in ab_list:
                    if b in home_ga_goals: home_ga_goals[b] += 1
        else:
            # aggregate by league (all home teams in subset)
            for b in hb_list:
                if b in home_gf_goals: home_gf_goals[b] += 1
            for b in ab_list:
                if b in home_ga_goals: home_ga_goals[b] += 1

        # AWAY side logic
        if sel_away and sel_away != 'Tutte':
            if away_name == sel_away:
                for b in ab_list:
                    if b in away_gf_goals: away_gf_goals[b] += 1
                for b in hb_list:
                    if b in away_ga_goals: away_ga_goals[b] += 1
        else:
            # aggregate by league (all away teams in subset)
            for b in ab_list:
                if b in away_gf_goals: away_gf_goals[b] += 1
            for b in hb_list:
                if b in away_ga_goals: away_ga_goals[b] += 1

    rows = []
    for b in buckets:
        with_goal = counts_matches_with_goal[b]
        pct = round((with_goal / total_matches) * 100, 2) if total_matches else 0.0
        odd_min = odd_min_from_percent(pct)
        g2 = counts_matches_with_2plus[b]
        pct2 = round((g2 / total_matches) * 100, 2) if total_matches else 0.0
        odd_min2 = odd_min_from_percent(pct2) if pct2 > 0 else None

        rows.append([
            b,
            with_goal, pct, odd_min,
            pct2, odd_min2,
            home_gf_goals[b], home_ga_goals[b],
            away_gf_goals[b], away_ga_goals[b]
        ])

    tf_df = pd.DataFrame(rows, columns=[
        'Timeframe',
        'Partite con Gol', 'Percentuale %', 'Odd Minima',
        '>= 2 Gol %', 'Odd Minima >= 2 Gol',
        'Home GF (gol)', 'Home GA (gol)',
        'Away GF (gol)', 'Away GA (gol)'
    ])
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

def filter_live_matches(df, current_min, home_score, away_score, home_goal_mins, away_goal_mins, live_home_team=None, live_away_team=None, odds_home_range=None, odds_away_range=None):
    """
    Filtra il dataset storico per trovare partite che erano nello stesso stato live
    al minuto specificato, con l'aggiunta di filtri squadra specifici, considerando
    la posizione (home/away) delle squadre selezionate contro qualsiasi avversario,
    e applica filtri quote FT pre-match.
    """
    
    required_cols = {'home_team_goal_count_half_time', 'away_team_goal_count_half_time', 'home_team_goal_timings', 'away_team_goal_timings', 'home_team_goal_count', 'away_team_goal_count', 'home_team_name', 'away_team_name'}
    
    if odds_home_range: required_cols.add('odds_ft_home_team_win')
    if odds_away_range: required_cols.add('odds_ft_away_team_win')
        
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.error(f"Colonne dati mancanti per l'analisi live (Live/Quote): {', '.join(missing)}")
        return pd.DataFrame()

    def get_goal_count_at_minute(goal_timings_cell, minute_limit):
        """Conta i gol segnati prima o esattamente al minuto limite."""
        if pd.isna(goal_timings_cell): return 0
        count = 0
        for token in [t.strip() for t in str(goal_timings_cell).split(',') if t.strip()!='']:
            m = re.fullmatch(r"(\d+)'(\d+)", token)
            if m:
                base = int(m.group(1)); extra = int(m.group(2))
                goal_min = base + extra
            else:
                digits = re.sub(r"[^0-9]", "", token)
                if digits == "": continue
                goal_min = int(digits)
            
            if goal_min <= minute_limit:
                count += 1
        return count

    temp_df = df.copy()

    # 1. Filtra per Risultato al minuto attuale
    temp_df['home_goals_at_live_min'] = temp_df['home_team_goal_timings'].apply(lambda x: get_goal_count_at_minute(x, current_min))
    temp_df['away_goals_at_live_min'] = temp_df['away_team_goal_timings'].apply(lambda x: get_goal_count_at_minute(x, current_min))

    # Applica i filtri di risultato
    filtered_df_base = temp_df[
        (temp_df['home_goals_at_live_min'] == home_score) &
        (temp_df['away_goals_at_live_min'] == away_score)
    ].copy()
    
    # 2. Filtra per Team specifici (Logica OR/combinata per posizione)
    
    home_selected = live_home_team and live_home_team != 'Tutte'
    away_selected = live_away_team and live_away_team != 'Tutte'
    
    if home_selected and away_selected:
        # Combinato (OR): (Partite HOME specificate) OR (Partite AWAY specificate)
        mask = (filtered_df_base['home_team_name'] == live_home_team) | \
               (filtered_df_base['away_team_name'] == live_away_team)
    elif home_selected:
        # Solo Home: Tutte le partite dove quella squadra giocava in casa
        mask = (filtered_df_base['home_team_name'] == live_home_team)
    elif away_selected:
        # Solo Away: Tutte le partite dove quella squadra giocava in trasferta
        mask = (filtered_df_base['away_team_name'] == live_away_team)
    else:
        # Nessuna squadra selezionata: non filtrare per nome squadra
        mask = pd.Series([True] * len(filtered_df_base), index=filtered_df_base.index)

    filtered_df = filtered_df_base[mask].copy()

    # 3. Filtra per Quote Pre-Match (se specificate)
    if odds_home_range and 'odds_ft_home_team_win' in filtered_df.columns:
        min_o, max_o = odds_home_range
        filtered_df = filtered_df[
            (filtered_df['odds_ft_home_team_win'] >= min_o) &
            (filtered_df['odds_ft_home_team_win'] <= max_o)
        ]
        
    if odds_away_range and 'odds_ft_away_team_win' in filtered_df.columns:
        min_o, max_o = odds_away_range
        filtered_df = filtered_df[
            (filtered_df['odds_ft_away_team_win'] >= min_o) &
            (filtered_df['odds_ft_away_team_win'] <= max_o)
        ]
        
    # 4. Filtra per Goal Timing Specifici (solo se forniti)
    
    if home_goal_mins:
        required_home_tokens = [re.escape(t.strip()) for t in home_goal_mins.split(',') if t.strip()]
        filtered_df = filtered_df[filtered_df['home_team_goal_timings'].astype(str).str.contains('|'.join(required_home_tokens), na=False)]

    if away_goal_mins:
        required_away_tokens = [re.escape(t.strip()) for t in away_goal_mins.split(',') if t.strip()]
        filtered_df = filtered_df[filtered_df['away_team_goal_timings'].astype(str).str.contains('|'.join(required_away_tokens), na=False)]
            
    # Calcola i gol residui per le partite filtrate
    filtered_df['remaining_home_goals'] = filtered_df['home_team_goal_count'] - filtered_df['home_goals_at_live_min']
    filtered_df['remaining_away_goals'] = filtered_df['away_team_goal_count'] - filtered_df['away_goals_at_live_min']
    filtered_df['remaining_total_goals'] = filtered_df['remaining_home_goals'] + filtered_df['remaining_away_goals']
    
    # Calcola il risultato HT effettivo (che non dipende dal minuto live, ma è utile per l'analisi HT)
    filtered_df['live_home_ht'] = filtered_df['home_team_goal_count_half_time']
    filtered_df['live_away_ht'] = filtered_df['away_team_goal_count_half_time']
    filtered_df['live_minute'] = current_min

    return filtered_df

def calculate_full_time_market_stats_live(df_live):
    """
    Calcola le statistiche dei mercati FT (WinRate, Over/Under, BTTS, ecc.)
    sui risultati finali effettivi del campione storico filtrato.
    """
    if df_live.empty or not {'home_team_goal_count', 'away_team_goal_count', 'total_goals_at_full_time'}.issubset(df_live.columns):
        return None
    
    total_matches = len(df_live)
    
    ft_home = df_live['home_team_goal_count']
    ft_away = df_live['away_team_goal_count']
    ft_total = df_live['total_goals_at_full_time']
    
    # --- 1. Risultati Esatti FT ---
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
    
    # --- 2. WinRate & Doppia Chance FT ---
    home_w = (ft_home > ft_away).sum()
    draws = (ft_home == ft_away).sum()
    away_w = (ft_home < ft_away).sum()
    
    df_wr = pd.DataFrame({
        'Esito': ['1 FT (Vittoria Casa)','X FT (Pareggio)','2 FT (Vittoria Trasferta)'],
        'Conteggio': [home_w, draws, away_w]
    })
    df_wr['Percentuale %'] = (df_wr['Conteggio'] / total_matches * 100).round(2)
    df_wr['Odd Minima'] = df_wr['Percentuale %'].apply(odd_min_from_percent)
    
    count_1X = int(((ft_home > ft_away) | (ft_home == ft_away)).sum())
    count_X2 = int(((ft_home < ft_away) | (ft_home == ft_away)).sum())
    count_12 = int((ft_home != ft_away).sum())
    
    df_dc = pd.DataFrame({
        'Mercato': ['1X FT','X2 FT','12 FT'],
        'Conteggio': [count_1X, count_X2, count_12]
    })
    df_dc['Percentuale %'] = (df_dc['Conteggio'] / total_matches * 100).round(2)
    df_dc['Odd Minima'] = df_dc['Percentuale %'].apply(odd_min_from_percent)
    
    # --- 3. Over/Under FT ---
    goal_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    over_rows = []
    under_rows = []
    
    for gl in goal_lines:
        # Over
        over_count = int((ft_total > (gl - 0.5)).sum())
        over_pct = round(over_count / total_matches * 100, 2)
        over_rows.append([f"Over {gl} FT", over_count, over_pct, odd_min_from_percent(over_pct)])
        
        # Under
        under_count = int(total_matches - over_count)
        under_pct = round(under_count / total_matches * 100, 2)
        under_rows.append([f"Under {gl} FT", under_count, under_pct, odd_min_from_percent(under_pct)])

    df_over = pd.DataFrame(over_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
    df_under = pd.DataFrame(under_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
    
    # --- 4. BTTS FT ---
    btts_yes = int(((ft_home >= 1) & (ft_away >= 1)).sum())
    btts_no = int(total_matches - btts_yes)
    
    df_btts = pd.DataFrame({
        'Mercato': ['BTTS SI (FT)','BTTS NO (FT)'],
        'Conteggio': [btts_yes, btts_no]
    })
    df_btts['Percentuale %'] = (df_btts['Conteggio'] / total_matches * 100).round(2)
    df_btts['Odd Minima'] = df_btts['Percentuale %'].apply(odd_min_from_percent)
    
    # --- 5. Next Goal (dal Minuto Live) ---
    next_goal_home = 0
    next_goal_away = 0
    no_more_goals = 0
    
    for _, row in df_live.iterrows():
        # Dobbiamo usare il minuto live salvato nel DataFrame filtrato
        live_minute = row['live_minute']
        
        home_after = row['remaining_home_goals']
        away_after = row['remaining_away_goals']
        
        if home_after == 0 and away_after == 0:
            no_more_goals += 1
        elif home_after > 0 or away_after > 0:
            
            h_mins = [m for m in parse_minutes_numeric(row.get('home_team_goal_timings')) if m > live_minute]
            a_mins = [m for m in parse_minutes_numeric(row.get('away_team_goal_timings')) if m > live_minute]
            
            earliest_h = min(h_mins) if h_mins else float('inf')
            earliest_a = min(a_mins) if a_mins else float('inf')
            
            if earliest_h == float('inf') and earliest_a == float('inf'):
                # Caso teoricamente impossibile se remaining > 0, ma per sicurezza
                no_more_goals += 1
            elif earliest_h < earliest_a:
                next_goal_home += 1
            elif earliest_a < earliest_h:
                next_goal_away += 1
            elif earliest_h == earliest_a and earliest_h != float('inf'): # Stesso minuto (solo se c'è stato un gol)
                next_goal_home += 1
                next_goal_away += 1
            else:
                no_more_goals += 1

                
    df_next_score = pd.DataFrame({
        'Mercato': ['Casa Segna Prossimo Gol', 'Trasferta Segna Prossimo Gol', 'Nessun Altro Gol'],
        'Conteggio': [next_goal_home, next_goal_away, no_more_goals]
    })
    df_next_score['Percentuale %'] = (df_next_score['Conteggio'] / total_matches * 100).round(2)
    df_next_score['Odd Minima'] = df_next_score['Percentuale %'].apply(odd_min_from_percent)

    return {
        'Risultati Esatti FT': df_cs_ft,
        'WinRate': df_wr,
        'Doppia Chance': df_dc,
        'Over FT': df_over,
        'Under FT': df_under,
        'BTTS FT': df_btts,
        'Next Goal': df_next_score
    }

def calculate_half_time_market_stats_live(df_live):
    """
    Calcola le statistiche dei mercati HT (WinRate, Over/Under, BTTS, ecc.)
    sui risultati Half Time effettivi del campione storico filtrato.
    """
    if df_live.empty or not {'live_home_ht', 'live_away_ht'}.issubset(df_live.columns):
        return None
    
    total_matches = len(df_live)
    
    ht_home = df_live['live_home_ht']
    ht_away = df_live['live_away_ht']
    ht_total = ht_home + ht_away
    
    # --- 1. Risultati Esatti HT ---
    betfair_order = ["0 - 0","0 - 1","0 - 2","0 - 3",
                     "1 - 0","1 - 1","1 - 2","1 - 3",
                     "2 - 0","2 - 1","2 - 2","2 - 3",
                     "3 - 0","3 - 1","3 - 2","3 - 3",
                     "Any Other Home Win","Any Other Away Win","Any Other Draw"]
    labels = [sh_cs_label(h, a) for h, a in zip(ht_home, ht_away)]
    dist = pd.Series(labels).value_counts(dropna=False).reindex(betfair_order, fill_value=0)
    df_cs_ht = pd.DataFrame({'HT (Betfair)': dist.index, 'Conteggio': dist.values})
    df_cs_ht['Percentuale %'] = (df_cs_ht['Conteggio'] / total_matches * 100).round(2)
    df_cs_ht['Odd Minima'] = df_cs_ht['Percentuale %'].apply(odd_min_from_percent)
    df_cs_ht['order'] = df_cs_ht['HT (Betfair)'].apply(lambda x: betfair_order.index(x))
    df_cs_ht = df_cs_ht.sort_values('order').drop(columns=['order'])
    
    # --- 2. WinRate & Doppia Chance HT ---
    home_w = (ht_home > ht_away).sum()
    draws = (ht_home == ht_away).sum()
    away_w = (ht_home < ht_away).sum()
    df_wr = pd.DataFrame({
        'Esito': ['1 HT (Vittoria Casa)','X HT (Pareggio)','2 HT (Vittoria Trasferta)'],
        'Conteggio': [home_w, draws, away_w]
    })
    df_wr['Percentuale %'] = (df_wr['Conteggio'] / total_matches * 100).round(2)
    df_wr['Odd Minima'] = df_wr['Percentuale %'].apply(odd_min_from_percent)
    count_1X = int(((ht_home > ht_away) | (ht_home == ht_away)).sum())
    count_X2 = int(((ht_home < ht_away) | (ht_home == ht_away)).sum())
    count_12 = int((ht_home != ht_away).sum())
    df_dc = pd.DataFrame({
        'Mercato': ['1X HT','X2 HT','12 HT'],
        'Conteggio': [count_1X, count_X2, count_12]
    })
    df_dc['Percentuale %'] = (df_dc['Conteggio'] / total_matches * 100).round(2)
    df_dc['Odd Minima'] = df_dc['Percentuale %'].apply(odd_min_from_percent)
    
    # --- 3. Over/Under HT ---
    goal_lines = [0.5, 1.5, 2.5, 3.5]
    over_rows = []
    under_rows = []
    for gl in goal_lines:
        # Over
        over_count = int((ht_total > (gl - 0.5)).sum())
        over_pct = round(over_count / total_matches * 100, 2)
        over_rows.append([f"Over {gl} HT", over_count, over_pct, odd_min_from_percent(over_pct)])
        
        # Under
        under_count = int(total_matches - over_count)
        under_pct = round(under_count / total_matches * 100, 2)
        under_rows.append([f"Under {gl} HT", under_count, under_pct, odd_min_from_percent(under_pct)])
    df_over = pd.DataFrame(over_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
    df_under = pd.DataFrame(under_rows, columns=['Mercato','Conteggio','Percentuale %','Odd Minima'])
    
    # --- 4. BTTS HT ---
    btts_yes = int(((ht_home >= 1) & (ht_away >= 1)).sum())
    btts_no = int(total_matches - btts_yes)
    df_btts = pd.DataFrame({
        'Mercato': ['BTTS SI (HT)','BTTS NO (HT)'],
        'Conteggio': [btts_yes, btts_no]
    })
    df_btts['Percentuale %'] = (df_btts['Conteggio'] / total_matches * 100).round(2)
    df_btts['Odd Minima'] = df_btts['Percentuale %'].apply(odd_min_from_percent)
    return {
        'Risultati Esatti HT': df_cs_ht,
        'WinRate': df_wr,
        'Doppia Chance': df_dc,
        'Over HT': df_over,
        'Under HT': df_under,
        'BTTS HT': df_btts,
    }

def calculate_remaining_timeframe_stats(df_live, current_min):
    """
    Calcola le statistiche dei gol segnati solo nel tempo residuo.
    """
    if df_live.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    total_live_matches = len(df_live)

    def count_future_goals_by_bucket(goal_timings_cell, current_min, step):
        buckets = gen_buckets(step)
        counts = {b: 0 for b in buckets}
        if pd.isna(goal_timings_cell):
            return counts
        
        goal_minutes = parse_minutes_numeric(goal_timings_cell)
        future_goals = [m for m in goal_minutes if m > current_min]
        
        for m in future_goals:
            label = bucket_label_5min(m) if step == 5 else bucket_label_15min(m)
            if label in counts:
                counts[label] += 1
        return counts

    def calculate_stats_for_step(step):
        buckets = gen_buckets(step)
        counts_matches_with_goal = {b: 0 for b in buckets}
        home_gf_goals = {b: 0 for b in buckets}
        away_gf_goals = {b: 0 for b in buckets}

        for _, row in df_live.iterrows():
            future_home_buckets = count_future_goals_by_bucket(row['home_team_goal_timings'], current_min, step)
            future_away_buckets = count_future_goals_by_bucket(row['away_team_goal_timings'], current_min, step)

            # Check per almeno un gol in un timeframe
            for b in buckets:
                if future_home_buckets[b] > 0 or future_away_buckets[b] > 0:
                    counts_matches_with_goal[b] += 1

            # Gol fatti/subiti dalle squadre selezionate
            # Questa logica non è precisa se le squadre non sono fisse, ma la lasciamo per coerenza con le altre sezioni.
            # Idealmente andrebbe rifatta per tracciare la squadra 'di casa' del campione
            # in base ai filtri applicati
            for b in buckets:
                home_gf_goals[b] += future_home_buckets[b]
                away_gf_goals[b] += future_away_buckets[b]

        rows = []
        for b in buckets:
            with_goal_count = counts_matches_with_goal[b]
            pct = (with_goal_count / total_live_matches * 100) if total_live_matches > 0 else 0
            odd_min = odd_min_from_percent(pct)
            
            rows.append([
                b,
                with_goal_count, pct, odd_min,
                home_gf_goals[b], away_gf_goals[b]
            ])
            
        return pd.DataFrame(rows, columns=[
            'Timeframe',
            'Partite con Gol (Futuro)', 'Percentuale %', 'Odd Minima',
            'Home GF (Futuro)', 'Away GF (Futuro)'
        ])
    
    df_tf15 = calculate_stats_for_step(15)
    df_tf5 = calculate_stats_for_step(5)
    
    return df_tf15, df_tf5


def analyze_value_bets(df):
    """
    Analizza i mercati per identificare potenziali Value Bet.
    """
    if df.empty:
        return pd.DataFrame(), None

    total_matches = len(df)
    
    # Check per le colonne necessarie
    required_odds_cols = {
        'odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win',
        'odds_over_05', 'odds_over_15', 'odds_over_25', 'odds_over_35',
        'odds_btts_yes', 'odds_btts_no'
    }
    
    if not required_odds_cols.issubset(df.columns):
        missing = required_odds_cols - set(df.columns)
        st.warning(f"Attenzione: Colonne quote mancanti per l'analisi Value Bet: {', '.join(missing)}. Riprova con un file che le contenga.")
        return pd.DataFrame(), None

    # Calcola le percentuali storiche per ogni mercato
    historical_data = {
        # FT Win Rate
        '1 FT': ((df['home_team_goal_count'] > df['away_team_goal_count']).sum() / total_matches) * 100,
        'X FT': ((df['home_team_goal_count'] == df['away_team_goal_count']).sum() / total_matches) * 100,
        '2 FT': ((df['home_team_goal_count'] < df['away_team_goal_count']).sum() / total_matches) * 100,
        # Overs
        'O 0.5 FT': ((df['total_goals_at_full_time'] > 0.5).sum() / total_matches) * 100,
        'O 1.5 FT': ((df['total_goals_at_full_time'] > 1.5).sum() / total_matches) * 100,
        'O 2.5 FT': ((df['total_goals_at_full_time'] > 2.5).sum() / total_matches) * 100,
        'O 3.5 FT': ((df['total_goals_at_full_time'] > 3.5).sum() / total_matches) * 100,
        # BTTS
        'BTTS SI FT': ((df['home_team_goal_count'] > 0) & (df['away_team_goal_count'] > 0)).sum() / total_matches * 100,
        'BTTS NO FT': ((df['home_team_goal_count'] == 0) | (df['away_team_goal_count'] == 0)).sum() / total_matches * 100,
    }
    
    # Mapping tra mercati e colonne quote
    odds_columns = {
        '1 FT': 'odds_ft_home_team_win',
        'X FT': 'odds_ft_draw',
        '2 FT': 'odds_ft_away_team_win',
        'O 0.5 FT': 'odds_over_05',
        'O 1.5 FT': 'odds_over_15',
        'O 2.5 FT': 'odds_over_25',
        'O 3.5 FT': 'odds_over_35',
        'BTTS SI FT': 'odds_btts_yes',
        'BTTS NO FT': 'odds_btts_no',
    }
    
    # Analizza ogni mercato
    value_bets = []
    
    for market, historical_pct in historical_data.items():
        if market not in odds_columns: continue
            
        odds_col = odds_columns[market]
        
        # Filtra solo le righe dove la colonna quote è presente e valida
        valid_odds_df = df[df[odds_col].notna()]
        if valid_odds_df.empty:
            continue
            
        # Calcola la media delle quote di mercato per questo campione
        market_odd_avg = valid_odds_df[odds_col].mean()

        # Calcola l'Odd Minima
        historical_odd_min = odd_min_from_percent(historical_pct)
        
        if historical_odd_min is not None and market_odd_avg > historical_odd_min:
            # Calcola l'incremento percentuale dell'odd di mercato rispetto all'odd minima
            odd_increment = ((market_odd_avg - historical_odd_min) / historical_odd_min) * 100
            
            value_bets.append({
                'Mercato': market,
                'Percentuale Storica %': historical_pct,
                'Odd Minima Storica': historical_odd_min,
                'Odd di Mercato Media': market_odd_avg,
                'Incremento %': odd_increment
            })

    value_bets_df = pd.DataFrame(value_bets)
    
    # Ordina per incremento
    if not value_bets_df.empty:
        value_bets_df = value_bets_df.sort_values(by='Incremento %', ascending=False)
        value_bets_df.reset_index(drop=True, inplace=True)
    
    return value_bets_df, total_matches

# ---------- Streamlit App Logic ----------

if "df" not in st.session_state:
    st.session_state.df = None
    
uploaded_file = st.sidebar.file_uploader("Carica un file CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=';', on_bad_lines='skip')
        st.session_state.df = df.copy()
        
        # Rileva e converte colonne di date se presenti
        if 'date' in df.columns:
            st.session_state.df['date'] = pd.to_datetime(st.session_state.df['date'], errors='coerce', dayfirst=True)
            if 'anno' not in st.session_state.df.columns:
                st.session_state.df['anno'] = st.session_state.df['date'].dt.year
        
        st.success("File caricato con successo!")

    except Exception as e:
        st.error(f"Errore durante il caricamento del file: {e}")


if st.session_state.df is not None:
    df = st.session_state.df
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtri")

    # Filtri globali
    with st.sidebar.expander("Filtri Globali"):
        all_leagues = ['Tutte'] + sorted(df['league_name'].unique().tolist())
        selected_league = st.selectbox("Seleziona Campionato", all_leagues)

        all_years = ['Tutti'] + sorted(df['anno'].unique().tolist(), reverse=True) if 'anno' in df.columns else ['Tutti']
        selected_year = st.selectbox("Seleziona Anno", all_years)
        
        min_matches = st.number_input("Numero Minimo Partite nel Campione", min_value=1, value=100)

    # Applica i filtri globali
    df_filtered = df.copy()
    if selected_league != 'Tutte':
        df_filtered = df_filtered[df_filtered['league_name'] == selected_league]
    if selected_year != 'Tutti' and 'anno' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['anno'] == selected_year]
        
    st.session_state.df_filtered = df_filtered

    # Se il campione è troppo piccolo, non mostrare i risultati
    if len(st.session_state.df_filtered) < min_matches:
        st.warning(f"Campione troppo piccolo (solo {len(st.session_state.df_filtered)} partite). Modifica i filtri.")
    else:
        # Mostra direttamente il Capitolo 5
        st.markdown("### Capitolo 5: Value Bet")
        st.write("Confronta le quote medie di mercato con le quote minime storiche per identificare potenziali Value Bet.")
        
        # Filtri per Value Bet
        min_matches_vb = st.number_input("Numero Minimo Partite per Mercato", min_value=1, value=50, key='min_matches_vb')
        
        # Applica filtro sul numero minimo di partite
        df_for_vb = st.session_state.df_filtered.copy()
        if len(df_for_vb) < min_matches_vb:
            st.warning(f"Il campione filtrato non è sufficiente per analizzare le Value Bet ({len(df_for_vb)} partite). Aumenta il numero di partite nei filtri globali.")
        else:
            value_bets_df, total_vb_matches = analyze_value_bets(df_for_vb)
            
            st.markdown(f"**Analisi su campione di**: {total_vb_matches} partite")
            
            if not value_bets_df.empty:
                # Filtra per minimo incremento
                value_bet_min_increment = st.number_input("Incremento minimo (%)", min_value=0.0, value=10.0, step=1.0)
                final_vb_df = value_bets_df[value_bets_df['Incremento %'] >= value_bet_min_increment].copy()
                
                if not final_vb_df.empty:
                    st.markdown("#### Risultati Value Bet")
                    st.dataframe(final_vb_df.style.format({
                        'Percentuale Storica %': '{:.2f}%',
                        'Odd Minima Storica': '{:.2f}',
                        'Odd di Mercato Media': '{:.2f}',
                        'Incremento %': '{:.2f}%'
                    }), use_container_width=True)
                else:
                    st.info("Nessuna Value Bet trovata con i filtri selezionati.")
            else:
                st.warning("Dati insufficienti o quote di mercato mancanti per l'analisi delle Value Bet.")
