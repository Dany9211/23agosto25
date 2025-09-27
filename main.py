import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

st.set_page_config(page_title="Filtri Dati Calcio", layout="wide")
st.title("⚽ Dashboard Filtri partite squadre Calcio⚽")
st.write("Carica il tuo file CSV per iniziare l'analisi.")

# ---------- Utility ----------

# ---------- Helper per confronto Value Bet ----------
def compare_book_odd_to_odd_min(book_odd, odd_min):
    try:
        if odd_min is None or pd.isna(odd_min):
            return None
        b = float(book_odd) if book_odd is not None else None
        o = float(odd_min) if odd_min is not None else None
        if b is None or o is None:
            return None
        if abs(b - o) < 1e-6:
            return None
        return 'BACK' if b > o else 'LAY'
    except Exception:
        return None

def format_edge_message(market_name, book_odd, odd_min):
    if odd_min is None or pd.isna(odd_min):
        return f"{market_name}: Odd minima non disponibile."
    res = compare_book_odd_to_odd_min(book_odd, odd_min)
    if res == 'BACK':
        return f"{market_name}: BOOK {book_odd:.2f}  >  OddMin {odd_min:.2f} → VALUE (BACK)"
    elif res == 'LAY':
        return f"{market_name}: BOOK {book_odd:.2f}  <  OddMin {odd_min:.2f} → VALUE (LAY)"
    else:
        return f"{market_name}: BOOK {book_odd:.2f} ~= OddMin {odd_min:.2f} → Nessun edge"
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
            elif earliest_h == earliest_a and earliest_h != float('inf'):
                # Stesso minuto (solo se c'è stato un gol)
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
    goal_lines = [0.5, 1.5, 2.5]
    over_rows = []
    under_rows = []
    for gl in goal_lines:
        over_count = int((ht_total > (gl - 0.5)).sum())
        over_pct = round(over_count / total_matches * 100, 2)
        over_rows.append([f"Over {gl} HT", over_count, over_pct, odd_min_from_percent(over_pct)])
        
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

def calculate_remaining_timeframe_stats(df, current_min):
    total_matches = len(df)
    
    # Calculate goal stats for 15-minute intervals
    df_15_min = df.copy()
    df_15_min['remaining_goal_timings'] = df_15_min.apply(
        lambda row: [m for m in parse_minutes_numeric(row['home_team_goal_timings']) + parse_minutes_numeric(row['away_team_goal_timings']) if m > current_min],
        axis=1
    )
    
    buckets_15 = gen_buckets(15)
    counts_with_goal_15 = {b: 0 for b in buckets_15}
    counts_with_2plus_15 = {b: 0 for b in buckets_15}
    
    for _, row in df_15_min.iterrows():
        rem_buckets = [bucket_label_15min(m) for m in row['remaining_goal_timings']]
        per_bucket = {b: rem_buckets.count(b) for b in buckets_15}
        for b, c in per_bucket.items():
            if c >= 1:
                counts_with_goal_15[b] += 1
            if c >= 2:
                counts_with_2plus_15[b] += 1
                
    rows_15 = []
    for b in buckets_15:
        with_goal = counts_with_goal_15[b]
        pct = round((with_goal / total_matches) * 100, 2) if total_matches else 0.0
        odd_min = odd_min_from_percent(pct)
        g2 = counts_with_2plus_15[b]
        pct2 = round((g2 / total_matches) * 100, 2) if total_matches else 0.0
        odd_min2 = odd_min_from_percent(pct2) if pct2 > 0 else None
        
        rows_15.append([b, with_goal, pct, odd_min, g2, pct2, odd_min2])
        
    df_tf15_rem = pd.DataFrame(rows_15, columns=[
        'Timeframe',
        'Partite con Gol (Futuro)', 'Percentuale %', 'Odd Minima',
        '>= 2 Gol (Futuro)', '>= 2 Gol %', 'Odd Minima >= 2 Gol'
    ])
    
    # Calculate goal stats for 5-minute intervals
    df_5_min = df.copy()
    df_5_min['remaining_goal_timings'] = df_5_min.apply(
        lambda row: [m for m in parse_minutes_numeric(row['home_team_goal_timings']) + parse_minutes_numeric(row['away_team_goal_timings']) if m > current_min],
        axis=1
    )
    
    buckets_5 = gen_buckets(5)
    counts_with_goal_5 = {b: 0 for b in buckets_5}
    counts_with_2plus_5 = {b: 0 for b in buckets_5}
    
    for _, row in df_5_min.iterrows():
        rem_buckets = [bucket_label_5min(m) for m in row['remaining_goal_timings']]
        per_bucket = {b: rem_buckets.count(b) for b in buckets_5}
        for b, c in per_bucket.items():
            if c >= 1:
                counts_with_goal_5[b] += 1
            if c >= 2:
                counts_with_2plus_5[b] += 1
                
    rows_5 = []
    for b in buckets_5:
        with_goal = counts_with_goal_5[b]
        pct = round((with_goal / total_matches) * 100, 2) if total_matches else 0.0
        odd_min = odd_min_from_percent(pct)
        g2 = counts_with_2plus_5[b]
        pct2 = round((g2 / total_matches) * 100, 2) if total_matches else 0.0
        odd_min2 = odd_min_from_percent(pct2) if pct2 > 0 else None
        
        rows_5.append([b, with_goal, pct, odd_min, g2, pct2, odd_min2])
    
    df_tf5_rem = pd.DataFrame(rows_5, columns=[
        'Timeframe',
        'Partite con Gol (Futuro)', 'Percentuale %', 'Odd Minima',
        '>= 2 Gol (Futuro)', '>= 2 Gol %', 'Odd Minima >= 2 Gol'
    ])
    
    return df_tf15_rem, df_tf5_rem


# ---------- Dashboard Main Section ----------

uploaded_file = st.sidebar.file_uploader("Carica un file CSV", type=["csv"], help="Il file CSV deve contenere i dati delle partite. Assicurati che le colonne necessarie per l'analisi (es. goal_timings, odds, etc.) siano presenti.")

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        # Rimuovi eventuali spazi bianchi dai nomi delle colonne
        df.columns = df.columns.str.strip()
        # Converti la colonna 'date' in datetime, se esiste
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df

    data_df = load_data(uploaded_file)
    
    st.markdown("---")
    st.subheader("Riepilogo Dati Caricati")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Numero di partite caricate", value=len(data_df))
    
    with col2:
        num_teams = len(pd.unique(data_df[['home_team_name', 'away_team_name']].values.ravel('K')))
        st.metric(label="Numero di squadre uniche", value=num_teams)
        
    with col3:
        num_leagues = len(data_df['league_name'].unique()) if 'league_name' in data_df.columns else "N/A"
        st.metric(label="Numero di campionati unici", value=num_leagues)

    st.markdown("---")
    st.markdown("#### **Filtri Base Dati**")
    
    # Multi-select per i campionati
    all_leagues = ['Tutti'] + sorted(list(data_df['league_name'].unique())) if 'league_name' in data_df.columns else ['Tutti']
    selected_leagues = st.multiselect("Seleziona uno o più campionati:", all_leagues, default=['Tutti'])

    # Multi-select per le squadre (home e away)
    all_teams = ['Tutte'] + sorted(list(data_df[['home_team_name', 'away_team_name']].values.ravel('K')))
    col_team1, col_team2 = st.columns(2)
    with col_team1:
        selected_home_team = st.selectbox("Seleziona la squadra di casa:", all_teams, key="home_select")
    with col_team2:
        selected_away_team = st.selectbox("Seleziona la squadra in trasferta:", all_teams, key="away_select")

    # Filtra il DataFrame in base alle selezioni
    filtered_df = data_df.copy()
    
    if 'Tutti' not in selected_leagues:
        filtered_df = filtered_df[filtered_df['league_name'].isin(selected_leagues)]
        
    if selected_home_team != 'Tutte':
        filtered_df = filtered_df[filtered_df['home_team_name'] == selected_home_team]
        
    if selected_away_team != 'Tutte':
        filtered_df = filtered_df[filtered_df['away_team_name'] == selected_away_team]

    st.markdown("---")
    st.markdown("#### **A. Analisi Risultati FT per il campione filtrato**")

    # Calcola le statistiche FT
    ft_stats = calculate_full_time_market_stats_live(filtered_df)
    
    if ft_stats:
        
        col_ft_wr, col_ft_dc = st.columns(2)
        with col_ft_wr:
            st.markdown("##### **Percentuale Esiti FT**")
            st.dataframe(ft_stats['WinRate'].style.background_gradient(subset=['Percentuale %'], cmap="RdYlGn"), use_container_width=True)
            
        with col_ft_dc:
            st.markdown("##### **Percentuale Doppia Chance FT**")
            st.dataframe(ft_stats['Doppia Chance'].style.background_gradient(subset=['Percentuale %'], cmap="RdYlGn"), use_container_width=True)

        col_ft_ov, col_ft_un = st.columns(2)
        with col_ft_ov:
            st.markdown("##### **Over FT**")
            st.dataframe(ft_stats['Over FT'].style.background_gradient(subset=['Percentuale %'], cmap="RdYlGn"), use_container_width=True)
        with col_ft_un:
            st.markdown("##### **Under FT**")
            st.dataframe(ft_stats['Under FT'].style.background_gradient(subset=['Percentuale %'], cmap="RdYlGn"), use_container_width=True)
            
        st.markdown("---")
        st.markdown("##### **Risultati Esatti FT**")
        st.dataframe(ft_stats['Risultati Esatti FT'].style.background_gradient(subset=['Percentuale %'], cmap="RdYlGn"), use_container_width=True)
    else:
        st.warning("Per calcolare le statistiche dei risultati FT, assicurati che il file CSV contenga le colonne 'home_team_goal_count', 'away_team_goal_count' e 'total_goals_at_full_time'.")


    st.markdown("---")
    st.markdown("#### **B. Analisi Live: Filtri Aggiuntivi**")
    st.markdown("Usa questi filtri per trovare partite storiche che si trovavano in una situazione live simile.")
    
    col_live1, col_live2 = st.columns(2)
    with col_live1:
        current_min = st.number_input("Minuto Corrente:", min_value=0, max_value=120, value=60)
        home_score = st.number_input("Gol Squadra Casa:", min_value=0, value=0)
    with col_live2:
        away_score = st.number_input("Gol Squadra Trasferta:", min_value=0, value=0)
        
    home_goal_mins_input = st.text_input("Minutaggio gol casa (es. 25, 45+2):", help="Inserisci i minuti separati da virgola. Lascia vuoto se non rilevante.")
    away_goal_mins_input = st.text_input("Minutaggio gol trasferta (es. 25, 45+2):", help="Inserisci i minuti separati da virgola. Lascia vuoto se non rilevante.")
    
    live_filtered_df = filter_live_matches(data_df, current_min, home_score, away_score, home_goal_mins_input, away_goal_mins_input, selected_home_team, selected_away_team)
    
    st.markdown(f"**Partite trovate con i filtri live:** {len(live_filtered_df)}")
    
    if not live_filtered_df.empty:
        st.markdown("---")
        st.markdown("#### C. Distribuzione Gol nel Minutaggio Futuro (dopo il minuto corrente)")
        
        df_tf15_rem, df_tf5_rem = calculate_remaining_timeframe_stats(live_filtered_df, current_min)
        
        col_tf15, col_tf5 = st.columns(2)
        with col_tf15:
            st.markdown("**Ogni 15 minuti (Tempo Residuo)**")
            # Nascondi i campi GF/GA e Odd Minima per chiarezza sul sample
            df_tf15_display = df_tf15_rem.drop(columns=['Odd Minima'], errors='ignore').copy()
            df_tf15_display.rename(columns={'Partite con Gol (Futuro)': 'Partite con Gol'}, inplace=True)
            st.dataframe(style_table(df_tf15_display, ['Percentuale %']), use_container_width=True)
        with col_tf5:
            st.markdown("**Ogni 5 minuti (Tempo Residuo)**")
            df_tf5_display = df_tf5_rem.drop(columns=['Odd Minima'], errors='ignore').copy()
            df_tf5_display.rename(columns={'Partite con Gol (Futuro)': 'Partite con Gol'}, inplace=True)
            st.dataframe(style_table(df_tf5_display, ['Percentuale %']), use_container_width=True)

        st.markdown("---")
        st.markdown("#### D. Anteprima Partite Live Trovate")
        st.dataframe(live_filtered_df.head(20).drop(columns=['remaining_home_goals', 'remaining_away_goals', 'remaining_total_goals', 'live_home_ht', 'live_away_ht', 'home_goals_at_live_min', 'away_goals_at_live_min'], errors='ignore'))
    
# ---------- Value Bet Checker (NON filtrante) ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Value Bet Checker (solo confronto, NON filtra)")

book_home_odd = st.sidebar.number_input("Book Quota Home (BACK)", min_value=1.01, value=2.00, step=0.01, key="book_home_odd")
book_draw_odd = st.sidebar.number_input("Book Quota Draw (BACK)", min_value=1.01, value=3.50, step=0.01, key="book_draw_odd")
book_away_odd = st.sidebar.number_input("Book Quota Away (BACK)", min_value=1.01, value=4.00, step=0.01, key="book_away_odd")

st.sidebar.markdown("**Mercati Soggetti a confronto**")
book_over25_odd = st.sidebar.number_input("Book Quota Over 2.5 (BACK)", min_value=1.01, value=1.85, step=0.01, key="book_over25_odd")
book_under25_odd = st.sidebar.number_input("Book Quota Under 2.5 (BACK)", min_value=1.01, value=2.10, step=0.01, key="book_under25_odd")
book_btts_yes_odd = st.sidebar.number_input("Book Quota BTTS SI (BACK)", min_value=1.01, value=1.70, step=0.01, key="book_btts_yes_odd")
book_btts_no_odd = st.sidebar.number_input("Book Quota BTTS NO (BACK)", min_value=1.01, value=2.00, step=0.01, key="book_btts_no_odd")
book_sh_first_odd = st.sidebar.number_input("Book Prossimo Gol (SH) (BACK)", min_value=1.01, value=2.50, step=0.01, key="book_sh_first_odd")
book_ht_over05_odd = st.sidebar.number_input("Book Over 0.5 HT (BACK)", min_value=1.01, value=1.35, step=0.01, key="book_ht_over05_odd")

if st.sidebar.button("Controlla Value Bet"):
    if live_filtered_df.empty:
        st.sidebar.warning("Carica un file o applica dei filtri live per analizzare i dati.")
    else:
        # Calcola le statistiche FT e HT sul campione live filtrato
        ft_stats_live = calculate_full_time_market_stats_live(live_filtered_df)
        ht_stats_live = calculate_half_time_market_stats_live(live_filtered_df)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Risultati Value Bet")
        
        # Estrai le quote minime
        min_odd_home = ft_stats_live['WinRate'][ft_stats_live['WinRate']['Esito'] == '1 FT (Vittoria Casa)']['Odd Minima'].iloc[0] if not ft_stats_live['WinRate'].empty else None
        min_odd_draw = ft_stats_live['WinRate'][ft_stats_live['WinRate']['Esito'] == 'X FT (Pareggio)']['Odd Minima'].iloc[0] if not ft_stats_live['WinRate'].empty else None
        min_odd_away = ft_stats_live['WinRate'][ft_stats_live['WinRate']['Esito'] == '2 FT (Vittoria Trasferta)']['Odd Minima'].iloc[0] if not ft_stats_live['WinRate'].empty else None
        
        min_odd_over25 = ft_stats_live['Over FT'][ft_stats_live['Over FT']['Mercato'] == 'Over 2.5 FT']['Odd Minima'].iloc[0] if not ft_stats_live['Over FT'].empty else None
        min_odd_under25 = ft_stats_live['Under FT'][ft_stats_live['Under FT']['Mercato'] == 'Under 2.5 FT']['Odd Minima'].iloc[0] if not ft_stats_live['Under FT'].empty else None
        
        min_odd_btts_yes = ft_stats_live['BTTS FT'][ft_stats_live['BTTS FT']['Mercato'] == 'BTTS SI (FT)']['Odd Minima'].iloc[0] if not ft_stats_live['BTTS FT'].empty else None
        min_odd_btts_no = ft_stats_live['BTTS FT'][ft_stats_live['BTTS FT']['Mercato'] == 'BTTS NO (FT)']['Odd Minima'].iloc[0] if not ft_stats_live['BTTS FT'].empty else None
        
        # Calcola prossima squadra che segna (se possibile)
        next_goal_home_pct = ft_stats_live['Next Goal'][ft_stats_live['Next Goal']['Mercato'] == 'Casa Segna Prossimo Gol']['Percentuale %'].iloc[0] if not ft_stats_live['Next Goal'].empty else 0
        min_odd_sh_first = odd_min_from_percent(next_goal_home_pct)
        
        min_odd_ht_over05 = ht_stats_live['Over HT'][ht_stats_live['Over HT']['Mercato'] == 'Over 0.5 HT']['Odd Minima'].iloc[0] if not ht_stats_live['Over HT'].empty else None

        # Mostra i confronti
        st.sidebar.write(format_edge_message('1 FT (Casa)', book_home_odd, min_odd_home))
        st.sidebar.write(format_edge_message('X FT (Pareggio)', book_draw_odd, min_odd_draw))
        st.sidebar.write(format_edge_message('2 FT (Trasferta)', book_away_odd, min_odd_away))
        
        st.sidebar.write("---")
        st.sidebar.write(format_edge_message('Over 2.5 FT', book_over25_odd, min_odd_over25))
        st.sidebar.write(format_edge_message('Under 2.5 FT', book_under25_odd, min_odd_under25))
        
        st.sidebar.write("---")
        st.sidebar.write(format_edge_message('BTTS SI FT', book_btts_yes_odd, min_odd_btts_yes))
        st.sidebar.write(format_edge_message('BTTS NO FT', book_btts_no_odd, min_odd_btts_no))
        
        st.sidebar.write("---")
        st.sidebar.write(format_edge_message('Prossimo Gol Casa', book_sh_first_odd, min_odd_sh_first))
        
        st.sidebar.write("---")
        st.sidebar.write(format_edge_message('Over 0.5 HT', book_ht_over05_odd, min_odd_ht_over05))

else:
    st.info("Carica un file CSV per iniziare l'analisi o utilizza i filtri a sinistra.")
