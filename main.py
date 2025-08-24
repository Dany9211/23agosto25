import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisi Campionati Next Gol e stats live", layout="wide")
st.title("Analisi Tabella 23agosto25")

# --- Funzione connessione al database ---
@st.cache_data
def run_query(query: str):
    """
    Esegue una query SQL e restituisce i risultati come DataFrame.
    La funzione è cacheata per evitare di riconnettersi al database
    ogni volta che l'applicazione si aggiorna.
    """
    try:
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            port=st.secrets["postgres"]["port"],
            dbname=st.secrets["postgres"]["dbname"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
            sslmode="require"
        )
        df = pd.read_sql(query, conn)
        conn.close()
        st.info(f"Query eseguita: {query}")
        st.info(f"Colonne caricate dal database (RAW): {df.columns.tolist()}") # Debugging: mostra colonne grezze
        return df
    except Exception as e:
        st.error(f"Errore di connessione o esecuzione query al database: {e}")
        st.stop() # Ferma l'esecuzione se c'è un errore grave al DB
        return pd.DataFrame()

# --- Caricamento dati iniziali ---
try:
    # La query SELECT * dovrebbe funzionare se i nomi delle colonne corrispondono
    df = run_query('SELECT * FROM "23agosto25";')
    
    if df.empty:
        st.warning("Il DataFrame caricato dal database è vuoto. Controlla la tabella in Supabase.")
        st.stop()

    st.write(f"**Righe iniziali nel dataset:** {len(df)}")
    st.write("Prime 5 righe del DataFrame:", df.head()) # Debugging: mostra le prime righe
    st.write("Tutte le colonne nel DataFrame:", df.columns.tolist()) # Debugging: mostra tutte le colonne caricate


    # Converti la colonna 'anno' in numerico subito dopo il caricamento
    # I valori non convertibili diventeranno NaN.
    # Assicurati che il nome 'anno' sia esattamente come quello nel database (case-sensitive)
    if "anno" in df.columns:
        df["anno"] = pd.to_numeric(df["anno"], errors='coerce')
        # Puoi anche decidere di eliminare le righe con anni non validi se non ti servono
        # df = df.dropna(subset=["anno"])
        st.success("Colonna 'anno' convertita a numerico con successo.")
    else:
        st.error("ERRORE: La colonna 'anno' non è stata trovata nel DataFrame. Controlla il nome della colonna in Supabase (sensibile al maiuscolo/minuscolo) e riprova.")
        st.stop() # Ferma l'esecuzione se 'anno' è critica e non trovata
        
except Exception as e:
    st.error(f"Errore critico durante il caricamento o la preparazione del database: {e}")
    st.stop()

# --- Aggiunta colonne risultato_ft e risultato_ht ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)
if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

filters = {}

# --- FILTRI INIZIALI ---
st.sidebar.header("Filtri Dati")

# Filtro League (Campionato) - Deve essere il primo per filtrare le squadre
if "league" in df.columns:
    leagues = ["Tutte"] + sorted(df["league"].dropna().unique())
    selected_league = st.sidebar.selectbox("Seleziona Campionato", leagues)
    if selected_league != "Tutte":
        filters["league"] = selected_league
    
    # Crea un DataFrame temporaneo per filtrare le squadre in base al campionato
    if selected_league != "Tutte":
        filtered_teams_df = df[df["league"] == selected_league]
    else:
        filtered_teams_df = df.copy()
else:
    filtered_teams_df = df.copy()
    selected_league = "Tutte"
    st.sidebar.warning("La colonna 'league' non è presente nel dataset.")

# Filtro Anno - AGGIORNATO
if "anno" in df.columns:
    # Ora la colonna 'anno' è già numerica nel DataFrame 'df'
    # Rimuovi i valori NaN per l'elenco delle opzioni della selectbox e converti in int
    anni_validi = df["anno"].dropna().astype(int).unique()
    anni = ["Tutte"] + sorted(anni_validi.tolist())
    
    selected_anno = st.sidebar.selectbox("Seleziona Anno", anni)
    if selected_anno != "Tutte":
        filters["anno"] = int(selected_anno) # Memorizza come int per coerenza
else:
    st.sidebar.warning("La colonna 'anno' non è presente nel dataset (dopo il caricamento iniziale).")

# Filtro Giornata - AGGIORNATO
if "giornata" in df.columns:
    # Converte la colonna 'giornata' in numerico, forzando gli errori a NaN
    giornata_numeric = pd.to_numeric(df["giornata"], errors='coerce')
    
    # Rimuove i valori NaN per calcolare min e max
    giornata_numeric = giornata_numeric.dropna()

    if not giornata_numeric.empty:
        giornata_min = int(giornata_numeric.min())
        giornata_max = int(giornata_numeric.max())
        
        # Aggiungi un controllo per assicurarti che min_value sia minore o uguale a max_value
        if giornata_min > giornata_max:
            giornata_min, giornata_max = giornata_max, giornata_min # Scambia se sono invertiti
            st.sidebar.warning("I valori min e max per la giornata sono stati scambiati per correggere il range.")
            
        giornata_range = st.sidebar.slider(
            "Seleziona Giornata",
            min_value=giornata_min,
            max_value=giornata_max,
            value=(giornata_min, giornata_max) # Imposta il valore iniziale all'intero range
        )
        filters["giornata"] = giornata_range
    else:
        st.sidebar.warning("La colonna 'giornata' non contiene valori numerici validi dopo la pulizia.")
else:
    st.sidebar.warning("La colonna 'giornata' non è presente nel dataset.")


# --- FILTRI SQUADRE (ora dinamici) ---
# Modificato per supportare la logica Home vs Away
if "home_team" in filtered_teams_df.columns and "away_team" in filtered_teams_df.columns:
    home_teams = ["Tutte"] + sorted(filtered_teams_df["home_team"].dropna().unique())
    selected_home = st.sidebar.selectbox("Seleziona Squadra Home", home_teams)
    away_teams = ["Tutte"] + sorted(filtered_teams_df["away_team"].dropna().unique())
    selected_away = st.sidebar.selectbox("Seleziona Squadra Away", away_teams)

    # Applica i filtri delle squadre con logica AND
    if selected_home != "Tutte":
        filters["home_team"] = selected_home
    if selected_away != "Tutte":
        filters["away_team"] = selected_away
else:
    st.sidebar.warning("Le colonne 'home_team' o 'away_team' non sono presenti per il filtro squadre.")
    selected_home = "Tutte" # Imposta un valore di default per evitare errori
    selected_away = "Tutte" # Imposta un valore di default per evitare errori


# --- NUOVO FILTRO: Risultato HT ---
if "risultato_ht" in df.columns:
    ht_results = sorted(df["risultato_ht"].dropna().unique())
    selected_ht_results = st.sidebar.multiselect("Seleziona Risultato HT", ht_results, default=None)
    if selected_ht_results:
        filters["risultato_ht"] = selected_ht_results
else:
    st.sidebar.info("La colonna 'risultato_ht' non è disponibile per il filtro.")

# --- FUNZIONE per filtri range ---
def add_range_filter(col_name, label=None):
    if col_name in df.columns:
        col_temp = pd.to_numeric(df[col_name].astype(str).str.replace(",", "."), errors="coerce")
        col_min = float(col_temp.min(skipna=True))
        col_max = float(col_temp.max(skipna=True))
        
        st.sidebar.write(f"Range attuale {label or col_name}: {col_min} - {col_max}")
        min_val = st.sidebar.text_input(f"Min {label or col_name}", value="", key=f"min_{col_name}")
        max_val = st.sidebar.text_input(f"Max {label or col_name}", value="", key=f"max_{col_name}")
        
        if min_val.strip() != "" and max_val.strip() != "":
            try:
                filters[col_name] = (float(min_val), float(max_val))
            except ValueError:
                st.sidebar.warning(f"Valori non validi per {label or col_name}. Inserisci numeri.")
    else:
        st.sidebar.info(f"La colonna '{col_name}' non è disponibile per il filtro range.")


st.sidebar.header("Filtri Quote")
for col in ["odd_home", "odd_draw", "odd_away"]:
    add_range_filter(col)

# --- APPLICA FILTRI AL DATAFRAME PRINCIPALE ---
filtered_df = df.copy()
for col, val in filters.items():
    if col in ["odd_home", "odd_draw", "odd_away"]:
        if col in filtered_df.columns:
            mask = pd.to_numeric(filtered_df[col].astype(str).str.replace(",", "."), errors="coerce").between(val[0], val[1])
            filtered_df = filtered_df[mask.fillna(False)] # Usa False per escludere i NaN dai range
    elif col == "giornata":
        if col in filtered_df.columns:
            mask = pd.to_numeric(filtered_df[col], errors="coerce").between(val[0], val[1])
            filtered_df = filtered_df[mask.fillna(False)]
    elif col == "risultato_ht":
        if col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col].isin(val)]
    else:
        # Questa logica funziona bene sia per stringhe che per numeri (come 'anno' ora)
        if col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col] == val]

st.subheader("Dati Filtrati")
st.write(f"**Righe visualizzate:** {len(filtered_df)}")

# --- NUOVA SEZIONE: Riepilogo Risultati per Anno ---
st.markdown("---")
st.subheader("Riepilogo partite per Anno")
if not filtered_df.empty and "anno" in filtered_df.columns:
    # Assicurati che 'anno' sia numerico e rimuovi NaN per il conteggio
    partite_per_anno = filtered_df["anno"].dropna().astype(int).value_counts().sort_index()
    riepilogo_df = pd.DataFrame(partite_per_anno).reset_index()
    riepilogo_df.columns = ["Anno", "Partite Trovate"]
    st.table(riepilogo_df)
else:
    st.info("Nessuna partita trovata o la colonna 'anno' non è disponibile nel dataset filtrato.")
st.markdown("---")
# --- FINE NUOVA SEZIONE ---

st.dataframe(filtered_df.head(50))


# --- Funzione per calcolare le probabilità di Vittoria/Sconfitta dopo il primo gol ---
def calcola_first_to_score_outcome(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    # Assicurati che le colonne siano numeriche
    df_to_analyze["gol_home_ft"] = pd.to_numeric(df_to_analyze["gol_home_ft"], errors='coerce')
    df_to_analyze["gol_away_ft"] = pd.to_numeric(df_to_analyze["gol_away_ft"], errors='coerce')

    risultati = {
        "Casa Segna Primo e Vince": 0,
        "Casa Segna Primo e Non Vince": 0,
        "Trasferta Segna Prima e Vince": 0,
        "Trasferta Segna Prima e Non Vince": 0,
        "Nessun Gol": 0
    }
    
    total_matches_with_goals = 0 # Contiamo solo le partite con almeno un gol per le percentuali
    
    # Filtra solo le righe che hanno almeno un gol_home_ft o gol_away_ft valido
    df_filtered_for_goals = df_to_analyze.dropna(subset=["gol_home_ft", "gol_away_ft"])
    
    for _, row in df_filtered_for_goals.iterrows():
        gol_home_str = str(row.get("minutaggio_gol_home", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit()]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        home_vince = row["gol_home_ft"] > row["gol_away_ft"]
        away_vince = row["gol_away_ft"] > row["gol_home_ft"]
        
        if min_home_goal < min_away_goal:
            total_matches_with_goals += 1
            # Home segna per primo
            if home_vince:
                risultati["Casa Segna Primo e Vince"] += 1
            else:
                risultati["Casa Segna Primo e Non Vince"] += 1
        elif min_away_goal < min_home_goal:
            total_matches_with_goals += 1
            # Away segna per primo
            if away_vince:
                risultati["Trasferta Segna Prima e Vince"] += 1
            else:
                risultati["Trasferta Segna Prima e Non Vince"] += 1
        else:
            # Se entrambi sono inf, significa nessun gol, altrimenti c'è un pareggio nel primo gol
            if min_home_goal == float('inf'):
                risultati["Nessun Gol"] += 1
            # else: caso di gol contemporaneo, che è raro e non facilmente gestibile qui

    stats = []
    # Usiamo total_matches_with_goals per calcolare le percentuali degli esiti in cui c'è stato almeno un gol
    # Per "Nessun Gol", la percentuale è sul totale delle partite analizzate (len(df_to_analyze))
    for esito, count in risultati.items():
        if esito == "Nessun Gol":
            perc = round((count / len(df_to_analyze)) * 100, 2) if len(df_to_analyze) > 0 else 0
        else:
            perc = round((count / total_matches_with_goals) * 100, 2) if total_matches_with_goals > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- Nuova funzione per analizzare l'esito del secondo gol dopo il primo ---
def calcola_first_to_score_next_goal_outcome(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {
        "Casa Segna Primo e Segna di Nuovo": 0,
        "Casa Segna Primo e Subisce Gol": 0,
        "Trasferta Segna Prima e Segna di Nuovo": 0,
        "Trasferta Segna Prima e Subisce Gol": 0,
        "Solo un gol o nessuno": 0
    }
    
    total_matches = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol_home", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        gol_home = sorted([int(x) for x in gol_home_str.split(";") if x.isdigit()])
        gol_away = sorted([int(x) for x in gol_away_str.split(";") if x.isdigit()])

        all_goals = []
        if gol_home:
            all_goals.extend([ (t, 'home') for t in gol_home ])
        if gol_away:
            all_goals.extend([ (t, 'away') for t in gol_away ])
        
        if len(all_goals) < 2:
            risultati["Solo un gol o nessuno"] += 1
            continue
            
        # Ordina tutti i gol per minuto
        all_goals.sort()
        
        first_goal = all_goals[0]
        second_goal = all_goals[1]
        
        first_scorer = first_goal[1]
        second_scorer = second_goal[1]
        
        if first_scorer == 'home':
            if second_scorer == 'home':
                risultati["Casa Segna Primo e Segna di Nuovo"] += 1
            else:
                risultati["Casa Segna Primo e Subisce Gol"] += 1
        elif first_scorer == 'away':
            if second_scorer == 'away':
                risultati["Trasferta Segna Prima e Segna di Nuovo"] += 1
            else:
                risultati["Trasferta Segna Prima e Subisce Gol"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- Funzione per calcolare i mercati di Doppia Chance ---
def calcola_double_chance(df_to_analyze, period):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    
    df_double_chance = df_to_analyze.copy()
    
    if period == 'ft':
        # Controllo delle colonne prima di accedervi
        if "gol_home_ft" not in df_double_chance.columns or "gol_away_ft" not in df_double_chance.columns:
            st.warning(f"Colonne 'gol_home_ft' o 'gol_away_ft' mancanti per calcolo Double Chance FT.")
            return pd.DataFrame()
        df_double_chance["gol_home"] = pd.to_numeric(df_double_chance["gol_home_ft"], errors='coerce')
        df_double_chance["gol_away"] = pd.to_numeric(df_double_chance["gol_away_ft"], errors='coerce')
    elif period == 'ht':
        # Controllo delle colonne prima di accedervi
        if "gol_home_ht" not in df_double_chance.columns or "gol_away_ht" not in df_double_chance.columns:
            st.warning(f"Colonne 'gol_home_ht' o 'gol_away_ht' mancanti per calcolo Double Chance HT.")
            return pd.DataFrame()
        df_double_chance["gol_home"] = pd.to_numeric(df_double_chance["gol_home_ht"], errors='coerce')
        df_double_chance["gol_away"] = pd.to_numeric(df_double_chance["gol_away_ht"], errors='coerce')
    elif period == 'sh':
        # Controllo delle colonne prima di accedervi
        if "gol_home_ft" not in df_double_chance.columns or "gol_home_ht" not in df_double_chance.columns or \
           "gol_away_ft" not in df_double_chance.columns or "gol_away_ht" not in df_double_chance.columns:
            st.warning(f"Colonne gol_ft/ht mancanti per calcolo Double Chance SH.")
            return pd.DataFrame()
            
        df_double_chance["gol_home_sh"] = pd.to_numeric(df_double_chance["gol_home_ft"], errors='coerce') - pd.to_numeric(df_double_chance["gol_home_ht"], errors='coerce')
        df_double_chance["gol_away_sh"] = pd.to_numeric(df_double_chance["gol_away_ft"], errors='coerce') - pd.to_numeric(df_double_chance["gol_away_ht"], errors='coerce')
        df_double_chance["gol_home"] = df_double_chance["gol_home_sh"]
        df_double_chance["gol_away"] = df_double_chance["gol_away_sh"]
    else:
        st.error("Periodo non valido per il calcolo della doppia chance.")
        return pd.DataFrame()
        
    # Rimuovi i NaN dalle colonne dei gol temporanee per il calcolo
    df_double_chance = df_double_chance.dropna(subset=["gol_home", "gol_away"])
    
    total_matches = len(df_double_chance)
    if total_matches == 0:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    
    # 1X (Home Win or Draw)
    count_1x = ((df_double_chance["gol_home"] >= df_double_chance["gol_away"])).sum()
    
    # 12 (Home Win or Away Win)
    count_12 = ((df_double_chance["gol_home"] != df_double_chance["gol_away"])).sum()
    
    # X2 (Draw or Away Win)
    count_x2 = ((df_double_chance["gol_away"] >= df_double_chance["gol_home"])).sum()
    
    data = [
        ["1X", count_1x, round((count_1x / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["12", count_12, round((count_12 / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["X2", count_x2, round((count_x2 / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats


# --- Funzione per calcolare le stats SH ---
def calcola_stats_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Controllo delle colonne prima di procedere
    required_cols_sh = ["gol_home_ft", "gol_home_ht", "gol_away_ft", "gol_away_ht"]
    if not all(col in df_to_analyze.columns for col in required_cols_sh):
        st.warning("Colonne mancanti per calcolo statistiche SH.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_sh = df_to_analyze.copy()
    
    # Calcolo dei gol nel secondo tempo
    df_sh["gol_home_sh"] = pd.to_numeric(df_sh["gol_home_ft"], errors='coerce') - pd.to_numeric(df_sh["gol_home_ht"], errors='coerce')
    df_sh["gol_away_sh"] = pd.to_numeric(df_sh["gol_away_ft"], errors='coerce') - pd.to_numeric(df_sh["gol_away_ht"], errors='coerce')
    
    # Filtra le righe dove i gol SH sono NaN (risultato di errori nella conversione o dati mancanti)
    df_sh = df_sh.dropna(subset=["gol_home_sh", "gol_away_sh"])

    if df_sh.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Winrate SH
    risultati_sh = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
    for _, row in df_sh.iterrows():
        if row["gol_home_sh"] > row["gol_away_sh"]:
            risultati_sh["1 (Casa)"] += 1
        elif row["gol_home_sh"] < row["gol_away_sh"]:
            risultati_sh["2 (Trasferta)"] += 1
        else:
            risultati_sh["X (Pareggio)"] += 1
    
    total_sh_matches = len(df_sh)
    stats_sh_winrate = []
    for esito, count in risultati_sh.items():
        perc = round((count / total_sh_matches) * 100, 2) if total_sh_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats_sh_winrate.append((esito, count, perc, odd_min))
    df_winrate_sh = pd.DataFrame(stats_sh_winrate, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
    
    # Over Goals SH
    over_sh_data = []
    df_sh["tot_goals_sh"] = df_sh["gol_home_sh"] + df_sh["gol_away_sh"]
    for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        count = (df_sh["tot_goals_sh"] > t).sum()
        perc = round((count / len(df_sh)) * 100, 2)
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        over_sh_data.append([f"Over {t} SH", count, perc, odd_min])
    df_over_sh = pd.DataFrame(over_sh_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    # BTTS SH
    btts_sh_count = ((df_sh["gol_home_sh"] > 0) & (df_sh["gol_away_sh"] > 0)).sum()
    no_btts_sh_count = len(df_sh) - btts_sh_count
    btts_sh_data = [
        ["BTTS SI SH", btts_sh_count, round((btts_sh_count / total_sh_matches) * 100, 2) if total_sh_matches > 0 else 0],
        ["BTTS NO SH", no_btts_sh_count, round((no_btts_sh_count / total_sh_matches) * 100, 2) if total_sh_matches > 0 else 0]
    ]
    df_btts_sh = pd.DataFrame(btts_sh_data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_btts_sh["Odd Minima"] = df_btts_sh["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_winrate_sh, df_over_sh, df_btts_sh


# --- Nuova funzione per calcolare le stats SH complete ---
def calcola_first_to_score_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    if "minutaggio_gol_home" not in df_to_analyze.columns or "minutaggio_gol_away" not in df_to_analyze.columns:
        st.warning("Colonne 'minutaggio_gol_home' o 'minutaggio_gol_away' mancanti per First to Score SH.")
        return pd.DataFrame()

    risultati = {"Home Team": 0, "Away Team": 0, "No Goals SH": 0}
    totale_partite = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol_home", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        # Considera solo i gol segnati nel secondo tempo (minuto > 45)
        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit() and int(x) > 45]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit() and int(x) > 45]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        if min_home_goal < min_away_goal:
            risultati["Home Team"] += 1
        elif min_away_goal < min_home_goal:
            risultati["Away Team"] += 1
        else:  
            if min_home_goal == float('inf'): # Nessun gol nel SH
                risultati["No Goals SH"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

def calcola_first_to_score_outcome_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    required_cols = ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht", "minutaggio_gol_home", "minutaggio_gol_away"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per First to Score + Risultato Finale SH: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame()

    # Assicurati che le colonne siano numeriche
    df_to_analyze["gol_home_ft"] = pd.to_numeric(df_to_analyze["gol_home_ft"], errors='coerce')
    df_to_analyze["gol_away_ft"] = pd.to_numeric(df_to_analyze["gol_away_ft"], errors='coerce')
    df_to_analyze["gol_home_ht"] = pd.to_numeric(df_to_analyze["gol_home_ht"], errors='coerce')
    df_to_analyze["gol_away_ht"] = pd.to_numeric(df_to_analyze["gol_away_ht"], errors='coerce')
    
    risultati = {
        "Casa Segna Primo SH e Vince": 0,
        "Casa Segna Primo SH e Non Vince": 0,
        "Trasferta Segna Prima SH e Vince": 0,
        "Trasferta Segna Prima SH e Non Vince": 0,
        "Nessun Gol SH": 0
    }
    
    total_matches = len(df_to_analyze)
    total_matches_with_sh_goals = 0

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol_home", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        gol_home_sh = [int(x) for x in gol_home_str.split(";") if x.isdigit() and int(x) > 45]
        gol_away_sh = [int(x) for x in gol_away_str.split(";") if x.isdigit() and int(x) > 45]

        min_home_goal = min(gol_home_sh) if gol_home_sh else float('inf')
        min_away_goal = min(gol_away_sh) if gol_away_sh else float('inf')
        
        home_vince = row["gol_home_ft"] > row["gol_away_ft"]
        away_vince = row["gol_away_ft"] > row["gol_home_ft"]
        
        if min_home_goal < min_away_goal:
            total_matches_with_sh_goals += 1
            if home_vince:
                risultati["Casa Segna Primo SH e Vince"] += 1
            else:
                risultati["Casa Segna Primo SH e Non Vince"] += 1
        elif min_away_goal < min_home_goal:
            total_matches_with_sh_goals += 1
            if away_vince:
                risultati["Trasferta Segna Prima SH e Vince"] += 1
            else:
                risultati["Trasferta Segna Prima SH e Non Vince"] += 1
        else:
            if min_home_goal == float('inf'):
                risultati["Nessun Gol SH"] += 1

    stats = []
    for esito, count in risultati.items():
        if esito == "Nessun Gol SH":
            perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        else:
            perc = round((count / total_matches_with_sh_goals) * 100, 2) if total_matches_with_sh_goals > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

def calcola_first_to_score_next_goal_outcome_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    if "minutaggio_gol_home" not in df_to_analyze.columns or "minutaggio_gol_away" not in df_to_analyze.columns:
        st.warning("Colonne 'minutaggio_gol_home' o 'minutaggio_gol_away' mancanti per First to Score + Prossimo Gol SH.")
        return pd.DataFrame()

    risultati = {
        "Casa Segna Primo SH e Segna di Nuovo SH": 0,
        "Casa Segna Primo SH e Subisce Gol SH": 0,
        "Trasferta Segna Prima SH e Segna di Nuovo SH": 0,
        "Trasferta Segna Prima SH e Subisce Gol SH": 0,
        "Solo un gol SH o nessuno": 0
    }
    
    total_matches = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol_home", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        gol_home_sh = sorted([int(x) for x in gol_home_str.split(";") if x.isdigit() and int(x) > 45])
        gol_away_sh = sorted([int(x) for x in gol_away_str.split(";") if x.isdigit() and int(x) > 45])

        all_goals = []
        if gol_home_sh:
            all_goals.extend([ (t, 'home') for t in gol_home_sh ])
        if gol_away_sh:
            all_goals.extend([ (t, 'away') for t in gol_away_sh ])
        
        if len(all_goals) < 2:
            risultati["Solo un gol SH o nessuno"] += 1
            continue
            
        all_goals.sort()
        
        first_goal = all_goals[0]
        second_goal = all_goals[1]
        
        first_scorer = first_goal[1]
        second_scorer = second_goal[1]
        
        if first_scorer == 'home':
            if second_scorer == 'home':
                risultati["Casa Segna Primo SH e Segna di Nuovo SH"] += 1
            else:
                risultati["Casa Segna Primo SH e Subisce Gol SH"] += 1
        elif first_scorer == 'away':
            if second_scorer == 'away':
                risultati["Trasferta Segna Prima SH e Segna di Nuovo SH"] += 1
            else:
                risultati["Trasferta Segna Prima SH e Subisce Gol SH"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

def calcola_to_score_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()

    required_cols = ["gol_home_ft", "gol_home_ht", "gol_away_ft", "gol_away_ht"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per To Score SH: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame()

    df_to_score = df_to_analyze.copy()

    df_to_score["gol_home_sh"] = pd.to_numeric(df_to_score["gol_home_ft"], errors='coerce') - pd.to_numeric(df_to_score["gol_home_ht"], errors='coerce')
    df_to_score["gol_away_sh"] = pd.to_numeric(df_to_score["gol_away_ft"], errors='coerce') - pd.to_numeric(df_to_score["gol_away_ht"], errors='coerce')
    
    df_to_score = df_to_score.dropna(subset=["gol_home_sh", "gol_away_sh"])

    home_to_score_count = (df_to_score["gol_home_sh"] > 0).sum()
    away_to_score_count = (df_to_score["gol_away_sh"] > 0).sum()
    
    total_matches = len(df_to_score)
    if total_matches == 0:
        return pd.DataFrame() # Restituisce DataFrame vuoto se non ci sono match validi
    
    data = [
        ["Home Team to Score SH", home_to_score_count, round((home_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Away Team to Score SH", away_to_score_count, round((away_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

def calcola_clean_sheet_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()
    
    required_cols = ["gol_home_ft", "gol_home_ht", "gol_away_ft", "gol_away_ht"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per Clean Sheet SH: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame()

    df_clean_sheet = df_to_analyze.copy()
    
    df_clean_sheet["gol_home_sh"] = pd.to_numeric(df_clean_sheet["gol_home_ft"], errors='coerce') - pd.to_numeric(df_clean_sheet["gol_home_ht"], errors='coerce')
    df_clean_sheet["gol_away_sh"] = pd.to_numeric(df_clean_sheet["gol_away_ft"], errors='coerce') - pd.to_numeric(df_clean_sheet["gol_away_ht"], errors='coerce')
    
    df_clean_sheet = df_clean_sheet.dropna(subset=["gol_home_sh", "gol_away_sh"])

    home_clean_sheet_count = (df_clean_sheet["gol_away_sh"] == 0).sum()
    away_clean_sheet_count = (df_clean_sheet["gol_home_sh"] == 0).sum()
    
    total_matches = len(df_clean_sheet)
    if total_matches == 0:
        return pd.DataFrame()
    
    data = [
        ["Clean Sheet SH (Casa)", home_clean_sheet_count, round((home_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Clean Sheet SH (Trasferta)", away_clean_sheet_count, round((away_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- Funzione per calcolare le percentuali di gol fatti/subiti per squadra/periodo ---
def calcola_goals_per_team_period(df_to_analyze, team_type, action_type, period):
    if df_to_analyze.empty:
        return pd.DataFrame()
    
    df_temp = df_to_analyze.copy()
    
    scored_col = None
    conceded_col = None

    if period == 'ft':
        if "gol_home_ft" not in df_temp.columns or "gol_away_ft" not in df_temp.columns:
            st.warning(f"Colonne gol_ft mancanti per Goals Fatti/Subiti FT.")
            return pd.DataFrame()
        scored_col = "gol_home_ft" if team_type == 'home' else "gol_away_ft"
        conceded_col = "gol_away_ft" if team_type == 'home' else "gol_home_ft"
    elif period == 'ht':
        if "gol_home_ht" not in df_temp.columns or "gol_away_ht" not in df_temp.columns:
            st.warning(f"Colonne gol_ht mancanti per Goals Fatti/Subiti HT.")
            return pd.DataFrame()
        scored_col = "gol_home_ht" if team_type == 'home' else "gol_away_ht"
        conceded_col = "gol_away_ht" if team_type == 'home' else "gol_home_ht"
    elif period == 'sh':
        required_cols_sh_calc = ["gol_home_ft", "gol_home_ht", "gol_away_ft", "gol_away_ht"]
        if not all(col in df_temp.columns for col in required_cols_sh_calc):
            st.warning(f"Colonne gol_ft/ht mancanti per Goals Fatti/Subiti SH.")
            return pd.DataFrame()
        df_temp["gol_home_sh"] = pd.to_numeric(df_temp["gol_home_ft"], errors='coerce') - pd.to_numeric(df_temp["gol_home_ht"], errors='coerce')
        df_temp["gol_away_sh"] = pd.to_numeric(df_temp["gol_away_ft"], errors='coerce') - pd.to_numeric(df_temp["gol_away_ht"], errors='coerce')
        scored_col = "gol_home_sh" if team_type == 'home' else "gol_away_sh"
        conceded_col = "gol_away_sh" if team_type == 'home' else "gol_home_sh"
    else:
        return pd.DataFrame()
        
    col_to_analyze = scored_col if action_type == 'fatti' else conceded_col
    
    # Assicurati che la colonna da analizzare esista e sia numerica
    if col_to_analyze not in df_temp.columns:
        st.warning(f"La colonna '{col_to_analyze}' non è disponibile per l'analisi.")
        return pd.DataFrame()

    df_temp[col_to_analyze] = pd.to_numeric(df_temp[col_to_analyze], errors='coerce')
    df_temp = df_temp.dropna(subset=[col_to_analyze]) # Rimuovi NaN prima del calcolo

    total_matches = len(df_temp)
    if total_matches == 0:
        return pd.DataFrame()
    
    ranges = [0.5, 1.5]
    data = []
    
    for r in ranges:
        count = (df_temp[col_to_analyze] > r).sum()
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        data.append([f"Over {r}", count, perc, odd_min])
        
    df_results = pd.DataFrame(data, columns=[f"Mercato (Over {period})", "Conteggio", "Percentuale %", "Odd Minima"])
    return df_results


# --- FUNZIONE WINRATE ---
def calcola_winrate(df, col_risultato):
    if col_risultato not in df.columns:
        st.warning(f"Colonna '{col_risultato}' mancante per il calcolo del WinRate.")
        return pd.DataFrame()

    df_valid = df[df[col_risultato].notna() & (df[col_risultato].astype(str).str.contains("-"))].copy() # Converti a str prima di contains
    if df_valid.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])

    risultati = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
    for ris in df_valid[col_risultato]:
        try:
            home, away = map(int, ris.split("-"))
            if home > away:
                risultati["1 (Casa)"] += 1
            elif home < away:
                risultati["2 (Trasferta)"] += 1
            else:
                risultati["X (Pareggio)"] += 1
        except ValueError: # Per risultati non validi dopo il parsing a int
            continue
    
    totale = len(df_valid)
    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale) * 100, 2) if totale > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])

# --- FUNZIONE CALCOLO FIRST TO SCORE ---
def calcola_first_to_score(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    if "minutaggio_gol_home" not in df_to_analyze.columns or "minutaggio_gol_away" not in df_to_analyze.columns:
        st.warning("Colonne 'minutaggio_gol_home' o 'minutaggio_gol_away' mancanti per First to Score.")
        return pd.DataFrame()

    risultati = {"Home Team": 0, "Away Team": 0, "No Goals": 0}
    totale_partite = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol_home", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit()]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        if min_home_goal < min_away_goal:
            risultati["Home Team"] += 1
        elif min_away_goal < min_home_goal:
            risultati["Away Team"] += 1
        else:  
            if min_home_goal == float('inf'):
                risultati["No Goals"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- NUOVA FUNZIONE CALCOLO FIRST TO SCORE HT ---
def calcola_first_to_score_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    if "minutaggio_gol_home" not in df_to_analyze.columns or "minutaggio_gol_away" not in df_to_analyze.columns:
        st.warning("Colonne 'minutaggio_gol_home' o 'minutaggio_gol_away' mancanti per First to Score HT.")
        return pd.DataFrame()

    risultati = {"Home Team": 0, "Away Team": 0, "No Goals": 0}
    totale_partite = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol_home", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        # Considera solo i gol segnati nel primo tempo (minuto <= 45)
        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit() and int(x) <= 45]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit() and int(x) <= 45]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        if min_home_goal < min_away_goal:
            risultati["Home Team"] += 1
        elif min_away_goal < min_home_goal:
            risultati["Away Team"] += 1
        else:  
            if min_home_goal == float('inf'): # Nessun gol nel HT
                risultati["No Goals"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- FUNZIONE RISULTATI ESATTI ---
def mostra_risultati_esatti(df, col_risultato, titolo):
    if col_risultato not in df.columns:
        st.warning(f"Colonna '{col_risultato}' mancante per mostrare i Risultati Esatti.")
        return
        
    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]
    df_valid = df[df[col_risultato].notna() & (df[col_risultato].astype(str).str.contains("-"))].copy()

    if df_valid.empty:
        st.subheader(f"Risultati Esatti {titolo} (0 partite)")
        st.info("Nessun dato valido per i risultati esatti in questo filtro.")
        return

    def classifica_risultato(ris):
        try:
            home, away = map(int, ris.split("-"))
        except ValueError:
            return "Altro" # In caso di stringhe non convertibili in int
        if ris in risultati_interessanti:
            return ris
        if home > away:
            return "Altro risultato casa vince"
        elif home < away:
            return "Altro risultato ospite vince"
        else:
            return "Altro pareggio"

    df_valid["classificato"] = df_valid[col_risultato].apply(classifica_risultato)
    distribuzione = df_valid["classificato"].value_counts().reset_index()
    distribuzione.columns = [titolo, "Conteggio"]
    distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df_valid) * 100).round(2)
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

    st.subheader(f"Risultati Esatti {titolo} ({len(df_valid)} partite)")
    styled_df = distribuzione.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (15 MIN) ---
def mostra_distribuzione_timeband(df_to_analyze):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 15 minuti è vuoto.")
        return
    
    if "minutaggio_gol_home" not in df_to_analyze.columns or "minutaggio_gol_away" not in df_to_analyze.columns:
        st.warning("Colonne 'minutaggio_gol_home' o 'minutaggio_gol_away' mancanti per la distribuzione Timeband.")
        return

    intervalli = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 150)]
    label_intervalli = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
    risultati = []
    totale_partite = len(df_to_analyze)
    for (start, end), label in zip(intervalli, label_intervalli):
        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol_home", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([label, partite_con_gol, perc, odd_min])
    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"])
    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- NUOVA FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (5 MIN) ---
def mostra_distribuzione_timeband_5min(df_to_analyze):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 5 minuti è vuoto.")
        return
    
    if "minutaggio_gol_home" not in df_to_analyze.columns or "minutaggio_gol_away" not in df_to_analyze.columns:
        st.warning("Colonne 'minutaggio_gol_home' o 'minutaggio_gol_away' mancanti per la distribuzione Timeband 5min.")
        return

    intervalli = [(0,5), (6,10), (11,15), (16,20), (21,25), (26,30), (31,35), (36,40), (41,45), (46,50), (51,55), (56,60), (61,65), (66,70), (71,75), (76,80), (81,85), (86,90), (91, 150)]
    label_intervalli = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", "71-75", "76-80", "81-85", "86-90", "90+"]
    risultati = []
    totale_partite = len(df_to_analyze)
    for (start, end), label in zip(intervalli, label_intervalli):
        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol_home", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([label, partite_con_gol, perc, odd_min])
    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"])
    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- FUNZIONE NEXT GOAL ---
def calcola_next_goal(df_to_analyze, start_min, end_min):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    if "minutaggio_gol_home" not in df_to_analyze.columns or "minutaggio_gol_away" not in df_to_analyze.columns:
        st.warning("Colonne 'minutaggio_gol_home' o 'minutaggio_gol_away' mancanti per Next Goal.")
        return pd.DataFrame()

    risultati = {"Prossimo Gol: Home": 0, "Prossimo Gol: Away": 0, "Nessun prossimo gol": 0}
    totale_partite = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home = [int(x) for x in str(row.get("minutaggio_gol_home", "")).split(";") if x.isdigit()]
        gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]

        next_home_goal = min([g for g in gol_home if start_min <= g <= end_min] or [float('inf')])
        next_away_goal = min([g for g in gol_away if start_min <= g <= end_min] or [float('inf')])
        
        if next_home_goal < next_away_goal:
            risultati["Prossimo Gol: Home"] += 1
        elif next_away_goal < next_home_goal:
            risultati["Prossimo Gol: Away"] += 1
        else:
            if next_home_goal == float('inf'): # Nessun gol nel range specificato
                risultati["Nessun prossimo gol"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- NUOVE FUNZIONI PER ANALISI RIMONTE ---
def calcola_rimonte(df_to_analyze, titolo_analisi):
    if df_to_analyze.empty:
        return pd.DataFrame(), {} # Restituisce un dict vuoto invece di una lista vuota

    required_cols = ["gol_home_ht", "gol_away_ht", "gol_home_ft", "gol_away_ft", "home_team", "away_team"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per Analisi Rimonte: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame(), {}

    partite_rimonta_parziale = []
    partite_rimonta_completa = []
    
    df_rimonte = df_to_analyze.copy()
    
    # Aggiungi colonne per i gol HT e FT e converti in numerico
    df_rimonte["gol_home_ht"] = pd.to_numeric(df_rimonte["gol_home_ht"], errors='coerce')
    df_rimonte["gol_away_ht"] = pd.to_numeric(df_rimonte["gol_away_ht"], errors='coerce')
    df_rimonte["gol_home_ft"] = pd.to_numeric(df_rimonte["gol_home_ft"], errors='coerce')
    df_rimonte["gol_away_ft"] = pd.to_numeric(df_rimonte["gol_away_ft"], errors='coerce')

    # Rimuovi le righe con NaN nelle colonne dei gol dopo la conversione
    df_rimonte = df_rimonte.dropna(subset=["gol_home_ht", "gol_away_ht", "gol_home_ft", "gol_away_ft"])

    if df_rimonte.empty:
        return pd.DataFrame(), {}

    def check_comeback(row):
        # Rimonta Home
        if row["gol_home_ht"] < row["gol_away_ht"] and row["gol_home_ft"] > row["gol_away_ft"]:
            return "Completa - Home"
        if row["gol_home_ht"] < row["gol_away_ht"] and row["gol_home_ft"] == row["gol_away_ft"]:
            return "Parziale - Home"
        # Rimonta Away
        if row["gol_away_ht"] < row["gol_home_ht"] and row["gol_away_ft"] > row["gol_home_ft"]:
            return "Completa - Away"
        if row["gol_away_ht"] < row["gol_home_ht"] and row["gol_away_ft"] == row["gol_home_ft"]:
            return "Parziale - Away"
        return "Nessuna"

    df_rimonte["rimonta"] = df_rimonte.apply(check_comeback, axis=1)
    
    # Filtra e conta i risultati
    rimonte_completa_home = (df_rimonte["rimonta"] == "Completa - Home").sum()
    rimonte_parziale_home = (df_rimonte["rimonta"] == "Parziale - Home").sum()
    rimonte_completa_away = (df_rimonte["rimonta"] == "Completa - Away").sum()
    rimonte_parziale_away = (df_rimonte["rimonta"] == "Parziale - Away").sum()

    totale = len(df_rimonte)
    
    rimonte_data = [
        ["Rimonta Completa (Home)", rimonte_completa_home, round((rimonte_completa_home / totale) * 100, 2) if totale > 0 else 0],
        ["Rimonta Parziale (Home)", rimonte_parziale_home, round((rimonte_parziale_home / totale) * 100, 2) if totale > 0 else 0],
        ["Rimonta Completa (Away)", rimonte_completa_away, round((rimonte_completa_away / totale) * 100, 2) if totale > 0 else 0],
        ["Rimonta Parziale (Away)", rimonte_parziale_away, round((rimonte_parziale_away / totale) * 100, 2) if totale > 0 else 0]
    ]

    df_rimonte_stats = pd.DataFrame(rimonte_data, columns=["Tipo Rimonta", "Conteggio", "Percentuale %"])
    df_rimonte_stats["Odd Minima"] = df_rimonte_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    # Crea la lista di squadre per ogni tipo di rimonta
    squadre_rimonta_completa_home = df_rimonte[df_rimonte["rimonta"] == "Completa - Home"]["home_team"].tolist()
    squadre_rimonta_parziale_home = df_rimonte[df_rimonte["rimonta"] == "Parziale - Home"]["home_team"].tolist()
    squadre_rimonta_completa_away = df_rimonte[df_rimonte["rimonta"] == "Completa - Away"]["away_team"].tolist()
    squadre_rimonta_parziale_away = df_rimonte[df_rimonte["rimonta"] == "Parziale - Away"]["away_team"].tolist()
    
    squadre_rimonte = {
        "Rimonta Completa (Home)": squadre_rimonta_completa_home,
        "Rimonta Parziale (Home)": squadre_rimonta_parziale_home,
        "Rimonta Completa (Away)": squadre_rimonta_completa_away,
        "Rimonta Parziale (Away)": squadre_rimonta_parziale_away
    }

    return df_rimonte_stats, squadre_rimonte

# --- NUOVA FUNZIONE PER TO SCORE ---
def calcola_to_score(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()

    required_cols = ["gol_home_ft", "gol_away_ft"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per To Score FT: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame()

    df_to_score = df_to_analyze.copy()

    df_to_score["gol_home_ft"] = pd.to_numeric(df_to_score["gol_home_ft"], errors='coerce')
    df_to_score["gol_away_ft"] = pd.to_numeric(df_to_score["gol_away_ft"], errors='coerce')
    
    df_to_score = df_to_score.dropna(subset=["gol_home_ft", "gol_away_ft"])

    home_to_score_count = (df_to_score["gol_home_ft"] > 0).sum()
    away_to_score_count = (df_to_score["gol_away_ft"] > 0).sum()
    
    total_matches = len(df_to_score)
    if total_matches == 0:
        return pd.DataFrame()
    
    data = [
        ["Home Team to Score", home_to_score_count, round((home_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Away Team to Score", away_to_score_count, round((away_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER TO SCORE HT ---
def calcola_to_score_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()

    required_cols = ["gol_home_ht", "gol_away_ht"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per To Score HT: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame()

    df_to_score = df_to_analyze.copy()

    df_to_score["gol_home_ht"] = pd.to_numeric(df_to_score["gol_home_ht"], errors='coerce')
    df_to_score["gol_away_ht"] = pd.to_numeric(df_to_score["gol_away_ht"], errors='coerce')
    
    df_to_score = df_to_score.dropna(subset=["gol_home_ht", "gol_away_ht"])

    home_to_score_count = (df_to_score["gol_home_ht"] > 0).sum()
    away_to_score_count = (df_to_score["gol_away_ht"] > 0).sum()
    
    total_matches = len(df_to_score)
    if total_matches == 0:
        return pd.DataFrame()
    
    data = [
        ["Home Team to Score", home_to_score_count, round((home_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Away Team to Score", away_to_score_count, round((away_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER BTTS HT ---
def calcola_btts_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()

    required_cols = ["gol_home_ht", "gol_away_ht"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per BTTS HT: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame()

    df_btts_ht = df_to_analyze.copy()
    df_btts_ht["gol_home_ht"] = pd.to_numeric(df_btts_ht["gol_home_ht"], errors='coerce')
    df_btts_ht["gol_away_ht"] = pd.to_numeric(df_btts_ht["gol_away_ht"], errors='coerce')
    
    df_btts_ht = df_btts_ht.dropna(subset=["gol_home_ht", "gol_away_ht"])

    btts_count = ((df_btts_ht["gol_home_ht"] > 0) & (df_btts_ht["gol_away_ht"] > 0)).sum()
    no_btts_count = len(df_btts_ht) - btts_count
    
    total_matches = len(df_btts_ht)
    if total_matches == 0:
        return pd.DataFrame()
    
    data = [
        ["BTTS SI HT", btts_count, round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO HT", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER BTTS FT ---
def calcola_btts_ft(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()

    required_cols = ["gol_home_ft", "gol_away_ft"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per BTTS FT: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame()

    df_btts_ft = df_to_analyze.copy()
    df_btts_ft["gol_home_ft"] = pd.to_numeric(df_btts_ft["gol_home_ft"], errors='coerce')
    df_btts_ft["gol_away_ft"] = pd.to_numeric(df_btts_ft["gol_away_ft"], errors='coerce')
    
    df_btts_ft = df_btts_ft.dropna(subset=["gol_home_ft", "gol_away_ft"])

    btts_count = ((df_btts_ft["gol_home_ft"] > 0) & (df_btts_ft["gol_away_ft"] > 0)).sum()
    no_btts_count = len(df_btts_ft) - btts_count
    
    total_matches = len(df_btts_ft)
    if total_matches == 0:
        return pd.DataFrame()
    
    data = [
        ["BTTS SI FT", btts_count, round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO FT", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER BTTS DINAMICO ---
def calcola_btts_dinamico(df_to_analyze, start_min, risultati_correnti):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    required_cols = ["minutaggio_gol_home", "minutaggio_gol_away", "gol_home_ft", "gol_away_ft"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per BTTS Dinamico: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame()

    total_matches = len(df_to_analyze)
    btts_si_count = 0
    
    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol_home", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))
        
        gol_home_before = sum(1 for g in [int(x) for x in gol_home_str.split(";") if x.isdigit()] if g < start_min)
        gol_away_before = sum(1 for g in [int(x) for x in gol_away_str.split(";") if x.isdigit()] if g < start_min)
        
        gol_home_ft = int(row.get("gol_home_ft", 0))
        gol_away_ft = int(row.get("gol_away_ft", 0))
        
        # Logica per BTTS SI dinamico
        btts_si = False
        # Assumiamo che i risultati_correnti siano stati filtrati in base al minuto di inizio
        # Questo è un esempio semplificato, la logica "dinamica" del BTTS dovrebbe considerare se ENTRAMBE le squadre hanno segnato
        # *dopo* il minuto iniziale e dato il risultato iniziale.
        # Ad esempio, se il risultato corrente è "0-0" e a fine partita è "1-1", allora è BTTS SI.
        # Se il risultato corrente è "1-0" e a fine partita è "1-1", allora è BTTS SI (Away ha segnato).
        
        # Questa logica deve essere più robusta per riflettere il BTTS *dopo* un certo minuto, dati i gol preesistenti.
        # Per ora, la logica proposta dal tuo codice originale sembra voler verificare se il BTTS è avvenuto *a prescindere* dal risultato iniziale.
        # Ho interpretato la tua intenzione come: "Se data una situazione iniziale (risultati_correnti), è poi successo il BTTS FT?"
        
        # Semplificazione: se alla fine della partita entrambe hanno segnato, è BTTS SI.
        # Per un BTTS *dinamico* più preciso, dovremmo tenere conto del punteggio al `start_min`.
        
        # Per ora, la logica è basata sul risultato finale e sul fatto che entrambe le squadre abbiano segnato.
        # La parte `risultati_correnti` nel tuo codice originale per BTTS dinamico sembrava più complessa.
        # Cercherò di mantenerla più aderente alla semplice condizione BTTS FT per il DF filtrato.
        
        # Se vogliamo un BTTS che si sia verificato *dopo* il minuto start_min:
        home_scored_after_start = (gol_home_ft > gol_home_before)
        away_scored_after_start = (gol_away_ft > gol_away_before)

        # Se entrambe le squadre hanno segnato almeno un gol DOPO il minuto iniziale
        # O se una squadra aveva già segnato e l'altra segna dopo
        if (home_scored_after_start and away_scored_after_start) or \
           (gol_home_before > 0 and away_scored_after_start) or \
           (gol_away_before > 0 and home_scored_after_start):
           btts_si = True
        
        # Se entrambi avevano già segnato al minuto start_min, il BTTS era già SI.
        if gol_home_before > 0 and gol_away_before > 0:
            btts_si = True
        
        if btts_si:
            btts_si_count += 1

    btts_no_count = total_matches - btts_si_count

    data = [
        ["BTTS SI (Dinamica)", btts_si_count, round((btts_si_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO (Dinamica)", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

    return df_stats
    
# --- NUOVA FUNZIONE PER BTTS HT DINAMICO ---
def calcola_btts_ht_dinamico(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    required_cols = ["gol_home_ht", "gol_away_ht"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per BTTS HT Dinamico: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame()

    df_btts_ht_dinamico = df_to_analyze.copy()
    
    # Assicurati che le colonne siano numeriche
    df_btts_ht_dinamico["gol_home_ht"] = pd.to_numeric(df_btts_ht_dinamico["gol_home_ht"], errors='coerce')
    df_btts_ht_dinamico["gol_away_ht"] = pd.to_numeric(df_btts_ht_dinamico["gol_away_ht"], errors='coerce')
    
    df_btts_ht_dinamico = df_btts_ht_dinamico.dropna(subset=["gol_home_ht", "gol_away_ht"])

    btts_count = ((df_btts_ht_dinamico["gol_home_ht"] > 0) & (df_btts_ht_dinamico["gol_away_ht"] > 0)).sum()
    no_btts_count = len(df_btts_ht_dinamico) - btts_count
    
    total_matches = len(df_btts_ht_dinamico)
    if total_matches == 0:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    
    data = [
        ["BTTS SI HT (Dinamica)", btts_count, round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO HT (Dinamica)", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER CLEAN SHEET ---
def calcola_clean_sheet(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()
    
    required_cols = ["gol_home_ft", "gol_away_ft"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per Clean Sheet FT: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame()

    df_clean_sheet = df_to_analyze.copy()
    
    df_clean_sheet["gol_home_ft"] = pd.to_numeric(df_clean_sheet["gol_home_ft"], errors='coerce')
    df_clean_sheet["gol_away_ft"] = pd.to_numeric(df_clean_sheet["gol_away_ft"], errors='coerce')
    
    df_clean_sheet = df_clean_sheet.dropna(subset=["gol_home_ft", "gol_away_ft"])

    home_clean_sheet_count = (df_clean_sheet["gol_away_ft"] == 0).sum()
    away_clean_sheet_count = (df_clean_sheet["gol_home_ft"] == 0).sum()
    
    total_matches = len(df_clean_sheet)
    if total_matches == 0:
        return pd.DataFrame()
    
    data = [
        ["Clean Sheet (Casa)", home_clean_sheet_count, round((home_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Clean Sheet (Trasferta)", away_clean_sheet_count, round((away_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER COMBO MARKETS ---
def calcola_combo_stats(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame()
        
    required_cols = ["gol_home_ft", "gol_away_ft"]
    if not all(col in df_to_analyze.columns for col in required_cols):
        st.warning(f"Colonne mancanti per Combo Markets: {', '.join([col for col in required_cols if col not in df_to_analyze.columns])}")
        return pd.DataFrame()

    df_combo = df_to_analyze.copy()

    df_combo["gol_home_ft"] = pd.to_numeric(df_combo["gol_home_ft"], errors='coerce')
    df_combo["gol_away_ft"] = pd.to_numeric(df_combo["gol_away_ft"], errors='coerce')
    
    df_combo = df_combo.dropna(subset=["gol_home_ft", "gol_away_ft"]) # Rimuovi NaN prima del calcolo
    if df_combo.empty:
        return pd.DataFrame()
        
    df_combo["tot_goals_ft"] = df_combo["gol_home_ft"] + df_combo["gol_away_ft"]
    
    # BTTS SI + Over 2.5
    btts_over_2_5_count = ((df_combo["gol_home_ft"] > 0) & (df_combo["gol_away_ft"] > 0) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    # Home Win + Over 2.5
    home_win_over_2_5_count = ((df_combo["gol_home_ft"] > df_combo["gol_away_ft"]) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    # Away Win + Over 2.5
    away_win_over_2_5_count = ((df_combo["gol_away_ft"] > df_combo["gol_home_ft"]) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    total_matches = len(df_combo)
    if total_matches == 0:
        return pd.DataFrame()
    
    data = [
        ["BTTS SI + Over 2.5", btts_over_2_5_count, round((btts_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Casa vince + Over 2.5", home_win_over_2_5_count, round((home_win_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Ospite vince + Over 2.5", away_win_over_2_5_count, round((away_win_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER MULTI GOL ---
def calcola_multi_gol(df_to_analyze, col_gol, titolo):
    if df_to_analyze.empty:
        return pd.DataFrame()
    
    if col_gol not in df_to_analyze.columns:
        st.warning(f"Colonna '{col_gol}' mancante per Multi Gol.")
        return pd.DataFrame()

    df_multi_gol = df_to_analyze.copy()
    df_multi_gol[col_gol] = pd.to_numeric(df_multi_gol[col_gol], errors='coerce')
    
    df_multi_gol = df_multi_gol.dropna(subset=[col_gol]) # Rimuovi NaN
    if df_multi_gol.empty:
        return pd.DataFrame()
    
    total_matches = len(df_multi_gol)
    
    multi_gol_ranges = [
        ("0-1", lambda x: (x >= 0) & (x <= 1)),
        ("1-2", lambda x: (x >= 1) & (x <= 2)),
        ("2-3", lambda x: (x >= 2) & (x <= 3)),
        ("3+", lambda x: (x >= 3))
    ]
    
    data = []
    for label, condition in multi_gol_ranges:
        count = df_multi_gol[condition(df_multi_gol[col_gol])].shape[0]
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        data.append([f"Multi Gol {label}", count, perc, odd_min])
        
    df_stats = pd.DataFrame(data, columns=[f"Mercato ({titolo})", "Conteggio", "Percentuale %", "Odd Minima"])
    return df_stats

# SEZIONE 1: Analisi Timeband per Campionato
st.subheader("1. Analisi Timeband per Campionato")
if selected_league != "Tutte":
    df_league_only = df[df["league"] == selected_league]
    st.write(f"Analisi basata su **{len(df_league_only)}** partite del campionato **{selected_league}**.")
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribuzione Gol per Timeframe (15min)**")
        mostra_distribuzione_timeband(df_league_only)
    with col2:
        st.write("**Distribuzione Gol per Timeframe (5min)**")
        mostra_distribuzione_timeband_5min(df_league_only)
else:
    st.info("Seleziona un campionato per visualizzare questa analisi.")

# SEZIONE 2: Analisi Timeband per Campionato e Quote
st.subheader("2. Analisi Timeband per Campionato e Quote")
st.write(f"Analisi basata su **{len(filtered_df)}** partite filtrate da tutti i parametri della sidebar.")
if not filtered_df.empty:
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribuzione Gol per Timeframe (15min)**")
        mostra_distribuzione_timeband(filtered_df)
    with col2:
        st.write("**Distribuzione Gol per Timeframe (5min)**")
        mostra_distribuzione_timeband_5min(filtered_df)
else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati per l'analisi Timeband.")


# NUOVA SEZIONE: Statistiche Pre-Match Complete (Filtri Sidebar)
st.subheader("3. Analisi Pre-Match Completa (Filtri Sidebar)")
st.write(f"Analisi completa basata su **{len(filtered_df)}** partite, considerando tutti i filtri del menu a sinistra.")
if not filtered_df.empty:
    
    # Calcolo e visualizzazione media gol
    st.subheader("Media Gol (Pre-Match)")
    df_prematch_goals = filtered_df.copy()
    
    required_cols_goals = ["gol_home_ht", "gol_away_ht", "gol_home_ft", "gol_away_ft"]
    if all(col in df_prematch_goals.columns for col in required_cols_goals):
        df_prematch_goals["gol_home_ht"] = pd.to_numeric(df_prematch_goals["gol_home_ht"], errors='coerce')
        df_prematch_goals["gol_away_ht"] = pd.to_numeric(df_prematch_goals["gol_away_ht"], errors='coerce')
        df_prematch_goals["gol_home_ft"] = pd.to_numeric(df_prematch_goals["gol_home_ft"], errors='coerce')
        df_prematch_goals["gol_away_ft"] = pd.to_numeric(df_prematch_goals["gol_away_ft"], errors='coerce')
        
        # Rimuovi le righe con NaN nelle colonne dei gol per un calcolo accurato delle medie
        df_prematch_goals_clean = df_prematch_goals.dropna(subset=required_cols_goals)
        
        if not df_prematch_goals_clean.empty:
            # Media gol HT
            avg_ht_goals = (df_prematch_goals_clean["gol_home_ht"] + df_prematch_goals_clean["gol_away_ht"]).mean()
            # Media gol FT
            avg_ft_goals = (df_prematch_goals_clean["gol_home_ft"] + df_prematch_goals_clean["gol_away_ft"]).mean()
            # Media gol SH (secondo tempo)
            avg_sh_goals = (df_prematch_goals_clean["gol_home_ft"] + df_prematch_goals_clean["gol_away_ft"] - df_prematch_goals_clean["gol_home_ht"] - df_prematch_goals_clean["gol_away_ht"]).mean()
            
            st.table(pd.DataFrame({
                "Periodo": ["HT", "FT", "SH"],
                "Media Gol": [f"{avg_ht_goals:.2f}", f"{avg_ft_goals:.2f}", f"{avg_sh_goals:.2f}"]
            }))
        else:
            st.info("Nessun dato valido per il calcolo della media gol dopo la pulizia.")
    else:
        st.warning(f"Colonne gol mancanti per calcolare la Media Gol (Pre-Match): {', '.join([col for col in required_cols_goals if col not in df_prematch_goals.columns])}")

    # --- Expander per Statistiche HT ---
    with st.expander("Mostra Statistiche HT"):
        mostra_risultati_esatti(filtered_df, "risultato_ht", f"HT ({len(filtered_df)})")
        st.subheader(f"WinRate HT ({len(filtered_df)})")
        df_winrate_ht = calcola_winrate(filtered_df, "risultato_ht")
        if not df_winrate_ht.empty:
            styled_df_ht = df_winrate_ht.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
            st.dataframe(styled_df_ht)
        else:
            st.info("Nessun dato per il WinRate HT.")

        st.subheader(f"Over Goals HT ({len(filtered_df)})")
        # Controllo delle colonne prima di procedere
        required_cols_over_ht = ["gol_home_ht", "gol_away_ht"]
        if all(col in filtered_df.columns for col in required_cols_over_ht):
            over_ht_data = []
            df_prematch_ht = filtered_df.copy()
            df_prematch_ht["tot_goals_ht"] = pd.to_numeric(df_prematch_ht["gol_home_ht"], errors='coerce') + pd.to_numeric(df_prematch_ht["gol_away_ht"], errors='coerce')
            df_prematch_ht = df_prematch_ht.dropna(subset=["tot_goals_ht"]) # Rimuovi NaN
            if not df_prematch_ht.empty:
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_prematch_ht["tot_goals_ht"] > t).sum()
                    perc = round((count / len(df_prematch_ht)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    over_ht_data.append([f"Over {t} HT", count, perc, odd_min])
                df_over_ht = pd.DataFrame(over_ht_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                styled_over_ht = df_over_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_over_ht)
            else:
                st.info("Nessun dato valido per Over Goals HT.")
        else:
            st.warning(f"Colonne gol_ht mancanti per Over Goals HT: {', '.join([col for col in required_cols_over_ht if col not in filtered_df.columns])}")

        st.subheader(f"BTTS HT ({len(filtered_df)})")
        df_btts_ht = calcola_btts_ht(filtered_df)
        if not df_btts_ht.empty:
            styled_df = df_btts_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per BTTS HT.")
        
        st.subheader(f"Doppia Chance HT ({len(filtered_df)})")
        df_dc_ht = calcola_double_chance(filtered_df, 'ht')
        if not df_dc_ht.empty:
            styled_df = df_dc_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per Doppia Chance HT.")

        st.subheader(f"First to Score HT ({len(filtered_df)})")
        df_fts_ht = calcola_first_to_score_ht(filtered_df)
        if not df_fts_ht.empty:
            styled_df = df_fts_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per First to Score HT.")

        st.subheader(f"To Score HT ({len(filtered_df)})")
        df_ts_ht = calcola_to_score_ht(filtered_df)
        if not df_ts_ht.empty:
            styled_df = df_ts_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per To Score HT.")

        st.subheader(f"Goals Fatti e Subiti HT ({len(filtered_df)})")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Fatti Casa")
            df_fatti_casa_ht = calcola_goals_per_team_period(filtered_df, 'home', 'fatti', 'ht')
            if not df_fatti_casa_ht.empty:
                st.dataframe(df_fatti_casa_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")
        with col2:
            st.markdown("#### Subiti Casa")
            df_subiti_casa_ht = calcola_goals_per_team_period(filtered_df, 'home', 'subiti', 'ht')
            if not df_subiti_casa_ht.empty:
                st.dataframe(df_subiti_casa_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Fatti Trasferta")
            df_fatti_away_ht = calcola_goals_per_team_period(filtered_df, 'away', 'fatti', 'ht')
            if not df_fatti_away_ht.empty:
                st.dataframe(df_fatti_away_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")
        with col4:
            st.markdown("#### Subiti Trasferta")
            df_subiti_away_ht = calcola_goals_per_team_period(filtered_df, 'away', 'subiti', 'ht')
            if not df_subiti_away_ht.empty:
                st.dataframe(df_subiti_away_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")
    
    # --- Nuove Expander per Statistiche SH ---
    with st.expander("Mostra Statistiche SH (Secondo Tempo)"):
        st.write(f"Analisi basata su **{len(filtered_df)}** partite.")
        df_winrate_sh_exp, df_over_sh_exp, df_btts_sh_exp = calcola_stats_sh(filtered_df)

        st.subheader(f"WinRate SH ({len(filtered_df)})")
        if not df_winrate_sh_exp.empty:
            styled_df = df_winrate_sh_exp.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per WinRate SH.")

        st.subheader(f"Over Goals SH ({len(filtered_df)})")
        if not df_over_sh_exp.empty:
            styled_df = df_over_sh_exp.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per Over Goals SH.")
            
        st.subheader(f"BTTS SH ({len(filtered_df)})")
        if not df_btts_sh_exp.empty:
            styled_df = df_btts_sh_exp.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per BTTS SH.")
            
        st.subheader(f"Doppia Chance SH ({len(filtered_df)})")
        df_dc_sh = calcola_double_chance(filtered_df, 'sh')
        if not df_dc_sh.empty:
            styled_df = df_dc_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per Doppia Chance SH.")
            
        st.subheader(f"First to Score SH ({len(filtered_df)})")
        df_fts_sh = calcola_first_to_score_sh(filtered_df)
        if not df_fts_sh.empty:
            styled_df = df_fts_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per First to Score SH.")

        st.subheader(f"First to Score + Risultato Finale SH ({len(filtered_df)})")
        df_fts_outcome_sh = calcola_first_to_score_outcome_sh(filtered_df)
        if not df_fts_outcome_sh.empty:
            styled_df = df_fts_outcome_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per First to Score + Risultato Finale SH.")

        st.subheader(f"First to Score + Risultato Prossimo Gol SH ({len(filtered_df)})")
        df_fts_next_sh = calcola_first_to_score_next_goal_outcome_sh(filtered_df)
        if not df_fts_next_sh.empty:
            styled_df = df_fts_next_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per First to Score + Risultato Prossimo Gol SH.")

        st.subheader(f"To Score SH ({len(filtered_df)})")
        df_ts_sh = calcola_to_score_sh(filtered_df)
        if not df_ts_sh.empty:
            styled_df = df_ts_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per To Score SH.")

        st.subheader(f"Clean Sheet SH ({len(filtered_df)})")
        df_cs_sh = calcola_clean_sheet_sh(filtered_df)
        if not df_cs_sh.empty:
            styled_df = df_cs_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per Clean Sheet SH.")
            
        st.subheader(f"Goals Fatti e Subiti SH ({len(filtered_df)})")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Fatti Casa")
            df_fatti_casa_sh = calcola_goals_per_team_period(filtered_df, 'home', 'fatti', 'sh')
            if not df_fatti_casa_sh.empty:
                st.dataframe(df_fatti_casa_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")
        with col2:
            st.markdown("#### Subiti Casa")
            df_subiti_casa_sh = calcola_goals_per_team_period(filtered_df, 'home', 'subiti', 'sh')
            if not df_subiti_casa_sh.empty:
                st.dataframe(df_subiti_casa_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Fatti Trasferta")
            df_fatti_away_sh = calcola_goals_per_team_period(filtered_df, 'away', 'fatti', 'sh')
            if not df_fatti_away_sh.empty:
                st.dataframe(df_fatti_away_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")
        with col4:
            st.markdown("#### Subiti Trasferta")
            df_subiti_away_sh = calcola_goals_per_team_period(filtered_df, 'away', 'subiti', 'sh')
            if not df_subiti_away_sh.empty:
                st.dataframe(df_subiti_away_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")

    # --- Expander per Statistiche FT ---
    with st.expander("Mostra Statistiche FT (Finale)"):
        mostra_risultati_esatti(filtered_df, "risultato_ft", f"FT ({len(filtered_df)})")
        st.subheader(f"WinRate FT ({len(filtered_df)})")
        df_winrate_ft = calcola_winrate(filtered_df, "risultato_ft")
        if not df_winrate_ft.empty:
            styled_df_ft = df_winrate_ft.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
            st.dataframe(styled_df_ft)
        else:
            st.info("Nessun dato per il WinRate FT.")
        
        st.subheader(f"Over Goals FT ({len(filtered_df)})")
        required_cols_over_ft = ["gol_home_ft", "gol_away_ft"]
        if all(col in filtered_df.columns for col in required_cols_over_ft):
            over_ft_data = []
            df_prematch_ft = filtered_df.copy()
            df_prematch_ft["tot_goals_ft"] = pd.to_numeric(df_prematch_ft["gol_home_ft"], errors='coerce') + pd.to_numeric(df_prematch_ft["gol_away_ft"], errors='coerce')
            df_prematch_ft = df_prematch_ft.dropna(subset=["tot_goals_ft"]) # Rimuovi NaN
            if not df_prematch_ft.empty:
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_prematch_ft["tot_goals_ft"] > t).sum()
                    perc = round((count / len(df_prematch_ft)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
                df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                styled_over_ft = df_over_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_over_ft)
            else:
                st.info("Nessun dato valido per Over Goals FT.")
        else:
            st.warning(f"Colonne gol_ft mancanti per Over Goals FT: {', '.join([col for col in required_cols_over_ft if col not in filtered_df.columns])}")

        st.subheader(f"BTTS FT ({len(filtered_df)})")
        df_btts_ft = calcola_btts_ft(filtered_df)
        if not df_btts_ft.empty:
            styled_df = df_btts_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per BTTS FT.")

        st.subheader(f"Doppia Chance FT ({len(filtered_df)})")
        df_dc_ft = calcola_double_chance(filtered_df, 'ft')
        if not df_dc_ft.empty:
            styled_df = df_dc_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per Doppia Chance FT.")

        st.subheader(f"Multi Gol (Pre-Match) ({len(filtered_df)})")
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Casa")
            df_multi_gol_home = calcola_multi_gol(filtered_df, "gol_home_ft", "Home")
            if not df_multi_gol_home.empty:
                styled_df = df_multi_gol_home.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            else: st.info("Nessun dato.")
        with col2:
            st.write("### Trasferta")
            df_multi_gol_away = calcola_multi_gol(filtered_df, "gol_away_ft", "Away")
            if not df_multi_gol_away.empty:
                styled_df = df_multi_gol_away.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            else: st.info("Nessun dato.")

        st.subheader(f"First to Score (Pre-Match) ({len(filtered_df)})")
        df_fts_prematch = calcola_first_to_score(filtered_df)
        if not df_fts_prematch.empty:
            styled_df = df_fts_prematch.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per First to Score (Pre-Match).")

        st.subheader(f"First to Score + Risultato Finale (Pre-Match) ({len(filtered_df)})")
        df_fts_outcome_prematch = calcola_first_to_score_outcome(filtered_df)
        if not df_fts_outcome_prematch.empty:
            styled_df = df_fts_outcome_prematch.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per First to Score + Risultato Finale (Pre-Match).")

        st.subheader(f"First to Score + Risultato Prossimo Gol (Pre-Match) ({len(filtered_df)})")
        df_fts_next_prematch = calcola_first_to_score_next_goal_outcome(filtered_df)
        if not df_fts_next_prematch.empty:
            styled_df = df_fts_next_prematch.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per First to Score + Risultato Prossimo Gol (Pre-Match).")

        st.subheader(f"To Score (Pre-Match) ({len(filtered_df)})")
        df_ts_prematch = calcola_to_score(filtered_df)
        if not df_ts_prematch.empty:
            styled_df = df_ts_prematch.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per To Score (Pre-Match).")

        st.subheader(f"Clean Sheet (Pre-Match) ({len(filtered_df)})")
        df_cs_prematch = calcola_clean_sheet(filtered_df)
        if not df_cs_prematch.empty:
            styled_df = df_cs_prematch.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per Clean Sheet (Pre-Match).")

        st.subheader(f"Combo Markets (Pre-Match) ({len(filtered_df)})")
        df_combo_prematch = calcola_combo_stats(filtered_df)
        if not df_combo_prematch.empty:
            styled_df = df_combo_prematch.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        else:
            st.info("Nessun dato per Combo Markets (Pre-Match).")

        st.subheader(f"Analisi Rimonte (Pre-Match) ({len(filtered_df)})")
        rimonte_stats, squadre_rimonte = calcola_rimonte(filtered_df, "Pre-Match")
        if not rimonte_stats.empty:
            styled_df = rimonte_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            st.markdown("**Squadre che hanno effettuato rimonte:**")
            for tipo, squadre in squadre_rimonte.items():
                if squadre:
                    st.markdown(f"**{tipo}:** {', '.join(squadre)}")
        else:
            st.info("Nessuna rimonta trovata nel dataset filtrato.")
            
        st.subheader(f"Goals Fatti e Subiti FT ({len(filtered_df)})")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Fatti Casa")
            df_fatti_casa_ft = calcola_goals_per_team_period(filtered_df, 'home', 'fatti', 'ft')
            if not df_fatti_casa_ft.empty:
                st.dataframe(df_fatti_casa_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")
        with col2:
            st.markdown("#### Subiti Casa")
            df_subiti_casa_ft = calcola_goals_per_team_period(filtered_df, 'home', 'subiti', 'ft')
            if not df_subiti_casa_ft.empty:
                st.dataframe(df_subiti_casa_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Fatti Trasferta")
            df_fatti_away_ft = calcola_goals_per_team_period(filtered_df, 'away', 'fatti', 'ft')
            if not df_fatti_away_ft.empty:
                st.dataframe(df_fatti_away_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")
        with col4:
            st.markdown("#### Subiti Trasferta")
            df_subiti_away_ft = calcola_goals_per_team_period(filtered_df, 'away', 'subiti', 'ft')
            if not df_subiti_away_ft.empty:
                st.dataframe(df_subiti_away_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else: st.info("Nessun dato.")

else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati per l'analisi pre-match.")

# SEZIONE 4: Analisi Timeband Dinamica (Minuto/Risultato)
st.subheader("4. Analisi Timeband Dinamica")
with st.expander("Mostra Analisi Dinamica (Minuto/Risultato)"):
    if not filtered_df.empty:
        # --- ANALISI DAL MINUTO (integrata) ---
        # Cursore unico per il range di minuti
        min_range = st.slider("Seleziona Range Minuti", 1, 90, (45, 90), key="dynamic_min_range")
        start_min, end_min = min_range[0], min_range[1]

        # Assicurati che ht_results esista e non sia vuoto
        ht_results_to_show = sorted(df["risultato_ht"].dropna().unique()) if "risultato_ht" in df.columns else []
        risultati_correnti = st.multiselect("Risultato corrente al minuto iniziale",
                                            ht_results_to_show,
                                            default=["0-0"] if "0-0" in ht_results_to_show else [],
                                            key="dynamic_current_results")

        partite_target = []
        for _, row in filtered_df.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol_home", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            home_fino = sum(1 for g in gol_home if g < start_min)
            away_fino = sum(1 for g in gol_away if g < start_min)
            risultato_fino = f"{home_fino}-{away_fino}"
            if risultato_fino in risultati_correnti:
                partite_target.append(row)

        if not partite_target:
            st.warning(f"Nessuna partita con risultato selezionato al minuto {start_min}.")
        else:
            df_target = pd.DataFrame(partite_target)
            st.write(f"**Partite trovate:** {len(df_target)}")

            # Calcolo e visualizzazione media gol dinamica
            st.subheader("Media Gol (Dinamica)")
            df_target_goals = df_target.copy()
            
            required_cols_dynamic_goals = ["gol_home_ht", "gol_away_ht", "gol_home_ft", "gol_away_ft"]
            if all(col in df_target_goals.columns for col in required_cols_dynamic_goals):
                df_target_goals["gol_home_ht"] = pd.to_numeric(df_target_goals["gol_home_ht"], errors='coerce')
                df_target_goals["gol_away_ht"] = pd.to_numeric(df_target_goals["gol_away_ht"], errors='coerce')
                df_target_goals["gol_home_ft"] = pd.to_numeric(df_target_goals["gol_home_ft"], errors='coerce')
                df_target_goals["gol_away_ft"] = pd.to_numeric(df_target_goals["gol_away_ft"], errors='coerce')
                
                df_target_goals_clean = df_target_goals.dropna(subset=required_cols_dynamic_goals)

                if not df_target_goals_clean.empty:
                    # Media gol HT
                    avg_ht_goals_dynamic = (df_target_goals_clean["gol_home_ht"] + df_target_goals_clean["gol_away_ht"]).mean()
                    # Media gol FT
                    avg_ft_goals_dynamic = (df_target_goals_clean["gol_home_ft"] + df_target_goals_clean["gol_away_ft"]).mean()
                    # Media gol SH (secondo tempo)
                    avg_sh_goals_dynamic = (df_target_goals_clean["gol_home_ft"] + df_target_goals_clean["gol_away_ft"] - df_target_goals_clean["gol_home_ht"] - df_target_goals_clean["gol_away_ht"]).mean()
                    
                    st.table(pd.DataFrame({
                        "Periodo": ["HT", "FT", "SH"],
                        "Media Gol": [f"{avg_ht_goals_dynamic:.2f}", f"{avg_ft_goals_dynamic:.2f}", f"{avg_sh_goals_dynamic:.2f}"]
                    }))
                else:
                    st.info("Nessun dato valido per la media gol dinamica dopo la pulizia.")
            else:
                st.warning(f"Colonne gol mancanti per calcolare la Media Gol (Dinamica): {', '.join([col for col in required_cols_dynamic_goals if col not in df_target_goals.columns])}")

            mostra_risultati_esatti(df_target, "risultato_ht", f"HT ({len(df_target)})")
            mostra_risultati_esatti(df_target, "risultato_ft", f"FT ({len(df_target)})")

            # WinRate
            st.subheader(f"WinRate (Dinamica) ({len(df_target)})")
            st.write("**HT:**")
            df_winrate_ht_dynamic = calcola_winrate(df_target, "risultato_ht")
            if not df_winrate_ht_dynamic.empty:
                styled_df_ht = df_winrate_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df_ht)
            else: st.info("Nessun dato.")
            st.write("**FT:**")
            df_winrate_ft_dynamic = calcola_winrate(df_target, "risultato_ft")
            if not df_winrate_ft_dynamic.empty:
                styled_df_ft = df_winrate_ft_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df_ft)
            else: st.info("Nessun dato.")
            
            # Over Goals HT e FT
            col1, col2 = st.columns(2)
            df_target_goals["tot_goals_ht"] = pd.to_numeric(df_target_goals["gol_home_ht"], errors='coerce') + pd.to_numeric(df_target_goals["gol_away_ht"], errors='coerce')
            df_target_goals["tot_goals_ft"] = pd.to_numeric(df_target_goals["gol_home_ft"], errors='coerce') + pd.to_numeric(df_target_goals["gol_away_ft"], errors='coerce')
            
            with col1:
                st.subheader(f"Over Goals HT (Dinamica) ({len(df_target)})")
                over_ht_data_dynamic = []
                df_target_goals_ht_clean = df_target_goals.dropna(subset=["tot_goals_ht"])
                if not df_target_goals_ht_clean.empty:
                    for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                        count = (df_target_goals_ht_clean["tot_goals_ht"] > t).sum()
                        perc = round((count / len(df_target_goals_ht_clean)) * 100, 2)
                        odd_min = round(100 / perc, 2) if perc > 0 else "-"
                        over_ht_data_dynamic.append([f"Over {t} HT", count, perc, odd_min])
                    df_over_ht_dynamic = pd.DataFrame(over_ht_data_dynamic, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                    styled_over_ht_dynamic = df_over_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_over_ht_dynamic)
                else: st.info("Nessun dato valido.")
            
            with col2:
                st.subheader(f"Over Goals FT (Dinamica) ({len(df_target)})")
                over_ft_data = []
                df_target_goals_ft_clean = df_target_goals.dropna(subset=["tot_goals_ft"])
                if not df_target_goals_ft_clean.empty:
                    for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                        count = (df_target_goals_ft_clean["tot_goals_ft"] > t).sum()
                        perc = round((count / len(df_target_goals_ft_clean)) * 100, 2)
                        odd_min = round(100 / perc, 2) if perc > 0 else "-"
                        over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
                    df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                    styled_over_ft = df_over_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_over_ft)
                else: st.info("Nessun dato valido.")
            
            # BTTS
            st.subheader(f"BTTS (Dinamica) ({len(df_target)})")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### HT")
                df_btts_ht_dynamic = calcola_btts_ht_dinamico(df_target)
                if not df_btts_ht_dynamic.empty:
                    styled_df = df_btts_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
            with col2:
                st.write("### FT")
                df_btts_ft_dynamic = calcola_btts_dinamico(df_target, start_min, risultati_correnti)
                if not df_btts_ft_dynamic.empty:
                    styled_df = df_btts_ft_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")

            # Doppia Chance Dinamica
            st.subheader(f"Doppia Chance (Dinamica) ({len(df_target)})")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### HT")
                df_dc_ht = calcola_double_chance(df_target, 'ht')
                if not df_dc_ht.empty:
                    styled_df = df_dc_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
            with col2:
                st.write("### FT")
                df_dc_ft = calcola_double_chance(df_target, 'ft')
                if not df_dc_ft.empty:
                    styled_df = df_dc_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")

            # Multi Gol
            st.subheader(f"Multi Gol (Dinamica) ({len(df_target)})")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Casa")
                df_mg_home_dyn = calcola_multi_gol(df_target, "gol_home_ft", "Home")
                if not df_mg_home_dyn.empty:
                    styled_df = df_mg_home_dyn.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
            with col2:
                st.write("### Trasferta")
                df_mg_away_dyn = calcola_multi_gol(df_target, "gol_away_ft", "Away")
                if not df_mg_away_dyn.empty:
                    styled_df = df_mg_away_dyn.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
            
            # First to Score nell'analisi dinamica (HT e FT)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"First to Score HT (Dinamica) ({len(df_target)})")
                df_fts_ht_dyn = calcola_first_to_score_ht(df_target)
                if not df_fts_ht_dyn.empty:
                    styled_df = df_fts_ht_dyn.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
            with col2:
                st.subheader(f"First to Score FT (Dinamica) ({len(df_target)})")
                df_fts_ft_dyn = calcola_first_to_score(df_target)
                if not df_fts_ft_dyn.empty:
                    styled_df = df_fts_ft_dyn.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
            
            # First to Score + Outcome Dinamica
            st.subheader(f"First to Score + Risultato Finale (Dinamica) ({len(df_target)})")
            df_fts_outcome_dyn = calcola_first_to_score_outcome(df_target)
            if not df_fts_outcome_dyn.empty:
                styled_df = df_fts_outcome_dyn.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            else: st.info("Nessun dato.")
            
            # First to Score + Next Goal Dinamica
            st.subheader(f"First to Score + Risultato Prossimo Gol (Dinamica) ({len(df_target)})")
            df_fts_next_dyn = calcola_first_to_score_next_goal_outcome(df_target)
            if not df_fts_next_dyn.empty:
                styled_df = df_fts_next_dyn.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            else: st.info("Nessun dato.")
            
            # To Score nell'analisi dinamica (HT e FT)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"To Score HT (Dinamica) ({len(df_target)})")
                df_ts_ht_dyn = calcola_to_score_ht(df_target)
                if not df_ts_ht_dyn.empty:
                    styled_df = df_ts_ht_dyn.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
            with col2:
                st.subheader(f"To Score FT (Dinamica) ({len(df_target)})")
                df_ts_ft_dyn = calcola_to_score(df_target)
                if not df_ts_ft_dyn.empty:
                    styled_df = df_ts_ft_dyn.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
            
            # Clean Sheet nell'analisi dinamica
            st.subheader(f"Clean Sheet (Dinamica) ({len(df_target)})")
            df_cs_dyn = calcola_clean_sheet(df_target)
            if not df_cs_dyn.empty:
                styled_df = df_cs_dyn.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            else: st.info("Nessun dato.")
            
            # Combo Markets nell'analisi dinamica
            st.subheader(f"Combo Markets (Dinamica) ({len(df_target)})")
            df_combo_dyn = calcola_combo_stats(df_target)
            if not df_combo_dyn.empty:
                styled_df = df_combo_dyn.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            else: st.info("Nessun dato.")
            
            # Next Goal nell'analisi dinamica
            st.subheader(f"Next Goal (Dinamica) ({len(df_target)})")
            df_next_goal_dyn = calcola_next_goal(df_target, start_min, end_min)
            if not df_next_goal_dyn.empty:
                styled_df = df_next_goal_dyn.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            else: st.info("Nessun dato.")
            
            # Analisi Rimonte Dinamica
            st.subheader(f"Analisi Rimonte (Dinamica) ({len(df_target)})")
            rimonte_stats, squadre_rimonte = calcola_rimonte(df_target, "Dinamica")
            if not rimonte_stats.empty:
                styled_df = rimonte_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
                
                st.markdown("**Squadre che hanno effettuato rimonte:**")
                for tipo, squadre in squadre_rimonte.items():
                    if squadre:
                        st.markdown(f"**{tipo}:** {', '.join(squadre)}")
            else:
                st.warning("Nessuna rimonta trovata nel dataset filtrato per questa analisi dinamica.")
            
            # Qui viene mostrata la timeband basata sull'analisi dinamica
            st.subheader("Distribuzione Gol per Timeframe (dinamica)")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**15min**")
                mostra_distribuzione_timeband(df_target)
            with col2:
                st.write("**5min**")
                mostra_distribuzione_timeband_5min(df_target)

    else:
        st.warning("Il dataset filtrato è vuoto o mancano le colonne necessarie per l'analisi dinamica.")

# --- SEZIONE 5: Analisi Head-to-Head (H2H) ---
st.subheader("5. Analisi Head-to-Head (H2H)")
st.write("Seleziona due squadre per analizzare i loro scontri diretti.")

# Recupera l'elenco completo di tutte le squadre disponibili nel dataset
if "home_team" in df.columns and "away_team" in df.columns:
    all_teams = sorted(list(set(df['home_team'].dropna().unique()) | set(df['away_team'].dropna().unique())))
    h2h_home_team = st.selectbox("Seleziona Squadra 1", ["Seleziona..."] + all_teams, key="h2h_team1")
    h2h_away_team = st.selectbox("Seleziona Squadra 2", ["Seleziona..."] + all_teams, key="h2h_team2")

    if h2h_home_team != "Seleziona..." and h2h_away_team != "Seleziona...":
        if h2h_home_team == h2h_away_team:
            st.warning("Seleziona due squadre diverse per l'analisi H2H.")
        else:
            # Filtra il DataFrame per trovare tutti i match tra le due squadre selezionate
            # NOTA: I filtri per le quote della sidebar non vengono applicati qui per avere il dataset H2H completo
            h2h_df = df[((df['home_team'] == h2h_home_team) & (df['away_team'] == h2h_away_team)) |
                        ((df['home_team'] == h2h_away_team) & (df['away_team'] == h2h_home_team))]
            
            if h2h_df.empty:
                st.warning(f"Nessuna partita trovata tra {h2h_home_team} e {h2h_away_team}.")
            else:
                st.write(f"Analisi basata su **{len(h2h_df)}** scontri diretti tra {h2h_home_team} e {h2h_away_team}.")

                # Esegui le stesse analisi pre-match, ma sul DataFrame H2H
                st.subheader(f"Statistiche H2H Complete tra {h2h_home_team} e {h2h_away_team} ({len(h2h_df)} partite)")
                
                # Media gol
                st.subheader("Media Gol (H2H)")
                df_h2h_goals = h2h_df.copy()
                required_cols_h2h_goals = ["gol_home_ht", "gol_away_ht", "gol_home_ft", "gol_away_ft"]
                if all(col in df_h2h_goals.columns for col in required_cols_h2h_goals):
                    df_h2h_goals["gol_home_ht"] = pd.to_numeric(df_h2h_goals["gol_home_ht"], errors='coerce')
                    df_h2h_goals["gol_away_ht"] = pd.to_numeric(df_h2h_goals["gol_away_ht"], errors='coerce')
                    df_h2h_goals["gol_home_ft"] = pd.to_numeric(df_h2h_goals["gol_home_ft"], errors='coerce')
                    df_h2h_goals["gol_away_ft"] = pd.to_numeric(df_h2h_goals["gol_away_ft"], errors='coerce')

                    df_h2h_goals_clean = df_h2h_goals.dropna(subset=required_cols_h2h_goals)

                    if not df_h2h_goals_clean.empty:
                        avg_ht_goals = (df_h2h_goals_clean["gol_home_ht"] + df_h2h_goals_clean["gol_away_ht"]).mean()
                        avg_ft_goals = (df_h2h_goals_clean["gol_home_ft"] + df_h2h_goals_clean["gol_away_ft"]).mean()
                        avg_sh_goals = (df_h2h_goals_clean["gol_home_ft"] + df_h2h_goals_clean["gol_away_ft"] - df_h2h_goals_clean["gol_home_ht"] - df_h2h_goals_clean["gol_away_ht"]).mean()
                        st.table(pd.DataFrame({
                            "Periodo": ["HT", "FT", "SH"],
                            "Media Gol": [f"{avg_ht_goals:.2f}", f"{avg_ft_goals:.2f}", f"{avg_sh_goals:.2f}"]
                        }))
                    else:
                        st.info("Nessun dato valido per la media gol H2H dopo la pulizia.")
                else:
                    st.warning(f"Colonne gol mancanti per calcolare la Media Gol (H2H): {', '.join([col for col in required_cols_h2h_goals if col not in df_h2h_goals.columns])}")
                
                # Risultati Esatti H2H
                mostra_risultati_esatti(h2h_df, "risultato_ht", f"HT H2H ({len(h2h_df)})")
                mostra_risultati_esatti(h2h_df, "risultato_ft", f"FT H2H ({len(h2h_df)})")

                # WinRate H2H
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"WinRate HT H2H ({len(h2h_df)})")
                    df_winrate_ht_h2h = calcola_winrate(h2h_df, "risultato_ht")
                    if not df_winrate_ht_h2h.empty:
                        styled_df_ht = df_winrate_ht_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                        st.dataframe(styled_df_ht)
                    else: st.info("Nessun dato.")
                with col2:
                    st.subheader(f"WinRate FT H2H ({len(h2h_df)})")
                    df_winrate_ft_h2h = calcola_winrate(h2h_df, "risultato_ft")
                    if not df_winrate_ft_h2h.empty:
                        styled_df_ft = df_winrate_ft_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                        st.dataframe(styled_df_ft)
                    else: st.info("Nessun dato.")
                
                # Doppia Chance H2H
                st.subheader(f"Doppia Chance (H2H) ({len(h2h_df)})")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### HT")
                    df_dc_ht_h2h = calcola_double_chance(h2h_df, 'ht')
                    if not df_dc_ht_h2h.empty:
                        styled_df = df_dc_ht_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                        st.dataframe(styled_df)
                    else: st.info("Nessun dato.")
                with col2:
                    st.write("### FT")
                    df_dc_ft_h2h = calcola_double_chance(h2h_df, 'ft')
                    if not df_dc_ft_h2h.empty:
                        styled_df = df_dc_ft_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                        st.dataframe(styled_df)
                    else: st.info("Nessun dato.")

                # Over Goals H2H
                col1, col2 = st.columns(2)
                if "gol_home_ht" in h2h_df.columns and "gol_away_ht" in h2h_df.columns and \
                   "gol_home_ft" in h2h_df.columns and "gol_away_ft" in h2h_df.columns:
                    df_h2h_goals["tot_goals_ht"] = pd.to_numeric(df_h2h_goals["gol_home_ht"], errors='coerce') + pd.to_numeric(df_h2h_goals["gol_away_ht"], errors='coerce')
                    df_h2h_goals["tot_goals_ft"] = pd.to_numeric(df_h2h_goals["gol_home_ft"], errors='coerce') + pd.to_numeric(df_h2h_goals["gol_away_ft"], errors='coerce')
                else:
                    st.warning("Colonne gol HT/FT mancanti per Over Goals H2H.")

                with col1:
                    st.subheader(f"Over Goals HT H2H ({len(h2h_df)})")
                    over_ht_data = []
                    df_h2h_goals_ht_clean = df_h2h_goals.dropna(subset=["tot_goals_ht"])
                    if not df_h2h_goals_ht_clean.empty:
                        for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                            count = (df_h2h_goals_ht_clean["tot_goals_ht"] > t).sum()
                            perc = round((count / len(df_h2h_goals_ht_clean)) * 100, 2)
                            odd_min = round(100 / perc, 2) if perc > 0 else "-"
                            over_ht_data.append([f"Over {t} HT", count, perc, odd_min])
                        df_over_ht = pd.DataFrame(over_ht_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                        styled_over_ht = df_over_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                        st.dataframe(styled_over_ht)
                    else: st.info("Nessun dato.")

                with col2:
                    st.subheader(f"Over Goals FT H2H ({len(h2h_df)})")
                    over_ft_data = []
                    df_h2h_goals_ft_clean = df_h2h_goals.dropna(subset=["tot_goals_ft"])
                    if not df_h2h_goals_ft_clean.empty:
                        for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                            count = (df_h2h_goals_ft_clean["tot_goals_ft"] > t).sum()
                            perc = round((count / len(df_h2h_goals_ft_clean)) * 100, 2)
                            odd_min = round(100 / perc, 2) if perc > 0 else "-"
                            over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
                        df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                        styled_over_ft = df_over_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                        st.dataframe(styled_over_ft)
                    else: st.info("Nessun dato.")
                
                # BTTS H2H
                st.subheader(f"BTTS (H2H) ({len(h2h_df)})")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### HT")
                    df_btts_ht_h2h = calcola_btts_ht(h2h_df)
                    if not df_btts_ht_h2h.empty:
                        styled_df = df_btts_ht_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                        st.dataframe(styled_df)
                    else: st.info("Nessun dato.")
                with col2:
                    st.write("### FT")
                    df_btts_ft_h2h = calcola_btts_ft(h2h_df)
                    if not df_btts_ft_h2h.empty:
                        styled_df = df_btts_ft_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                        st.dataframe(styled_df)
                    else: st.info("Nessun dato.")
                    
                # Multi Gol H2H
                st.subheader(f"Multi Gol (H2H) ({len(h2h_df)})")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Casa")
                    df_mg_home_h2h = calcola_multi_gol(h2h_df, "gol_home_ft", "Home")
                    if not df_mg_home_h2h.empty:
                        styled_df = df_mg_home_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                        st.dataframe(styled_df)
                    else: st.info("Nessun dato.")
                with col2:
                    st.write("### Trasferta")
                    df_mg_away_h2h = calcola_multi_gol(h2h_df, "gol_away_ft", "Away")
                    if not df_mg_away_h2h.empty:
                        styled_df = df_mg_away_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                        st.dataframe(styled_df)
                    else: st.info("Nessun dato.")

                # First to Score H2H
                st.subheader(f"First to Score (H2H) ({len(h2h_df)})")
                df_fts_h2h = calcola_first_to_score(h2h_df)
                if not df_fts_h2h.empty:
                    styled_df = df_fts_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
                
                # First to Score + Outcome H2H
                st.subheader(f"First to Score + Risultato Finale (H2H) ({len(h2h_df)})")
                df_fts_outcome_h2h = calcola_first_to_score_outcome(h2h_df)
                if not df_fts_outcome_h2h.empty:
                    styled_df = df_fts_outcome_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")

                # First to Score + Next Goal H2H
                st.subheader(f"First to Score + Risultato Prossimo Gol (H2H) ({len(h2h_df)})")
                df_fts_next_h2h = calcola_first_to_score_next_goal_outcome(h2h_df)
                if not df_fts_next_h2h.empty:
                    styled_df = df_fts_next_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
                
                # To Score H2H
                st.subheader(f"To Score (H2H) ({len(h2h_df)})")
                df_ts_h2h = calcola_to_score(h2h_df)
                if not df_ts_h2h.empty:
                    styled_df = df_ts_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
                
                # Clean Sheet H2H
                st.subheader(f"Clean Sheet (H2H) ({len(h2h_df)})")
                df_cs_h2h = calcola_clean_sheet(h2h_df)
                if not df_cs_h2h.empty:
                    styled_df = df_cs_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
                
                # Combo Markets H2H
                st.subheader(f"Combo Markets (H2H) ({len(h2h_df)})")
                df_combo_h2h = calcola_combo_stats(h2h_df)
                if not df_combo_h2h.empty:
                    styled_df = df_combo_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                else: st.info("Nessun dato.")
                
                # Analisi Rimonte H2H
                st.subheader(f"Analisi Rimonte (H2H) ({len(h2h_df)})")
                rimonte_stats, squadre_rimonte = calcola_rimonte(h2h_df, "H2H")
                if not rimonte_stats.empty:
                    styled_df = rimonte_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                    st.dataframe(styled_df)
                    
                    st.markdown("**Squadre che hanno effettuato rimonte:**")
                    for tipo, squadre in squadre_rimonte.items():
                        if squadre:
                            st.markdown(f"**{tipo}:** {', '.join(squadre)}")
                else:
                    st.warning("Nessuna rimonta trovata nel dataset filtrato.")
    else:
        st.warning("Le colonne 'home_team' o 'away_team' non sono presenti per l'analisi H2H.")

# --- SEZIONE 6: Backtesting Strategie ---
st.subheader("6. Backtesting Strategie")
st.write("Testa una strategia di scommesse sui dati filtrati.")
if selected_home != "Tutte" and selected_away != "Tutte":
    st.write(f"I seguenti risultati di backtest si basano sulle partite in cui **{selected_home}** ha giocato in casa e **{selected_away}** ha giocato in trasferta.")
else:
    st.write("I seguenti risultati di backtest si basano su tutti i match filtrati.")

# Aggiungi un expander per contenere la logica di backtesting
with st.expander("Configura e avvia il Backtest"):
    
    if filtered_df.empty:
        st.warning("Il DataFrame filtrato è vuoto, non è possibile eseguire il backtest.")
    else:
        # Funzione per eseguire il backtest
        def esegui_backtest(df_to_analyze, market, strategy, stake):
            
            # Definizione dei mercati e delle colonne necessarie
            market_map = {
                "1 (Casa)": ("odd_home", lambda row: row["gol_home_ft"] > row["gol_away_ft"]),
                "X (Pareggio)": ("odd_draw", lambda row: row["gol_home_ft"] == row["gol_away_ft"]),
                "2 (Trasferta)": ("odd_away", lambda row: row["gol_home_ft"] < row["gol_away_ft"]),
                "Over 2.5 FT": ("odd_over_2_5", lambda row: (row["gol_home_ft"] + row["gol_away_ft"]) > 2.5),
                "BTTS SI FT": ("odd_btts_si", lambda row: (row["gol_home_ft"] > 0 and row["gol_away_ft"] > 0))
            }
            
            if market not in market_map:
                st.error(f"Mercato '{market}' non supportato nel backtest.")
                return 0, 0, 0, 0.0, 0.0, 0.0, 0.0

            odd_col, win_condition = market_map[market]
            
            # Controllo che le colonne necessarie esistano nel DataFrame
            required_cols = [odd_col, "gol_home_ft", "gol_away_ft"]
            for col in required_cols:
                if col not in df_to_analyze.columns:
                    st.warning(f"Impossibile eseguire il backtest: la colonna '{col}' non è presente nel dataset.")
                    return 0, 0, 0, 0.0, 0.0, 0.0, 0.0
            
            vincite = 0
            perdite = 0
            profit_loss = 0.0
            numero_scommesse = 0
            
            # Rimuovi le righe con valori nulli nelle colonne chiave
            df_clean = df_to_analyze.dropna(subset=required_cols).copy()
            
            # Assicurati che le colonne quote e gol siano numeriche
            df_clean[odd_col] = pd.to_numeric(df_clean[odd_col].astype(str).str.replace(",", "."), errors='coerce').fillna(0)
            df_clean["gol_home_ft"] = pd.to_numeric(df_clean["gol_home_ft"], errors='coerce').fillna(0)
            df_clean["gol_away_ft"] = pd.to_numeric(df_clean["gol_away_ft"], errors='coerce').fillna(0)

            for _, row in df_clean.iterrows():
                try:
                    odd = row[odd_col]
                    
                    if odd > 0: # Ignora scommesse con quota pari a 0
                        is_winning = win_condition(row)
                        
                        if strategy == "Back":
                            if is_winning:
                                vincite += 1
                                profit_loss += (odd - 1) * stake
                            else:
                                perdite += 1
                                profit_loss -= stake
                        elif strategy == "Lay":
                            if is_winning:
                                perdite += 1
                                profit_loss -= (odd - 1) * stake
                            else:
                                vincite += 1
                                profit_loss += stake
                        
                        numero_scommesse += 1
                    
                except (ValueError, KeyError) as e:
                    # Gestione di righe con dati mancanti o non validi
                    st.warning(f"Errore durante l'elaborazione della riga per il backtest: {e}")
                    continue

            investimento_totale = numero_scommesse * stake
            roi = (profit_loss / investimento_totale) * 100 if investimento_totale > 0 else 0
            win_rate = (vincite / numero_scommesse) * 100 if numero_scommesse > 0 else 0
            odd_minima = 100 / win_rate if win_rate > 0 else 0
            
            return vincite, perdite, numero_scommesse, profit_loss, roi, win_rate, odd_minima

        # UI per il backtest
        backtest_market = st.selectbox(
            "Seleziona un mercato da testare",
            ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)", "Over 2.5 FT", "BTTS SI FT"],
            key="backtest_market_select"
        )
        backtest_strategy = st.selectbox(
            "Seleziona la strategia",
            ["Back", "Lay"],
            key="backtest_strategy_select"
        )
        stake = st.number_input("Stake per scommessa", min_value=1.0, value=1.0, step=0.5, key="backtest_stake_input")
        
        if st.button("Avvia Backtest", key="start_backtest_button"):
            vincite, perdite, numero_scommesse, profit_loss, roi, win_rate, odd_minima = esegui_backtest(filtered_df, backtest_market, backtest_strategy, stake)
            
            if numero_scommesse > 0:
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                col_met1.metric("Numero Scommesse", numero_scommesse)
                col_met2.metric("Vincite", vincite)
                col_met3.metric("Perdite", perdite)
                col_met4.metric("Profitto/Perdita", f"{profit_loss:.2f} €")
                
                col_met5, col_met6 = st.columns(2)
                col_met5.metric("ROI", f"{roi:.2f} %")
                col_met6.metric("Win Rate", f"{win_rate:.2f} %")
                st.metric("Odd Minima per profitto", f"{odd_minima:.2f}")
            elif numero_scommesse == 0:
                st.info("Nessuna scommessa idonea trovata con i filtri e il mercato selezionati.")

