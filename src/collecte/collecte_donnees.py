"""
========================================================
MODULE 1 : COLLECTE DES DONNÉES
========================================================
Sources :
  - API OpenMeteo (météo réelle, gratuite, sans clé)
  - Génération simulée réaliste : parcelles, sol, intrants, rendements
========================================================
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# ─── CONFIG ───────────────────────────────────────────
# Coordonnées : Lomé, Togo (ou adapter à votre région)
LAT = 6.1375
LON = 1.2123
START_DATE = "2023-01-01"
END_DATE   = "2024-12-31"

DATA_RAW = os.path.join(os.path.dirname(__file__), "../../data/raw")
os.makedirs(DATA_RAW, exist_ok=True)


# ══════════════════════════════════════════════════════
# 1. DONNÉES MÉTÉO — API Open-Meteo (gratuite, sans clé)
# ══════════════════════════════════════════════════════
def collecter_meteo(lat=LAT, lon=LON, start=START_DATE, end=END_DATE):
    """
    Collecte les données météo journalières via l'API Open-Meteo.
    Variables : température min/max/moy, précipitations, humidité,
                rayonnement solaire, évapotranspiration.
    """
    print("[COLLECTE] Récupération des données météo via Open-Meteo...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "relative_humidity_2m_mean" if False else "precipitation_hours",
            "et0_fao_evapotranspiration",
            "sunshine_duration",
            "windspeed_10m_max",
        ],
        "timezone": "Africa/Abidjan"
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data["daily"])
        df.rename(columns={
            "time": "date",
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "temperature_2m_mean": "temp_moy",
            "precipitation_sum": "precipitation_mm",
            "precipitation_hours": "heures_pluie",
            "et0_fao_evapotranspiration": "evapotranspiration",
            "sunshine_duration": "ensoleillement_s",
            "windspeed_10m_max": "vent_max_kmh",
        }, inplace=True)

        df["date"] = pd.to_datetime(df["date"])
        df["ensoleillement_h"] = df["ensoleillement_s"] / 3600
        df.drop(columns=["ensoleillement_s"], inplace=True)

        # Enrichissement immédiat : indices dérivés
        df["stress_thermique"] = ((df["temp_max"] > 35) | (df["temp_min"] < 15)).astype(int)
        df["bilan_hydrique"]   = df["precipitation_mm"] - df["evapotranspiration"]
        df["mois"] = df["date"].dt.month
        df["semaine"] = df["date"].dt.isocalendar().week.astype(int)
        df["saison"] = df["mois"].map({
            12:"Harmattan", 1:"Harmattan", 2:"Harmattan",
            3:"Grande Saison Sèche", 4:"Grande Saison Sèche",
            5:"Grande Saison Pluies", 6:"Grande Saison Pluies",
            7:"Grande Saison Pluies", 8:"Petite Saison Sèche",
            9:"Petite Saison Pluies", 10:"Petite Saison Pluies",
            11:"Petite Saison Pluies"
        })

        path = os.path.join(DATA_RAW, "meteo.csv")
        df.to_csv(path, index=False)
        print(f"  ✓ {len(df)} jours de données météo → {path}")
        return df

    except Exception as e:
        print(f"  ✗ Erreur API météo : {e}")
        print("  → Génération de données météo simulées...")
        return _simuler_meteo(start, end)


def _simuler_meteo(start, end):
    """Fallback : génère des données météo réalistes si l'API échoue."""
    dates = pd.date_range(start, end, freq="D")
    np.random.seed(42)
    n = len(dates)
    mois = dates.month

    temp_base  = 28 + 4 * np.sin(2 * np.pi * (mois - 3) / 12)
    precip_base = np.where((mois >= 5) & (mois <= 10), 4.5, 0.5)

    df = pd.DataFrame({
        "date": dates,
        "temp_max": temp_base + np.random.normal(3, 1.5, n),
        "temp_min": temp_base - np.random.normal(6, 1.5, n),
        "temp_moy": temp_base + np.random.normal(0, 1, n),
        "precipitation_mm": np.random.exponential(precip_base),
        "heures_pluie": np.random.uniform(0, 4, n),
        "evapotranspiration": np.random.uniform(3, 6, n),
        "ensoleillement_h": np.random.uniform(5, 10, n),
        "vent_max_kmh": np.random.uniform(5, 25, n),
    })
    df["stress_thermique"] = ((df["temp_max"] > 35) | (df["temp_min"] < 15)).astype(int)
    df["bilan_hydrique"]   = df["precipitation_mm"] - df["evapotranspiration"]
    df["mois"] = df["date"].dt.month
    df["semaine"] = df["date"].dt.isocalendar().week.astype(int)
    df["saison"] = df["mois"].map({
        12:"Harmattan", 1:"Harmattan", 2:"Harmattan",
        3:"Grande Saison Sèche", 4:"Grande Saison Sèche",
        5:"Grande Saison Pluies", 6:"Grande Saison Pluies",
        7:"Grande Saison Pluies", 8:"Petite Saison Sèche",
        9:"Petite Saison Pluies", 10:"Petite Saison Pluies",
        11:"Petite Saison Pluies"
    })
    path = os.path.join(DATA_RAW, "meteo.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ {len(df)} jours simulés → {path}")
    return df


# ══════════════════════════════════════════════════════
# 2. DONNÉES PARCELLES ET SOL
# ══════════════════════════════════════════════════════
def generer_parcelles():
    """
    Génère les données fixes des parcelles : superficie, culture, sol.
    Représente une ferme de ~50 ha divisée en 10 parcelles.
    """
    print("[COLLECTE] Génération des données parcelles / sol...")
    np.random.seed(10)

    parcelles = []
    cultures   = ["Maïs", "Tomate", "Manioc", "Haricot", "Sorgho"]
    types_sol  = ["Argilo-limoneux", "Sableux", "Limoneux", "Argileux"]

    for i in range(1, 11):
        culture = np.random.choice(cultures)
        parcelles.append({
            "parcelle_id": f"P-{i:02d}",
            "superficie_ha": round(np.random.uniform(3, 7), 2),
            "culture": culture,
            "type_sol": np.random.choice(types_sol),
            "ph_sol": round(np.random.uniform(5.5, 7.5), 1),
            "azote_N_ppm": round(np.random.uniform(15, 60), 1),
            "phosphore_P_ppm": round(np.random.uniform(10, 45), 1),
            "potassium_K_ppm": round(np.random.uniform(80, 200), 1),
            "matiere_organique_pct": round(np.random.uniform(1.5, 4.5), 2),
            "capacite_retention_eau": round(np.random.uniform(0.25, 0.45), 2),
            "annee_derniere_analyse": np.random.choice([2022, 2023]),
        })

    df = pd.DataFrame(parcelles)
    path = os.path.join(DATA_RAW, "parcelles.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ {len(df)} parcelles générées → {path}")
    return df


# ══════════════════════════════════════════════════════
# 3. DONNÉES DE PRODUCTION (semis, intrants, récoltes)
# ══════════════════════════════════════════════════════
def generer_production(df_parcelles, df_meteo):
    """
    Génère les données de production par parcelle et par cycle cultural.
    Le rendement est simulé de façon réaliste selon la météo et le sol.
    """
    print("[COLLECTE] Génération des données de production...")
    np.random.seed(99)

    rendements_base = {
        "Maïs":    {"min": 1.5, "max": 4.5, "unite": "t/ha"},
        "Tomate":  {"min": 15,  "max": 45,  "unite": "t/ha"},
        "Manioc":  {"min": 8,   "max": 20,  "unite": "t/ha"},
        "Haricot": {"min": 0.8, "max": 2.2, "unite": "t/ha"},
        "Sorgho":  {"min": 1.0, "max": 3.0, "unite": "t/ha"},
    }

    productions = []
    annees = [2023, 2024]
    cycles = {"Grande Saison Pluies": ("05-01", "08-15"), "Petite Saison Pluies": ("09-01", "11-30")}

    for _, parc in df_parcelles.iterrows():
        for annee in annees:
            for saison_nom, (debut, fin) in cycles.items():
                date_semis  = pd.to_datetime(f"{annee}-{debut}")
                date_recolte = pd.to_datetime(f"{annee}-{fin}")

                # Calcul du score météo sur la période
                mask = (df_meteo["date"] >= date_semis) & (df_meteo["date"] <= date_recolte)
                periode = df_meteo[mask]

                if len(periode) == 0:
                    continue

                score_pluie  = np.clip(periode["precipitation_mm"].mean() / 5, 0.5, 1.3)
                score_stress = 1 - (periode["stress_thermique"].mean() * 0.4)
                score_sol    = (parc["ph_sol"] / 7) * (parc["matiere_organique_pct"] / 3)
                score_global = score_pluie * score_stress * score_sol

                base = rendements_base[parc["culture"]]
                rendement = round(
                    np.random.uniform(base["min"], base["max"]) * np.clip(score_global, 0.5, 1.4),
                    2
                )

                intrant_engrais = round(np.random.uniform(80, 200), 1)
                intrant_eau     = round(np.random.uniform(200, 600), 1)
                intrant_pesticide = round(np.random.uniform(2, 8), 2)

                productions.append({
                    "parcelle_id": parc["parcelle_id"],
                    "culture": parc["culture"],
                    "annee": annee,
                    "saison": saison_nom,
                    "date_semis": date_semis,
                    "date_recolte": date_recolte,
                    "rendement_t_ha": rendement,
                    "production_totale_t": round(rendement * parc["superficie_ha"], 2),
                    "engrais_kg_ha": intrant_engrais,
                    "eau_irrigation_m3_ha": intrant_eau,
                    "pesticide_L_ha": intrant_pesticide,
                    "score_meteo": round(score_pluie, 3),
                    "score_stress_thermique": round(score_stress, 3),
                    "score_sol": round(score_sol, 3),
                })

    df = pd.DataFrame(productions)
    path = os.path.join(DATA_RAW, "production.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ {len(df)} enregistrements de production → {path}")
    return df


# ══════════════════════════════════════════════════════
# 4. DONNÉES MARCHÉ (prix de vente)
# ══════════════════════════════════════════════════════
def generer_prix_marche():
    """Génère des séries de prix hebdomadaires par culture."""
    print("[COLLECTE] Génération des données de marché...")
    np.random.seed(77)

    prix_base = {
        "Maïs":    150,   # FCFA/kg
        "Tomate":  300,
        "Manioc":  80,
        "Haricot": 500,
        "Sorgho":  120,
    }

    semaines = pd.date_range("2023-01-01", "2024-12-31", freq="W")
    records  = []

    for culture, base in prix_base.items():
        saisonnalite = 1 + 0.25 * np.sin(2 * np.pi * (semaines.month - 3) / 12)
        bruit        = np.random.normal(1, 0.08, len(semaines))
        prix         = base * saisonnalite * bruit

        for i, sem in enumerate(semaines):
            records.append({
                "semaine": sem,
                "culture": culture,
                "prix_fcfa_kg": round(prix[i], 0),
                "demande_index": round(np.random.uniform(0.6, 1.4), 2),
            })

    df = pd.DataFrame(records)
    path = os.path.join(DATA_RAW, "marche.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ {len(df)} cotations de marché → {path}")
    return df


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def collecter_toutes_donnees():
    print("\n" + "="*60)
    print("  PIPELINE — ÉTAPE 1 : COLLECTE DES DONNÉES")
    print("="*60)

    df_meteo      = collecter_meteo()
    df_parcelles  = generer_parcelles()
    df_production = generer_production(df_parcelles, df_meteo)
    df_marche     = generer_prix_marche()

    print("\n[RÉSUMÉ COLLECTE]")
    print(f"  Météo      : {len(df_meteo)} jours")
    print(f"  Parcelles  : {len(df_parcelles)} parcelles")
    print(f"  Production : {len(df_production)} cycles")
    print(f"  Marché     : {len(df_marche)} cotations")
    print("  → Collecte terminée ✓\n")

    return df_meteo, df_parcelles, df_production, df_marche


if __name__ == "__main__":
    collecter_toutes_donnees()
