"""
════════════════════════════════════════════════════════════════
FASTAPI — API REST Ferme Agricole
════════════════════════════════════════════════════════════════
Endpoints :
  GET  /                        — Santé de l'API
  GET  /parcelles               — Liste des parcelles
  GET  /production              — Données de production
  GET  /meteo                   — Données météo récentes
  GET  /alertes                 — Alertes actives
  GET  /stats/performance       — Statistiques de performance
  POST /predict/rendement       — Prédiction ML du rendement
  GET  /marche/previsions       — Prévisions des prix
════════════════════════════════════════════════════════════════
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import os, joblib
from sqlalchemy import create_engine, text
from datetime import datetime

# ── Config ────────────────────────────────────────────────────
DB_URL = (
    f"postgresql://"
    f"{os.getenv('DB_USER','ferme')}:{os.getenv('DB_PASS','ferme2024')}"
    f"@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5432')}"
    f"/{os.getenv('DB_NAME','ferme_agricole')}"
)

DATA_DIR   = os.getenv("DATA_DIR", "/data")
MODELS_DIR = os.path.join(DATA_DIR, "models")

app = FastAPI(
    title="Ferme Agricole — API Big Data",
    description="API REST pour le système de pilotage agricole par la donnée",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Connexion DB ──────────────────────────────────────────────
def get_engine():
    return create_engine(DB_URL)

def query_df(sql: str, params: dict = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# ── Schémas Pydantic ──────────────────────────────────────────
class RendementInput(BaseModel):
    culture:              str   = Field(..., example="Maïs")
    ph_sol:               float = Field(6.5,  ge=4.0, le=9.0)
    azote_N_ppm:          float = Field(35.0, ge=0)
    phosphore_P_ppm:      float = Field(25.0, ge=0)
    potassium_K_ppm:      float = Field(140.0, ge=0)
    matiere_organique_pct: float = Field(3.0, ge=0, le=10)
    temp_moy_cycle:       float = Field(28.0)
    pluie_totale_mm:      float = Field(280.0, ge=0)
    bilan_hydrique:       float = Field(0.0)
    jours_stress:         int   = Field(5,  ge=0)
    ensoleillement_h:     float = Field(7.0, ge=0)
    duree_cycle_jours:    int   = Field(120, ge=30)
    engrais_kg_ha:        float = Field(130.0, ge=0)
    eau_irrigation_m3_ha: float = Field(380.0, ge=0)

class RendementOutput(BaseModel):
    culture:              str
    rendement_predit_t_ha: float
    indice_fertilite_sol: float
    score_risque_climatique: float
    niveau_performance:   str
    timestamp:            str


# ════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.get("/", tags=["Santé"])
def health_check():
    return {
        "status": "online",
        "service": "Ferme Agricole API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/parcelles", tags=["Données"])
def get_parcelles(culture: Optional[str] = Query(None, description="Filtrer par culture")):
    """Retourne la liste des parcelles avec leurs caractéristiques de sol."""
    try:
        sql = "SELECT * FROM parcelles"
        if culture:
            sql += " WHERE culture = :culture"
        df = query_df(sql, {"culture": culture} if culture else None)
        return {"count": len(df), "data": df.to_dict(orient="records")}
    except Exception as e:
        # Fallback fichier CSV si DB indisponible
        path = os.path.join(DATA_DIR, "enriched", "parcelles.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if culture:
                df = df[df["culture"] == culture]
            return {"count": len(df), "data": df.to_dict(orient="records"), "source": "csv"}
        raise HTTPException(status_code=503, detail=f"DB indisponible : {e}")


@app.get("/production", tags=["Données"])
def get_production(
    annee:    Optional[int] = Query(None),
    culture:  Optional[str] = Query(None),
    parcelle: Optional[str] = Query(None),
    limit:    int = Query(100, le=1000),
):
    """Retourne les données de production enrichies."""
    try:
        conditions = []
        params = {}
        if annee:
            conditions.append("annee = :annee")
            params["annee"] = annee
        if culture:
            conditions.append("culture = :culture")
            params["culture"] = culture
        if parcelle:
            conditions.append("parcelle_id = :parcelle")
            params["parcelle"] = parcelle

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql   = f"SELECT * FROM dataset_enrichi{where} LIMIT :limit"
        params["limit"] = limit

        df = query_df(sql, params)
        return {"count": len(df), "data": df.to_dict(orient="records")}
    except Exception as e:
        path = os.path.join(DATA_DIR, "enriched", "dataset_enrichi.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if annee:    df = df[df["annee"] == annee]
            if culture:  df = df[df["culture"] == culture]
            if parcelle: df = df[df["parcelle_id"] == parcelle]
            return {"count": len(df), "data": df.head(limit).to_dict(orient="records"), "source": "csv"}
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/meteo", tags=["Données"])
def get_meteo(
    start: Optional[str] = Query(None, description="Date début YYYY-MM-DD"),
    end:   Optional[str] = Query(None, description="Date fin YYYY-MM-DD"),
    limit: int = Query(90, le=730),
):
    """Retourne les données météo nettoyées."""
    try:
        sql = "SELECT * FROM meteo ORDER BY date DESC LIMIT :limit"
        df  = query_df(sql, {"limit": limit})
        return {"count": len(df), "data": df.to_dict(orient="records")}
    except Exception as e:
        path = os.path.join(DATA_DIR, "enriched", "meteo_nettoyee.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            if start: df = df[df["date"] >= start]
            if end:   df = df[df["date"] <= end]
            return {"count": len(df), "data": df.tail(limit).to_dict(orient="records"), "source": "csv"}
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/alertes", tags=["Alertes"])
def get_alertes(resolue: Optional[bool] = Query(False)):
    """Retourne les alertes actives (déficit hydrique, stress thermique)."""
    try:
        sql = "SELECT * FROM alertes WHERE resolue = :resolue ORDER BY date_alerte DESC"
        df  = query_df(sql, {"resolue": resolue})
        return {"count": len(df), "data": df.to_dict(orient="records")}
    except Exception as e:
        path = os.path.join(DATA_DIR, "enriched", "dataset_enrichi.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            alertes = []
            for _, row in df.iterrows():
                if row.get("alerte_deficit_hydrique", 0) == 1:
                    alertes.append({"parcelle_id": row["parcelle_id"], "type": "DEFICIT_HYDRIQUE",
                                    "severite": "CRITIQUE", "culture": row["culture"]})
                if row.get("alerte_stress_thermique", 0) == 1:
                    alertes.append({"parcelle_id": row["parcelle_id"], "type": "STRESS_THERMIQUE",
                                    "severite": "AVERTISSEMENT", "culture": row["culture"]})
            return {"count": len(alertes), "data": alertes, "source": "csv"}
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/stats/performance", tags=["Statistiques"])
def get_stats_performance():
    """Retourne les statistiques globales de performance de la ferme."""
    try:
        sql = "SELECT * FROM v_performance_parcelle ORDER BY rendement_moy DESC"
        df  = query_df(sql)
    except Exception:
        path = os.path.join(DATA_DIR, "enriched", "dataset_enrichi.csv")
        df   = pd.read_csv(path)
        df   = df.groupby(["parcelle_id","culture","annee"]).agg(
            rendement_moy=("rendement_t_ha","mean"),
            marge_moy=("marge_nette_fcfa_ha","mean"),
            nb_alertes_hydrique=("alerte_deficit_hydrique","sum"),
            nb_alertes_thermique=("alerte_stress_thermique","sum"),
        ).reset_index()

    return {
        "ferme": {
            "rendement_global_moy": round(float(df["rendement_moy"].mean()), 3),
            "marge_global_moy":     round(float(df["marge_moy"].mean()), 0),
            "nb_parcelles":         int(df["parcelle_id"].nunique()),
            "nb_alertes_total":     int(df["nb_alertes_hydrique"].sum() + df["nb_alertes_thermique"].sum()),
        },
        "par_parcelle": df.round(2).to_dict(orient="records"),
    }


@app.post("/predict/rendement", response_model=RendementOutput, tags=["ML"])
def predict_rendement(data: RendementInput):
    """
    Prédit le rendement d'une parcelle via le modèle Random Forest.
    Retourne le rendement prédit, les indices calculés et le niveau de performance.
    """
    model_path = os.path.join(MODELS_DIR, "rf_rendement.pkl")
    le_path    = os.path.join(MODELS_DIR, "label_encoder_culture.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Modèle non disponible. Lancez le pipeline ML.")

    model = joblib.load(model_path)
    le    = joblib.load(le_path)

    try:
        culture_enc = le.transform([data.culture])[0]
    except ValueError:
        raise HTTPException(status_code=400,
                            detail=f"Culture inconnue. Valeurs acceptées : {list(le.classes_)}")

    # Calcul indices
    ind_fertilite = (
        (data.ph_sol / 7.0) * 0.25 +
        min(data.azote_N_ppm / 50, 1.0) * 0.30 +
        min(data.phosphore_P_ppm / 40, 1.0) * 0.20 +
        min(data.potassium_K_ppm / 180, 1.0) * 0.15 +
        (data.matiere_organique_pct / 4.5) * 0.10
    )
    score_risque = (
        (data.jours_stress / data.duree_cycle_jours) * 0.5 +
        min(abs(data.bilan_hydrique) / 300, 1.0) * 0.5
    )

    X = pd.DataFrame([{
        "indice_fertilite_sol":       ind_fertilite,
        "score_risque_climatique":    score_risque,
        "meteo_temp_moy_cycle":       data.temp_moy_cycle,
        "meteo_pluie_totale_mm":      data.pluie_totale_mm,
        "meteo_bilan_hydrique":       data.bilan_hydrique,
        "meteo_jours_stress":         data.jours_stress,
        "meteo_ensoleillement_moy_h": data.ensoleillement_h,
        "duree_cycle_jours":          data.duree_cycle_jours,
        "engrais_kg_ha":              data.engrais_kg_ha,
        "eau_irrigation_m3_ha":       data.eau_irrigation_m3_ha,
        "ph_sol":                     data.ph_sol,
        "azote_N_ppm":                data.azote_N_ppm,
        "phosphore_P_ppm":            data.phosphore_P_ppm,
        "potassium_K_ppm":            data.potassium_K_ppm,
        "matiere_organique_pct":      data.matiere_organique_pct,
        "culture_enc":                culture_enc,
    }])

    rendement = round(float(model.predict(X)[0]), 3)

    if rendement < 5:       niveau = "Faible"
    elif rendement < 20:    niveau = "Moyen"
    else:                   niveau = "Élevé"

    return RendementOutput(
        culture=data.culture,
        rendement_predit_t_ha=rendement,
        indice_fertilite_sol=round(ind_fertilite, 4),
        score_risque_climatique=round(score_risque, 4),
        niveau_performance=niveau,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/marche/previsions", tags=["Marché"])
def get_previsions_prix(culture: Optional[str] = Query(None)):
    """Retourne les prévisions de prix Prophet sur 26 semaines."""
    path = os.path.join(MODELS_DIR, "previsions_prix.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Prévisions non disponibles. Lancez le pipeline ML.")
    df = pd.read_csv(path, parse_dates=["date"])
    if culture:
        df = df[df["culture"] == culture]
    return {"count": len(df), "data": df.round(0).to_dict(orient="records")}
