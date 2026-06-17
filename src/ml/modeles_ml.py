"""
========================================================
MODULE 3 : MODÈLES MACHINE LEARNING
========================================================
Modèles :
  1. Random Forest — Prédiction du rendement
  2. K-Means       — Segmentation des parcelles
  3. Prophet       — Prévision des prix de marché
  4. Isolation Forest — Détection d'anomalies
========================================================
"""

import pandas as pd
import numpy as np
import os, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_ENRICHED = os.path.join(os.path.dirname(__file__), "../../data/enriched")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "../../data/models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════
# 1. RANDOM FOREST — Prédiction du rendement
# ══════════════════════════════════════════════════════
def entrainer_random_forest(df):
    print("[ML] Entraînement Random Forest — Prédiction rendement...")

    FEATURES = [
        "indice_fertilite_sol", "score_risque_climatique",
        "meteo_temp_moy_cycle", "meteo_pluie_totale_mm",
        "meteo_bilan_hydrique", "meteo_jours_stress",
        "meteo_ensoleillement_moy_h", "duree_cycle_jours",
        "engrais_kg_ha", "eau_irrigation_m3_ha",
        "ph_sol", "azote_N_ppm", "phosphore_P_ppm",
        "potassium_K_ppm", "matiere_organique_pct",
    ]
    TARGET = "rendement_t_ha"

    # Encodage de la culture
    le = LabelEncoder()
    df = df.copy()
    df["culture_enc"] = le.fit_transform(df["culture"])
    FEATURES.append("culture_enc")

    df_ml = df[FEATURES + [TARGET]].dropna()
    X, y  = df_ml[FEATURES], df_ml[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    print(f"  MAE  : {mae:.3f} t/ha")
    print(f"  RMSE : {rmse:.3f} t/ha")
    print(f"  R²   : {r2:.3f}")
    print(f"  R² CV (5-fold) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Importance des variables
    importances = pd.DataFrame({
        "feature":    FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # Sauvegarde
    joblib.dump(model, os.path.join(MODELS_DIR, "rf_rendement.pkl"))
    joblib.dump(le,    os.path.join(MODELS_DIR, "label_encoder_culture.pkl"))
    importances.to_csv(os.path.join(MODELS_DIR, "rf_importances.csv"), index=False)

    metriques = {"MAE": round(mae,3), "RMSE": round(rmse,3),
                 "R2": round(r2,3), "R2_CV": round(cv_scores.mean(),3)}

    resultats = pd.DataFrame({
        "rendement_reel": y_test.values,
        "rendement_predit": y_pred.round(2)
    })
    resultats.to_csv(os.path.join(MODELS_DIR, "rf_predictions.csv"), index=False)

    print(f"  ✓ Modèle sauvegardé → rf_rendement.pkl")
    return model, importances, metriques


# ══════════════════════════════════════════════════════
# 2. K-MEANS — Segmentation des parcelles
# ══════════════════════════════════════════════════════
def segmenter_parcelles(df):
    print("[ML] K-Means — Segmentation des parcelles...")

    # Agrégation par parcelle
    agg = df.groupby("parcelle_id").agg(
        rendement_moy=("rendement_t_ha", "mean"),
        rendement_std=("rendement_t_ha", "std"),
        marge_moy=("marge_nette_fcfa_ha", "mean"),
        indice_sol=("indice_fertilite_sol", "mean"),
        risque_moy=("score_risque_climatique", "mean"),
        nb_alertes=("alerte_deficit_hydrique", "sum"),
        culture=("culture", "first"),
    ).reset_index().fillna(0)

    features_seg = ["rendement_moy", "rendement_std", "marge_moy",
                    "indice_sol", "risque_moy", "nb_alertes"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg[features_seg])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    agg["segment"] = kmeans.fit_predict(X_scaled)

    labels = {0: "Haute Performance", 1: "Performance Moyenne", 2: "Sous-Performance"}
    # Réordonner les segments selon le rendement moyen
    ordre = agg.groupby("segment")["rendement_moy"].mean().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(ordre.index)}
    agg["segment"] = agg["segment"].map(mapping)
    agg["segment_label"] = agg["segment"].map(labels)

    agg.to_csv(os.path.join(MODELS_DIR, "segmentation_parcelles.csv"), index=False)
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_parcelles.pkl"))

    print(f"  Segments détectés :")
    for seg, label in labels.items():
        n = (agg["segment"] == seg).sum()
        rend = agg[agg["segment"] == seg]["rendement_moy"].mean()
        print(f"    Segment {seg} — {label} : {n} parcelles, rendement moy. {rend:.2f} t/ha")

    print(f"  ✓ Segmentation sauvegardée → segmentation_parcelles.csv")
    return agg


# ══════════════════════════════════════════════════════
# 3. PROPHET — Prévision des prix de marché
# ══════════════════════════════════════════════════════
def prevoir_prix_marche():
    print("[ML] Prophet — Prévision des prix de marché...")

    try:
        from prophet import Prophet
    except ImportError:
        print("  ✗ Prophet non installé. Pip install prophet.")
        return None

    df_marche = pd.read_csv(os.path.join(DATA_ENRICHED, "marche.csv"),
                            parse_dates=["semaine"])
    cultures  = df_marche["culture"].unique()
    previsions_all = []

    for culture in cultures:
        df_c = df_marche[df_marche["culture"] == culture][["semaine", "prix_fcfa_kg"]].copy()
        df_c.columns = ["ds", "y"]
        df_c = df_c.dropna().sort_values("ds")

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.1,
        )
        model.fit(df_c)

        future   = model.make_future_dataframe(periods=26, freq="W")
        forecast = model.predict(future)

        forecast["culture"] = culture
        previsions_all.append(forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "culture"]].tail(26))

    df_prev = pd.concat(previsions_all, ignore_index=True)
    df_prev.columns = ["date", "prix_predit", "prix_min", "prix_max", "culture"]
    df_prev.to_csv(os.path.join(MODELS_DIR, "previsions_prix.csv"), index=False)

    print(f"  ✓ Prévisions 26 semaines pour {len(cultures)} cultures → previsions_prix.csv")
    return df_prev


# ══════════════════════════════════════════════════════
# 4. ISOLATION FOREST — Détection d'anomalies
# ══════════════════════════════════════════════════════
def detecter_anomalies(df):
    print("[ML] Isolation Forest — Détection d'anomalies...")

    features = ["rendement_t_ha", "engrais_kg_ha", "eau_irrigation_m3_ha",
                "meteo_pluie_totale_mm", "meteo_jours_stress"]
    df_ano = df[features].dropna()

    iso = IsolationForest(contamination=0.05, random_state=42)
    predictions = iso.fit_predict(df_ano)

    df_result = df.loc[df_ano.index].copy()
    df_result["anomalie"] = (predictions == -1).astype(int)
    df_result["score_anomalie"] = iso.score_samples(df_ano).round(4)

    nb_anomalies = df_result["anomalie"].sum()
    df_result.to_csv(os.path.join(MODELS_DIR, "anomalies_detectees.csv"), index=False)

    print(f"  ✓ {nb_anomalies} anomalies détectées sur {len(df_result)} cycles "
          f"({nb_anomalies/len(df_result)*100:.1f}%)")
    return df_result


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def entrainer_tous_modeles():
    print("\n" + "="*60)
    print("  PIPELINE — ÉTAPE 3 : MODÈLES MACHINE LEARNING")
    print("="*60)

    df = pd.read_csv(os.path.join(DATA_ENRICHED, "dataset_enrichi.csv"),
                     parse_dates=["date_semis", "date_recolte"])

    rf_model, importances, metriques_rf = entrainer_random_forest(df)
    df_segments = segmenter_parcelles(df)
    df_prev     = prevoir_prix_marche()
    df_anomalies = detecter_anomalies(df)

    print("\n[RÉSUMÉ ML]")
    print(f"  Random Forest R²   : {metriques_rf['R2']}")
    print(f"  Random Forest MAE  : {metriques_rf['MAE']} t/ha")
    print(f"  Segments parcelles : 3 clusters identifiés")
    print(f"  Anomalies détectées: {df_anomalies['anomalie'].sum()}")
    print("  → Modèles ML entraînés ✓\n")

    return rf_model, df_segments, df_prev, df_anomalies, importances, metriques_rf


if __name__ == "__main__":
    entrainer_tous_modeles()
