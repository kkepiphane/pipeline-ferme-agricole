"""
════════════════════════════════════════════════════════════════
DAG AIRFLOW — Pipeline Ferme Agricole Big Data
════════════════════════════════════════════════════════════════
Orchestration complète :
  collect_data → enrich_data → train_models → load_to_postgres → generate_alerts

Schedule : quotidien à 6h (données météo fresh)
════════════════════════════════════════════════════════════════
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging

log = logging.getLogger(__name__)

# ── Paramètres du DAG ────────────────────────────────────────
default_args = {
    "owner":            "ferme-bigdata",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

dag = DAG(
    dag_id="ferme_agricole_pipeline",
    default_args=default_args,
    description="Pipeline Big Data — Collecte, Enrichissement, ML, Alertes",
    schedule_interval="0 6 * * *",       # Tous les jours à 6h
    start_date=days_ago(1),
    catchup=False,
    tags=["ferme", "bigdata", "agriculture"],
)


# ════════════════════════════════════════════════════════════════
# TÂCHE 1 — Collecte des données
# ════════════════════════════════════════════════════════════════
def task_collecter(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from collecte.collecte_donnees import collecter_toutes_donnees

    log.info("═" * 50)
    log.info("TÂCHE 1 : Collecte des données")
    log.info("═" * 50)

    df_meteo, df_parcelles, df_production, df_marche = collecter_toutes_donnees()

    # Partager les stats via XCom pour la tâche suivante
    context["ti"].xcom_push(key="nb_jours_meteo",   value=len(df_meteo))
    context["ti"].xcom_push(key="nb_cycles_prod",   value=len(df_production))
    context["ti"].xcom_push(key="nb_cotations_marche", value=len(df_marche))

    log.info(f"Collecte OK — météo:{len(df_meteo)} prod:{len(df_production)}")


# ════════════════════════════════════════════════════════════════
# TÂCHE 2 — Enrichissement ETL (PySpark)
# ════════════════════════════════════════════════════════════════
def task_enrichir(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from enrichissement.enrichissement_spark import enrichir_avec_spark

    log.info("═" * 50)
    log.info("TÂCHE 2 : Enrichissement PySpark")
    log.info("═" * 50)

    nb_jours = context["ti"].xcom_pull(key="nb_jours_meteo", task_ids="collecter_donnees")
    log.info(f"Enrichissement sur base de {nb_jours} jours météo")

    nb_lignes, nb_colonnes = enrichir_avec_spark()
    context["ti"].xcom_push(key="enrichi_shape", value=f"{nb_lignes}×{nb_colonnes}")
    log.info(f"Enrichissement OK — {nb_lignes} lignes × {nb_colonnes} colonnes")


# ════════════════════════════════════════════════════════════════
# TÂCHE 3 — Chargement PostgreSQL
# ════════════════════════════════════════════════════════════════
def task_charger_postgres(**context):
    import sys, os
    sys.path.insert(0, "/opt/airflow/src")
    import pandas as pd
    from sqlalchemy import create_engine

    log.info("═" * 50)
    log.info("TÂCHE 3 : Chargement PostgreSQL")
    log.info("═" * 50)

    DB_URL = (
        f"postgresql://ferme:ferme2024@postgres:5432/ferme_agricole"
    )
    engine = create_engine(DB_URL)

    DATA_ENRICHED = "/opt/airflow/data/enriched"
    DATA_RAW      = "/opt/airflow/data/raw"

    tables = {
        "meteo":           (os.path.join(DATA_ENRICHED, "meteo_nettoyee.csv"),     "replace"),
        "dataset_enrichi": (os.path.join(DATA_ENRICHED, "dataset_enrichi.csv"),   "replace"),
        "marche":          (os.path.join(DATA_RAW,      "marche.csv"),             "replace"),
    }

    for table, (path, mode) in tables.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.to_sql(table, engine, if_exists=mode, index=False)
            log.info(f"  ✓ Table {table} chargée — {len(df)} lignes")
        else:
            log.warning(f"  ✗ Fichier non trouvé : {path}")

    log.info("Chargement PostgreSQL OK")


# ════════════════════════════════════════════════════════════════
# TÂCHE 4 — Entraînement des modèles ML
# ════════════════════════════════════════════════════════════════
def task_entrainer_ml(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from ml.modeles_ml import entrainer_tous_modeles

    log.info("═" * 50)
    log.info("TÂCHE 4 : Entraînement des modèles ML")
    log.info("═" * 50)

    _, _, _, _, _, metriques = entrainer_tous_modeles()

    context["ti"].xcom_push(key="rf_r2",  value=metriques["R2"])
    context["ti"].xcom_push(key="rf_mae", value=metriques["MAE"])
    log.info(f"ML OK — R²={metriques['R2']}  MAE={metriques['MAE']}")


# ════════════════════════════════════════════════════════════════
# TÂCHE 5 — Génération des alertes
# ════════════════════════════════════════════════════════════════
def task_generer_alertes(**context):
    import sys, os
    sys.path.insert(0, "/opt/airflow/src")
    import pandas as pd
    from sqlalchemy import create_engine, text
    from datetime import datetime

    log.info("═" * 50)
    log.info("TÂCHE 5 : Génération des alertes")
    log.info("═" * 50)

    engine = create_engine("postgresql://ferme:ferme2024@postgres:5432/ferme_agricole")

    df = pd.read_csv("/opt/airflow/data/enriched/dataset_enrichi.csv")
    alertes = []

    for _, row in df.iterrows():
        if row.get("alerte_deficit_hydrique", 0) == 1:
            alertes.append({
                "parcelle_id": row["parcelle_id"],
                "type_alerte": "DEFICIT_HYDRIQUE",
                "severite":    "CRITIQUE",
                "message":     f"Bilan hydrique de {row['meteo_bilan_hydrique']:.1f} mm — irrigation urgente",
                "valeur":      row["meteo_bilan_hydrique"],
                "seuil":       -100.0,
                "date_alerte": datetime.now(),
                "resolue":     False,
            })
        if row.get("alerte_stress_thermique", 0) == 1:
            alertes.append({
                "parcelle_id": row["parcelle_id"],
                "type_alerte": "STRESS_THERMIQUE",
                "severite":    "AVERTISSEMENT",
                "message":     f"{row['meteo_jours_stress']} jours de stress thermique détectés",
                "valeur":      row["meteo_jours_stress"],
                "seuil":       10.0,
                "date_alerte": datetime.now(),
                "resolue":     False,
            })

    if alertes:
        df_alertes = pd.DataFrame(alertes)
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE alertes"))
            conn.commit()
        df_alertes.to_sql("alertes", engine, if_exists="append", index=False)
        log.info(f"  ✓ {len(alertes)} alertes générées et stockées en DB")
    else:
        log.info("  ✓ Aucune alerte active")

    r2  = context["ti"].xcom_pull(key="rf_r2",  task_ids="entrainer_modeles")
    mae = context["ti"].xcom_pull(key="rf_mae", task_ids="entrainer_modeles")
    log.info(f"\n{'═'*50}")
    log.info("PIPELINE TERMINÉ AVEC SUCCÈS")
    log.info(f"  Random Forest R² : {r2}")
    log.info(f"  Random Forest MAE: {mae} t/ha")
    log.info(f"  Alertes générées : {len(alertes)}")
    log.info("═" * 50)


# ════════════════════════════════════════════════════════════════
# DÉFINITION DES TÂCHES ET DÉPENDANCES
# ════════════════════════════════════════════════════════════════
t1_collecter = PythonOperator(
    task_id="collecter_donnees",
    python_callable=task_collecter,
    dag=dag,
)

t2_enrichir = PythonOperator(
    task_id="enrichir_donnees",
    python_callable=task_enrichir,
    dag=dag,
)

t3_postgres = PythonOperator(
    task_id="charger_postgres",
    python_callable=task_charger_postgres,
    dag=dag,
)

t4_ml = PythonOperator(
    task_id="entrainer_modeles",
    python_callable=task_entrainer_ml,
    dag=dag,
)

t5_alertes = PythonOperator(
    task_id="generer_alertes",
    python_callable=task_generer_alertes,
    dag=dag,
)

# ── Chaîne d'exécution ───────────────────────────────────────
#   collect → enrich → postgres → ml → alertes
t1_collecter >> t2_enrichir >> t3_postgres >> t4_ml >> t5_alertes
