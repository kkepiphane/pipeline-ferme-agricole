"""
════════════════════════════════════════════════════════════════
MODULE ETL — ENRICHISSEMENT PYSPARK
════════════════════════════════════════════════════════════════
Pipeline de transformation des données brutes vers
le dataset enrichi final, via Apache Spark.
════════════════════════════════════════════════════════════════
"""

import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, DateType
)
from pyspark.sql.window import Window

DATA_RAW      = os.path.join(os.path.dirname(__file__), "../../data/raw")
DATA_ENRICHED = os.path.join(os.path.dirname(__file__), "../../data/enriched")
os.makedirs(DATA_ENRICHED, exist_ok=True)


def creer_spark_session():
    """Initialise la SparkSession avec config optimisée."""
    spark = (
        SparkSession.builder
        .appName("FermeAgricole-ETL")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"  ✓ Spark {spark.version} initialisé")
    return spark


def charger_donnees_brutes(spark):
    """Charge les CSV bruts dans des DataFrames Spark."""
    print("[SPARK] Chargement des données brutes...")

    meteo = spark.read.csv(
        os.path.join(DATA_RAW, "meteo.csv"),
        header=True, inferSchema=True
    )
    parcelles = spark.read.csv(
        os.path.join(DATA_RAW, "parcelles.csv"),
        header=True, inferSchema=True
    )
    production = spark.read.csv(
        os.path.join(DATA_RAW, "production.csv"),
        header=True, inferSchema=True
    )
    marche = spark.read.csv(
        os.path.join(DATA_RAW, "marche.csv"),
        header=True, inferSchema=True
    )

    print(f"  Météo      : {meteo.count()} lignes")
    print(f"  Parcelles  : {parcelles.count()} lignes")
    print(f"  Production : {production.count()} lignes")
    print(f"  Marché     : {marche.count()} lignes")

    return meteo, parcelles, production, marche


def nettoyer_meteo_spark(meteo):
    """Nettoyage et validation des données météo avec Spark."""
    print("[SPARK] Nettoyage météo...")

    # Suppression des doublons
    meteo = meteo.dropDuplicates(["date"])

    # Correction valeurs aberrantes
    meteo = meteo.withColumn(
        "temp_max", F.when(F.col("temp_max") > 50, 50).otherwise(F.col("temp_max"))
    ).withColumn(
        "temp_min", F.when(F.col("temp_min") < 0, 0).otherwise(F.col("temp_min"))
    ).withColumn(
        "precipitation_mm",
        F.when(F.col("precipitation_mm") < 0, 0).otherwise(F.col("precipitation_mm"))
    )

    # Imputation des nulls par la médiane (via approxQuantile)
    cols_num = ["temp_max", "temp_min", "temp_moy", "precipitation_mm", "evapotranspiration"]
    for col in cols_num:
        mediane = meteo.approxQuantile(col, [0.5], 0.01)[0]
        meteo = meteo.withColumn(
            col, F.when(F.col(col).isNull(), mediane).otherwise(F.col(col))
        )

    nb_apres = meteo.count()
    print(f"  ✓ Météo nettoyée : {nb_apres} lignes valides")
    return meteo


def agreger_meteo_par_cycle_spark(production, meteo):
    """
    Agrégation météo sur la période de chaque cycle cultural.
    Utilise un cross-join filtré (join conditionnel sur dates).
    """
    print("[SPARK] Agrégation météo par cycle cultural...")

    # Cast des colonnes date
    production = production.withColumn("date_semis",   F.to_date("date_semis")) \
                           .withColumn("date_recolte", F.to_date("date_recolte"))
    meteo      = meteo.withColumn("date", F.to_date("date"))

    # Cross join + filtre période
    prod_alias  = production.alias("prod")
    meteo_alias = meteo.alias("met")

    joined = prod_alias.join(
        meteo_alias,
        (F.col("met.date") >= F.col("prod.date_semis")) &
        (F.col("met.date") <= F.col("prod.date_recolte")),
        "left"
    )

    meteo_agg = joined.groupBy(
        "prod.parcelle_id", "prod.annee", "prod.saison"
    ).agg(
        F.round(F.avg("met.temp_moy"),         2).alias("meteo_temp_moy_cycle"),
        F.round(F.max("met.temp_max"),          2).alias("meteo_temp_max_max"),
        F.round(F.sum("met.stress_thermique"),  0).cast(IntegerType()).alias("meteo_jours_stress"),
        F.round(F.sum("met.precipitation_mm"),  1).alias("meteo_pluie_totale_mm"),
        F.round(F.avg("met.precipitation_mm"),  2).alias("meteo_pluie_moy_mm"),
        F.round(F.sum("met.bilan_hydrique"),    1).alias("meteo_bilan_hydrique"),
        F.round(F.sum("met.evapotranspiration"),1).alias("meteo_etp_totale"),
        F.round(F.avg("met.ensoleillement_h"),  2).alias("meteo_ensoleillement_moy_h"),
    )

    print(f"  ✓ {meteo_agg.count()} agrégations météo produites")
    return meteo_agg


def enrichir_dataset_spark(production, parcelles, meteo_agg, marche):
    """Fusion et création des variables enrichies avec Spark SQL."""
    print("[SPARK] Enrichissement et création des variables dérivées...")

    # Prix moyen par culture × année
    marche = marche.withColumn("annee", F.year(F.to_date("semaine")))
    prix_moy = marche.groupBy("culture", "annee").agg(
        F.round(F.avg("prix_fcfa_kg"), 0).alias("prix_moy_fcfa_kg")
    )

    # Jointures
    df = production.join(parcelles, on="parcelle_id", how="left") \
                   .join(meteo_agg,  on=["parcelle_id", "annee", "saison"], how="left") \
                   .join(prix_moy,   on=["culture", "annee"], how="left")

    # ── Variables enrichies ──────────────────────────────────
    df = df.withColumn(
        "indice_fertilite_sol",
        F.round(
            (F.col("ph_sol") / 7.0) * 0.25 +
            F.least(F.col("azote_N_ppm") / 50, F.lit(1.0)) * 0.30 +
            F.least(F.col("phosphore_P_ppm") / 40, F.lit(1.0)) * 0.20 +
            F.least(F.col("potassium_K_ppm") / 180, F.lit(1.0)) * 0.15 +
            (F.col("matiere_organique_pct") / 4.5) * 0.10,
            4
        )
    ).withColumn(
        "score_risque_climatique",
        F.round(
            (F.col("meteo_jours_stress").cast(DoubleType()) /
             F.datediff(F.to_date("date_recolte"), F.to_date("date_semis")).cast(DoubleType())) * 0.5 +
            F.least(F.abs(F.col("meteo_bilan_hydrique")) / 300, F.lit(1.0)) * 0.5,
            4
        )
    ).withColumn(
        "efficience_eau",
        F.round(
            F.when(F.col("meteo_pluie_totale_mm") > 0,
                   F.col("rendement_t_ha") / (F.col("meteo_pluie_totale_mm") / 100))
            .otherwise(F.lit(None).cast(DoubleType())),
            4
        )
    ).withColumn(
        "efficience_engrais",
        F.round(F.col("rendement_t_ha") / (F.col("engrais_kg_ha") / 100), 4)
    ).withColumn(
        "revenu_brut_fcfa_ha",
        F.round(F.col("rendement_t_ha") * 1000 * F.col("prix_moy_fcfa_kg"), 0)
    ).withColumn(
        "cout_intrants_fcfa_ha",
        F.round(
            F.col("engrais_kg_ha") * 350 +
            F.col("eau_irrigation_m3_ha") * 50 +
            F.col("pesticide_L_ha") * 2500,
            0
        )
    ).withColumn(
        "marge_nette_fcfa_ha",
        F.round(F.col("revenu_brut_fcfa_ha") - F.col("cout_intrants_fcfa_ha"), 0)
    ).withColumn(
        "alerte_deficit_hydrique",
        F.when(F.col("meteo_bilan_hydrique") < -100, 1).otherwise(0)
    ).withColumn(
        "alerte_stress_thermique",
        F.when(F.col("meteo_jours_stress") > 10, 1).otherwise(0)
    )

    nb_lignes  = df.count()
    nb_colonnes = len(df.columns)
    print(f"  ✓ Dataset enrichi : {nb_lignes} lignes × {nb_colonnes} colonnes")
    print(f"  ✓ Alertes déficit hydrique  : {df.filter(F.col('alerte_deficit_hydrique') == 1).count()}")
    print(f"  ✓ Alertes stress thermique  : {df.filter(F.col('alerte_stress_thermique') == 1).count()}")
    return df, nb_lignes, nb_colonnes


def sauvegarder_enrichi(df_spark, meteo_spark):
    """Conversion Spark → Pandas pour sauvegarde CSV."""
    print("[SPARK] Sauvegarde des données enrichies...")

    # Dataset principal
    df_pandas = df_spark.toPandas()
    df_pandas.to_csv(os.path.join(DATA_ENRICHED, "dataset_enrichi.csv"), index=False)

    # Météo nettoyée
    meteo_pandas = meteo_spark.toPandas()
    meteo_pandas.to_csv(os.path.join(DATA_ENRICHED, "meteo_nettoyee.csv"), index=False)

    # Météo hebdo (agrégation Pandas pour le dashboard)
    meteo_pandas["date"] = pd.to_datetime(meteo_pandas["date"])
    meteo_pandas["semaine_debut"] = meteo_pandas["date"] - pd.to_timedelta(
        meteo_pandas["date"].dt.dayofweek, unit="d"
    )
    hebdo = meteo_pandas.groupby("semaine_debut").agg(
        temp_moy=("temp_moy","mean"), temp_max=("temp_max","max"),
        temp_min=("temp_min","min"), pluie_totale=("precipitation_mm","sum"),
        jours_stress=("stress_thermique","sum"), bilan_hydrique=("bilan_hydrique","sum"),
    ).reset_index().round(2)
    hebdo.to_csv(os.path.join(DATA_ENRICHED, "meteo_hebdo.csv"), index=False)

    print(f"  ✓ dataset_enrichi.csv  — {len(df_pandas)} lignes")
    print(f"  ✓ meteo_nettoyee.csv   — {len(meteo_pandas)} lignes")
    print(f"  ✓ meteo_hebdo.csv      — {len(hebdo)} semaines")


def enrichir_avec_spark():
    """Point d'entrée principal — retourne (nb_lignes, nb_colonnes)."""
    print("\n" + "="*60)
    print("  PIPELINE SPARK — ENRICHISSEMENT DES DONNÉES")
    print("="*60)

    spark = creer_spark_session()
    try:
        meteo, parcelles, production, marche = charger_donnees_brutes(spark)
        meteo      = nettoyer_meteo_spark(meteo)
        meteo_agg  = agreger_meteo_par_cycle_spark(production, meteo)
        df_enrich, nb_lignes, nb_colonnes = enrichir_dataset_spark(
            production, parcelles, meteo_agg, marche
        )
        sauvegarder_enrichi(df_enrich, meteo)
        print("  → Enrichissement Spark terminé ✓\n")
        return nb_lignes, nb_colonnes
    finally:
        spark.stop()


if __name__ == "__main__":
    enrichir_avec_spark()
