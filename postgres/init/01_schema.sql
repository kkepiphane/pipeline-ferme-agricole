-- ═══════════════════════════════════════════════════════════
--  FERME AGRICOLE — SCHÉMA POSTGRESQL
--  Tables : parcelles, meteo, production, marche, alertes
-- ═══════════════════════════════════════════════════════════

-- Créer la base Airflow séparément
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO ferme;

\c ferme_agricole;

-- ── Table parcelles ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS parcelles (
    parcelle_id       VARCHAR(10) PRIMARY KEY,
    superficie_ha     NUMERIC(6,2),
    culture           VARCHAR(50),
    type_sol          VARCHAR(50),
    ph_sol            NUMERIC(4,2),
    azote_N_ppm       NUMERIC(6,1),
    phosphore_P_ppm   NUMERIC(6,1),
    potassium_K_ppm   NUMERIC(6,1),
    matiere_organique_pct NUMERIC(5,2),
    capacite_retention_eau NUMERIC(5,3),
    annee_derniere_analyse INTEGER,
    created_at        TIMESTAMP DEFAULT NOW()
);

-- ── Table meteo ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS meteo (
    id               SERIAL PRIMARY KEY,
    date             DATE NOT NULL UNIQUE,
    temp_max         NUMERIC(5,2),
    temp_min         NUMERIC(5,2),
    temp_moy         NUMERIC(5,2),
    precipitation_mm NUMERIC(7,2),
    heures_pluie     NUMERIC(5,2),
    evapotranspiration NUMERIC(6,2),
    ensoleillement_h NUMERIC(5,2),
    vent_max_kmh     NUMERIC(5,2),
    stress_thermique INTEGER DEFAULT 0,
    bilan_hydrique   NUMERIC(7,2),
    mois             INTEGER,
    semaine          INTEGER,
    saison           VARCHAR(50)
);

-- ── Table production ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS production (
    id                    SERIAL PRIMARY KEY,
    parcelle_id           VARCHAR(10) REFERENCES parcelles(parcelle_id),
    culture               VARCHAR(50),
    annee                 INTEGER,
    saison                VARCHAR(50),
    date_semis            DATE,
    date_recolte          DATE,
    rendement_t_ha        NUMERIC(8,3),
    production_totale_t   NUMERIC(10,3),
    engrais_kg_ha         NUMERIC(8,2),
    eau_irrigation_m3_ha  NUMERIC(8,2),
    pesticide_L_ha        NUMERIC(8,3),
    score_meteo           NUMERIC(5,3),
    score_stress_thermique NUMERIC(5,3),
    score_sol             NUMERIC(5,3),
    created_at            TIMESTAMP DEFAULT NOW()
);

-- ── Table dataset_enrichi ────────────────────────────────────
CREATE TABLE IF NOT EXISTS dataset_enrichi (
    id                        SERIAL PRIMARY KEY,
    parcelle_id               VARCHAR(10),
    culture                   VARCHAR(50),
    annee                     INTEGER,
    saison                    VARCHAR(50),
    rendement_t_ha            NUMERIC(8,3),
    production_totale_t       NUMERIC(10,3),
    indice_fertilite_sol      NUMERIC(6,4),
    score_risque_climatique   NUMERIC(6,4),
    efficience_eau            NUMERIC(8,4),
    efficience_engrais        NUMERIC(8,4),
    revenu_brut_fcfa_ha       NUMERIC(12,0),
    cout_intrants_fcfa_ha     NUMERIC(12,0),
    marge_nette_fcfa_ha       NUMERIC(12,0),
    alerte_deficit_hydrique   INTEGER DEFAULT 0,
    alerte_stress_thermique   INTEGER DEFAULT 0,
    categorie_performance     VARCHAR(20),
    meteo_pluie_totale_mm     NUMERIC(8,2),
    meteo_bilan_hydrique      NUMERIC(8,2),
    meteo_jours_stress        INTEGER,
    meteo_temp_moy_cycle      NUMERIC(5,2),
    created_at                TIMESTAMP DEFAULT NOW()
);

-- ── Table marche ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS marche (
    id              SERIAL PRIMARY KEY,
    semaine         DATE,
    culture         VARCHAR(50),
    prix_fcfa_kg    NUMERIC(8,0),
    demande_index   NUMERIC(5,2)
);

-- ── Table alertes ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alertes (
    id              SERIAL PRIMARY KEY,
    parcelle_id     VARCHAR(10),
    type_alerte     VARCHAR(50),
    severite        VARCHAR(20),
    message         TEXT,
    valeur          NUMERIC(10,3),
    seuil           NUMERIC(10,3),
    date_alerte     TIMESTAMP DEFAULT NOW(),
    resolue         BOOLEAN DEFAULT FALSE
);

-- ── Index pour les requêtes fréquentes ───────────────────────
CREATE INDEX IF NOT EXISTS idx_meteo_date        ON meteo(date);
CREATE INDEX IF NOT EXISTS idx_production_parcelle ON production(parcelle_id);
CREATE INDEX IF NOT EXISTS idx_enrichi_culture   ON dataset_enrichi(culture);
CREATE INDEX IF NOT EXISTS idx_enrichi_annee     ON dataset_enrichi(annee);
CREATE INDEX IF NOT EXISTS idx_alertes_parcelle  ON alertes(parcelle_id);
CREATE INDEX IF NOT EXISTS idx_marche_culture    ON marche(culture);

-- ── Vue synthèse performance ──────────────────────────────────
CREATE OR REPLACE VIEW v_performance_parcelle AS
SELECT
    parcelle_id,
    culture,
    annee,
    AVG(rendement_t_ha)           AS rendement_moy,
    AVG(marge_nette_fcfa_ha)      AS marge_moy,
    AVG(indice_fertilite_sol)     AS fertilite_moy,
    SUM(alerte_deficit_hydrique)  AS nb_alertes_hydrique,
    SUM(alerte_stress_thermique)  AS nb_alertes_thermique
FROM dataset_enrichi
GROUP BY parcelle_id, culture, annee;

GRANT ALL PRIVILEGES ON ALL TABLES    IN SCHEMA public TO ferme;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ferme;
