# 🌾 Ferme Agricole — Système Big Data Complet

> **Amélioration de la Production Agricole par l'Enrichissement des Données**
> Stack : Python · PySpark · Airflow · PostgreSQL · FastAPI · Streamlit · Docker

---

## 📁 Architecture du Projet

```
ferme_agricole/
│
├── docker-compose.yml           ← Infrastructure complète (4 services)
├── requirements.txt
│
├── docker/
│   ├── Dockerfile.api           ← Image FastAPI
│   └── Dockerfile.dashboard     ← Image Streamlit
│
├── postgres/
│   └── init/01_schema.sql       ← Schéma PostgreSQL + vues
│
├── airflow/
│   └── dags/pipeline_ferme.py   ← DAG orchestration complète
│
├── notebooks/
│   └── analyse_ml.ipynb         ← Notebook Jupyter (EDA + 4 modèles ML)
│
├── src/
│   ├── collecte/
│   │   └── collecte_donnees.py  ← API Open-Meteo + génération données
│   ├── enrichissement/
│   │   └── enrichissement_spark.py  ← ETL PySpark (production)
│   ├── ml/
│   │   └── modeles_ml.py        ← RF · K-Means · Prophet · IsoForest
│   ├── api/
│   │   └── main.py              ← FastAPI REST (8 endpoints)
│   └── dashboard/
│       └── dashboard.py         ← Streamlit 6 pages
│
└── data/
    ├── raw/                     ← CSV bruts (gitignore)
    ├── enriched/                ← CSV enrichis (gitignore)
    └── models/                  ← Modèles .pkl (gitignore)
```

---

## 🚀 Démarrage Rapide

### Docker Compose (seul mode supporté)

```bash
# 1. Lancer tous les services
docker compose up -d

# 2. Vérifier que tout tourne
docker compose ps

# 3. Déclencher le pipeline manuellement via Airflow UI
#    → http://localhost:8080  (admin / admin)
#    → DAG : ferme_agricole_pipeline → Trigger
```

**Services disponibles :**

| Service | URL | Identifiants |
|---------|-----|--------------|
| Airflow | http://localhost:8080 | admin / admin |
| FastAPI (Swagger) | http://localhost:8000/docs | — |
| Streamlit | http://localhost:8501 | — |
| PostgreSQL | localhost:5432 | ferme / ferme2024 |

> Le notebook ML (`notebooks/analyse_ml.ipynb`) tourne en local (Jupyter installé sur la machine, via `requirements.txt`), pas dans Docker. Il se connecte à PostgreSQL sur `localhost:5432`.

---

## 🔬 Description des Modules

### 📥 Collecte (`collecte_donnees.py`)
- **API Open-Meteo** : 2 ans de données météo réelles (temp, pluies, ETP, ensoleillement)
- Génération réaliste de 10 parcelles avec caractéristiques de sol
- 40 cycles culturaux avec rendements basés sur scores météo × sol
- 525 cotations de prix de marché avec saisonnalité

### ⚙️ Enrichissement ETL (`enrichissement_spark.py`)
```
Lecture CSV → SparkSession → Nettoyage → Cross-join conditionnel
→ Agrégation par cycle → Variables dérivées → Sauvegarde
```
Variables créées : `indice_fertilite_sol`, `score_risque_climatique`,
`efficience_eau`, `revenu_brut_fcfa_ha`, `marge_nette_fcfa_ha`,
`alerte_deficit_hydrique`, `alerte_stress_thermique`

### 🗄️ PostgreSQL (`01_schema.sql`)
Tables : `parcelles` · `meteo` · `production` · `dataset_enrichi` · `marche` · `alertes`
Vue : `v_performance_parcelle` (agrégation prête à l'emploi)

### 🔄 Airflow DAG (`pipeline_ferme.py`)
```
collecter_donnees → enrichir_donnees → charger_postgres → entrainer_modeles → generer_alertes
```
- Schedule : quotidien à 6h
- XCom entre tâches pour partage de métriques
- Retry automatique (2 tentatives, délai 5 min)

### 🤖 Machine Learning (`analyse_ml.ipynb`)

| Modèle | Objectif | Métriques |
|--------|----------|-----------|
| **Random Forest** | Prédire le rendement (t/ha) | R², MAE, RMSE, CV 5-fold |
| **K-Means** | Segmenter les parcelles | Elbow, Score Silhouette |
| **Prophet** | Prévoir les prix (26 sem.) | MAPE |
| **Isolation Forest** | Détecter les anomalies | Contamination 5% |

### 🌐 FastAPI (`main.py`)
```
GET  /                    → Health check
GET  /parcelles           → Liste parcelles
GET  /production          → Données enrichies (filtres: annee, culture, parcelle)
GET  /meteo               → Données météo
GET  /alertes             → Alertes actives
GET  /stats/performance   → KPIs globaux ferme
POST /predict/rendement   → Prédiction ML temps réel
GET  /marche/previsions   → Prévisions Prophet
```

### 📊 Streamlit Dashboard (`dashboard.py`)
6 pages : Vue d'ensemble · Météo & Alertes · Parcelles · ML · Marché · Simulateur

---

## 💰 Monétisation (SaaS)

| Horizon | Modèle | Description |
|---------|--------|-------------|
| Court terme | Consulting | Déploiement chez d'autres fermes |
| Moyen terme | **SaaS** | Abonnement dashboard (15k–35k FCFA/mois) |
| Long terme | **API Data** | Accès payant via token |
| Long terme | **Marketplace** | Commission mise en relation |

---

## 🛠️ Stack Technologique

```
Python 3.11    Langage principal
PySpark 3.5    ETL et traitement à grande échelle
Airflow 2.9    Orchestration du pipeline (DAG)
PostgreSQL 16  Stockage structuré des données enrichies
FastAPI        API REST (SaaS + intégrations)
Streamlit      Dashboard interactif
Docker         Containerisation et déploiement
Scikit-learn   Random Forest · K-Means · Isolation Forest
Prophet        Prévisions séries temporelles
```

---

*Projet Big Data — Agriculture de Précision | UCAO*
