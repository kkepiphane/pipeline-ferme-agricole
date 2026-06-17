"""
========================================================
MODULE 4 : DASHBOARD STREAMLIT
========================================================
Interface interactive de pilotage agricole :
  - Vue d'ensemble de la ferme
  - Analyse météo et alertes
  - Performance des parcelles
  - Prévisions et ML
  - Simulateur de rendement
========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, joblib, warnings
warnings.filterwarnings("ignore")

# ─── CHEMINS ─────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
ENRICHED = os.path.join(BASE, "../../data/enriched")
RAW      = os.path.join(BASE, "../../data/raw")
MODELS   = os.path.join(BASE, "../../data/models")

# ─── CONFIG PAGE ─────────────────────────────────────
st.set_page_config(
    page_title="FermeData — Tableau de Bord",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS CUSTOM ──────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: linear-gradient(135deg, #1F5C2E, #2E8B57);
    border-radius: 12px; padding: 18px; color: white;
    text-align: center; margin: 4px;
  }
  .metric-val  { font-size: 2em; font-weight: bold; }
  .metric-lbl  { font-size: 0.85em; opacity: 0.85; margin-top: 4px; }
  .alert-red   { background: #fee2e2; border-left: 4px solid #dc2626;
                  padding: 10px 16px; border-radius: 6px; margin: 6px 0; }
  .alert-orange{ background: #ffedd5; border-left: 4px solid #ea580c;
                  padding: 10px 16px; border-radius: 6px; margin: 6px 0; }
  .alert-green { background: #dcfce7; border-left: 4px solid #16a34a;
                  padding: 10px 16px; border-radius: 6px; margin: 6px 0; }
  .section-title { font-size: 1.3em; font-weight: 700; color: #1F5C2E;
                    border-bottom: 2px solid #2E8B57; padding-bottom: 6px; margin: 20px 0 12px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES (mis en cache)
# ══════════════════════════════════════════════════════
@st.cache_data
def charger_donnees():
    df       = pd.read_csv(os.path.join(ENRICHED, "dataset_enrichi.csv"),
                           parse_dates=["date_semis", "date_recolte"])
    meteo    = pd.read_csv(os.path.join(ENRICHED, "meteo_nettoyee.csv"), parse_dates=["date"])
    hebdo    = pd.read_csv(os.path.join(ENRICHED, "meteo_hebdo.csv"), parse_dates=["semaine_debut"])
    parcelles= pd.read_csv(os.path.join(RAW, "parcelles.csv"))
    marche   = pd.read_csv(os.path.join(RAW, "marche.csv"), parse_dates=["semaine"])
    return df, meteo, hebdo, parcelles, marche

@st.cache_data
def charger_resultats_ml():
    out = {}
    for f, key in [("rf_importances.csv","importances"), ("rf_predictions.csv","predictions"),
                   ("segmentation_parcelles.csv","segments"), ("anomalies_detectees.csv","anomalies")]:
        path = os.path.join(MODELS, f)
        if os.path.exists(path):
            out[key] = pd.read_csv(path)
    path_prev = os.path.join(MODELS, "previsions_prix.csv")
    if os.path.exists(path_prev):
        out["previsions"] = pd.read_csv(path_prev, parse_dates=["date"])
    path_m = os.path.join(MODELS, "rf_rendement.pkl")
    if os.path.exists(path_m):
        out["rf_model"] = joblib.load(path_m)
        out["le_culture"] = joblib.load(os.path.join(MODELS, "label_encoder_culture.pkl"))
    return out


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
def sidebar():
    st.sidebar.image("https://img.icons8.com/color/96/farm.png", width=80)
    st.sidebar.title("🌾 FermeData")
    st.sidebar.caption("Pilotage agricole par la donnée")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigation", [
        "🏠 Vue d'ensemble",
        "🌦️ Météo & Alertes",
        "📦 Parcelles & Production",
        "🤖 Analyse ML",
        "💰 Marché & Rentabilité",
        "🔮 Simulateur",
    ])
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Ferme Pilote — Lomé, Togo**")
    st.sidebar.caption("50 ha · 10 parcelles · 5 cultures")
    return page


# ══════════════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════
def page_vue_ensemble(df, meteo, ml):
    st.title("🌾 Tableau de Bord — Ferme Agricole")
    st.caption("Pilotage intelligent basé sur l'enrichissement des données")

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    rend_moy  = df["rendement_t_ha"].mean()
    marge_moy = df["marge_nette_fcfa_ha"].mean() / 1000
    nb_alertes_h = df["alerte_deficit_hydrique"].sum()
    nb_alertes_t = df["alerte_stress_thermique"].sum()
    prod_totale  = df["production_totale_t"].sum()

    for col, val, lbl, icon in zip(
        [c1, c2, c3, c4, c5],
        [f"{rend_moy:.2f}", f"{marge_moy:.0f}k", f"{nb_alertes_h}", f"{nb_alertes_t}", f"{prod_totale:.0f}"],
        ["Rendement moy. (t/ha)", "Marge moy. (FCFA/ha)", "Alertes Hydrique", "Alertes Thermique", "Production totale (t)"],
        ["📈", "💰", "💧", "🌡️", "🌽"]
    ):
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-val'>{icon} {val}</div>
          <div class='metric-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Alertes actives
    alertes_h = df[df["alerte_deficit_hydrique"] == 1][["parcelle_id","culture","saison","annee","meteo_bilan_hydrique"]]
    alertes_t = df[df["alerte_stress_thermique"] == 1][["parcelle_id","culture","saison","annee","meteo_jours_stress"]]

    if len(alertes_h) > 0:
        st.markdown(f"<div class='alert-red'>🚨 <b>{len(alertes_h)} alerte(s) Déficit Hydrique</b> — "
                    f"Parcelles : {', '.join(alertes_h['parcelle_id'].unique())}</div>", unsafe_allow_html=True)
    if len(alertes_t) > 0:
        st.markdown(f"<div class='alert-orange'>⚠️ <b>{len(alertes_t)} alerte(s) Stress Thermique</b> — "
                    f"Parcelles : {', '.join(alertes_t['parcelle_id'].unique())}</div>", unsafe_allow_html=True)

    # Graphiques résumé
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Rendement moyen par culture</div>", unsafe_allow_html=True)
        rend_cult = df.groupby("culture")["rendement_t_ha"].mean().reset_index().sort_values("rendement_t_ha", ascending=True)
        fig = px.bar(rend_cult, x="rendement_t_ha", y="culture", orientation="h",
                     color="rendement_t_ha", color_continuous_scale="Greens",
                     labels={"rendement_t_ha": "t/ha", "culture": "Culture"})
        fig.update_layout(showlegend=False, height=300, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Performance par saison</div>", unsafe_allow_html=True)
        saison_rend = df.groupby(["annee","saison"])["rendement_t_ha"].mean().reset_index()
        saison_rend["periode"] = saison_rend["annee"].astype(str) + " — " + saison_rend["saison"]
        fig2 = px.bar(saison_rend, x="periode", y="rendement_t_ha",
                      color="saison", barmode="group",
                      color_discrete_sequence=["#2E8B57", "#7FB069"],
                      labels={"rendement_t_ha": "t/ha", "periode": ""})
        fig2.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), legend_title="Saison")
        st.plotly_chart(fig2, use_container_width=True)

    # Série temporelle météo
    st.markdown("<div class='section-title'>Températures et précipitations (2 ans)</div>", unsafe_allow_html=True)
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Scatter(x=meteo["date"], y=meteo["temp_moy"],
                              name="Temp. moy. (°C)", line=dict(color="#e67e22", width=1.5)), secondary_y=False)
    fig3.add_trace(go.Bar(x=meteo["date"], y=meteo["precipitation_mm"],
                          name="Précipitations (mm)", marker_color="rgba(46,139,87,0.5)"), secondary_y=True)
    fig3.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10), legend=dict(x=0, y=1.1, orientation="h"))
    fig3.update_yaxes(title_text="°C", secondary_y=False)
    fig3.update_yaxes(title_text="mm", secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 2 — MÉTÉO & ALERTES
# ══════════════════════════════════════════════════════
def page_meteo(meteo, hebdo, df):
    st.title("🌦️ Météo & Alertes Climatiques")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temp. moy. annuelle", f"{meteo['temp_moy'].mean():.1f}°C")
    col2.metric("Précip. totale", f"{meteo['precipitation_mm'].sum():.0f} mm")
    col3.metric("Jours stress thermique", f"{meteo['stress_thermique'].sum()}")
    col4.metric("Bilan hydrique moy.", f"{meteo['bilan_hydrique'].mean():.1f} mm/j")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Température max hebdomadaire")
        fig = px.line(hebdo, x="semaine_debut", y=["temp_max","temp_min"],
                      color_discrete_map={"temp_max":"#e74c3c","temp_min":"#3498db"},
                      labels={"value":"°C","semaine_debut":"Semaine","variable":"Mesure"})
        fig.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Seuil stress 35°C")
        fig.update_layout(height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Bilan hydrique hebdomadaire")
        colors = ["#dc2626" if v < 0 else "#16a34a" for v in hebdo["bilan_hydrique"]]
        fig2 = go.Figure(go.Bar(x=hebdo["semaine_debut"], y=hebdo["bilan_hydrique"],
                                marker_color=colors, name="Bilan hydrique"))
        fig2.add_hline(y=0, line_color="black", line_width=1)
        fig2.update_layout(height=320, margin=dict(t=10,b=10),
                           yaxis_title="Bilan (mm)", xaxis_title="Semaine")
        st.plotly_chart(fig2, use_container_width=True)

    # Alertes détaillées
    st.subheader("📋 Détail des alertes par cycle cultural")
    alertes = df[(df["alerte_deficit_hydrique"]==1) | (df["alerte_stress_thermique"]==1)][
        ["parcelle_id","culture","annee","saison","rendement_t_ha",
         "meteo_bilan_hydrique","meteo_jours_stress",
         "alerte_deficit_hydrique","alerte_stress_thermique"]
    ].copy()
    alertes.columns = ["Parcelle","Culture","Année","Saison","Rendement t/ha",
                       "Bilan Hydrique","Jours Stress","Alerte Hydrique","Alerte Thermique"]
    st.dataframe(alertes.style.highlight_max(
        subset=["Alerte Hydrique","Alerte Thermique"], color="#fee2e2"), use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 3 — PARCELLES & PRODUCTION
# ══════════════════════════════════════════════════════
def page_parcelles(df, parcelles, ml):
    st.title("📦 Parcelles & Production")

    filtre_culture = st.multiselect("Filtrer par culture", df["culture"].unique(),
                                    default=df["culture"].unique().tolist())
    df_f = df[df["culture"].isin(filtre_culture)]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rendement par parcelle (boxplot)")
        fig = px.box(df_f, x="parcelle_id", y="rendement_t_ha", color="culture",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     labels={"rendement_t_ha":"t/ha","parcelle_id":"Parcelle"})
        fig.update_layout(height=350, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Indice de fertilité du sol")
        sol_data = df_f.groupby("parcelle_id")["indice_fertilite_sol"].mean().reset_index()
        fig2 = px.bar(sol_data, x="parcelle_id", y="indice_fertilite_sol",
                      color="indice_fertilite_sol", color_continuous_scale="Greens",
                      labels={"indice_fertilite_sol":"Indice Fertilité","parcelle_id":"Parcelle"})
        fig2.update_layout(height=350, margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Segmentation ML
    if "segments" in ml:
        st.subheader("🗂️ Segmentation des parcelles (K-Means)")
        seg = ml["segments"]
        col_colors = {"Haute Performance":"#16a34a","Performance Moyenne":"#ca8a04","Sous-Performance":"#dc2626"}
        fig3 = px.scatter(seg, x="rendement_moy", y="marge_moy",
                          size="indice_sol", color="segment_label",
                          color_discrete_map=col_colors, text="parcelle_id",
                          hover_data=["culture","nb_alertes"],
                          labels={"rendement_moy":"Rendement moy. (t/ha)","marge_moy":"Marge moy. (FCFA/ha)",
                                  "segment_label":"Segment"})
        fig3.update_traces(textposition="top center")
        fig3.update_layout(height=400, margin=dict(t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(seg[["parcelle_id","culture","segment_label","rendement_moy","marge_moy",
                           "indice_sol","risque_moy","nb_alertes"]].round(2), use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 4 — ANALYSE ML
# ══════════════════════════════════════════════════════
def page_ml(ml):
    st.title("🤖 Analyse Machine Learning")

    if "importances" in ml:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Importance des variables (Random Forest)")
            imp = ml["importances"].head(12)
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Greens",
                         labels={"importance":"Importance","feature":"Variable"})
            fig.update_layout(height=400, margin=dict(t=10,b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Prédictions vs Réalité")
            if "predictions" in ml:
                pred = ml["predictions"]
                fig2 = px.scatter(pred, x="rendement_reel", y="rendement_predit",
                                  color_discrete_sequence=["#2E8B57"],
                                  labels={"rendement_reel":"Rendement réel (t/ha)",
                                          "rendement_predit":"Rendement prédit (t/ha)"})
                # Droite OLS via numpy (évite la dépendance statsmodels/scipy)
                x_vals = pred["rendement_reel"].values
                y_vals = pred["rendement_predit"].values
                m, b = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                fig2.add_scatter(x=x_line, y=m * x_line + b,
                                 mode="lines", name="Tendance OLS",
                                 line=dict(color="#1a5c38", width=2))
                # Ligne parfaite
                vmin = pred[["rendement_reel","rendement_predit"]].min().min()
                vmax = pred[["rendement_reel","rendement_predit"]].max().max()
                fig2.add_shape(type="line", x0=vmin, y0=vmin, x1=vmax, y1=vmax,
                               line=dict(color="red", dash="dash"))
                fig2.update_layout(height=400, margin=dict(t=10,b=10))
                st.plotly_chart(fig2, use_container_width=True)

    # Anomalies
    if "anomalies" in ml:
        st.subheader("🔍 Anomalies détectées (Isolation Forest)")
        ano = ml["anomalies"]
        nb  = ano["anomalie"].sum()
        st.info(f"**{nb} anomalies** détectées sur {len(ano)} cycles culturaux "
                f"({nb/len(ano)*100:.1f}% de contamination)")

        fig3 = px.scatter(ano, x="rendement_t_ha", y="engrais_kg_ha",
                          color=ano["anomalie"].map({0:"Normal",1:"Anomalie"}),
                          color_discrete_map={"Normal":"#2E8B57","Anomalie":"#dc2626"},
                          hover_data=["parcelle_id","culture","saison","annee"],
                          labels={"rendement_t_ha":"Rendement (t/ha)",
                                  "engrais_kg_ha":"Engrais (kg/ha)",
                                  "color":"Statut"})
        fig3.update_layout(height=380, margin=dict(t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 5 — MARCHÉ & RENTABILITÉ
# ══════════════════════════════════════════════════════
def page_marche(df, marche, ml):
    st.title("💰 Marché & Rentabilité")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Évolution des prix par culture")
        culture_sel = st.selectbox("Culture", marche["culture"].unique())
        mdf = marche[marche["culture"] == culture_sel]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mdf["semaine"], y=mdf["prix_fcfa_kg"],
                                 name="Prix réel", line=dict(color="#2E8B57")))
        if "previsions" in ml:
            prev = ml["previsions"][ml["previsions"]["culture"] == culture_sel]
            fig.add_trace(go.Scatter(x=prev["date"], y=prev["prix_predit"],
                                     name="Prévision Prophet", line=dict(color="#e67e22", dash="dash")))
            fig.add_trace(go.Scatter(
                x=pd.concat([prev["date"], prev["date"].iloc[::-1]]),
                y=pd.concat([prev["prix_max"], prev["prix_min"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(230,126,34,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="Intervalle confiance"
            ))
        fig.update_layout(height=360, margin=dict(t=10,b=10),
                          yaxis_title="FCFA/kg", xaxis_title="Semaine",
                          legend=dict(x=0,y=1,orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Marge nette par culture")
        marge = df.groupby("culture").agg(
            marge_moy=("marge_nette_fcfa_ha","mean"),
            revenu_moy=("revenu_brut_fcfa_ha","mean"),
            cout_moy=("cout_intrants_fcfa_ha","mean"),
        ).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Revenu brut", x=marge["culture"],
                              y=marge["revenu_moy"]/1000, marker_color="#2E8B57"))
        fig2.add_trace(go.Bar(name="Coût intrants", x=marge["culture"],
                              y=marge["cout_moy"]/1000, marker_color="#e74c3c"))
        fig2.add_trace(go.Bar(name="Marge nette", x=marge["culture"],
                              y=marge["marge_moy"]/1000, marker_color="#3498db"))
        fig2.update_layout(barmode="group", height=360, margin=dict(t=10,b=10),
                           yaxis_title="FCFA (×1000)/ha")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 Tableau de rentabilité détaillé")
    rent = df.groupby(["culture","annee","saison"]).agg(
        rendement_moy=("rendement_t_ha","mean"),
        production_totale=("production_totale_t","sum"),
        revenu_brut=("revenu_brut_fcfa_ha","mean"),
        cout_intrants=("cout_intrants_fcfa_ha","mean"),
        marge_nette=("marge_nette_fcfa_ha","mean"),
    ).reset_index().round(0)
    st.dataframe(rent, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 6 — SIMULATEUR DE RENDEMENT
# ══════════════════════════════════════════════════════
def page_simulateur(ml):
    st.title("🔮 Simulateur de Rendement")
    st.info("Ajustez les paramètres pour prédire le rendement estimé via le modèle Random Forest.")

    if "rf_model" not in ml:
        st.warning("Modèle Random Forest non trouvé. Lancez d'abord le pipeline ML.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🌱 Culture & Sol**")
        culture = st.selectbox("Culture", ["Maïs","Tomate","Manioc","Haricot","Sorgho"])
        ph_sol  = st.slider("pH du sol", 5.0, 8.0, 6.5, 0.1)
        azote   = st.slider("Azote N (ppm)", 10, 70, 35)
        phosphore = st.slider("Phosphore P (ppm)", 5, 50, 25)
        potassium = st.slider("Potassium K (ppm)", 50, 220, 140)
        matiere_org = st.slider("Matière organique (%)", 1.0, 5.0, 3.0, 0.1)

    with col2:
        st.markdown("**🌦️ Conditions Météo**")
        temp_moy  = st.slider("Temp. moy. cycle (°C)", 18.0, 38.0, 28.0, 0.5)
        pluie     = st.slider("Précipitations totales (mm)", 50, 600, 280)
        bilan_h   = st.slider("Bilan hydrique (mm)", -200, 200, 0)
        jours_stress = st.slider("Jours de stress thermique", 0, 30, 5)
        ensoleil  = st.slider("Ensoleillement moy. (h/j)", 3.0, 10.0, 7.0, 0.5)
        duree     = st.slider("Durée du cycle (jours)", 60, 180, 120)

    with col3:
        st.markdown("**⚗️ Intrants**")
        engrais = st.slider("Engrais (kg/ha)", 50, 250, 130)
        eau_irrig = st.slider("Eau irrigation (m³/ha)", 100, 700, 380)

    # Calcul indices dérivés
    ind_fertilite = (
        (ph_sol / 7.0) * 0.25 +
        np.clip(azote / 50, 0, 1) * 0.30 +
        np.clip(phosphore / 40, 0, 1) * 0.20 +
        np.clip(potassium / 180, 0, 1) * 0.15 +
        (matiere_org / 4.5) * 0.10
    )
    score_risque = (
        (jours_stress / duree) * 0.5 +
        np.clip(abs(bilan_h) / 300, 0, 1) * 0.5
    )

    culture_enc = ml["le_culture"].transform([culture])[0]

    X_sim = pd.DataFrame([{
        "indice_fertilite_sol": ind_fertilite,
        "score_risque_climatique": score_risque,
        "meteo_temp_moy_cycle": temp_moy,
        "meteo_pluie_totale_mm": pluie,
        "meteo_bilan_hydrique": bilan_h,
        "meteo_jours_stress": jours_stress,
        "meteo_ensoleillement_moy_h": ensoleil,
        "duree_cycle_jours": duree,
        "engrais_kg_ha": engrais,
        "eau_irrigation_m3_ha": eau_irrig,
        "ph_sol": ph_sol,
        "azote_N_ppm": azote,
        "phosphore_P_ppm": phosphore,
        "potassium_K_ppm": potassium,
        "matiere_organique_pct": matiere_org,
        "culture_enc": culture_enc,
    }])

    rendement_predit = ml["rf_model"].predict(X_sim)[0]

    st.markdown("---")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("🌾 Rendement prédit", f"{rendement_predit:.2f} t/ha")
    col_r2.metric("🌿 Indice fertilité sol", f"{ind_fertilite:.3f}")
    col_r3.metric("⚠️ Score risque climatique", f"{score_risque:.3f}")

    # Jauge visuelle
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rendement_predit,
        title={"text": f"Rendement prédit — {culture}"},
        gauge={
            "axis": {"range": [0, 50]},
            "bar": {"color": "#2E8B57"},
            "steps": [
                {"range": [0, 15], "color": "#fee2e2"},
                {"range": [15, 30], "color": "#fef9c3"},
                {"range": [30, 50], "color": "#dcfce7"},
            ],
            "threshold": {"line": {"color": "red","width":3}, "thickness":0.75, "value": rendement_predit}
        }
    ))
    fig.update_layout(height=300, margin=dict(t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main():
    try:
        df, meteo, hebdo, parcelles, marche = charger_donnees()
        ml = charger_resultats_ml()
    except FileNotFoundError:
        st.error("⚠️ Données non trouvées. Lancez d'abord `pipeline.py` pour générer les données.")
        st.code("python src/pipeline.py", language="bash")
        return

    page = sidebar()

    if page == "🏠 Vue d'ensemble":
        page_vue_ensemble(df, meteo, ml)
    elif page == "🌦️ Météo & Alertes":
        page_meteo(meteo, hebdo, df)
    elif page == "📦 Parcelles & Production":
        page_parcelles(df, parcelles, ml)
    elif page == "🤖 Analyse ML":
        page_ml(ml)
    elif page == "💰 Marché & Rentabilité":
        page_marche(df, marche, ml)
    elif page == "🔮 Simulateur":
        page_simulateur(ml)


if __name__ == "__main__":
    main()
