"""
Dashboard Streamlit -- prevision des exportations de phosphates et derives.

Lancement :
    streamlit run app.py

Reutilise integralement pipeline.py (aucune logique metier dupliquee) : les resultats
affiches ici sont exactement ceux du notebook et du CLI (cli.py). L'assistant
explicatif (assistant.py) est accessible partout via la bulle flottante en bas a droite.
"""
import base64
import html as _html
import os
from pathlib import Path

import markdown as _md
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import assistant
import cli as _cli
import pipeline as pl

# Pont optionnel st.secrets -> variable d'environnement, pour le LLM optionnel de
# assistant.py (voir sa docstring) : si une cle est definie dans .streamlit/secrets.toml,
# elle devient visible a assistant.py sans qu'il ait besoin de connaitre Streamlit. Sans
# secrets.toml (cas par defaut), ce bloc ne fait rien -- aucun impact sur le fonctionnement
# normal (100% routeur, sans cle).
try:
    if "ANTHROPIC_API_KEY" not in os.environ and "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

DATA_PATH = Path(__file__).parent / "data" / "data_phosphates.xlsx"
LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"
LOGO_B64 = base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else ""

# Identite visuelle du site : couleurs extraites directement du logo fourni (bleu marine
# des barres/anneau, vert de la fleche, vert menthe de la feuille).
BRAND_NAVY = "#0F4973"
BRAND_NAVY_DARK = "#0B3A5C"
BRAND_GREEN = "#4B8F7C"
BRAND_GREEN_DARK = "#3A7566"
BRAND_MINT = "#9CC3B6"

# Couleurs par modele : meme famille que l'identite du logo (bleu marine / vert / menthe),
# pour une coherence visuelle sur toute la page et tous les graphiques -- chaque modele
# recoit une teinte ou une nuance de cette meme famille plutot qu'une palette generique.
MODEL_COLOR = {
    "ARIMA": BRAND_NAVY,               # modele retenu -- couleur d'ancrage du logo
    "Random Forest": BRAND_GREEN,      # 2e meilleur modele -- couleur d'ancrage du logo
    "XGBoost": "#3E7CA8",              # nuance plus claire du bleu marine
    "Ridge": "#2F6B58",                # nuance plus foncee du vert
    "SARIMAX": BRAND_MINT,             # vert menthe du logo
    "Naive saisonnier": "#8B8F92",     # gris neutre -- convention pour une base naive
}
REF_COLOR = "#16232E"  # "encre" bleu marine tres fonce, pour les series reelles/historiques

st.set_page_config(page_title="Prevision Exportations Phosphates ML", page_icon="\U0001F4CA", layout="wide")

st.markdown(f"""
<style>
/* -- chrome Streamlit par defaut : masque pour une allure professionnelle -- */
#MainMenu {{visibility: hidden;}}
[data-testid="stToolbar"] {{visibility: hidden;}}
[data-testid="stDecoration"] {{display: none;}}
[data-testid="stStatusWidget"] {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* -- espace au-dessus du titre un peu reduit (marge par defaut de Streamlit jugee trop
   grande) -- */
.block-container {{ padding-top: 3.5rem !important; }}

/* -- identite (couleurs extraites du logo) -- */
.page-title {{
    font-size: 2.25rem; font-weight: 700; line-height: 1.25; margin: 0 0 4px;
    display: flex; align-items: center; gap: 10px;
}}
.page-title .grad-text {{
    background: linear-gradient(90deg, {BRAND_NAVY} 0%, {BRAND_GREEN} 60%, {BRAND_MINT} 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    color: {BRAND_NAVY};
}}
.retained-pill {{
    display: inline-flex; align-items: center; gap: 8px;
    background: {BRAND_GREEN}14; border: 1px solid {BRAND_GREEN}40;
    color: {BRAND_GREEN_DARK}; border-radius: 999px; padding: 6px 14px;
    font-size: 0.85rem; font-weight: 600;
}}
.retained-pill .dot {{width: 8px; height: 8px; border-radius: 50%; background: {BRAND_GREEN};}}

/* boutons / elements interactifs natifs -> bleu marine du logo */
.stButton > button[kind="primary"] {{
    background-color: {BRAND_NAVY} !important; border-color: {BRAND_NAVY} !important;
}}
/* bouton d'envoi du chat -> icone ronde verte (fleche "entree"), comme le bouton d'envoi de
   Claude, a la place du bouton texte "Envoyer". Note : le corps du popover est rendu par
   Streamlit dans un portail hors de .st-key-assistant_fab (pas un vrai descendant DOM) --
   selecteur global, sans risque puisque c'est le seul formulaire de toute l'appli. */
.stFormSubmitButton {{ display: flex; justify-content: flex-end; }}
/* selecteur descendant (pas d'enfant direct), conserve par prudence : un parametre help=...
   sur ce bouton inserait un span.stTooltipIcon/stTooltipHoverTarget entre .stFormSubmitButton
   et le <button> et casserait un selecteur en enfant direct (piege rencontre au round 10-bis) --
   le bouton n'a plus de help= depuis le round 13-ter (infobulle "Envoyer" retiree), mais le
   selecteur reste descendant pour rester robuste si l'un ou l'autre revient. */
.stFormSubmitButton button {{
    background-color: {BRAND_GREEN} !important; border-color: {BRAND_GREEN} !important;
    color: white !important;
    width: 38px !important; height: 38px !important; min-height: 38px !important;
    max-width: 38px !important; padding: 0 !important; border-radius: 50% !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    font-size: 1.15rem !important; line-height: 1 !important; font-weight: 700 !important;
}}
.stFormSubmitButton button:hover {{
    background-color: {BRAND_GREEN_DARK} !important; border-color: {BRAND_GREEN_DARK} !important;
}}
.stFormSubmitButton button p {{ margin: 0 !important; font-size: 1.15rem !important; color: white !important; }}
.stTabs [aria-selected="true"] {{ color: {BRAND_NAVY} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {BRAND_NAVY} !important; }}

/* -- bouton flottant de l'assistant : logo fourni en fond de cercle -- */
.st-key-assistant_fab {{
    position: fixed !important;
    bottom: 22px; right: 22px; left: auto !important;
    width: fit-content !important;
    z-index: 999999;
}}
.st-key-assistant_fab button {{
    border-radius: 50% !important;
    width: 58px !important; height: 58px !important;
    font-size: 1px !important; line-height: 1 !important;
    color: transparent !important;
    background-image: url(data:image/png;base64,{LOGO_B64}) !important;
    background-size: cover !important; background-position: center !important;
    background-color: white !important; border: none !important;
    box-shadow: 0 8px 22px rgba(15,73,115,0.35) !important;
    transition: transform .15s ease !important;
}}
.st-key-assistant_fab button:hover {{ transform: scale(1.06); }}
/* le corps du popover est rendu dans un portail (pas un descendant de .st-key-assistant_fab) ;
   selecteur global sans risque, un seul popover dans toute l'appli. IMPORTANT : ne jamais
   remettre "overflow: hidden" ici (ca a deja bloque tout defilement et rendu le champ de
   saisie inatteignable -- bug corrige au round 11). Le defilement lui-meme appartient
   desormais UNIQUEMENT a la zone de conversation interieure (.st-key-chat_scroll, round 13)
   -- le cadre exterieur du popover n'a plus de defilement ni de max-height propre : sa
   hauteur reste naturellement bornee puisque la conversation interieure est elle-meme
   plafonnee (voir plus bas). Round 13-bis : plus de bandeau-titre distinct en haut du popover
   (l'entete "Assistant du projet / Pose ta question ci-dessous" a ete retiree) -- la conversation
   demarre directement par le message de bienvenue, donc plus besoin de faire correspondre un
   border-radius d'entete : rien ne touche les coins arrondis du popover. */
[data-testid="stPopoverBody"] {{
    width: 420px !important; max-width: 92vw;
    border-radius: 16px !important;
    box-shadow: 0 16px 40px rgba(0,0,0,0.18) !important;
    padding: 0 !important;
    position: relative;
}}
.assistant-body {{ padding: 16px 20px 18px; }}
/* zone de conversation interieure (round 13) : hauteur naturelle jusqu'a un plafond, PAS une
   hauteur fixe -- colle au contenu (comme une conversation Claude, pas de grand vide entre le
   dernier message et le champ de saisie s'il y a peu d'echanges), et ne defile QUE cette zone
   une fois le plafond atteint (jamais le cadre exterieur du popover). Marge au-dessus/dessous
   pour separer visuellement le cadre exterieur (popover) du cadre interieur (conversation). */
.st-key-chat_scroll {{
    max-height: 320px !important; overflow-y: auto !important;
    margin: 8px 0 10px !important;
}}
.chip-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 10px; }}
/* l'indication native "Press Enter to submit form" (Streamlit) sous le champ de saisie du
   chat n'apporte rien ici (le bouton rond suffit) -- masquee. */
[data-testid="stPopoverBody"] [data-testid="InputInstructions"] {{ display: none !important; }}

/* -- onglet Le projet : quelques phrases mises en avant en vert (identite du logo) -- */
.le-projet-section strong {{ color: {BRAND_GREEN_DARK}; }}
/* -- onglet Vue d'ensemble : meme traitement (mots-cles en gras vert), pour le paragraphe de
   lecture rapide du graphique + de la vue d'ensemble (round 14 : simple paragraphe, sans titre,
   au lieu de plusieurs blocs avec emojis). -- */
.vue-ensemble-lecture strong {{ color: {BRAND_GREEN_DARK}; }}

/* -- onglet Prevision : tableau (2 colonnes, texte centre dans chaque cellule) --
   agrandi au round 13-quater (jugee trop petite a cote du selecteur de modele) : police et
   paddings augmentes, hauteur de defilement etendue. */
.forecast-table-wrap {{ max-height: 340px; overflow-y: auto; margin: 8px 0 14px; }}
.forecast-table {{ width: 100%; border-collapse: collapse; font-size: 1.05rem; }}
.forecast-table th, .forecast-table td {{ text-align: center; padding: 11px 16px; }}
.forecast-table thead th {{
    background: {BRAND_NAVY}; color: white; font-weight: 600; position: sticky; top: 0;
    font-size: 1rem;
}}
.forecast-table tbody tr:nth-child(even) {{ background: #F7F8F9; }}
.forecast-table tbody td {{ border-bottom: 1px solid rgba(15,73,115,0.08); font-weight: 500; }}

/* -- onglet Pipeline : schema du flux CRISP-DM -- */
.pipeline-flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 10px; margin: 8px 0 22px; }}
.pipeline-step {{
    flex: 1 1 170px; min-width: 160px;
    background: #F7F8F9; border: 1px solid rgba(15,73,115,0.14); border-radius: 12px;
    padding: 14px 14px 12px;
}}
.pipeline-num {{
    width: 30px; height: 30px; border-radius: 50%; background: {BRAND_NAVY}14;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; margin-bottom: 9px;
}}
.pipeline-title {{ font-weight: 700; font-size: 0.92rem; margin-bottom: 5px; color: {BRAND_NAVY_DARK}; }}
.pipeline-desc {{ font-size: 0.78rem; line-height: 1.45; color: #3a3a3a; }}
.pipeline-arrow {{
    display: flex; align-items: center; justify-content: center;
    color: {BRAND_GREEN}; font-size: 1.3rem; flex: 0 0 auto; opacity: 0.65;
}}

/* -- style de conversation "ChatGPT" : bulle grise a droite pour l'utilisateur,
      texte simple sans bulle a gauche pour l'assistant -- */
.cg-row {{ display: flex; margin: 10px 0; align-items: flex-start; gap: 8px; }}
.cg-row.user {{ justify-content: flex-end; }}
.cg-row.assistant {{ justify-content: flex-start; }}
.cg-avatar {{
    width: 24px; height: 24px; border-radius: 50%; flex: none; margin-top: 1px;
    background-image: url(data:image/png;base64,{LOGO_B64});
    background-size: cover; background-position: center;
}}
.cg-bubble {{ font-size: 0.88rem; line-height: 1.55; max-width: 82%; }}
.cg-bubble.user {{
    background: #eef0ef; color: #111; padding: 9px 13px;
    border-radius: 16px; border-bottom-right-radius: 4px;
}}
.cg-bubble.assistant {{ background: transparent; color: inherit; padding: 2px 0; }}
.cg-bubble.assistant p {{ margin: 0 0 6px; }}
.cg-bubble.assistant p:last-child {{ margin-bottom: 0; }}
.cg-bubble.assistant ul {{ margin: 2px 0 6px 18px; padding: 0; }}
.cg-bubble.assistant li {{ margin-bottom: 2px; }}
.cg-bubble.assistant strong {{ color: {BRAND_GREEN_DARK}; }}

/* -- bulle d'accueil (teaser) pres du bouton flottant -- */
.st-key-assistant_teaser {{
    position: fixed !important;
    bottom: 90px; right: 22px; left: auto !important;
    width: 248px !important;
    z-index: 999998;
}}
.teaser-bubble {{
    background: white; color: #1a1a1a; border-radius: 14px; border-bottom-right-radius: 4px;
    padding: 10px 12px; font-size: 0.82rem; line-height: 1.4;
    box-shadow: 0 10px 26px rgba(0,0,0,0.16); border: 1px solid rgba(0,0,0,0.06);
}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Chargement du pipeline...")
def load_all():
    # Reutilise le cache disque de la CLI (models/artifacts.joblib) : le zip livre les
    # modeles deja entraines, donc le chargement est quasi instantane. En cas de cache
    # absent ou incompatible (ex. versions de librairies differentes), reentrainement
    # automatique -- une seule fois, ~30-45s -- puis le cache est recree.
    try:
        artifacts = _cli.train(str(DATA_PATH))
    except Exception:
        _cli.ARTIFACTS_PATH.unlink(missing_ok=True)
        artifacts = _cli.train(str(DATA_PATH), force=True)
    return (artifacts["state"], artifacts["mw_results"], artifacts["mw_mean_rmse"],
            artifacts["best_model_name"], artifacts["final_models"])


state, mw_results, mw_mean_rmse, best_model_name, final_models = load_all()
df = state["df"]


# ---------------------------------------------------------------------------
# En-tete
# ---------------------------------------------------------------------------
top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown(
        '<div class="page-title"><span>\U0001F4CA</span>'
        '<span class="grad-text">Prevision des exportations de phosphates et dérivés</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Le phosphate, pilier de l'economie marocaine -- ce dashboard mobilise le Machine "
        "Learning pour anticiper ses exportations, mois apres mois."
    )
with top_r:
    st.markdown(
        f'<div style="text-align:right; padding-top: 18px;">'
        f'<span class="retained-pill"><span class="dot"></span>Modele retenu : {best_model_name}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

tabs = st.tabs(["Vue d'ensemble", "Comparaison des modeles", "Prevision", "Pipeline", "Le projet"])


# ---------------------------------------------------------------------------
# Onglet 1 -- Vue d'ensemble
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Observations", len(df))
    c2.metric("Periode", f"{df.index.min():%m/%y}–{df.index.max():%m/%y}")
    c3.metric("RMSE (modele retenu)", f"{state['results'].loc[best_model_name, 'RMSE']:.0f} M DH")
    c4.metric("Variables exogenes", len(pl.EXOG_RAW))
    c5.metric("Exogenes retenues (CCF)", len(state["selected_lags"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[pl.TARGET], mode="lines",
                              line=dict(color=MODEL_COLOR["ARIMA"], width=2),
                              name="Exportations reelles"))
    fig.add_vrect(x0=state["split_date"], x1=df.index.max(),
                  fillcolor="grey", opacity=0.08, line_width=0,
                  annotation_text="periode test", annotation_position="top left")
    fig.update_layout(title="Exportations de phosphates et dérivés (Millions DH)",
                       height=380, margin=dict(l=10, r=10, t=40, b=10),
                       yaxis_title="Millions DH", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Lecture rapide -- interpretation concise du graphique ET des indicateurs de la vue
    # d'ensemble ci-dessus (KPI Observations/Periode/RMSE/Variables exogenes/Exogenes CCF),
    # entierement calculee a partir de la serie et de l'etat reels (rien d'invente, meme
    # principe de "grounding" que l'assistant conversationnel). Round 14 : un seul paragraphe
    # simple, sans titre, mots-cles en gras colore (meme traitement que l'onglet Le projet,
    # via la classe CSS .vue-ensemble-lecture) -- remplace les 4 blocs avec emojis precedents.
    _target = df[pl.TARGET]
    _first_val, _last_val = _target.iloc[0], _target.iloc[-1]
    _first_date, _last_date = _target.index[0], _target.index[-1]
    _growth_pct = (_last_val / _first_val - 1) * 100 if _first_val else float("nan")
    _peak_val, _peak_date = _target.max(), _target.idxmax()
    _trough_val, _trough_date = _target.min(), _target.idxmin()
    _recent_12 = _target.iloc[-12:].mean()
    _prior_12 = (_target.iloc[-24:-12].mean() if len(_target) >= 24 else _target.iloc[:-12].mean())
    _recent_change_pct = (_recent_12 / _prior_12 - 1) * 100 if _prior_12 else float("nan")
    _cv = (_target.std() / _target.mean() * 100) if _target.mean() else float("nan")
    _sens = "en hausse" if _recent_change_pct >= 0 else "en baisse"
    _rmse_val = state["results"].loc[best_model_name, "RMSE"]
    _rmse_pct = (_rmse_val / _target.mean() * 100) if _target.mean() else float("nan")

    _lecture = (
        f"Sur {len(df)} observations mensuelles ({_first_date:%m/%Y} – {_last_date:%m/%Y}), les "
        f"exportations sont passees de {_first_val:.0f} M DH a {_last_val:.0f} M DH, soit "
        f"<strong>{_growth_pct:+.0f}%</strong> sur toute la periode -- point haut en {_peak_date:%m/%Y} "
        f"({_peak_val:.0f} M DH), point bas en {_trough_date:%m/%Y} ({_trough_val:.0f} M DH). Sur les "
        f"12 derniers mois, la moyenne s'etablit a {_recent_12:.0f} M DH, <strong>{_sens} de "
        f"{abs(_recent_change_pct):.0f}%</strong> par rapport aux 12 mois precedents, avec une "
        f"volatilite typique d'environ {_cv:.0f}% autour de la moyenne. Le modele retenu, "
        f"<strong>{best_model_name}</strong>, entraine sur "
        f"<strong>{len(state['selected_lags'])}/{len(pl.EXOG_RAW)} variables exogenes</strong> "
        f"retenues par correlation croisee (CCF), affiche un RMSE de test de "
        f"<strong>{_rmse_val:.0f} M DH</strong> (environ {_rmse_pct:.0f}% de la moyenne de la serie) "
        f"sur la <strong>periode de test</strong> grisee a droite du graphique (20% des derniers "
        f"mois, jamais vue a l'entrainement) -- demarche complete et comparaison des 6 modeles dans "
        f"les onglets Pipeline et Comparaison des modeles."
    )
    st.markdown(f'<div class="vue-ensemble-lecture"><p>{_lecture}</p></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Onglet 2 -- Comparaison des modeles
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Test unique -- RMSE par modele")
    st.caption("Erreur sur les 20% derniers mois (non vus a l'entrainement), classee par RMSE croissant.")

    results_sorted = state["results"].sort_values("RMSE")
    bar_colors = [MODEL_COLOR[m] for m in results_sorted.index]
    fig_bar = go.Figure(go.Bar(
        x=results_sorted.index, y=results_sorted["RMSE"], marker_color=bar_colors,
        text=[f"{v:.0f}" for v in results_sorted["RMSE"]], textposition="outside",
    ))
    fig_bar.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="RMSE (M DH)")
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("Tableau complet (RMSE, MAE, MAPE, R²)", expanded=False):
        st.dataframe(results_sorted.round(2), use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Predictions sur la periode test")
        default_models = list(state["top2"])
        chosen = st.multiselect("Modeles a afficher", options=pl.ALL_MODEL_NAMES,
                                 default=default_models, key="testpred_models")
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Scatter(x=state["y_test"].index, y=state["y_test"].values,
                                     mode="lines+markers", name="Reel",
                                     line=dict(color=REF_COLOR, width=2)))
        for m in chosen:
            pred = state["predictions"][m]
            fig_tp.add_trace(go.Scatter(
                x=state["y_test"].index, y=pred, mode="lines", name=m,
                line=dict(color=MODEL_COLOR[m], width=2, dash="dash" if m == "Naive saisonnier" else "solid"),
            ))
        fig_tp.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                              yaxis_title="Millions DH", hovermode="x unified")
        st.plotly_chart(fig_tp, use_container_width=True)

    with col_b:
        st.subheader("Validation multi-fenetres")
        st.caption(f"RMSE par fenetre glissante (7 x 12 mois), modeles {state['top2']}.")
        fig_mw = go.Figure()
        for m in state["top2"]:
            sub = mw_results[mw_results["modele"] == m].sort_values("fold")
            fig_mw.add_trace(go.Scatter(x=sub["fold"], y=sub["RMSE"], mode="lines+markers",
                                         name=m, line=dict(color=MODEL_COLOR[m], width=2)))
        fig_mw.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                              xaxis_title="Fenetre", yaxis_title="RMSE (M DH)", hovermode="x unified")
        st.plotly_chart(fig_mw, use_container_width=True)
        mcols = st.columns(len(mw_mean_rmse))
        for col, (name, val) in zip(mcols, mw_mean_rmse.items()):
            col.metric(name, f"{val:.0f} M DH")


# ---------------------------------------------------------------------------
# Onglet 3 -- Prevision
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Prevision")
    st.caption("Exogenes figees a leur derniere valeur connue (limite documentee du modele).")

    # fc2 (table de prevision) recoit plus de largeur que fc1 (selecteurs) -- round 13-quater,
    # la table etait jugee trop petite/serree cote-a-cote avec les selecteurs en 50/50.
    fc1, fc2 = st.columns([0.8, 1.2])
    with fc1:
        model_choice = st.selectbox("Modele", options=list(final_models.keys()),
                                     index=list(final_models.keys()).index(best_model_name)
                                     if best_model_name in final_models else 0)
        horizon = st.slider("Horizon (mois)", min_value=1, max_value=24, value=pl.N_FORECAST_DEFAULT)

    forecast = pl.forecast_with_model(state, final_models, model_choice, horizon)

    with fc2:
        # Tableau HTML custom (2 colonnes, texte centre dans chaque cellule) plutot que
        # st.dataframe : la date est formatee sans l'heure "00:00:00" (mois seul, les
        # donnees sont mensuelles) et le centrage des cellules n'est pas fiable avec le
        # rendu en grille natif de st.dataframe -- une table HTML donne un controle total.
        rows_html = "".join(
            f"<tr><td>{d:%Y-%m}</td><td>{v:.1f}</td></tr>" for d, v in forecast.items()
        )
        st.markdown(
            '<div class="forecast-table-wrap"><table class="forecast-table">'
            '<thead><tr><th>Date</th><th>Prevision (M DH)</th></tr></thead>'
            f"<tbody>{rows_html}</tbody></table></div>",
            unsafe_allow_html=True,
        )
        st.download_button("Telecharger la prevision (CSV)",
                            forecast.rename("prevision_millions_dh").round(1).to_csv().encode("utf-8"),
                            file_name=f"prevision_{model_choice.replace(' ', '_')}.csv", mime="text/csv")

    hist = df[pl.TARGET].iloc[-30:]
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines",
                                 name="Historique", line=dict(color=REF_COLOR, width=2)))
    bridge_x = [hist.index[-1]] + list(forecast.index)
    bridge_y = [hist.values[-1]] + list(forecast.values)
    fig_fc.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers",
                                 name=f"Prevision -- {model_choice}",
                                 line=dict(color=MODEL_COLOR[model_choice], width=2, dash="dash")))
    fig_fc.add_vline(x=hist.index[-1], line_dash="dot", line_color="grey")
    fig_fc.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                          yaxis_title="Millions DH", hovermode="x unified")
    st.plotly_chart(fig_fc, use_container_width=True)


# ---------------------------------------------------------------------------
# Onglet 4 -- Pipeline (visualisation du flux CRISP-DM, valeurs calculees en direct)
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Pipeline du projet")
    st.caption(
        "Les six phases du cycle CRISP-DM appliquees a ce projet, avec les chiffres reels "
        "calcules par le pipeline (rien n'est fige a l'avance : si les donnees ou seuils "
        "changent, ces valeurs se mettent a jour automatiquement)."
    )

    verdict_level = state["stationarity"].iloc[0]["verdict"]
    verdict_diff = state["stationarity"].iloc[1]["verdict"]
    roll_txt = ", ".join(str(w) for w in pl.ROLL_WINDOWS)
    models_txt = ", ".join(pl.ALL_MODEL_NAMES)

    steps = [
        ("🎯", "Business understanding",
         "Anticiper les exportations mensuelles de phosphates et derives, pilier de "
         "l'economie marocaine, pour eclairer le suivi de l'activite."),
        ("📊", "Data understanding",
         f"{len(df)} mois ({df.index.min():%m/%Y} → {df.index.max():%m/%Y}) &middot; "
         f"1 cible (Millions DH) + {len(pl.EXOG_RAW)} exogenes -- des produits remarquables "
         f"importes en Admission Temporaire (AT, valeurs en DH)."),
        ("🧹", "Data preparation",
         f"ADF + KPSS sur la cible (niveau : {verdict_level}, differenciee : {verdict_diff}) "
         f"&middot; CCF (seuil |r| ≥ {pl.CORR_THRESHOLD}) → {len(state['selected_lags'])}/"
         f"{len(pl.EXOG_RAW)} exogenes retenues &middot; ACF cible → {len(state['lag_target'])} "
         f"lags &middot; moyennes/ecarts-types mobiles ({roll_txt} mois)."),
        ("🤖", "Modeling",
         f"6 modeles compares : {models_txt}."),
        ("✅", "Evaluation",
         f"Test unique (20%) puis validation multi-fenetres (7 × 12 mois) → modele retenu : "
         f"{best_model_name}."),
        ("🚀", "Deployment",
         "Dashboard Streamlit interactif (Vue d'ensemble, comparaison des modeles, "
         "prevision) accompagne d'un assistant explicatif integre."),
    ]

    html_parts = ['<div class="pipeline-flow">']
    for i, (icon, title, desc) in enumerate(steps, start=1):
        html_parts.append(
            f'<div class="pipeline-step"><div class="pipeline-num">{icon}</div>'
            f'<div class="pipeline-title">{i}. {title}</div>'
            f'<div class="pipeline-desc">{desc}</div></div>'
        )
        if i < len(steps):
            html_parts.append('<div class="pipeline-arrow">&#8594;</div>')
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    st.markdown(
        "**Pourquoi ce decoupage.** La stationnarite et les correlations sont traitees comme deux "
        "sous-etapes distinctes de la preparation : l'ADF seul ne suffit pas a conclure (il est "
        "combine a KPSS), et l'ACF (cible vs son propre passe) ne dit rien de la relation avec les "
        "exogenes -- c'est le role de la CCF, calculee sur le train uniquement pour eviter toute "
        "fuite d'information vers le test."
    )


# ---------------------------------------------------------------------------
# Onglet 5 -- Le projet (presentation concise, remplace l'ancienne simulation what-if)
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Le projet")

    p1 = (
        "Ce projet vise a prevoir les exportations mensuelles de « Phosphates et derives », "
        "<strong>pilier de l'economie marocaine</strong>, a partir de leur historique et des "
        "importations en Admission Temporaire (AT) de <strong>produits remarquables</strong> "
        "entrant dans la transformation et la production des phosphates, utilisees comme "
        "variables exogenes -- une correlation entre ces importations et les exportations "
        "n'etablit pas de lien de causalite, elles refletent une dynamique d'activite plutot "
        f"qu'un mecanisme causal direct. Les donnees couvrent {len(df)} mois "
        f"({df.index.min():%m/%Y} a {df.index.max():%m/%Y}), avec {len(pl.EXOG_RAW)} variables "
        "explicatives."
    )
    p2 = (
        "La demarche suit les <strong>six phases du cycle CRISP-DM</strong> (business &amp; data "
        "understanding, data preparation, modeling, evaluation, deployment -- detail dans "
        "l'onglet Pipeline) : tests de stationnarite (ADF + KPSS) sur la cible, selection des "
        "lags exogenes par correlation croisee (CCF, calculee sur le train uniquement) et des "
        "lags de la cible par magnitude ACF, puis comparaison de six modeles -- un naif "
        "saisonnier, trois modeles de Machine Learning sur variables retardees (Ridge, Random "
        "Forest, XGBoost), et deux modeles autoregressifs (ARIMA, sans exogenes ; SARIMAX, avec "
        "les memes exogenes que les modeles ML) -- afin de <strong>mesurer separement l'apport "
        "de la modelisation classique et celui des variables exogenes</strong>."
    )
    p3 = (
        f"Le modele retenu, <strong>{best_model_name}</strong>, obtient le RMSE moyen le plus "
        "bas sur une validation multi-fenetres (7 fenetres glissantes de 12 mois) -- "
        "<strong>un resultat plus robuste qu'un score sur une seule periode de test</strong>, "
        "celui a retenir en priorite."
    )
    p4 = (
        "<strong>Axes d'amélioration identifiés :</strong><br>prevoir reellement les "
        "variables exogenes futures plutot que de les figer a leur derniere valeur connue, "
        "reevaluer le modele a mesure que de nouvelles observations arrivent, et surveiller "
        "les changements de regime de la serie."
    )
    st.markdown(
        f'<div class="le-projet-section"><p>{p1}</p><p>{p2}</p><p>{p3}</p><p>{p4}</p></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Assistant -- bulle flottante persistante (visible sur tous les onglets)
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Bonjour \U0001F44B, je suis l'assistant du projet. Comment puis-je vous aider ?"}
    ]
def render_chat_message(role, content):
    """Rendu 'style ChatGPT' : bulle grise a droite pour l'utilisateur, texte simple
    (sans bulle) a gauche pour l'assistant, avec un petit avatar."""
    if role == "user":
        safe = _html.escape(content).replace("\n", "<br>")
        st.markdown(f'<div class="cg-row user"><div class="cg-bubble user">{safe}</div></div>',
                    unsafe_allow_html=True)
    else:
        body_html = _md.markdown(content)
        st.markdown(
            f'<div class="cg-row assistant"><div class="cg-avatar"></div>'
            f'<div class="cg-bubble assistant">{body_html}</div></div>',
            unsafe_allow_html=True,
        )


# bulle d'accueil pres du bouton, signalant que ce logo est bien un assistant/chatbot.
# Round 13-quater : affichee EN PERMANENCE tant que le popover est ferme -- meme apres qu'une
# conversation a deja eu lieu -- pour que ce role reste identifiable a tout moment quand le
# widget est ferme (avant, elle disparaissait definitivement des le premier message, laissant
# un simple logo rond sans indice). Masquee uniquement pendant que le popover est OUVERT (le
# message de bienvenue est alors deja visible DANS la conversation, pas besoin des deux a la
# fois). L'ouverture/fermeture d'un st.popover est purement cote navigateur (pas de rerun
# Python associe) : on ne peut donc pas s'appuyer sur session_state ici, seul un petit script
# cote client peut observer l'apparition/disparition du popover.
with st.container(key="assistant_teaser"):
    st.markdown(
        '<div class="teaser-bubble">\U0001F44B Bonjour, je suis l\'assistant du projet.'
        '<br>Comment puis-je vous aider ?</div>',
        unsafe_allow_html=True,
    )
components.html(
    """<script>
    (function () {
        const doc = window.parent.document;
        function syncTeaser() {
            const teaser = doc.querySelector('.st-key-assistant_teaser');
            if (!teaser) return;
            const popoverOpen = !!doc.querySelector('[data-testid="stPopoverBody"]');
            teaser.style.display = popoverOpen ? 'none' : '';
        }
        syncTeaser();
        const observer = new MutationObserver(syncTeaser);
        observer.observe(doc.body, {childList: true, subtree: true});
    })();
    </script>""",
    height=0,
)

with st.container(key="assistant_fab"):
    with st.popover("\U0001F4AC"):
        # Round 13-bis : plus de bandeau-titre separe ("Assistant du projet / Pose ta
        # question ci-dessous") -- la conversation demarre directement par le message
        # de bienvenue (premiere entree de chat_history), comme une vraie conversation.
        st.markdown('<div class="assistant-body">', unsafe_allow_html=True)

        # Principe "conversation Claude" : l'ecran d'accueil (juste le message de bienvenue)
        # s'affiche sans defilement, hauteur naturelle -- des que la conversation demarre, la
        # zone colle a son contenu (PAS de hauteur fixe : `key=` seul, sans `height=`, la
        # hauteur/le defilement sont geres par CSS -- `.st-key-chat_scroll` plus haut -- en
        # max-height + overflow-y, pour ne jamais laisser de grand vide entre le dernier
        # message et le champ de saisie quand la conversation est courte), et ne defile que
        # cette zone (jamais le cadre exterieur du popover) une fois le plafond atteint,
        # auto-scrollee vers le dernier message.
        if len(st.session_state.chat_history) <= 1:
            for msg in st.session_state.chat_history:
                render_chat_message(msg["role"], msg["content"])
        else:
            with st.container(key="chat_scroll"):
                for msg in st.session_state.chat_history:
                    render_chat_message(msg["role"], msg["content"])
            # Le commentaire {n} force un contenu different a chaque nouveau message, pour
            # que Streamlit recharge reellement l'iframe (sinon le script ne se reexecute
            # pas quand le HTML est identique au rendu precedent). Plusieurs tentatives
            # espacees (au lieu d'une seule a 150ms) : le rendu du dernier message (surtout
            # une reponse longue) peut prendre un peu plus de temps que prevu selon la
            # machine, et une seule tentative trop tot laissait parfois la conversation pas
            # tout a fait defilee jusqu'en bas.
            components.html(
                f"""<script>
                function scrollChatToBottom() {{
                    const doc = window.parent.document;
                    const box = doc.querySelector('.st-key-chat_scroll');
                    if (!box) return;
                    const scrollable = box.querySelector('[data-testid="stVerticalBlockBorderWrapper"]') || box;
                    scrollable.scrollTop = scrollable.scrollHeight;
                }}
                [80, 200, 400, 700].forEach(function (delay) {{
                    setTimeout(scrollChatToBottom, delay);
                }});
                </script>
                <!-- n={len(st.session_state.chat_history)} -->""",
                height=0,
            )

        # Les suggestions ne s'affichent que sur l'ecran d'accueil (avant toute conversation) --
        # une fois un echange lance, elles sont masquees pour laisser toute la place au champ
        # de saisie (elles reprenaient sinon de la hauteur utile en continu, contribuant au
        # bug "pas de place pour ecrire" sur un ecran pas assez haut).
        chip_clicked = None
        if len(st.session_state.chat_history) <= 1:
            st.markdown('<div class="chip-row">', unsafe_allow_html=True)
            chip_cols = st.columns(2)
            for i, ex in enumerate(assistant.EXAMPLE_QUESTIONS[:4]):
                if chip_cols[i % 2].button(ex, key=f"chip_{i}", use_container_width=True):
                    chip_clicked = ex
            st.markdown("</div>", unsafe_allow_html=True)

        with st.form("assistant_form", clear_on_submit=True):
            in_col, btn_col = st.columns([6, 1])
            with in_col:
                user_q = st.text_input("Message", label_visibility="collapsed",
                                        placeholder="Comment puis-je vous aider ?")
            with btn_col:
                # Bouton d'envoi icone seule (fleche "entree"), rond et vert, comme le bouton
                # d'envoi d'une conversation Claude -- pas de texte "Envoyer" (ni infobulle,
                # round 13-ter : l'icone seule suffit, demande explicite de retrait).
                sent = st.form_submit_button("↵", use_container_width=True)

        question = chip_clicked or (user_q if sent and user_q else None)
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            reply = assistant.answer(question, state, final_models, best_model_name, mw_mean_rmse)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
