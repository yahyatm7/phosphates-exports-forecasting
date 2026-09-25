"""
Assistant conversationnel explicatif -- repond aux questions sur le projet et ses
resultats en s'appuyant directement sur le pipeline. Architecture **hybride routeur + LLM** :

1. Un routeur d'intentions par mots-cles (regles precises + filet de secours par
   recouvrement lexical, insensible aux accents, + rapprochement flou pour les fautes de
   frappe) traite les questions "simples" -- tout ce qui touche un sujet connu du projet.
   Rapide, gratuit, deterministe, et **toujours grounded** (les chiffres viennent directement
   du pipeline, jamais inventes). Une question floue (un seul mot-cle faible, ou reconnue
   seulement par rapprochement flou) recoit une reponse **avec une nuance explicite**
   ("Je pense que vous voulez parler de ... ") plutot qu'une affirmation aussi ferme qu'une
   correspondance precise -- l'assistant reste honnete sur son niveau de confiance tout en
   restant utile.
2. Si aucun sujet connu ne matche, meme approximativement (question complexe, hors du
   perimetre couvert par le routeur), et **seulement si une cle API Anthropic est
   configuree** (variable d'environnement `ANTHROPIC_API_KEY`, optionnelle), un appel LLM
   prend le relai -- avec les resultats reels du pipeline injectes dans son prompt systeme
   pour eviter les hallucinations. Sans cle configuree (cas par defaut), l'assistant se
   comporte exactement comme avant : 100% routeur, aucune dependance externe, aucun cout.

Utilise par l'assistant flottant de app.py.

Architecture du routeur : chaque sujet est une fonction `_h_xxx(q, ctx)` -> texte markdown,
associee a un libelle lisible dans `_TOPIC_LABELS` (pour les reponses nuancees). `answer()`
essaie d'abord des regles precises (motifs regex, priorite aux sujets qui se recoupent, ex.
"meilleur modele" avant "modele" seul) ; puis, si rien ne matche, le filet de secours par
recouvrement de mots-cles (`_fallback`, insensible aux accents) sur TOUS les sujets connus --
reponse nuancee si le recouvrement est faible (un seul mot-cle) ; puis, si toujours rien, un
rapprochement flou tolerant aux fautes de frappe (`_fuzzy_fallback`, via `difflib`) ; et enfin,
en dernier recours, le LLM optionnel decrit ci-dessus.
"""
import difflib
import os
import re
import unicodedata

import pipeline as pl


def _strip_accents(s):
    """Retire les accents (é -> e, etc.) pour un rapprochement de mots-cles insensible aux
    accents -- les questions tapees vite omettent souvent les accents, et les mots-cles de
    ce fichier sont eux-memes tous ecrits sans accent."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

EXAMPLE_QUESTIONS = [
    "Quel est le meilleur modele et pourquoi ?",
    "Compare les resultats des 6 modeles",
    "Qu'est-ce que la CCF et a quoi sert-elle ici ?",
    "La serie est-elle stationnaire ?",
    "Quelles sont les variables explicatives ?",
    "Quelle est la prevision a 6 mois ?",
    "Quelles sont les limites du projet ?",
    "Explique les 6 phases du CRISP-DM",
]

MODEL_ALIASES = {
    "arima": "ARIMA", "sarimax": "SARIMAX", "ridge": "Ridge",
    "random forest": "Random Forest", "randomforest": "Random Forest", "rf": "Random Forest",
    "xgboost": "XGBoost", "xgb": "XGBoost",
}

# Libelle d'origine (avant renommage technique) de chaque variable exogene -- pour une
# reponse lisible sur les "produits remarquables".
EXOG_LABELS = {code: label for label, code in pl.COL_MAP.items() if code != "target"}


def _find_model_in_text(text):
    text_low = text.lower()
    for alias, name in MODEL_ALIASES.items():
        if alias in text_low:
            return name
    return None


def _find_horizon_in_text(text, default=6):
    m = re.search(r"(\d{1,2})\s*mois", text.lower())
    if m:
        n = int(m.group(1))
        return max(1, min(n, 24))
    return default


# ---------------------------------------------------------------------------
# Sujets -- chaque handler recoit (q, ctx) et renvoie du markdown.
# ctx = {"state":..., "final_models":..., "best_model_name":..., "mw_mean_rmse":...}
# ---------------------------------------------------------------------------

def _h_greeting(q, ctx):
    return (
        "Bonjour ! Je peux repondre librement a vos questions sur le projet -- pas "
        "seulement celles suggerees. Par exemple : la demarche (CRISP-DM), la comparaison "
        "des modeles, la selection des variables (CCF/ACF), la stationnarite, les variables "
        "exogenes (\"produits remarquables\"), une prevision chiffree, ou les limites du "
        "projet. Posez votre question comme vous le souhaitez."
    )


def _h_help(q, ctx):
    return (
        "Je reponds a partir des resultats reellement calcules par le pipeline (pas de "
        "reponse inventee). Sujets couverts : objectif et demarche du projet (cycle "
        "CRISP-DM), donnees et variables exogenes, stationnarite (ADF/KPSS), selection des "
        "lags (CCF/ACF), comparaison des 6 modeles, prevision chiffree (modele et horizon au "
        "choix), mode what-if (via la CLI), limites et axes d'amelioration. Vous pouvez "
        "poser vos questions librement, en francais courant -- pas besoin de mots-cles exacts."
    )


def _h_thanks(q, ctx):
    return "Avec plaisir. N'hesitez pas si vous avez d'autres questions sur le projet."


def _h_objectif(q, ctx):
    state = ctx["state"]
    df = state["df"]
    return (
        f"Le projet vise a prevoir les exportations mensuelles de « Phosphates et derives » "
        f"(cible, en Millions DH) a partir de leur historique et de variables exogenes : des "
        f"importations en Admission Temporaire (AT) de **produits remarquables** ({len(pl.EXOG_RAW)} "
        f"au total) contribuant directement ou indirectement a la transformation et a la "
        f"production des phosphates (engrais, acide phosphorique, machines, hydrocarbures...). "
        f"Une correlation entre ces importations et les exportations n'etablit pas de lien de "
        f"causalite : elle reflete une dynamique d'activite globale, pas un mecanisme causal "
        f"direct. Les donnees couvrent {len(df)} mois ({df.index.min():%m/%Y} a "
        f"{df.index.max():%m/%Y})."
    )


def _h_variables(q, ctx):
    noms = ", ".join(sorted(EXOG_LABELS.values(), key=str.lower))
    return (
        f"La cible est « Phosphates et derives » (exportations, en **Millions DH**). Les "
        f"{len(pl.EXOG_RAW)} variables explicatives sont des **produits remarquables** importes "
        f"en Admission Temporaire (AT), contribuant directement ou indirectement a la "
        f"transformation et a la production des phosphates -- leurs valeurs sont en **DH** "
        f"(et non en Millions DH comme la cible, attention a l'echelle). Liste : {noms}."
    )


def _h_best_model(q, ctx):
    state, best_model_name, mw_mean_rmse = ctx["state"], ctx["best_model_name"], ctx["mw_mean_rmse"]
    r = state["results"].loc[best_model_name]
    return (
        f"**{best_model_name}** est le modele retenu : c'est celui qui obtient le RMSE moyen "
        f"le plus bas sur la validation multi-fenetres ({mw_mean_rmse.loc[best_model_name]:.0f} M DH "
        f"en moyenne sur 7 fenetres de 12 mois), ce qui confirme son bon score sur le test unique "
        f"(RMSE {r['RMSE']:.0f} M DH, MAE {r['MAE']:.0f} M DH, MAPE {r['MAPE']:.1f}%). "
        f"La validation multi-fenetres sert precisement a verifier qu'un modele n'est pas seulement "
        f"bon sur une seule coupure temporelle par chance."
    )


def _h_comparaison(q, ctx):
    state = ctx["state"]
    res = state["results"].sort_values("RMSE")
    lines = [f"{i+1}. **{name}** -- RMSE {row.RMSE:.0f} M DH, R² {row.R2:.2f}"
             for i, (name, row) in enumerate(res.iterrows())]
    return "Classement par RMSE (test unique, 20% derniers mois) :\n\n" + "\n".join(lines)


def _h_sarimax_pourquoi(q, ctx):
    state = ctx["state"]
    r = state["results"].loc["SARIMAX"]
    return (
        f"SARIMAX est le modele le moins bon (RMSE {r['RMSE']:.0f} M DH) : il combine un ordre "
        f"ARIMA, une composante saisonniere et {len(state['selected_lags'])} exogenes pour "
        f"seulement {len(state['train'])} mois d'entrainement -- un ratio parametres/observations "
        f"defavorable qui favorise le sur-ajustement plutot que la generalisation."
    )


def _h_sarimax(q, ctx):
    state = ctx["state"]
    r = state["results"].loc["SARIMAX"]
    return (
        f"SARIMAX (ordre {state['sarimax_order']}, saisonnalite {state['sarimax_seasonal']}) "
        f"est le seul modele autoregressif a integrer les exogenes decalees. RMSE test unique : "
        f"{r['RMSE']:.0f} M DH -- le moins bon des 6 modeles ici (voir \"pourquoi SARIMAX sous-performe\")."
    )


def _h_ccf(q, ctx):
    state = ctx["state"]
    n_sel = len(state["selected_lags"])
    return (
        f"La CCF (correlation croisee) mesure, pour chaque variable exogene, sa correlation avec "
        f"la cible decalee de 0 a 12 mois -- elle relie la cible aux valeurs **passees des autres "
        f"variables**, contrairement a l'ACF qui ne regarde que le passe de la cible elle-meme. "
        f"Calculee sur les donnees d'entrainement uniquement (pour eviter toute fuite), elle retient "
        f"{n_sel} des {len(pl.EXOG_RAW)} exogenes dont le meilleur lag depasse le seuil |CCF| >= 0,20. "
        f"Affichee en grille de graphiques par variable plutot qu'en heatmap unique : la heatmap "
        f"n'aurait montre qu'une intensite par case, sans la forme du signal lag par lag."
    )


def _h_acf(q, ctx):
    state = ctx["state"]
    return (
        f"L'ACF (autocorrelation) mesure la correlation de la cible **avec son propre passe** -- "
        f"utile pour choisir ses propres lags, distinct de la CCF (cible vs exogenes). Les "
        f"{pl.N_TARGET_LAGS} lags de plus forte magnitude |ACF| sur la cible differenciee (train) "
        f"sont retenus : {state['lag_target']}."
    )


def _h_lags(q, ctx):
    state = ctx["state"]
    return (
        f"Deux familles de lags dans ce projet : les lags **exogenes** (choisis par CCF, "
        f"{len(state['selected_lags'])} variables retenues, seuil |CCF| >= 0,20) et les lags de la "
        f"**cible elle-meme** (choisis par ACF, {state['lag_target']}). Les deux sont calcules "
        f"uniquement sur les donnees d'entrainement pour eviter toute fuite d'information."
    )


def _h_stationnarite(q, ctx):
    state = ctx["state"]
    rows = state["stationarity"]
    lines = [f"- {r['serie']} : ADF p={r['adf_pvalue']:.3f}, KPSS p={r['kpss_pvalue']:.3f} -> **{r['verdict']}**"
             for _, r in rows.iterrows()]
    return (
        "Un seul test (ADF) ne suffit pas a conclure : ADF (H0 = non stationnaire) et KPSS "
        "(H0 = stationnaire) sont complementaires et donc croises ici :\n\n" + "\n".join(lines)
    )


def _h_whatif(q, ctx):
    return (
        "Le mode what-if remplace la persistance par defaut (chaque exogene figee a sa derniere "
        "valeur connue) par une hypothese : pourcentage (`+20%`, impossible si la derniere valeur "
        "est nulle) ou valeur absolue en Millions DH. Deux limites intrinseques : sans effet sur "
        "ARIMA (aucune exogene utilisee), et sans effet si le lag CCF de la variable depasse "
        "l'horizon de prevision. Disponible via la CLI : `python cli.py predict --what-if "
        "\"engrais=+20%\"`."
    )


def _h_differenciation(q, ctx):
    return (
        "La differenciation (predire la variation mois-sur-mois plutot que le niveau) a ete testee "
        "comme transformation des features ML, en plus de l'approche en niveau. Resultat : pas de "
        "gain net constant selon le modele (Ridge et XGBoost reagissent differemment) -- les "
        "features en niveau restent donc le choix retenu pour le pipeline final."
    )


def _h_metrics(q, ctx):
    return (
        "Quatre indicateurs sont utilises pour comparer les modeles : **RMSE** (racine de "
        "l'erreur quadratique moyenne, en Millions DH -- penalise fort les grosses erreurs), "
        "**MAE** (erreur absolue moyenne, en Millions DH -- plus lisible, moins sensible aux "
        "valeurs extremes), **MAPE** (erreur absolue moyenne en %, comparable entre echelles) et "
        "**R²** (part de variance expliquee -- ici souvent negatif, ce qui est attendu sur une "
        "serie courte et volatile : cela signifie que le modele fait moins bien qu'une simple "
        "moyenne constante, pas qu'il est inutilise -- le RMSE/MAPE restent les criteres de choix)."
    )


def _h_limites(q, ctx):
    ctx_best = ctx["best_model_name"]
    return (
        f"Limites et axes d'amelioration identifies : (1) les variables exogenes futures ne sont "
        f"pas reellement prevues, elles sont figees a leur derniere valeur connue (le mode what-if "
        f"permet de tester des hypotheses manuellement, via la CLI) ; (2) le modele retenu "
        f"({ctx_best}) devrait etre reevalue a mesure que de nouvelles observations arrivent, "
        f"plutot que fige definitivement ; (3) aucune surveillance automatique des changements de "
        f"regime de la serie (rupture structurelle) n'est mise en place."
    )


def _h_crisp_dm(q, ctx):
    state, best = ctx["state"], ctx["best_model_name"]
    verdict_level = state["stationarity"].iloc[0]["verdict"]
    return (
        "Le projet suit les 6 phases classiques du cycle **CRISP-DM** :\n\n"
        "1. 🎯 **Business understanding** -- anticiper les exportations de phosphates et derives, "
        "pilier de l'economie marocaine.\n"
        f"2. 📊 **Data understanding** -- {len(state['df'])} mois, 1 cible + {len(pl.EXOG_RAW)} "
        f"exogenes (\"produits remarquables\" importes en AT).\n"
        f"3. 🧹 **Data preparation** -- stationnarite (ADF+KPSS, niveau : {verdict_level}), "
        f"selection CCF (exogenes) et ACF (cible), feature engineering (lags + moyennes mobiles).\n"
        "4. 🤖 **Modeling** -- 6 modeles entraines (Naif saisonnier, Ridge, Random Forest, "
        "XGBoost, ARIMA, SARIMAX).\n"
        "5. ✅ **Evaluation** -- test unique (20%) puis validation multi-fenetres (7 x 12 mois) "
        f"-> modele retenu : **{best}**.\n"
        "6. 🚀 **Deployment** -- dashboard Streamlit interactif + assistant explicatif.\n\n"
        "Le detail avec les chiffres reels est dans l'onglet **Pipeline**."
    )


def _h_prevision(q, ctx):
    state, final_models, best_model_name = ctx["state"], ctx["final_models"], ctx["best_model_name"]
    model_name = _find_model_in_text(q) or best_model_name
    horizon = _find_horizon_in_text(q, default=pl.N_FORECAST_DEFAULT)
    if model_name not in final_models:
        return f"Modele inconnu. Choix possibles : {', '.join(final_models.keys())}."
    forecast = pl.forecast_with_model(state, final_models, model_name, horizon)
    lines = [f"- {d.strftime('%Y-%m')} : {v:.0f} M DH" for d, v in forecast.items()]
    note = "" if model_name != "ARIMA" else "\n\n(ARIMA est autoregressif pur -- un scenario what-if n'aurait aucun effet sur cette prevision.)"
    return (
        f"Prevision a {horizon} mois avec **{model_name}** (exogenes figees a leur derniere valeur "
        f"connue) :\n\n" + "\n".join(lines) + note
    )


def _grounded_facts(ctx):
    """Resume factuel du pipeline (donnees reelles, pas de texte invente) -- injecte dans le
    prompt systeme du LLM optionnel pour qu'il reponde UNIQUEMENT a partir de ces faits, plutot
    que de deviner ou d'inventer des chiffres."""
    state = ctx["state"]
    df = state["df"]
    res = state["results"].sort_values("RMSE").round(1)
    lignes_res = "\n".join(
        f"- {name} : RMSE={row.RMSE}, MAE={row.MAE}, MAPE={row.MAPE}%, R2={row.R2}"
        for name, row in res.iterrows()
    )
    noms_exog = ", ".join(sorted(EXOG_LABELS.values(), key=str.lower))
    return (
        f"Projet : prevision des exportations mensuelles de « Phosphates et derives » "
        f"(cible, Millions DH), INSEA.\n"
        f"Donnees : {len(df)} mois ({df.index.min():%m/%Y} a {df.index.max():%m/%Y}).\n"
        f"Variables exogenes ({len(pl.EXOG_RAW)}, \"produits remarquables\" importes en "
        f"Admission Temporaire, valeurs en DH) : {noms_exog}.\n"
        f"Demarche : cycle CRISP-DM (business & data understanding, data preparation -- "
        f"stationnarite ADF+KPSS, selection CCF/ACF --, modeling -- 6 modeles --, evaluation "
        f"-- test unique puis multi-fenetres --, deployment -- dashboard Streamlit).\n"
        f"Resultats (test unique, tries par RMSE croissant) :\n{lignes_res}\n"
        f"Modele retenu (RMSE moyen le plus bas sur validation multi-fenetres, 7 fenetres de "
        f"12 mois) : {ctx['best_model_name']}.\n"
        f"Limites connues : exogenes futures figees a leur derniere valeur connue (pas de "
        f"prevision reelle des exogenes), pas de reevaluation continue du modele, pas de "
        f"surveillance des changements de regime."
    )


def _try_llm(question, ctx):
    """Filet de secours optionnel : si une cle API Anthropic est configuree (variable
    d'environnement ANTHROPIC_API_KEY), tente une reponse LLM pour une question que le
    routeur n'a pas reconnue -- avec les faits reels du pipeline injectes dans le prompt
    systeme pour eviter toute hallucination. Retourne None (et laisse `answer()` afficher
    son message d'invitation a reformuler) si : pas de cle configuree, package `anthropic`
    non installe, ou tout probleme reseau/API -- l'assistant reste donc pleinement
    fonctionnel meme sans cette dependance optionnelle."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        system_prompt = (
            "Tu es l'assistant explicatif du dashboard d'un projet academique INSEA de "
            "prevision des exportations de phosphates. Reponds en francais, de facon "
            "concise (5-8 lignes maximum) et professionnelle. Appuie-toi UNIQUEMENT sur "
            "les faits ci-dessous -- ne donne et n'invente JAMAIS un chiffre qui n'y figure "
            "pas. Si la question sort du perimetre de ces faits (hors sujet du projet), "
            "dis-le clairement plutot que d'inventer une reponse. Si la question est floue "
            "ou ambigue, ne refuse pas sechement : propose ton interpretation la plus "
            "probable (\"Je pense que vous voulez dire ... \") puis reponds sur cette base, "
            "en invitant a preciser si tu t'es trompe.\n\n" + _grounded_facts(ctx)
        )
        resp = client.messages.create(
            model=model, max_tokens=500, system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text")).strip()
        return text or None
    except Exception:
        return None


# Table de dispatch : (regex de declenchement precis) -> handler. Ordre = priorite (les
# sujets les plus specifiques d'abord pour eviter qu'un mot generique comme "modele" ne
# capture une question qui visait autre chose).
_RULES = [
    (re.compile(r"\b(bonjour|salut|hello|coucou|bonsoir)\b"), _h_greeting),
    (re.compile(r"\b(merci|top|parfait genial|super travail)\b"), _h_thanks),
    (re.compile(r"aide|capacit|que peux.tu|qui es.tu|es tu un chatbot|es-tu un llm"), _h_help),
    (re.compile(r"meilleur|retenu|quel mod[eè]le"), _h_best_model),
    (re.compile(r"compar|classement|r[eé]sultats|tableau"), _h_comparaison),
    (re.compile(r"sarimax.*(pourquoi|sous.perform|mauvais|pire|faible)"), _h_sarimax_pourquoi),
    (re.compile(r"sarimax"), _h_sarimax),
    (re.compile(r"\bccf\b|corr[eé]lation crois[eé]e"), _h_ccf),
    (re.compile(r"\bacf\b|(lag.*cible)"), _h_acf),
    (re.compile(r"\blag\b|lags?\b"), _h_lags),
    (re.compile(r"stationnaire|\badf\b|\bkpss\b"), _h_stationnarite),
    (re.compile(r"what.if|what if|sc[eé]nario"), _h_whatif),
    (re.compile(r"diff[eé]renc"), _h_differenciation),
    (re.compile(r"\brmse\b|\bmae\b|\bmape\b|\br2\b|r²|indicateur|metrique"), _h_metrics),
    (re.compile(r"limite|am[eé]lioration|faiblesse|inconv[eé]nient"), _h_limites),
    (re.compile(r"crisp|phase|cycle|\bbusiness understanding\b"), _h_crisp_dm),
    (re.compile(r"colonnes?|variables? explicative|variables? exog|produits? remarquable|quelles? variables"), _h_variables),
    (re.compile(r"objectif|but du projet|pourquoi ce projet|de quoi s'agit"), _h_objectif),
    (re.compile(r"d[eé]marche|m[eé]thode|comment.*fonctionne|pipeline"), _h_crisp_dm),
    (re.compile(r"pr[eé]vision|pr[eé]voir|combien"), _h_prevision),
]

# Sujets pour le filet de secours (recouvrement de mots-cles) -- reutilise les memes
# handlers, mais avec un ensemble de mots-cles plus large pour capter une reformulation
# qui n'aurait declenche aucune regle precise ci-dessus. Renforce (round 11) avec davantage
# de synonymes/variantes par sujet, pour couvrir plus de reformulations naturelles.
_FALLBACK_TOPICS = [
    ({"bonjour", "salut", "hello", "coucou", "bonsoir", "hey"}, _h_greeting),
    ({"merci", "sympa", "cool", "top", "genial", "nickel"}, _h_thanks),
    ({"aide", "capacite", "capacites", "chatbot", "assistant", "faire", "fonctionnalites",
      "sais", "connais", "peux", "utilite", "usage"}, _h_help),
    ({"meilleur", "modele", "retenu", "choix", "gagnant", "top1", "numero1", "premier",
      "recommande"}, _h_best_model),
    ({"comparaison", "comparer", "compare", "classement", "resultats", "tableau",
      "performance", "versus", "difference"}, _h_comparaison),
    ({"sarimax"}, _h_sarimax),
    ({"ccf", "correlation", "correlations", "croisee", "croisees"}, _h_ccf),
    ({"acf", "autocorrelation", "autocorrelations"}, _h_acf),
    ({"lag", "lags", "decalage", "decalages", "retard", "retards"}, _h_lags),
    ({"stationnaire", "stationnarite", "stationnaires", "adf", "kpss", "tendance",
      "racine", "unitaire"}, _h_stationnarite),
    ({"whatif", "what-if", "scenario", "hypothese", "hypotheses", "simulation",
      "simuler"}, _h_whatif),
    ({"differenciation", "differencier", "differencie", "difference", "niveau"}, _h_differenciation),
    ({"rmse", "mae", "mape", "r2", "metrique", "metriques", "indicateur", "indicateurs",
      "erreur", "precision", "score", "scores", "qualite", "fiabilite"}, _h_metrics),
    ({"limite", "limites", "limitation", "limitations", "amelioration", "ameliorations",
      "faiblesse", "faiblesses", "inconvenient", "probleme", "contrainte",
      "contraintes"}, _h_limites),
    ({"crisp", "phase", "phases", "cycle", "etapes", "etape", "demarche", "methode",
      "methodologie", "pipeline", "workflow", "processus"}, _h_crisp_dm),
    ({"colonne", "colonnes", "variable", "variables", "exogene", "exogenes", "produit",
      "produits", "remarquable", "remarquables", "importation", "importations", "feature",
      "features", "entree", "entrees", "predicteur", "predicteurs"}, _h_variables),
    ({"objectif", "but", "sujet", "projet", "presentation", "resume", "contexte",
      "description"}, _h_objectif),
    ({"prevision", "prevoir", "predire", "predite", "futur", "prochain", "mois",
      "horizon", "estimation", "anticipation"}, _h_prevision),
]

# Libelle lisible par handler -- utilise dans les reponses nuancees ("Je pense que vous
# voulez parler de ...") pour les questions floues, capturees par un recouvrement faible ou
# un rapprochement approximatif plutot qu'une correspondance nette.
_TOPIC_LABELS = {
    _h_greeting: "un salut",
    _h_thanks: "un remerciement",
    _h_help: "l'aide / les capacites de l'assistant",
    _h_best_model: "le modele retenu",
    _h_comparaison: "la comparaison des 6 modeles",
    _h_sarimax_pourquoi: "SARIMAX",
    _h_sarimax: "SARIMAX",
    _h_ccf: "la CCF (correlation croisee)",
    _h_acf: "l'ACF (autocorrelation)",
    _h_lags: "les lags utilises dans le pipeline",
    _h_stationnarite: "la stationnarite de la serie",
    _h_whatif: "le mode what-if",
    _h_differenciation: "la differenciation (niveau vs variation)",
    _h_metrics: "les indicateurs de performance (RMSE/MAE/MAPE/R²)",
    _h_limites: "les limites du projet",
    _h_crisp_dm: "la demarche CRISP-DM",
    _h_variables: "les variables explicatives (\"produits remarquables\")",
    _h_objectif: "l'objectif du projet",
    _h_prevision: "la prevision chiffree",
}

# Vocabulaire complet des mots-cles connus (union de tous les sujets), pour le rapprochement
# flou (fautes de frappe) ci-dessous.
_ALL_KEYWORDS = set()
for _kws, _handler in _FALLBACK_TOPICS:
    _ALL_KEYWORDS |= _kws


def _tokenize(q):
    """Tokenise et normalise (accents retires) une question pour le matching par mots-cles."""
    return {_strip_accents(t) for t in re.findall(r"[a-zàâäéèêëïîôöùûüç]+", q.lower())}


def _fallback(q):
    """Filet de secours : score chaque sujet connu par recouvrement de mots-cles (insensible
    aux accents) avec la question, et renvoie (handler, score) pour le sujet le mieux note
    -- au lieu de s'arreter des la premiere regle manquee, une question reformulee, mal
    orthographiee ou hors des exemples suggeres a de bonnes chances d'obtenir une vraie
    reponse. Le score sert ensuite a `answer()` pour decider si la reponse doit etre nuancee
    (score faible = question probablement floue) ou directe (score fort)."""
    tokens = _tokenize(q)
    best_score, best_handler = 0, None
    for keywords, handler in _FALLBACK_TOPICS:
        score = len(tokens & keywords)
        if score > best_score:
            best_score, best_handler = score, handler
    return best_handler, best_score


def _fuzzy_fallback(q):
    """Dernier filet avant le LLM : rapprochement APPROXIMATIF (tolerant aux fautes de
    frappe/orthographe) entre les mots de la question et le vocabulaire connu du routeur, via
    `difflib`. N'intervient que si `_fallback` n'a rien trouve du tout (aucun mot-cle exact) --
    permet de recuperer une question du style "stationnairte" ou "previsionn" sans mot-cle
    exact. Seuil de similarite volontairement eleve (0.82) et mots courts ignores (< 4
    caracteres) pour eviter les faux rapprochements hasardeux."""
    tokens = _tokenize(q)
    scores = {}
    for tok in tokens:
        if len(tok) < 4:
            continue
        close = difflib.get_close_matches(tok, _ALL_KEYWORDS, n=1, cutoff=0.82)
        if not close:
            continue
        matched_kw = close[0]
        for keywords, handler in _FALLBACK_TOPICS:
            if matched_kw in keywords:
                scores[handler] = scores.get(handler, 0) + 1
                break
    if not scores:
        return None
    return max(scores, key=scores.get)


def answer(question, state, final_models, best_model_name, mw_mean_rmse):
    """Retourne une reponse texte (markdown) a une question libre sur le projet. Architecture
    hybride, en 4 paliers de confiance decroissante :
    (1) regles precises (_RULES) -> reponse directe, confiance maximale ;
    (2) filet de secours par mots-cles (_fallback) avec un recouvrement fort (score >= 2)
        -> reponse directe, toujours confiante ;
    (3) filet de secours avec un recouvrement faible (score == 1, question probablement
        floue) OU rapprochement flou tolerant aux fautes de frappe (_fuzzy_fallback, aucun
        mot-cle exact mais un mot proche) -> reponse NUANCEE ("Je pense que vous voulez
        parler de ... "), pour rester honnete sur le niveau de confiance sans pour autant
        laisser la question sans reponse ;
    (4) si rien du tout n'est reconnu, meme approximativement : LLM optionnel (voir
        `_try_llm`), UNIQUEMENT si une cle API est configuree, sinon message final invitant
        a reformuler plutot que de bloquer sechement."""
    q = question.strip().lower()
    if not q:
        return "Posez-moi une question sur le projet, librement -- les exemples ci-dessous ne sont que des idees de depart."

    ctx = {
        "state": state, "final_models": final_models,
        "best_model_name": best_model_name, "mw_mean_rmse": mw_mean_rmse,
    }

    for pattern, handler in _RULES:
        if pattern.search(q):
            return handler(q, ctx)

    fb_handler, fb_score = _fallback(q)
    if fb_handler is not None:
        if fb_score >= 2:
            return fb_handler(q, ctx)
        # Un seul mot-cle en commun : correspondance plausible mais pas franche -- la
        # question est probablement floue ou tres elliptique. On repond quand meme, mais en
        # signalant l'interpretation plutot qu'en l'affirmant comme une certitude.
        label = _TOPIC_LABELS.get(fb_handler, "ce sujet")
        return (
            f"Je pense que votre question porte sur **{label}** -- n'hesitez pas a preciser "
            f"si ce n'est pas le cas :\n\n" + fb_handler(q, ctx)
        )

    fuzzy_handler = _fuzzy_fallback(q)
    if fuzzy_handler is not None:
        label = _TOPIC_LABELS.get(fuzzy_handler, "ce sujet")
        return (
            f"Votre question n'est pas totalement claire pour moi (peut-etre une coquille "
            f"ou une formulation inhabituelle), mais je pense qu'elle porte peut-etre sur "
            f"**{label}** -- reformulez si je me trompe :\n\n" + fuzzy_handler(q, ctx)
        )

    # Question non reconnue par le routeur, meme approximativement -> probablement complexe
    # ou hors motifs connus : tenter le LLM optionnel (no-op silencieux si aucune cle n'est
    # configuree).
    llm_reply = _try_llm(question, ctx)
    if llm_reply:
        return llm_reply

    return (
        "Je n'ai pas de reponse toute prete pour cette formulation precise -- je m'appuie sur "
        "les resultats reels du pipeline (et, si une cle API est configuree, un LLM pour les "
        "questions plus complexes -- non configure ici). Essayez de reformuler en citant le "
        "sujet (ex. \"modeles\", \"stationnarite\", \"variables\", \"prevision\", \"limites\", "
        "\"CRISP-DM\"), ou consultez les onglets **Pipeline** et **Le projet** pour une vue "
        "d'ensemble."
    )
