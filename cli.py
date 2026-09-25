#!/usr/bin/env python3
"""
CLI de prevision des exportations de phosphates et derives.

Usage :
    python cli.py train                                 # entraine et sauvegarde les modeles
    python cli.py evaluate                               # tableaux de resultats (entraine si besoin)
    python cli.py predict --horizon 6                     # prevision a 6 mois (meilleur modele)
    python cli.py predict --horizon 6 --model "Random Forest"
    python cli.py predict --horizon 6 --what-if "engrais=+20,phosphates_bruts=-10"
    python cli.py predict --horizon 6 --output forecast.csv

Le premier appel a `evaluate` ou `predict` entraine automatiquement et sauvegarde les
artefacts dans models/artifacts.joblib ; les appels suivants les rechargent (rapide),
sauf --retrain qui force un nouvel entrainement.
"""
import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

import pipeline as pl
import plots

ARTIFACTS_PATH = Path(__file__).parent / "models" / "artifacts.joblib"

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger("phosphates-cli")


def train(data_path, force=False):
    if ARTIFACTS_PATH.exists() and not force:
        log.info("Artefacts deja presents (%s) -- utilise --retrain pour forcer.", ARTIFACTS_PATH)
        return load_artifacts()

    log.info("Entrainement du pipeline complet (donnees -> CCF/lags -> 6 modeles)...")
    state = pl.run_full_pipeline(data_path)
    log.info("Modele ARIMA : ordre %s (convergence : %s)", state["arima_order"],
              "ok" if state["arima_converged"] else "non atteinte")
    log.info("Modele SARIMAX : ordre %s %s (convergence : %s)", state["sarimax_order"],
              state["sarimax_seasonal"], "ok" if state["sarimax_converged"] else "non atteinte")

    log.info("Evaluation multi-fenetres (top2 = %s)...", state["top2"])
    mw_results, mw_mean_rmse, best_model_name = pl.multi_window_eval(state)
    log.info("Modele retenu (RMSE moyen multi-fenetres) : %s", best_model_name)

    log.info("Reentrainement des 6 modeles sur les donnees completes...")
    final_models = pl.refit_all_models_full_data(state)

    artifacts = {
        "state": state,
        "final_models": final_models,
        "mw_results": mw_results,
        "mw_mean_rmse": mw_mean_rmse,
        "best_model_name": best_model_name,
    }
    ARTIFACTS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifacts, ARTIFACTS_PATH)
    log.info("Artefacts sauvegardes dans %s", ARTIFACTS_PATH)
    return artifacts


def load_artifacts():
    return joblib.load(ARTIFACTS_PATH)


def cmd_train(args):
    artifacts = train(args.data, force=True)
    print("\n=== Test unique (derniers 20%) ===")
    print(artifacts["state"]["results"].round(2).to_string())
    print(f"\nModele retenu pour le deploiement : {artifacts['best_model_name']}")
    path1 = plots.plot_results_comparison(artifacts["state"]["results"])
    path2 = plots.plot_multi_window(artifacts["mw_results"])
    log.info("Graphiques sauvegardes : %s , %s", path1, path2)


def cmd_evaluate(args):
    artifacts = train(args.data, force=args.retrain)
    state = artifacts["state"]

    print("\n=== Test unique (derniers 20%) ===")
    print(state["results"].round(2).to_string())
    print(f"\ntop2 (RMSE test unique) : {state['top2']}")

    print("\n=== Multi-fenetres (RMSE moyen sur 7 fenetres, top2 uniquement) ===")
    print(artifacts["mw_mean_rmse"].round(2).to_string())
    print(f"\nModele retenu pour le deploiement : {artifacts['best_model_name']}")

    path1 = plots.plot_results_comparison(state["results"])
    path2 = plots.plot_multi_window(artifacts["mw_results"])
    log.info("Graphiques sauvegardes : %s , %s", path1, path2)


def _parse_what_if(raw):
    if not raw:
        return None
    overrides = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"format invalide : {part!r} (attendu var=+20%% ou var=5.2)")
        col, value = part.split("=", 1)
        col, value = col.strip(), value.strip()
        mode = "pct" if value.endswith("%") else "abs"
        value = value[:-1] if mode == "pct" else value
        try:
            overrides[col] = (mode, float(value))
        except ValueError:
            raise argparse.ArgumentTypeError(f"valeur invalide dans : {part!r}")
    return overrides


def cmd_predict(args):
    artifacts = train(args.data, force=args.retrain)
    state = artifacts["state"]
    final_models = artifacts["final_models"]
    model_name = args.model or artifacts["best_model_name"]

    if model_name not in final_models:
        log.error("Modele inconnu : %s (choix possibles : %s)", model_name, list(final_models.keys()))
        sys.exit(1)

    overrides = _parse_what_if(args.what_if)
    if overrides:
        unknown = set(overrides) - set(pl.EXOG_RAW)
        if unknown:
            log.error("Variable(s) exogene(s) inconnue(s) : %s", unknown)
            log.error("Variables disponibles : %s", pl.EXOG_RAW)
            sys.exit(1)
        if model_name == "ARIMA":
            log.warning("--what-if n'a aucun effet sur ARIMA (autoregressif pur, aucune exogene).")
        else:
            for col in overrides:
                lag = state["selected_lags"].get(col)
                if lag is not None and lag >= args.horizon:
                    log.warning(
                        "%s (lag=%d) n'aura aucun effet avec --horizon %d : il faut un horizon "
                        ">= %d pour que cette variable influence la prevision.",
                        col, lag, args.horizon, lag + 1,
                    )
        try:
            future_exog = pl.what_if_exog_frame(state["df"], args.horizon, overrides)
        except ValueError as exc:
            log.error(str(exc))
            sys.exit(1)
        log.info("Mode what-if : %s", overrides)
    else:
        future_exog = None

    forecast = pl.forecast_with_model(state, final_models, model_name, args.horizon, future_exog)

    def _fmt(mode, value):
        return f"{value:+g}%" if mode == "pct" else f"={value:g}"

    what_if_str = ", ".join(f"{k}{_fmt(mode, value)}" for k, (mode, value) in overrides.items()) if overrides else None

    print(f"\nPrevision a {args.horizon} mois -- modele : {model_name}"
          f"{' (what-if : ' + what_if_str + ')' if what_if_str else ''}")
    forecast_df = forecast.rename("prevision_millions_dh").round(1).to_frame()
    print(forecast_df.to_string())

    if args.output:
        forecast_df.to_csv(args.output, index_label="date")
        log.info("Prevision sauvegardee dans %s", args.output)

    what_if_label = what_if_str if overrides and model_name != "ARIMA" else None
    plot_path = plots.plot_forecast(state["df"], forecast, model_name, what_if_label=what_if_label)
    log.info("Graphique sauvegarde : %s", plot_path)


def main():
    parser = argparse.ArgumentParser(description="Prevision des exportations de phosphates et derives (ML + ARIMA/SARIMAX).")
    parser.add_argument("--data", default=str(Path(__file__).parent / "data" / "data_phosphates.xlsx"),
                         help="Chemin vers le fichier de donnees Excel.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Entraine les 6 modeles et sauvegarde les artefacts (ecrase l'existant).")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Affiche les tableaux de resultats (test unique + multi-fenetres).")
    p_eval.add_argument("--retrain", action="store_true", help="Force un reentrainement meme si des artefacts existent.")
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Prevision a N mois.")
    p_pred.add_argument("--horizon", type=int, default=pl.N_FORECAST_DEFAULT, help="Nombre de mois a prevoir (defaut : 6).")
    p_pred.add_argument("--model", default=None,
                         help="Modele a utiliser (Ridge, Random Forest, XGBoost, ARIMA, SARIMAX). "
                              "Par defaut : le modele retenu par le pipeline (RMSE moyen multi-fenetres).")
    p_pred.add_argument("--what-if", default=None,
                         help="Ajuste des exogenes au lieu de persister leur derniere valeur connue. "
                              "Var=+20%% pour un pourcentage (echoue si la derniere valeur est nulle), "
                              "var=5.2 pour une valeur absolue (Millions DH). Ex. \"engrais=+20%%,phosphates_bruts=3\". "
                              "Sans effet sur ARIMA ; sans effet non plus si le lag CCF de la variable "
                              "depasse l'horizon (avertissement affiche le cas echeant).")
    p_pred.add_argument("--output", default=None, help="Chemin CSV ou sauvegarder la prevision.")
    p_pred.add_argument("--retrain", action="store_true", help="Force un reentrainement meme si des artefacts existent.")
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
