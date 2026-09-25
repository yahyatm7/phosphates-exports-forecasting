"""Graphiques generes par le CLI (evaluate / predict) -- sauvegardes en PNG dans outputs/,
pour que le pipeline produise des visuels et pas seulement du texte dans le terminal."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # rendu fichier, pas d'ecran requis
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent / "outputs"


def _save(fig, name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_results_comparison(results, name="results_comparison.png"):
    """Barres RMSE par modele (test unique), triees, meilleur en evidence."""
    ordered = results.sort_values("RMSE")
    colors = ["tab:green" if i == 0 else "tab:blue" for i in range(len(ordered))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(ordered.index, ordered["RMSE"], color=colors)
    for bar, val in zip(bars, ordered["RMSE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (Millions DH)")
    ax.set_title("Comparaison des modeles -- test unique (derniers 20%)")
    plt.xticks(rotation=30, ha="right")
    return _save(fig, name)


def plot_multi_window(mw_results, name="multi_window_r2.png"):
    """R2 par fenetre de test pour les modeles evalues en multi-fenetres (top2)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model_name in mw_results["modele"].unique():
        sub = mw_results[mw_results["modele"] == model_name]
        labels = [f"{d1}\n{d2}" for d1, d2 in zip(sub["debut_test"], sub["fin_test"])]
        ax.plot(labels, sub["R2"], marker="o", label=model_name)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_ylabel("R²")
    ax.set_title("R² par fenetre de test (expanding window, 12 mois)")
    ax.legend()
    plt.xticks(rotation=45, ha="right", fontsize=8)
    return _save(fig, name)


def plot_forecast(df, forecast, model_name, target_col="target", history_months=36,
                   name=None, what_if_label=None):
    """Historique + prevision, meme style que le notebook (section 6.2)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index[-history_months:], df[target_col].iloc[-history_months:], label="historique")
    label = f"prevision ({model_name})" + (f" -- {what_if_label}" if what_if_label else "")
    ax.plot(forecast.index, forecast.values, label=label, linestyle="--", marker="o", color="tab:orange")
    ax.axvline(df.index[-1], color="gray", linestyle=":")
    ax.set_ylabel("Millions DH")
    ax.set_title(f"Prevision a {len(forecast)} mois -- {model_name}")
    ax.legend()
    return _save(fig, name or f"forecast_{model_name.replace(' ', '_').lower()}.png")
