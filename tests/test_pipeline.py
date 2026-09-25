"""Tests de sanite pour pipeline.py. Lancer avec : pytest (depuis la racine du projet)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import pipeline as pl

DATA_PATH = Path(__file__).parent.parent / "data" / "data_phosphates.xlsx"


@pytest.fixture(scope="module")
def state():
    return pl.run_full_pipeline(str(DATA_PATH))


def test_data_loads_and_columns_present():
    df = pl.load_data(str(DATA_PATH))
    assert pl.TARGET in df.columns
    assert all(c in df.columns for c in pl.EXOG_RAW)
    assert len(pl.EXOG_RAW) == 14
    assert df.index.freqstr == "MS"


def test_time_split_is_chronological_and_no_overlap():
    df = pl.load_data(str(DATA_PATH))
    train, test, split_date = pl.time_split(df)
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(df)


def test_lag_selection_uses_train_only_no_leakage(state):
    # tous les lags choisis doivent provenir de EXOG_RAW, avec un lag >= 1 (jamais 0 :
    # une valeur contemporaine n'est pas disponible en prevision)
    for col, lag in state["selected_lags"].items():
        assert col in pl.EXOG_RAW
        assert lag >= 1

    assert len(state["lag_target"]) == pl.N_TARGET_LAGS
    assert all(1 <= lag <= pl.MAX_LAG for lag in state["lag_target"])


def test_results_table_has_all_models(state):
    assert set(state["results"].index) == set(pl.ALL_MODEL_NAMES)
    # RMSE toujours positif et fini
    assert (state["results"]["RMSE"] > 0).all()
    assert state["results"]["RMSE"].notna().all()


def test_arima_is_the_best_model_on_this_dataset(state):
    # verite empirique connue sur ce jeu de donnees (voir notebook) -- si ce test casse,
    # soit les donnees ont change, soit une regression a ete introduite dans le pipeline
    assert state["results"]["RMSE"].idxmin() == "ARIMA"


def test_multi_window_eval_runs_and_agrees_with_best_model(state):
    mw_results, mw_mean_rmse, best_model_name = pl.multi_window_eval(state)
    assert set(mw_results["modele"].unique()) == set(state["top2"])
    assert best_model_name == mw_mean_rmse.index[0]


def test_refit_and_forecast_shapes(state):
    final_models = pl.refit_all_models_full_data(state)
    assert set(final_models.keys()) == set(pl.ALL_MODEL_NAMES) - {"Naive saisonnier"}

    forecast = pl.forecast_with_model(state, final_models, "ARIMA", n_steps=6)
    assert len(forecast) == 6
    assert forecast.notna().all()


def test_what_if_frame_percentage_and_absolute(state):
    df = state["df"]
    frame_abs = pl.what_if_exog_frame(df, 6, {"phosphates_bruts": ("abs", 5.0)})
    assert (frame_abs["phosphates_bruts"] == 5.0).all()

    with pytest.raises(ValueError):
        # derniere valeur connue nulle pour phosphates_bruts -> pourcentage refuse
        pl.what_if_exog_frame(df, 6, {"phosphates_bruts": ("pct", 20.0)})

    with pytest.raises(ValueError):
        pl.what_if_exog_frame(df, 6, {"variable_inexistante": ("abs", 1.0)})


def test_what_if_only_affects_forecast_after_the_lag_horizon(state):
    final_models = pl.refit_all_models_full_data(state)
    lag = state["selected_lags"]["phosphates_bruts"]  # 2 sur ce jeu de donnees

    baseline = pl.forecast_with_model(state, final_models, "Random Forest", n_steps=6)
    future_exog = pl.what_if_exog_frame(state["df"], 6, {"phosphates_bruts": ("abs", 999.0)})
    whatif = pl.forecast_with_model(state, final_models, "Random Forest", n_steps=6, future_exog_raw=future_exog)

    # les mois avant que le lag n'atteigne la periode future doivent etre inchanges
    for i in range(lag):
        assert baseline.iloc[i] == pytest.approx(whatif.iloc[i])
    # au moins un mois apres doit differer
    assert not (baseline.iloc[lag:] == whatif.iloc[lag:]).all()


def test_arima_forecast_ignores_what_if(state):
    final_models = pl.refit_all_models_full_data(state)
    baseline = pl.forecast_with_model(state, final_models, "ARIMA", n_steps=6)
    future_exog = pl.what_if_exog_frame(state["df"], 6, {"phosphates_bruts": ("abs", 999.0)})
    whatif = pl.forecast_with_model(state, final_models, "ARIMA", n_steps=6, future_exog_raw=future_exog)
    assert (baseline.values == whatif.values).all()
