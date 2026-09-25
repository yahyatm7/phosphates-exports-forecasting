"""
Pipeline ML pour la prevision des exportations de phosphates et derives.

Reprend exactement la logique du notebook `Prevision_Exportations_Phosphates_MAJ_5.ipynb`
(demarche CRISP-DM, feature engineering, choix des modeles, evaluation) sous forme de
fonctions reutilisables par l'application Streamlit. Aucune logique metier n'est modifiee
par rapport au notebook -- seule la mise en forme (fonctions au lieu de cellules) change.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from xgboost import XGBRegressor

DEFAULT_DATA_PATH = str(Path(__file__).parent / "data" / "data_phosphates.xlsx")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

COL_MAP = {
    "TOURTEAUX ET AUTRES RÉSIDUS DES INDUSTRIES ALIMENTAIRES": "tourteaux",
    "ACIDE PHOSPHORIQUE": "acide_phosphorique",
    "ENGRAIS NATURELS ET CHIMIQUES": "engrais",
    "PRODUITS CHIMIQUES": "produits_chimiques",
    "GAZ DE PÉTROLE ET AUTRES HYDROCARBURES": "gaz_petrole",
    "HOUILLES, COKES ET COMBUSTIBLES SOLIDES SIMILAIRES": "houilles_cokes",
    "Huile brute de pétrole": "huile_brute",
    "DÉCHETS DE MATIERES TEXTILES": "dechets_textiles",
    "SOUS-PRODUITS ANIMAUX NON COMESTIBLES": "sous_produits_animaux",
    "AUTRES PRODUITS BRUTS D'ORIGINE MINÉRALE": "produits_bruts_mineraux",
    "PHOSPHATES": "phosphates_bruts",
    "Machines et appareils électriques à usages divers": "machines_electriques",
    "MACHINES ET MATÉRIEL DE GÉNIE CIVIL ET DE CONSTRUCTION": "machines_genie_civil",
    "Machines, appareils pour industries alimentaires": "machines_agroalimentaire",
    "Phosphates et derivés": "target",
}
TARGET = "target"
EXOG_RAW = [c for c in COL_MAP.values() if c != "target"]

MAX_LAG = 12
CORR_THRESHOLD = 0.20
N_TARGET_LAGS = 5  # nombre de lags cible retenus (memes dimensionnalite que l'ancien choix fixe)
ROLL_WINDOWS = [3, 6, 12]
N_FORECAST_DEFAULT = 6

ARIMA_ORDERS = [(1, 1, 0), (2, 1, 0), (1, 1, 1), (2, 1, 1), (0, 1, 2), (3, 1, 0), (1, 1, 2)]
SARIMAX_ORDERS = [(1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 0)]
SARIMAX_SEASONAL = [(0, 1, 1, 12), (1, 1, 0, 12), (1, 1, 1, 12)]

ML_MODEL_NAMES = ["Ridge", "Random Forest", "XGBoost"]
ALL_MODEL_NAMES = ["Naive saisonnier"] + ML_MODEL_NAMES + ["ARIMA", "SARIMAX"]


# ---------------------------------------------------------------------------
# 1. Donnees
# ---------------------------------------------------------------------------

def load_data(path=DEFAULT_DATA_PATH):
    raw = pd.read_excel(path)
    df = raw.rename(columns=COL_MAP)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True).set_index("DATE").asfreq("MS")
    df[EXOG_RAW] = df[EXOG_RAW] / 1e6  # DH -> Millions DH, homogene avec la cible
    return df


def time_split(df, train_frac=0.8):
    split_idx = int(len(df) * train_frac)
    split_date = df.index[split_idx]
    return df.iloc[:split_idx], df.iloc[split_idx:], split_date


# ---------------------------------------------------------------------------
# 2. Stationnarite (ADF + KPSS, comme le notebook)
# ---------------------------------------------------------------------------

def stationarity_tests(series, label):
    from statsmodels.tsa.stattools import adfuller, kpss

    series = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p = adfuller(series, autolag="AIC")[:2]
        kpss_stat, kpss_p = kpss(series, regression="c", nlags="auto")[:2]

    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p > 0.05
    if adf_stationary and kpss_stationary:
        verdict = "stationnaire"
    elif not adf_stationary and not kpss_stationary:
        verdict = "non stationnaire"
    else:
        verdict = "incertain (ADF et KPSS en desaccord)"

    return {
        "serie": label, "adf_stat": adf_stat, "adf_pvalue": adf_p,
        "kpss_stat": kpss_stat, "kpss_pvalue": kpss_p, "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 3. Correlation croisee (CCF) et selection des lags exogenes (train uniquement)
# ---------------------------------------------------------------------------

def cross_corr(target_series, exog_series, max_lag):
    return pd.Series(
        [target_series.corr(exog_series.shift(k)) for k in range(0, max_lag + 1)],
        index=range(0, max_lag + 1),
    )


def compute_ccf_matrix(series_df, max_lag=MAX_LAG):
    return pd.DataFrame(
        {col: cross_corr(series_df[TARGET], series_df[col], max_lag) for col in EXOG_RAW}
    )


def select_lags(ccf_matrix, corr_threshold=CORR_THRESHOLD):
    selected = {}
    for col in EXOG_RAW:
        corr_by_lag = ccf_matrix[col].drop(index=0)
        best_lag = corr_by_lag.abs().idxmax()
        best_corr = corr_by_lag.loc[best_lag]
        if abs(best_corr) >= corr_threshold:
            selected[col] = int(best_lag)
    return selected


def select_target_lags(train_diff_target, max_lag=MAX_LAG, n_lags=N_TARGET_LAGS):
    """Choisit les n_lags lags de plus forte magnitude ACF sur la cible differenciee
    (train), au lieu d'une liste fixe arbitraire -- coherent avec la selection des lags
    exogenes par CCF. Seule une minorite de lags passe la significativite stricte a 95%
    sur ce jeu de donnees modeste : on classe donc par magnitude (meme logique exploratoire
    que le seuil CCF 0.20, pas un test de significativite)."""
    from statsmodels.tsa.stattools import acf as _acf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acf_vals = _acf(train_diff_target, nlags=max_lag)
    acf_by_lag = pd.Series(acf_vals[1:], index=range(1, max_lag + 1))
    return sorted(acf_by_lag.abs().sort_values(ascending=False).index[:n_lags].tolist())


# ---------------------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------------------

def build_features(df, selected_lags, lag_target, roll_windows=ROLL_WINDOWS):
    features = pd.DataFrame(index=df.index)
    features[TARGET] = df[TARGET]

    for k in lag_target:
        features[f"target_lag{k}"] = df[TARGET].shift(k)

    for w in roll_windows:
        features[f"target_roll_mean{w}"] = df[TARGET].shift(1).rolling(w).mean()
        features[f"target_roll_std{w}"] = df[TARGET].shift(1).rolling(w).std()

    for col, lag in selected_lags.items():
        features[f"{col}_lag{lag}"] = df[col].shift(lag)

    features["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    features = features.dropna()
    feature_cols = [c for c in features.columns if c != TARGET]
    return features, feature_cols


# ---------------------------------------------------------------------------
# 5. ARIMA / SARIMAX -- fit robuste (maxiter releve + garde-fou convergence)
# ---------------------------------------------------------------------------

def fit_quiet(mod, use_sarimax, maxiter=200):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if use_sarimax:
            res = mod.fit(disp=False, maxiter=maxiter)
        else:
            res = mod.fit(method_kwargs={"maxiter": maxiter})
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return res, converged


def select_order_aic(endog, exog, orders, seasonal_orders=None, use_sarimax=False, maxiter=200):
    best_converged, best_any = None, None
    for order in orders:
        for sorder in (seasonal_orders or [(0, 0, 0, 0)]):
            try:
                mod = (SARIMAX(endog, exog=exog, order=order, seasonal_order=sorder,
                               enforce_stationarity=False, enforce_invertibility=False)
                       if use_sarimax else ARIMA(endog, order=order))
                res, converged = fit_quiet(mod, use_sarimax, maxiter=maxiter)
                candidate = (res.aic, order, sorder, converged)
                if best_any is None or candidate[0] < best_any[0]:
                    best_any = candidate
                if converged and (best_converged is None or candidate[0] < best_converged[0]):
                    best_converged = candidate
            except Exception:
                continue
    return best_converged if best_converged is not None else best_any


def rolling_one_step(endog_train, endog_test, order, seasonal_order=None,
                      exog_train=None, exog_test=None, use_sarimax=False, maxiter=200):
    mod = (SARIMAX(endog_train, exog=exog_train, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False)
           if use_sarimax else ARIMA(endog_train, order=order))
    res, _ = fit_quiet(mod, use_sarimax, maxiter=maxiter)
    preds = []
    for i in range(len(endog_test)):
        step_exog = exog_test.iloc[[i]] if use_sarimax else None
        preds.append(res.get_forecast(steps=1, exog=step_exog).predicted_mean.iloc[0])
        res = res.append(endog_test.iloc[[i]], exog=step_exog, refit=False)
    return np.array(preds)


# ---------------------------------------------------------------------------
# 6. Entrainement + evaluation (test unique, comme section 5.1 du notebook)
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R2": r2_score(y_true, y_pred),
    }


def model_configs_dict():
    return {
        "Ridge": (
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(random_state=RANDOM_STATE))]),
            [{"ridge__alpha": [a]} for a in np.logspace(-2, 3, 15)],
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            [
                {"n_estimators": [200], "max_depth": [4], "min_samples_leaf": [2]},
                {"n_estimators": [300], "max_depth": [6], "min_samples_leaf": [2]},
                {"n_estimators": [400], "max_depth": [8], "min_samples_leaf": [3]},
            ],
        ),
        "XGBoost": (
            XGBRegressor(random_state=RANDOM_STATE, subsample=0.8, colsample_bytree=0.8),
            [
                {"n_estimators": [200], "max_depth": [2], "learning_rate": [0.05]},
                {"n_estimators": [300], "max_depth": [3], "learning_rate": [0.05]},
                {"n_estimators": [300], "max_depth": [3], "learning_rate": [0.1]},
            ],
        ),
    }


def run_full_pipeline(data_path=DEFAULT_DATA_PATH):
    """Reproduit integralement les sections 1 a 5 du notebook. Retourne un dict avec
    tout ce dont l'app a besoin (donnees, features, modeles entraines, resultats)."""
    df = load_data(data_path)
    train_raw, test_raw, split_date = time_split(df)

    stationarity = pd.DataFrame([
        stationarity_tests(train_raw[TARGET], "target (niveau, train)"),
        stationarity_tests(train_raw[TARGET].diff(), "target (differenciee ordre 1, train)"),
    ])

    train_diff = train_raw.diff().dropna()
    ccf_matrix = compute_ccf_matrix(train_diff)
    selected_lags = select_lags(ccf_matrix)

    ccf_matrix_level = compute_ccf_matrix(train_raw)
    selected_lags_level = select_lags(ccf_matrix_level)

    lag_target = select_target_lags(train_diff[TARGET])
    features, feature_cols = build_features(df, selected_lags, lag_target)
    train = features[features.index < split_date]
    test = features[features.index >= split_date]
    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    predictions = {"Naive saisonnier": df[TARGET].shift(12).reindex(test.index).values}
    tscv = TimeSeriesSplit(n_splits=5)
    fitted_models = {}
    model_params = {}
    for name, (estimator, grid) in model_configs_dict().items():
        search = GridSearchCV(estimator, grid, cv=tscv, scoring="neg_mean_squared_error")
        search.fit(X_train, y_train)
        fitted_models[name] = search.best_estimator_
        model_params[name] = search.best_params_
        predictions[name] = search.best_estimator_.predict(X_test)

    exog_feature_cols = [f"{col}_lag{lag}" for col, lag in selected_lags.items()]
    exog_mean, exog_std = train[exog_feature_cols].mean(), train[exog_feature_cols].std()
    train_exog_s = (train[exog_feature_cols] - exog_mean) / exog_std
    test_exog_s = (test[exog_feature_cols] - exog_mean) / exog_std

    _, arima_order, _, arima_converged = select_order_aic(train[TARGET], None, ARIMA_ORDERS)
    predictions["ARIMA"] = rolling_one_step(train[TARGET], test[TARGET], order=arima_order)

    _, sarimax_order, sarimax_seasonal, sarimax_converged = select_order_aic(
        train[TARGET], train_exog_s, SARIMAX_ORDERS, SARIMAX_SEASONAL, use_sarimax=True
    )
    predictions["SARIMAX"] = rolling_one_step(
        train[TARGET], test[TARGET], order=sarimax_order, seasonal_order=sarimax_seasonal,
        exog_train=train_exog_s, exog_test=test_exog_s, use_sarimax=True,
    )

    results = pd.DataFrame({name: evaluate(y_test, pred) for name, pred in predictions.items()}).T
    results = results.sort_values("RMSE")
    top2 = results.loc[ALL_MODEL_NAMES].sort_values("RMSE").index[:2].tolist()

    return dict(
        df=df, train_raw=train_raw, test_raw=test_raw, split_date=split_date,
        stationarity=stationarity, ccf_matrix=ccf_matrix, ccf_matrix_level=ccf_matrix_level,
        selected_lags=selected_lags, selected_lags_level=selected_lags_level, lag_target=lag_target,
        features=features, feature_cols=feature_cols,
        train=train, test=test, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        predictions=predictions, fitted_models=fitted_models, model_params=model_params,
        exog_feature_cols=exog_feature_cols, exog_mean=exog_mean, exog_std=exog_std,
        arima_order=arima_order, arima_converged=arima_converged,
        sarimax_order=sarimax_order, sarimax_seasonal=sarimax_seasonal, sarimax_converged=sarimax_converged,
        results=results, top2=top2,
    )


# ---------------------------------------------------------------------------
# 7. Evaluation multi-fenetres (section 5.2) -- limitee a top2, comme le notebook
# ---------------------------------------------------------------------------

def multi_window_eval(state, n_splits=7, test_size=12):
    X_all, y_all = state["features"][state["feature_cols"]], state["features"][TARGET]
    mw_cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    exog_feature_cols = state["exog_feature_cols"]

    rows = []
    for name in state["top2"]:
        for fold, (tr_idx, te_idx) in enumerate(mw_cv.split(X_all), start=1):
            X_tr, y_tr = X_all.iloc[tr_idx], y_all.iloc[tr_idx]
            X_te, y_te = X_all.iloc[te_idx], y_all.iloc[te_idx]
            if name in state["fitted_models"]:
                model = clone(state["fitted_models"][name])
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
            elif name == "ARIMA":
                pred = rolling_one_step(y_tr, y_te, order=state["arima_order"])
            else:  # SARIMAX
                mu, sd = X_tr[exog_feature_cols].mean(), X_tr[exog_feature_cols].std()
                pred = rolling_one_step(
                    y_tr, y_te, order=state["sarimax_order"], seasonal_order=state["sarimax_seasonal"],
                    exog_train=(X_tr[exog_feature_cols] - mu) / sd, exog_test=(X_te[exog_feature_cols] - mu) / sd,
                    use_sarimax=True,
                )
            rows.append({
                "modele": name, "fold": fold,
                "debut_test": X_te.index[0].date(), "fin_test": X_te.index[-1].date(),
                "RMSE": np.sqrt(mean_squared_error(y_te, pred)), "R2": r2_score(y_te, pred),
            })

    mw_results = pd.DataFrame(rows)
    mw_mean_rmse = mw_results.groupby("modele")["RMSE"].mean().sort_values()
    best_model_name = mw_mean_rmse.index[0]
    return mw_results, mw_mean_rmse, best_model_name


# ---------------------------------------------------------------------------
# 8. Reentrainement sur donnees completes -- generalise a n'importe quel modele
#    (le notebook ne le fait que pour best_model_name en 6.1 ; ici on le fait
#    pour les 6, pour permettre le selecteur interactif du dashboard).
# ---------------------------------------------------------------------------

def refit_all_models_full_data(state):
    X_full = pd.concat([state["X_train"], state["X_test"]])
    y_full = pd.concat([state["y_train"], state["y_test"]])

    final_models = {}
    for name in ML_MODEL_NAMES:
        model = clone(state["fitted_models"][name])
        model.fit(X_full, y_full)
        final_models[name] = model

    final_models["ARIMA"], _ = fit_quiet(ARIMA(y_full, order=state["arima_order"]), use_sarimax=False)

    exog_full = (X_full[state["exog_feature_cols"]] - state["exog_mean"]) / state["exog_std"]
    final_models["SARIMAX"], _ = fit_quiet(
        SARIMAX(y_full, exog=exog_full, order=state["sarimax_order"], seasonal_order=state["sarimax_seasonal"],
                enforce_stationarity=False, enforce_invertibility=False),
        use_sarimax=True,
    )
    return final_models


# ---------------------------------------------------------------------------
# 9. Prevision -- persistance des exogenes (comme le notebook) ou valeurs "what-if"
# ---------------------------------------------------------------------------

def future_exog_frame(df, exog_cols, n_steps):
    last_date = df.index[-1]
    future_index = pd.date_range(last_date + pd.DateOffset(months=1), periods=n_steps, freq="MS")
    last_values = df[exog_cols].iloc[-1]
    return pd.DataFrame([last_values.values] * n_steps, index=future_index, columns=exog_cols)


def what_if_exog_frame(df, n_steps, overrides=None):
    """Comme future_exog_frame, mais avec une valeur ajustee pour certaines exogenes
    (les autres restent persistees a leur derniere valeur connue). overrides : dict
    {colonne: ("pct", valeur)} pour une variation en % de la derniere valeur connue,
    ou {colonne: ("abs", valeur)} pour fixer une valeur absolue (Millions DH) --
    necessaire pour les variables dont la derniere valeur connue est nulle (frequent
    ici : beaucoup d'importations AT sont a 0 la plupart des mois), pour lesquelles un
    pourcentage n'a mathematiquement aucun effet. Partagee par le CLI et le dashboard."""
    frame = future_exog_frame(df, EXOG_RAW, n_steps)
    if not overrides:
        return frame
    for col, (mode, value) in overrides.items():
        if col not in frame.columns:
            raise ValueError(f"variable exogene inconnue : {col!r} (attendu parmi {EXOG_RAW})")
        if mode == "pct":
            base = frame[col].iloc[0]
            if base == 0:
                raise ValueError(
                    f"{col!r} : derniere valeur connue nulle, un pourcentage n'a aucun effet -- "
                    f"utilisez une valeur absolue (ex. {col}=5) plutot qu'un pourcentage."
                )
            frame[col] = frame[col] * (1 + value / 100)
        elif mode == "abs":
            frame[col] = value
        else:
            raise ValueError(f"mode inconnu : {mode!r} (attendu 'pct' ou 'abs')")
    return frame


def recursive_forecast(model, history, future_exog_raw, feature_cols, selected_lags, lag_target, roll_windows):
    history = history.copy()
    forecasts = []

    for date in future_exog_raw.index:
        extended = pd.concat([history[EXOG_RAW], future_exog_raw.loc[[date]]])

        row = {}
        for k in lag_target:
            row[f"target_lag{k}"] = history[TARGET].iloc[-k]
        for w in roll_windows:
            recent = history[TARGET].iloc[-w:]
            row[f"target_roll_mean{w}"] = recent.mean()
            row[f"target_roll_std{w}"] = recent.std()
        for col, lag in selected_lags.items():
            row[f"{col}_lag{lag}"] = extended[col].iloc[-1 - lag]
        row["month_sin"] = np.sin(2 * np.pi * date.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * date.month / 12)

        X_row = pd.DataFrame([row], columns=feature_cols, index=[date])
        y_pred = model.predict(X_row)[0]
        forecasts.append(y_pred)

        new_row = future_exog_raw.loc[[date]].copy()
        new_row[TARGET] = y_pred
        history = pd.concat([history, new_row[[TARGET] + EXOG_RAW]])

    return pd.Series(forecasts, index=future_exog_raw.index)


def forecast_with_model(state, final_models, model_name, n_steps, future_exog_raw=None):
    """Prevision a n_steps mois avec le modele choisi. Si future_exog_raw est fourni
    (mode what-if), il remplace la persistance de la derniere valeur connue -- sans
    effet pour ARIMA, qui n'utilise aucune exogene."""
    df = state["df"]
    if future_exog_raw is None:
        future_exog_raw = future_exog_frame(df, EXOG_RAW, n_steps)
    else:
        n_steps = len(future_exog_raw)  # coherence si un cadre "what-if" est fourni

    model = final_models[model_name]

    if model_name in ML_MODEL_NAMES:
        forecast = recursive_forecast(
            model, df[[TARGET] + EXOG_RAW], future_exog_raw, state["feature_cols"],
            state["selected_lags"], state["lag_target"], ROLL_WINDOWS,
        )
    elif model_name == "ARIMA":
        forecast = model.get_forecast(steps=n_steps).predicted_mean
        forecast.index = future_exog_raw.index
    else:  # SARIMAX
        # Le lag doit etre applique sur la serie continue historique+future : pour un
        # horizon plus court que le lag (cas frequent, lags jusqu'a 12 vs horizon 6),
        # la "vraie" valeur passee est encore connue et doit etre utilisee telle quelle --
        # ce n'est qu'une fois que t-lag entre dans la periode future que la persistance
        # (ou le what-if) entre en jeu. Meme logique que recursive_forecast pour les ML.
        extended = pd.concat([df[EXOG_RAW], future_exog_raw])
        future_exog_feat = pd.DataFrame({
            f"{c}_lag{lag}": extended[c].shift(lag).reindex(future_exog_raw.index)
            for c, lag in state["selected_lags"].items()
        })
        future_exog_feat_s = (future_exog_feat - state["exog_mean"]) / state["exog_std"]
        forecast = model.get_forecast(steps=n_steps, exog=future_exog_feat_s).predicted_mean
        forecast.index = future_exog_raw.index

    return forecast
