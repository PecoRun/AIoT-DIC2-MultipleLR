"""Boston Housing 多元線性迴歸：以 Forward SFS 產生 k=1..13 的模型結果，輸出 boston_data.js。"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = fetch_openml(name="boston", version=1, as_frame=True, parser="auto").frame
df = df.astype(float)
X, y = df.drop(columns="MEDV"), df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []
selected_prev = []
for k in range(1, X.shape[1] + 1):
    if k == X.shape[1]:
        feats = list(X.columns)
    else:
        sfs = SequentialFeatureSelector(
            LinearRegression(), n_features_to_select=k, direction="forward", cv=5
        ).fit(X_train, y_train)
        feats = list(X.columns[sfs.get_support()])

    Xtr = sm.add_constant(X_train[feats])
    ols = sm.OLS(y_train, Xtr).fit()
    pred = ols.predict(sm.add_constant(X_test[feats], has_constant="add"))
    ci = ols.conf_int()

    coefs = []
    for i, name in enumerate(ols.params.index):
        is_const = name == "const"
        coefs.append({
            "feature": "Intercept (β₀)" if is_const else name,
            "coef": float(ols.params[name]),
            "stderr": float(ols.bse[name]),
            "tstat": float(ols.tvalues[name]),
            "pvalue": float(ols.pvalues[name]),
            "ci_lower": float(ci.loc[name, 0]),
            "ci_upper": float(ci.loc[name, 1]),
            "vif": None if is_const else float(variance_inflation_factor(Xtr.values, i)),
        })

    resid = y_test.values - pred.values
    order = np.argsort(-np.abs(resid))[:10]
    outliers = [{
        "rank": r + 1,
        "index": int(y_test.index[j]),
        "actual": float(y_test.values[j]),
        "predicted": float(pred.values[j]),
        "residual": float(resid[j]),
        "abs_error": float(abs(resid[j])),
        "pct_error": float(abs(resid[j]) / y_test.values[j] * 100),
    } for r, j in enumerate(order)]

    results.append({
        "k": k,
        "features": feats,
        "added": [f for f in feats if f not in selected_prev],
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "adj_r2": float(ols.rsquared_adj),
        "f_stat": float(ols.fvalue),
        "f_pvalue": float(ols.f_pvalue),
        "coefficients": coefs,
        "outliers": outliers,
    })
    selected_prev = feats

out = os.path.join(BASE_DIR, "boston_data.js")
with open(out, "w", encoding="utf-8") as f:
    f.write("window.BOSTON_DATA = " + json.dumps(results, ensure_ascii=False, indent=1) + ";\n")
best = min(results, key=lambda r: r["rmse"])
print(f"wrote {out}; best k={best['k']} RMSE={best['rmse']:.3f} R2={best['r2']:.4f}")
for r in results:
    print(r["k"], round(r["rmse"], 3), round(r["r2"], 4), r["added"])
