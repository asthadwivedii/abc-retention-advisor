"""
Model code shared by the Streamlit app (and mirrored in the Colab notebook).

ABC Ltd - Employee Attrition & Pay Benchmark
  * Logistic regression -> probability that an employee leaves (Attrition = Yes)
  * Linear regression   -> expected monthly income for a role/experience profile

Gender is deliberately left out of both models so the tool cannot
recommend different treatment by gender.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "employee_attrition.csv"
RANDOM_STATE = 42

# ---------------------------------------------------------------- features
ATTR_NUM = [
    "Age", "DistanceFromHome", "EnvironmentSatisfaction", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked",
    "PercentSalaryHike", "RelationshipSatisfaction", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]
ATTR_CAT = ["BusinessTravel", "Department", "JobRole", "MaritalStatus", "OverTime"]

INC_NUM = ["JobLevel", "TotalWorkingYears", "YearsAtCompany", "Age", "Education"]
INC_CAT = ["JobRole", "Department"]

# Plain-English labels for managers
LABELS = {
    "Age": "Age",
    "DistanceFromHome": "Commute distance",
    "EnvironmentSatisfaction": "Satisfaction with work environment",
    "JobInvolvement": "Job involvement",
    "JobLevel": "Job level / seniority",
    "JobSatisfaction": "Job satisfaction",
    "MonthlyIncome": "Monthly pay",
    "NumCompaniesWorked": "Number of previous employers",
    "PercentSalaryHike": "Last salary hike (%)",
    "RelationshipSatisfaction": "Relationship with colleagues",
    "StockOptionLevel": "Stock options / long-term incentive",
    "TotalWorkingYears": "Total work experience",
    "TrainingTimesLastYear": "Trainings attended last year",
    "WorkLifeBalance": "Work-life balance",
    "YearsAtCompany": "Years at ABC Ltd",
    "YearsInCurrentRole": "Years in current role",
    "YearsSinceLastPromotion": "Years since last promotion",
    "YearsWithCurrManager": "Years with current manager",
    "BusinessTravel": "Business travel",
    "Department": "Department",
    "JobRole": "Job role",
    "MaritalStatus": "Marital status",
    "OverTime": "Works overtime",
    "Education": "Education level",
}

# What a manager can actually do about a driver
ACTIONS = {
    "OverTime": "Review workload and overtime; rebalance tasks or add support.",
    "MonthlyIncome": "Benchmark pay against the expected salary below; consider a correction.",
    "StockOptionLevel": "Consider a retention bonus or long-term incentive.",
    "JobSatisfaction": "Hold a stay-interview: what would make the work more meaningful?",
    "EnvironmentSatisfaction": "Check team climate, tools and workspace issues.",
    "WorkLifeBalance": "Discuss flexible hours / remote days.",
    "YearsSinceLastPromotion": "Discuss a growth path or promotion timeline.",
    "JobInvolvement": "Give ownership of a visible project.",
    "BusinessTravel": "Reduce or rotate travel load.",
    "DistanceFromHome": "Offer remote/hybrid days or transport support.",
    "TrainingTimesLastYear": "Nominate for training or certification.",
    "RelationshipSatisfaction": "Check team relationships; consider a mentor or buddy.",
    "YearsWithCurrManager": "Schedule regular 1:1s to build the manager relationship.",
    "PercentSalaryHike": "Revisit the next increment cycle.",
    "JobLevel": "Map a clear path to the next level.",
}

SCALE_1_4 = {1: "Low", 2: "Medium", 3: "High", 4: "Very high"}


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
    return df


def _preprocessor(num, cat):
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat),
    ])


def train_models(df: pd.DataFrame | None = None) -> dict:
    """Train both models and return them with test-set metrics."""
    if df is None:
        df = load_data()

    # ---- Logistic regression: attrition
    X = df[ATTR_NUM + ATTR_CAT]
    y = df["AttritionFlag"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    logit = Pipeline([
        ("prep", _preprocessor(ATTR_NUM, ATTR_CAT)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
    ]).fit(Xtr, ytr)
    p = logit.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    logit_metrics = {
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred),
        "f1": f1_score(yte, pred),
        "roc_auc": roc_auc_score(yte, p),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "n_test": int(len(yte)),
    }

    # ---- Linear regression: expected monthly income
    Xi = df[INC_NUM + INC_CAT]
    yi = df["MonthlyIncome"]
    Xitr, Xite, yitr, yite = train_test_split(
        Xi, yi, test_size=0.25, random_state=RANDOM_STATE)
    linreg = Pipeline([
        ("prep", _preprocessor(INC_NUM, INC_CAT)),
        ("model", LinearRegression()),
    ]).fit(Xitr, yitr)
    pi = linreg.predict(Xite)
    lin_metrics = {
        "r2": r2_score(yite, pi),
        "mae": mean_absolute_error(yite, pi),
        "rmse": float(np.sqrt(np.mean((yite - pi) ** 2))),
        "n_test": int(len(yite)),
    }

    # Baseline (average employee in transformed space) for explanations
    baseline = logit.named_steps["prep"].transform(Xtr).mean(axis=0)

    return {
        "logit": logit, "linreg": linreg,
        "logit_metrics": logit_metrics, "lin_metrics": lin_metrics,
        "baseline": np.asarray(baseline).ravel(),
        "base_rate": float(y.mean()),
    }


def risk_band(p: float) -> str:
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Medium"
    return "High"


def explain(models: dict, row: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Contribution of each original feature to the log-odds of leaving,
    relative to an average employee. Positive = pushes risk up."""
    logit = models["logit"]
    prep = logit.named_steps["prep"]
    coefs = logit.named_steps["model"].coef_.ravel()
    x = np.asarray(prep.transform(row[ATTR_NUM + ATTR_CAT])).ravel()
    contrib = coefs * (x - models["baseline"])

    names = prep.get_feature_names_out()
    agg = {}
    for n, c in zip(names, contrib):
        base = n.split("__", 1)[1]
        orig = next((f for f in ATTR_CAT if base.startswith(f + "_")), base)
        agg[orig] = agg.get(orig, 0.0) + c

    out = (pd.DataFrame({"feature": list(agg), "impact": list(agg.values())})
           .assign(abs_impact=lambda d: d.impact.abs())
           .sort_values("abs_impact", ascending=False)
           .head(top_n))
    out["factor"] = out.feature.map(LABELS)
    out["direction"] = np.where(out.impact > 0, "Raises risk", "Lowers risk")
    out["value"] = [row.iloc[0][f] for f in out.feature]
    out["suggested_action"] = [
        ACTIONS.get(f, "-") if imp > 0 else "-" for f, imp in zip(out.feature, out.impact)]
    return out.drop(columns="abs_impact").reset_index(drop=True)


def coefficient_table(models: dict) -> pd.DataFrame:
    """Odds ratios from the logistic model, for the 'How it works' page."""
    logit = models["logit"]
    names = logit.named_steps["prep"].get_feature_names_out()
    coefs = logit.named_steps["model"].coef_.ravel()
    df = pd.DataFrame({"term": [n.split("__", 1)[1] for n in names], "coef": coefs})
    df["odds_ratio"] = np.exp(df.coef)
    return df.sort_values("coef", ascending=False).reset_index(drop=True)
