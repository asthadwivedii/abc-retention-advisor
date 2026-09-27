"""
ABC Ltd - Employee Retention Advisor
Streamlit app for non-technical managers.

Run locally:   streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_utils import (ATTR_CAT, ATTR_NUM, INC_CAT, INC_NUM, LABELS,
                         SCALE_1_4, coefficient_table, explain, load_data,
                         risk_band, train_models)

# Paste your Google Form link here after you create the survey
FEEDBACK_URL = "https://forms.gle/your-form-id"

st.set_page_config(page_title="ABC Ltd Retention Advisor", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_models():
    return train_models(get_data())


df = get_data()
models = get_models()
logit, linreg = models["logit"], models["linreg"]
BAND_COLOR = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}


def score(frame: pd.DataFrame) -> pd.DataFrame:
    """Add risk and expected-pay columns to any frame with the model columns."""
    out = frame.copy()
    out["Attrition risk (%)"] = (logit.predict_proba(out[ATTR_NUM + ATTR_CAT])[:, 1] * 100).round(1)
    out["Risk band"] = out["Attrition risk (%)"].apply(lambda p: risk_band(p / 100))
    out["Expected pay"] = linreg.predict(out[INC_NUM + INC_CAT]).round(0)
    out["Pay gap (%)"] = ((out["MonthlyIncome"] - out["Expected pay"]) / out["Expected pay"] * 100).round(1)
    return out


def gauge(p: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#263238"},
            "steps": [
                {"range": [0, 30], "color": "#c8e6c9"},
                {"range": [30, 60], "color": "#fff3c4"},
                {"range": [60, 100], "color": "#ffcdd2"},
            ],
            "threshold": {"line": {"color": "#1565c0", "width": 3},
                          "value": models["base_rate"] * 100},
        }))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=0))
    return fig


# ---------------------------------------------------------------- header
st.title("📊 ABC Ltd — Employee Retention Advisor")
st.caption(
    "Estimates how likely an employee is to leave in the coming year, explains why, "
    "and checks whether their pay is in line with similar roles. "
    "It supports your judgement — it does not replace it.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["👤 Check one employee", "👥 Check a team", "🔍 How the tool works", "📝 Give feedback"])

# ================================================================ TAB 1
with tab1:
    med = df[ATTR_NUM + INC_NUM].median()
    roles = sorted(df.JobRole.unique())
    dept_of_role = df.groupby("JobRole").Department.agg(lambda s: s.mode()[0]).to_dict()

    with st.form("employee"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Role & pay**")
            role = st.selectbox("Job role", roles, index=roles.index("Sales Executive"))
            dept = st.selectbox("Department", sorted(df.Department.unique()),
                                index=sorted(df.Department.unique()).index(dept_of_role[role]))
            level = st.select_slider("Job level (1 = entry, 5 = senior leader)", [1, 2, 3, 4, 5], value=2)
            income = st.number_input("Current monthly pay", 1000, 20000, 5000, step=250)
            hike = st.slider("Last salary hike (%)", 11, 25, 14)
            stock = st.select_slider("Stock options / LTI (0 = none)", [0, 1, 2, 3], value=0)
        with c2:
            st.markdown("**Experience**")
            age = st.slider("Age", 18, 60, 32)
            total_yrs = st.slider("Total work experience (years)", 0, 40, 8)
            yrs_co = st.slider("Years at ABC Ltd", 0, 40, 4)
            yrs_role = st.slider("Years in current role", 0, 18, 2)
            yrs_promo = st.slider("Years since last promotion", 0, 15, 1)
            yrs_mgr = st.slider("Years with current manager", 0, 17, 2)
            n_comp = st.slider("Number of previous employers", 0, 9, 2)
            edu = st.select_slider("Education", [1, 2, 3, 4, 5], value=3,
                                   format_func=lambda v: {1: "Below college", 2: "College", 3: "Bachelor",
                                                          4: "Master", 5: "Doctorate"}[v])
        with c3:
            st.markdown("**Engagement & conditions**")
            fmt = lambda v: f"{v} – {SCALE_1_4[v]}"
            job_sat = st.select_slider("Job satisfaction", [1, 2, 3, 4], value=3, format_func=fmt)
            env_sat = st.select_slider("Work environment satisfaction", [1, 2, 3, 4], value=3, format_func=fmt)
            rel_sat = st.select_slider("Relationship with colleagues", [1, 2, 3, 4], value=3, format_func=fmt)
            involve = st.select_slider("Job involvement", [1, 2, 3, 4], value=3, format_func=fmt)
            wlb = st.select_slider("Work-life balance", [1, 2, 3, 4], value=3, format_func=fmt)
            overtime = st.radio("Regularly works overtime?", ["No", "Yes"], horizontal=True)
            travel = st.selectbox("Business travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=1,
                                  format_func=lambda s: s.replace("_", " "))
            marital = st.selectbox("Marital status", ["Single", "Married", "Divorced"], index=1)
            dist = st.slider("Commute distance (km)", 1, 29, 7)
            training = st.slider("Trainings attended last year", 0, 6, 3)
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    emp = pd.DataFrame([{
        "Age": age, "DistanceFromHome": dist, "EnvironmentSatisfaction": env_sat,
        "JobInvolvement": involve, "JobLevel": level, "JobSatisfaction": job_sat,
        "MonthlyIncome": income, "NumCompaniesWorked": n_comp, "PercentSalaryHike": hike,
        "RelationshipSatisfaction": rel_sat, "StockOptionLevel": stock,
        "TotalWorkingYears": total_yrs, "TrainingTimesLastYear": training,
        "WorkLifeBalance": wlb, "YearsAtCompany": yrs_co, "YearsInCurrentRole": yrs_role,
        "YearsSinceLastPromotion": yrs_promo, "YearsWithCurrManager": yrs_mgr,
        "BusinessTravel": travel, "Department": dept, "JobRole": role,
        "MaritalStatus": marital, "OverTime": overtime, "Education": edu,
    }])

    if yrs_co > total_yrs or yrs_role > yrs_co or yrs_promo > yrs_co:
        st.warning("Check the experience inputs: years at ABC Ltd can't exceed total experience, "
                   "and years in role / since promotion can't exceed years at ABC Ltd.")

    s = score(emp).iloc[0]
    p = s["Attrition risk (%)"] / 100
    band = s["Risk band"]

    st.divider()
    r1, r2, r3 = st.columns([1.1, 1, 1])
    with r1:
        st.subheader("Chance of leaving")
        st.plotly_chart(gauge(p), width="stretch")
        st.markdown(
            f"<div style='text-align:center;font-size:1.2rem'>Risk level: "
            f"<b style='color:{BAND_COLOR[band]}'>{band}</b></div>", unsafe_allow_html=True)
        st.caption(f"Blue line = ABC Ltd average ({models['base_rate']*100:.0f}% of employees left).")
    with r2:
        st.subheader("Pay check")
        st.metric("Expected pay for this profile", f"{s['Expected pay']:,.0f}")
        st.metric("Current pay", f"{income:,.0f}", delta=f"{s['Pay gap (%)']:+.1f}% vs expected")
        if s["Pay gap (%)"] < -15:
            st.warning("Paid well below similar employees — worth reviewing for fairness.")
        elif s["Pay gap (%)"] > 15:
            st.info("Paid above similar employees.")
        else:
            st.success("Pay is in line with similar employees.")
        st.caption("Expected pay comes from the linear regression model "
                   "(role, level, experience, education).")
    with r3:
        st.subheader("Quick what-if")
        st.caption("See how the risk changes if you act.")
        w_ot = st.checkbox("Remove regular overtime", value=overtime == "Yes", disabled=overtime == "No")
        w_pay = st.slider("Raise pay by (%)", 0, 30, 0, step=5)
        w_promo = st.checkbox("Promote now (resets years since promotion)")
        w_stock = st.checkbox("Grant stock options / LTI (level 1)", disabled=stock > 0)
        what = emp.copy()
        if w_ot and overtime == "Yes":
            what["OverTime"] = "No"
        what["MonthlyIncome"] = income * (1 + w_pay / 100)
        if w_promo:
            what["YearsSinceLastPromotion"] = 0
        if w_stock and stock == 0:
            what["StockOptionLevel"] = 1
        p2 = logit.predict_proba(what[ATTR_NUM + ATTR_CAT])[0, 1]
        st.metric("Risk after these actions", f"{p2*100:.1f}%", delta=f"{(p2-p)*100:+.1f} pts",
                  delta_color="inverse")

    st.subheader("Why the tool thinks so")
    st.caption("The factors that move this employee's risk the most, compared with an average ABC Ltd employee.")
    ex = explain(models, emp, top_n=6)
    for _, r in ex.iterrows():
        icon = "🔺" if r.impact > 0 else "🔻"
        action = f" — *{r.suggested_action}*" if r.suggested_action != "-" else ""
        st.markdown(f"{icon} **{r.factor}** ({r.value}) · {r.direction.lower()}{action}")

    st.info("**Use this as one input.** The model is right about 8 in 10 times overall, "
            "but it catches only about 2 in 3 leavers and raises some false alarms. "
            "Talk to the employee before acting.")

# ================================================================ TAB 2
with tab2:
    st.subheader("Rank a team by attrition risk")
    st.caption("Upload a CSV with the same columns as the template, or try a sample of ABC Ltd employees.")
    template = df[ATTR_NUM + ATTR_CAT + ["Education"]].head(3)
    st.download_button("Download CSV template", template.to_csv(index=False), "team_template.csv", "text/csv")

    up = st.file_uploader("Upload team CSV", type="csv")
    use_sample = st.toggle("Use a sample of 40 ABC Ltd employees", value=up is None)

    team = None
    if up is not None:
        team = pd.read_csv(up)
        missing = set(ATTR_NUM + ATTR_CAT + ["Education"]) - set(team.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
            team = None
    elif use_sample:
        team = df.sample(40, random_state=7).reset_index(drop=True)
        team.insert(0, "Employee", [f"EMP-{1000+i}" for i in range(len(team))])

    if team is not None:
        scored = score(team).sort_values("Attrition risk (%)", ascending=False)
        a, b, c = st.columns(3)
        a.metric("Employees", len(scored))
        b.metric("High risk", int((scored["Risk band"] == "High").sum()))
        c.metric("Underpaid by 15%+", int((scored["Pay gap (%)"] < -15).sum()))
        show = [col for col in ["Employee"] if col in scored] + [
            "JobRole", "JobLevel", "OverTime", "MonthlyIncome", "Expected pay",
            "Pay gap (%)", "Attrition risk (%)", "Risk band"]
        st.dataframe(
            scored[show].style.map(
                lambda v: f"color:{BAND_COLOR.get(v, 'inherit')};font-weight:600", subset=["Risk band"]),
            width="stretch", hide_index=True)
        st.download_button("Download results", scored.to_csv(index=False), "team_risk.csv", "text/csv")

# ================================================================ TAB 3
with tab3:
    lm, im = models["logit_metrics"], models["lin_metrics"]
    st.subheader("Two simple, explainable models")
    st.markdown(
        "- **Logistic regression** estimates the chance an employee leaves (yes/no outcome).\n"
        "- **Linear regression** estimates the pay an employee *should* earn given role, level, "
        "experience and education.\n\n"
        "Both were trained on 1,470 past ABC Ltd employee records; 25% were held back to test them. "
        "Gender is deliberately excluded so the tool cannot treat people differently by gender.")

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{lm['accuracy']*100:.0f}%")
    b.metric("Leavers caught (recall)", f"{lm['recall']*100:.0f}%")
    c.metric("Ranking quality (AUC)", f"{lm['roc_auc']:.2f}")
    d.metric("Pay model fit (R²)", f"{im['r2']:.2f}")
    tn, fp, fn, tp = np.array(lm["confusion_matrix"]).ravel()
    st.markdown(
        f"On {lm['n_test']} test employees the tool flagged {tp+fp} as likely to leave: "
        f"**{tp} actually left** and {fp} stayed (false alarms). It missed {fn} leavers. "
        f"The pay model is off by about **{im['mae']:,.0f}** per month on average.")

    st.subheader("What drives attrition at ABC Ltd")
    st.caption("Odds ratio > 1 raises the chance of leaving; < 1 lowers it. "
               "Numeric factors are per one standard deviation.")
    ct = coefficient_table(models)
    ct = pd.concat([ct.head(8), ct.tail(8)])
    ct["label"] = ct.term.str.replace("_", ": ", n=1)
    fig = go.Figure(go.Bar(
        x=ct.odds_ratio, y=ct.label, orientation="h",
        marker_color=np.where(ct.odds_ratio > 1, "#c62828", "#2e7d32")))
    fig.add_vline(x=1, line_dash="dash")
    fig.update_layout(height=520, yaxis=dict(autorange="reversed"),
                      xaxis_title="Odds ratio", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Limits")
    st.markdown(
        "- Based on historical patterns; it cannot see personal events, offers from competitors, or team changes.\n"
        "- Satisfaction scores come from surveys and may be out of date.\n"
        "- A 'High' score is a prompt for a conversation, not a verdict.")

# ================================================================ TAB 4
with tab4:
    st.subheader("Help us improve this tool")
    st.markdown(
        "We are studying how managers decide whether to trust AI predictions. "
        "After trying the tool, please fill in a short survey (about 8 minutes).")
    st.link_button("Open the feedback survey", FEEDBACK_URL, type="primary")
    st.caption("Responses are anonymous and used only for an academic study at IRMA.")
