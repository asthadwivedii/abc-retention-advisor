# ABC Ltd — Employee Retention Advisor

A predictive tool that helps managers at ABC Ltd see **which employees are likely to leave, why, and whether their pay is in line with similar roles**. It was built for Assignment 2: *Predictive Analytics & Managerial AI Adoption*.

| | |
|---|---|
| **Business problem** | Employee attrition (about 16% of employees leave each year) |
| **Model 1: Linear regression** | Predicts expected monthly pay from role, level, experience and education. R² = 0.94 on test data |
| **Model 2: Logistic regression** | Predicts the probability that an employee leaves. Accuracy 80%, AUC 0.81, recall 68% |
| **App** | Streamlit, built for non-technical managers |
| **User study** | 32-respondent manager survey, analysed quantitatively and qualitatively |

> **Data:** Employee attrition dataset (1,470 records) from Kaggle, originally published by IBM. The company is called ABC Ltd throughout.

---

## Repository structure

```
abc-retention-advisor/
├── app.py                          ← Streamlit app (entry point)
├── model_utils.py                  ← model training + explanation code used by the app
├── requirements.txt
├── .streamlit/config.toml          ← theme
├── data/
│   └── employee_attrition.csv
├── models/                         ← trained models saved from Colab (evidence)
│   ├── attrition_logit.pkl
│   ├── income_linreg.pkl
│   └── metrics.json
├── notebooks/
│   ├── 01_ABC_Attrition_Models.ipynb      ← EDA + linear & logistic regression (Colab)
│   └── 02_Manager_Survey_Analysis.ipynb   ← user-study analysis (Colab)
└── survey/
    ├── questionnaire.md                   ← questions for the Google Form
    ├── survey_responses.csv               ← responses (currently SYNTHETIC placeholder)
    ├── generate_synthetic_survey.py       ← script that created the placeholder data
    └── survey_analysis.xlsx               ← analysis tables exported from notebook 02
```

---

## Step-by-step: what to do, in order

### Step 1: Run the models in Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → `notebooks/01_ABC_Attrition_Models.ipynb`.
2. **Runtime → Run all.** The data loads from a public URL, so no upload is needed.
3. Read through the outputs: EDA, the OLS and Logit summaries, the confusion matrix, ROC curve and threshold table.
4. The last cells save the `models/` folder. Download it from the Files panel on the left if you want your own copy.

### Step 2: Put the project on GitHub
1. Create a GitHub account if you don't have one → **New repository** → name it `abc-retention-advisor` → **Public** → Create.
2. **Add file → Upload files** → drag in *all* files and folders from this project (unzip first). Hidden folders like `.streamlit` may not upload by drag-and-drop; if so, create `.streamlit/config.toml` using **Add file → Create new file**. It is optional.
3. Click **Commit changes**.

*(Alternative: section 9 of notebook 01 contains a cell that pushes from Colab using a personal access token.)*

### Step 3: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
2. **Create app → Deploy a public app from GitHub** → Repository `your-username/abc-retention-advisor`, Branch `main`, Main file `app.py`.
3. Optional: under **Advanced settings**, set Python 3.11 or 3.12.
4. Click **Deploy**. The first build takes 2–4 minutes.
5. Copy the URL, for example `https://abc-retention-advisor.streamlit.app`. **This is your app link for submission.**

The app trains the two models from the CSV on start-up, so it never breaks because of scikit-learn version differences.

### Step 4: Run the manager survey
1. Create a Google Form using `survey/questionnaire.md`. Use the same codes as column headers (PU1 … BI2, C1_Scenario, Q1 … Q11). This lets the analysis notebook read the export directly.
2. Paste the form link into `FEEDBACK_URL` near the top of `app.py` and commit. The **Give feedback** tab will then open your form.
3. Send the app link and the form to managers, team leaders, supervisors and entrepreneurs. Aim for 25–40 responses.

### Step 5: Analyse the survey in Colab
1. Export the Google Form responses as CSV, rename columns to match the codes, and replace `survey/survey_responses.csv`.
2. Open `notebooks/02_Manager_Survey_Analysis.ipynb` in Colab. Upload `survey_responses.csv` to the Files panel, or set `GITHUB_CSV` to your repo's raw URL.
3. **Run all.** You get descriptive statistics, Cronbach's α, group comparisons, a regression of intention-to-use, the scenario analysis, thematic coding of open-ended answers, representative quotes, and `survey_analysis.xlsx`.
4. Update the Findings section at the end to match your real numbers.

### What to submit
- **App URL** from Step 3
- **Survey data**: `survey_responses.csv` (or the Excel export)
- **Analysis**: `02_Manager_Survey_Analysis.ipynb` and `survey_analysis.xlsx`, plus `01_ABC_Attrition_Models.ipynb` for the model

---

## Using the app (for managers)

- **Check one employee:** fill in role, pay, experience and engagement scores, then click **Predict**. You see:
  - the chance of leaving (Low / Medium / High)
  - the expected pay for the profile
  - the main reasons behind the score, with suggested actions
  - a what-if panel (remove overtime, raise pay, promote, grant stock options)
- **Check a team:** upload a CSV (a template is provided) or use the built-in sample to rank a team by risk and download the results.
- **How the tool works:** accuracy in plain language, what drives attrition, and the tool's limits.
- **Give feedback:** link to the survey.

**Design choices for trust:**
- Gender is excluded from both models.
- Every score comes with its reasons.
- The model's error rate is stated openly.
- The tool is positioned as a prompt for a conversation, not a decision.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
