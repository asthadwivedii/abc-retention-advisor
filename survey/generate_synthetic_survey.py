"""
Generates the SYNTHETIC manager survey used as placeholder data for the
qualitative study. Replace survey_responses.csv with real Google Form
responses (same column names) and re-run the analysis notebook.

python survey/generate_synthetic_survey.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
OUT = Path(__file__).parent / "survey_responses.csv"

# ------------------------------------------------------------ profile
roles = (["Manager"] * 10 + ["Team Leader"] * 7 + ["Supervisor"] * 5 +
         ["Entrepreneur"] * 4 + ["Working Professional"] * 4 + ["Senior Decision-maker"] * 2)
rng.shuffle(roles)
N = len(roles)
exp_base = {"Manager": (12, 4), "Team Leader": (7, 2.5), "Supervisor": (10, 5),
            "Entrepreneur": (11, 5), "Working Professional": (5, 2), "Senior Decision-maker": (20, 4)}
team_of = {"Manager": ["6–15", "16–50", "16–50", "50+"], "Team Leader": ["1–5", "6–15", "6–15"],
           "Supervisor": ["6–15", "16–50", "16–50"], "Entrepreneur": ["1–5", "6–15", "16–50"],
           "Working Professional": ["None", "None", "1–5"], "Senior Decision-maker": ["50+"]}
sectors = ["Manufacturing", "IT services", "FMCG", "Banking & NBFC", "Cooperative & Agri",
           "Retail", "Healthcare", "Consulting"]
ai_freq = ["Never", "Rarely", "Monthly", "Weekly", "Daily"]

rows = []
for i, role in enumerate(roles):
    mu, sd = exp_base[role]
    exp = int(np.clip(round(rng.normal(mu, sd)), 2, 30))
    tech = rng.normal(0, 1)                      # comfort with technology
    if role in ("Working Professional", "Team Leader"):
        tech += 0.4
    if exp > 15:
        tech -= 0.3
    freq = ai_freq[int(np.clip(round(2 + tech * 1.1 + rng.normal(0, .5)), 0, 4))]

    # latent constructs (1–5 scale)
    PU = 3.7 + 0.35 * tech + rng.normal(0, .45)
    EU = 4.1 + 0.3 * tech + rng.normal(0, .5)
    TR = 3.1 + 0.4 * tech - 0.03 * (exp - 10) + rng.normal(0, .65)
    EX = 4.3 + 0.15 * tech + rng.normal(0, .5)
    FR = 3.4 + 0.15 * (role in ("Supervisor", "Team Leader")) + rng.normal(0, .6)
    IN = 3.1 + 0.05 * (exp - 10) - 0.25 * tech + rng.normal(0, .5)
    OS = 2.8 + 0.4 * (role == "Entrepreneur") + rng.normal(0, .6)
    BI = 0.45 * PU + 0.35 * TR + 0.2 * OS - 0.2 * IN + 0.9 + rng.normal(0, .35)

    item = lambda base, sd=.38: int(np.clip(round(base + rng.normal(0, sd)), 1, 5))
    r = {
        "RespondentID": f"R{i+1:02d}", "Role": role, "ExperienceYears": exp,
        "Sector": rng.choice(sectors), "TeamSize": rng.choice(team_of[role]),
        "AIUseFrequency": freq,
        "ReviewMode": "Used it myself" if rng.random() < .7 else "Watched a demo",
        "PU1": item(PU + .2), "PU2": item(PU - .15), "PU3": item(PU),
        "EU1": item(EU), "EU2": item(EU + .15),
        "TR1": item(TR), "TR2": item(TR - .2), "TR3": item(TR + .1),
        "EX1": item(EX + .1), "EX2": item(EX - .3),
        "FR1": item(FR + .1), "FR2": item(FR - .1),
        "IN1": item(IN), "IN2": item(IN + .1),
        "OS1": item(OS - .2), "OS2": item(OS + .1),
        "BI1": item(BI), "BI2": item(BI - .1),
    }
    # scenario choice
    logits = np.array([0.6 * (TR - 3) - .3, 0.8 * (IN - 3) - .8, 1.2, 0.5 + .3 * (FR - 3)])
    pr = np.exp(logits) / np.exp(logits).sum()
    r["C1_Scenario"] = rng.choice([
        "Follow the tool: start retention actions now",
        "Follow my judgement: ignore the score",
        "Talk to the employee first, then decide",
        "Look for more data before acting"], p=pr)
    r["_tier"] = "pos" if BI >= 3.9 else ("neu" if BI >= 3.2 else "neg")
    rows.append(r)

df = pd.DataFrame(rows)

# ------------------------------------------------------------ open-ended text
T = {
 "Q1": {
  "pos": ["Very useful. Right now we only find out someone is leaving when the resignation mail comes.",
          "Useful as an early-warning system, especially for large teams where I can't track everyone closely.",
          "Quite useful for prioritising. I have 30+ people and can't do stay-interviews with all of them.",
          "Helpful because it puts overtime, promotion gap and pay in one screen.",
          "Useful for first-line managers who don't have HR analytics support.",
          "The team ranking page alone would save time before our monthly review.",
          "Useful. It turns a gut feeling into something I can discuss with HR."],
  "neu": ["Moderately useful. Good as a starting point, but managers still need to talk to people.",
          "Useful for HR more than for line managers, I think.",
          "Somewhat useful. The pay benchmark is more practical for me than the risk score.",
          "It helps, but only if the satisfaction scores are updated regularly.",
          "Useful for large teams; for a team of five I already know who is unhappy."],
  "neg": ["Limited. Most attrition in my unit is due to family relocation, which no model can see.",
          "Not very useful for us. Our HR data is patchy, so the inputs will be guesses.",
          "In my small business I talk to everyone daily, so I don't see much added value.",
          "Useful in theory, but managers here won't enter data into another tool."]},
 "Q2": {
  "pos": ["Yes, before appraisal and increment discussions to decide where to focus.",
          "Yes, if it connects to our HRMS so I don't have to enter data manually.",
          "Yes, mainly the what-if part — I can show HR that removing overtime reduces risk.",
          "Yes. It gives me a structured reason to start a retention conversation.",
          "Yes, monthly, to review my team and flag cases to HR early.",
          "I would, especially for new joiners in the first year, where attrition is highest for us."],
  "neu": ["Maybe, as a second opinion, not the main basis for decisions.",
          "Possibly, if my organisation officially approves it. Otherwise it's risky.",
          "I'd use the pay check but I'm unsure about acting on the risk percentage.",
          "Only if HR also uses it, so we're looking at the same numbers.",
          "Perhaps once a quarter, not regularly."],
  "neg": ["Probably not. My decisions are based on relationships, not scores.",
          "Not yet. I'd want to see it tested on our own people first.",
          "No — I worry employees would feel profiled if they found out.",
          "Not in the current form; entering 20 fields per person is too much work."]},
 "Q3": {
  "pos": ["The 'why' section with suggested actions. It doesn't just give a number.",
          "The what-if slider — seeing risk fall when overtime is removed is powerful.",
          "Simple language and the traffic-light colours.",
          "That it compares pay with similar employees. That's a real fairness check.",
          "Clean layout. I understood it without any training.",
          "The team view and the download option."],
  "neu": ["Easy to use. The gauge is clear.",
          "I like that it shows the model's limits honestly.",
          "The suggested actions are practical.",
          "The drivers list is helpful.",
          "It loads fast and the questions are in normal HR language.",
          "The 1–4 scales with labels are easy to fill.",
          "Good that gender is left out of the model.",
          "The blue line showing the company average gives useful context.",
          "It admits it can be wrong, which I appreciate."],
  "neg": ["The design is clean, I'll give it that.",
          "The pay benchmark is the only part I found directly useful.",
          "It's simple to use."]},
 "Q4": {
  "pos": ["Too many inputs for one employee. Should pull data automatically.",
          "I'd like a trend — is the risk going up or down over months?",
          "Missing factors like manager behaviour and team conflicts.",
          "Would like a mobile-friendly version.",
          "Needs a confidence range, not just one percentage."],
  "neu": ["It gives false alarms — about half the 'high risk' people actually stay.",
          "Satisfaction scores are subjective; the tool depends too much on them.",
          "No explanation of how the model was built, in simple terms.",
          "Doesn't consider external job market or competitor hiring.",
          "Some suggested actions are generic.",
          "No option to save an employee and compare later.",
          "The pay figures have no currency or grade mapping for our structure.",
          "I wanted to know how recent the training data is.",
          "The team upload needs a specific CSV format that our HRMS doesn't export."],
  "neg": ["It could make managers treat people as numbers.",
          "It misses one-third of leavers, so I can't rely on it.",
          "Data privacy is not addressed — who sees these scores?",
          "It doesn't know the local context of our region."]},
 "Q5": {
  "pos": ["Mostly yes, because the reasons it shows match what I have seen in my team.",
          "I trust it as a direction. The drivers like overtime make sense.",
          "Yes, since it's tested on past data and the accuracy is stated openly.",
          "Fairly, because I could check it against two people who actually left last year."],
  "neu": ["Partly. I'd trust the ranking more than the exact percentage.",
          "I trust it for patterns, not for one individual.",
          "Somewhat. 80% accuracy sounds good but the false alarms bother me.",
          "Depends on the data quality — garbage in, garbage out.",
          "Half-half. Some drivers made sense, others like marital status felt odd.",
          "I'd trust it more after seeing it run on my own team for a few months.",
          "Reasonably, but I would never act on it alone.",
          "To some extent. The reasons were logical but the percentage felt too precise."],
  "neg": ["Not fully. It's trained on another company's history, not ours.",
          "No. People are too complex to be predicted by 20 variables.",
          "I don't trust it yet because I can't see how it would handle a special case.",
          "Not really; missing a third of leavers is too many."]},
 "Q6": {
  "pos": ["A track record — if it correctly flags people in our own company for six months.",
          "Showing past cases where it was right and wrong.",
          "Integration with actual HR data rather than manual entry.",
          "A confidence range and a note when the data is unusual."],
  "neu": ["Being validated by HR and approved by management.",
          "Regular updates of the model with our latest data.",
          "Knowing who built it and how it is monitored.",
          "Clear explanations in plain language for every prediction.",
          "Peer managers vouching for it after using it.",
          "An option to give feedback when the prediction is wrong.",
          "A pilot in one department with results shared openly.",
          "Audit by an independent team for bias."],
  "neg": ["Seeing it tested on our employees first.",
          "Transparency on the data it uses and privacy safeguards.",
          "Being able to add my own observations to the prediction."]},
 "Q7": {
  "pos": ["I'd talk to the employee. The AI tells me where to look; I decide what to do.",
          "Treat the disagreement as a signal to dig deeper, not to pick a side.",
          "Check the drivers — if overtime is high and I hadn't noticed, the AI may be right.",
          "Have an informal conversation first, then decide."],
  "neu": ["Go with my experience but keep a watch on that person.",
          "Discuss with HR and look at more data before acting.",
          "Look at which factor is driving the score and verify it myself.",
          "Probably trust my judgement, but not ignore the warning completely.",
          "Ask the employee's peers or skip-level manager for their view.",
          "Wait a month and re-check the score before doing anything.",
          "Flag it to HR and let them take a neutral view.",
          "Use it as a reason for a routine 1:1, without mentioning the score."],
  "neg": ["I'd go with my experience. I know my people.",
          "My judgement, because I'm accountable, not the software.",
          "Experience first. The model doesn't know the personal situation."]},
 "Q8": ["Because they are accountable for the outcome and feel more in control with their own judgement.",
        "Intuition includes context the data doesn't have — family issues, team politics, market offers.",
        "Habit. Most senior managers built their careers without data tools.",
        "They don't understand how the model works, so it feels like a black box.",
        "Admitting a model knows better can feel like a threat to their expertise.",
        "Past experience with bad data and wrong reports has made them sceptical.",
        "Intuition is faster; the tool needs inputs and time.",
        "Because people decisions are personal, and a number feels cold.",
        "In our culture, the boss's experience is valued more than analysis.",
        "They have seen AI hype before and results not matching.",
        "Managers remember the one case where they were right and the data was wrong."],
 "Q9": ["Yes, a lot. Without reasons I couldn't justify the action to my boss or the employee.",
        "Definitely. The drivers section is the reason I would consider using it.",
        "Yes — if I can see why, I can check whether it makes sense.",
        "Yes, explainability builds trust more than accuracy numbers do.",
        "Somewhat. Explanations help, but they also need to be correct.",
        "Yes. A black-box score would never be accepted in our review meetings.",
        "It matters, because I have to explain any retention bonus to finance.",
        "Yes, but too much detail would confuse managers — keep it short.",
        "Not much for me; I care more about whether it has been right before."],
 "Q10": ["Yes. If I act on a wrong prediction and the employee feels targeted, it damages trust.",
         "Yes, especially because I'll be blamed, not the AI.",
         "A little. But not acting when the tool warned me is also a risk.",
         "Yes, which is why I'd use it only to start a conversation, not to take action directly.",
         "Not much — a wrong retention conversation costs very little.",
         "Yes. In a hierarchical setup, mistakes are remembered longer than successes.",
         "It does; managers prefer to be wrong with their own judgement than with a machine's.",
         "Somewhat. Clear company guidelines on how to use it would reduce the fear.",
         "No. I see it as supporting evidence, which actually reduces my risk."],
 "Q11": ["Poor data quality and outdated HR records.",
         "No training on AI tools for middle managers.",
         "Top management does not ask for data-based decisions.",
         "Privacy concerns — employees may resist being scored.",
         "Lack of time; we are already overloaded with reports.",
         "Fear of job loss or of AI reducing the manager's role.",
         "Budget constraints for small firms like mine.",
         "Resistance from senior people who prefer the old ways.",
         "No clear policy on who is accountable when AI is wrong.",
         "Tools are built by IT without asking managers what they need.",
         "Low digital literacy among supervisors on the shop floor.",
         "Too many disconnected systems — one more tool is a burden."],
}


def pick(pool, k, used):
    """Pick an answer, preferring ones not yet used, and sometimes add a second sentence."""
    fresh = [a for a in pool if a not in used[k]] or pool
    a = rng.choice(fresh)
    used[k].add(a)
    return a


used = {k: set() for k in T}
for k, pools in T.items():
    col, seen = [], set()
    for tier in df["_tier"]:
        pool = pools[tier] if isinstance(pools, dict) else pools
        for _ in range(30):
            ans = pick(pool, k, used)
            # sometimes (or when the single answer is taken) combine two for a richer reply
            if rng.random() < .25 or ans in seen:
                neg = ans.startswith(("No", "Not"))
                same = [a for a in pool if a != ans and a.startswith(("No", "Not")) == neg]
                if not same:
                    continue
                ans = f"{ans} {rng.choice(same)}"
            if ans not in seen:
                break
        seen.add(ans)
        col.append(ans)
    df[k] = col

df = df.drop(columns="_tier")
df.to_csv(OUT, index=False)
print(f"Wrote {len(df)} synthetic responses to {OUT}")
