"""Figure and tables for the OASIS leakage paper."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, TAB = ROOT / "results", ROOT / "paper" / "figures", ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
r = pd.read_csv(RES / "protocols.csv")
order = list(dict.fromkeys(r.protocol))
short = {p: p.split(" ", 1)[0] for p in order}
g = r.groupby(["protocol", "model"])[["auc", "bal_acc", "brier", "ece"]].agg(["mean", "std"])
rows = []
for p in order:
    for m in ["Logistic regression", "Random forest", "Gradient boosting"]:
        x = g.loc[(p, m)]
        rows.append({"Protocol": p if m == "Logistic regression" else "", "Model": m,
                     "AUC": f"{x[('auc', 'mean')]:.3f}", "Bal.\\ acc.": f"{x[('bal_acc', 'mean')]:.3f}",
                     "Brier": f"{x[('brier', 'mean')]:.3f}", "ECE": f"{x[('ece', 'mean')]:.3f}"})
(TAB / "protocols.tex").write_text(
    "\\begin{table*}[t]\n\\centering\n\\caption{Dementia (CDR $>$ 0) classification under five protocols, mean of five repetitions. "
    "P1--P4 are 5-fold cross-validation on OASIS-2 (373 visits, 150 participants); P5 trains on all of OASIS-2 and tests on "
    "OASIS-1 participants aged 60 or over (198).}\n\\label{tab:protocols}\n\\small\n"
    + pd.DataFrame(rows).to_latex(index=False, escape=False, column_format="llrrrr") + "\\end{table*}\n")

fig, ax = plt.subplots(figsize=(6.4, 2.4))
models = ["Logistic regression", "Random forest", "Gradient boosting"]
cols = ["#1f4e79", "#2e8b57", "#c0504d"]
x = np.arange(len(order))
for i, (m, c) in enumerate(zip(models, cols)):
    vals = [g.loc[(p, m)][("auc", "mean")] for p in order]
    ax.bar(x + (i - 1) * 0.26, vals, 0.26, color=c, label=m)
ax.set_xticks(x)
ax.set_xticklabels(["P1 visit split\n+ CDR", "P2 visit split\n+ MMSE", "P3 subject split\n+ MMSE", "P4 subject split\nno MMSE",
                    "P5 external\nno MMSE", "P5b external\n+ MMSE"], fontsize=6)
ax.set_ylim(0.5, 1.02)
ax.set_ylabel("AUC")
ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "protocols.pdf")

s = pd.read_csv(RES / "subgroups.csv")
s = s[(s.model == "Logistic regression") & s.protocol.str.startswith("P4")]
t = pd.DataFrame({"Group": s.group, "Level": s.level, "Visits": s.n_visits, "Participants": s.n_subjects,
                  "CDR$>$0": (s.pos_share * 100).round(0).astype(int).astype(str) + "\\%",
                  "AUC [95\\% CI]": [f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in zip(s.auc, s.auc_lo, s.auc_hi)]})
(TAB / "subgroups.tex").write_text(
    "\\begin{table}[t]\n\\centering\n\\caption{Subgroup AUC for the P4 logistic-regression model (out-of-fold predictions; "
    "participant-level bootstrap CIs).}\n\\label{tab:sub}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
    + t.to_latex(index=False, escape=False, column_format="llrrrr") + "\\end{table}\n")
print(pd.DataFrame(rows).to_string())
