"""How much of the reported accuracy for dementia prediction on OASIS comes from leakage?

Target: CDR > 0 at a visit (very mild dementia or worse). OASIS-2: 150 older adults, 373 MRI visits.
External test: OASIS-1 participants aged 60+ with a CDR rating.

Protocols, from the common notebook recipe to a defensible one:
  P1  visit-level random split, all features including CDR            (target leakage)
  P2  visit-level random split, CDR removed, MMSE kept                (subject leakage + a near-diagnostic test score)
  P3  subject-level split (GroupKFold), MMSE kept
  P4  subject-level split, MMSE removed: demographics + MRI volumetrics only
  P5  P4 model trained on all of OASIS-2, tested on OASIS-1 (external)

Models: logistic regression, random forest, gradient boosting. Metrics: AUC, balanced accuracy, Brier score,
expected calibration error, with subject-level bootstrap CIs; subgroup AUC by sex, age band and education.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEEDS = list(range(5))
DEMO_MRI = ["age", "female", "educ_cat", "ses", "etiv", "nwbv", "asf"]


def educ_cat(years: float) -> float:
    """OASIS-2 records years of education; OASIS-1 uses five categories. Map years to the OASIS-1 coding."""
    if pd.isna(years):
        return np.nan
    if years < 12:
        return 1
    if years == 12:
        return 2
    if years < 16:
        return 3
    if years == 16:
        return 4
    return 5


def load():
    o2 = pd.read_csv(ROOT / "data" / "oasis_longitudinal.csv")
    o2 = pd.DataFrame({"subject": o2["Subject ID"], "visit": o2.Visit, "age": o2.Age, "female": (o2["M/F"] == "F").astype(int),
                       "educ_cat": o2.EDUC.map(educ_cat), "ses": o2.SES, "mmse": o2.MMSE, "cdr": o2.CDR,
                       "etiv": o2.eTIV, "nwbv": o2.nWBV, "asf": o2.ASF})
    o1 = pd.read_csv(ROOT / "data" / "oasis_cross-sectional.csv")
    o1 = o1[o1.CDR.notna() & (o1.Age >= 60)]
    o1 = pd.DataFrame({"subject": o1.ID, "visit": 1, "age": o1.Age, "female": (o1["M/F"] == "F").astype(int),
                       "educ_cat": o1.Educ, "ses": o1.SES, "mmse": o1.MMSE, "cdr": o1.CDR, "etiv": o1.eTIV,
                       "nwbv": o1.nWBV, "asf": o1.ASF})
    for d in (o1, o2):
        d["y"] = (d.cdr > 0).astype(int)
    return o2.reset_index(drop=True), o1.reset_index(drop=True)


def models(seed):
    imp = SimpleImputer(strategy="median")
    return {
        "Logistic regression": make_pipeline(imp, StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random forest": make_pipeline(imp, RandomForestClassifier(200, min_samples_leaf=3, random_state=seed)),
        "Gradient boosting": make_pipeline(imp, HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                               max_iter=200, random_state=seed)),
    }


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def metrics(y, p):
    return {"auc": roc_auc_score(y, p), "bal_acc": balanced_accuracy_score(y, p >= 0.5), "brier": brier_score_loss(y, p),
            "ece": ece(np.asarray(y), np.asarray(p))}


PROTOCOLS = {
    "P1 visit split, all features incl. CDR": (False, DEMO_MRI + ["mmse", "cdr"]),
    "P2 visit split, MMSE, no CDR": (False, DEMO_MRI + ["mmse"]),
    "P3 subject split, MMSE": (True, DEMO_MRI + ["mmse"]),
    "P4 subject split, demographics + MRI": (True, DEMO_MRI),
}


def main():
    o2, o1 = load()
    meta = {"oasis2_visits": len(o2), "oasis2_subjects": int(o2.subject.nunique()), "oasis2_pos_share": float(o2.y.mean()),
            "oasis1_n": len(o1), "oasis1_pos_share": float(o1.y.mean()),
            "subjects_changing_label": int((o2.groupby("subject").y.nunique() > 1).sum())}
    rows, oof_store = [], {}
    for proto, (grouped, feats) in PROTOCOLS.items():
        for seed in SEEDS:
            if grouped:
                # GroupKFold has no shuffle in older sklearn; shuffle subjects ourselves.
                subj = o2.subject.unique()
                rng = np.random.default_rng(seed)
                perm = {s: i for i, s in enumerate(rng.permutation(subj))}
                groups = o2.subject.map(perm)
                splits = list(GroupKFold(5).split(o2, o2.y, groups))
            else:
                splits = list(KFold(5, shuffle=True, random_state=seed).split(o2))
            for name in models(seed):
                oof = np.zeros(len(o2))
                for tr, te in splits:
                    m = models(seed)[name].fit(o2.loc[tr, feats], o2.y[tr])
                    oof[te] = m.predict_proba(o2.loc[te, feats])[:, 1]
                rows.append({"protocol": proto, "seed": seed, "model": name, **metrics(o2.y, oof)})
                if seed == 0:
                    oof_store[(proto, name)] = oof
    # P5 external validation and P5b with MMSE, trained on all OASIS-2.
    for feats, label in ((DEMO_MRI, "P5 external OASIS-1, demographics + MRI"), (DEMO_MRI + ["mmse"], "P5b external OASIS-1, + MMSE")):
        for seed in SEEDS:
            for name, m in models(seed).items():
                m.fit(o2[feats], o2.y)
                p = m.predict_proba(o1[feats])[:, 1]
                rows.append({"protocol": label, "seed": seed, "model": name, **metrics(o1.y, p)})
                if seed == 0:
                    oof_store[(label, name)] = p
    res = pd.DataFrame(rows)
    res.to_csv(RES / "protocols.csv", index=False)

    # Subject-level bootstrap CIs for the P4 logistic-regression out-of-fold predictions, and subgroup AUCs.
    rng = np.random.default_rng(0)
    p4 = oof_store[("P4 subject split, demographics + MRI", "Logistic regression")]
    subj = o2.subject.values
    us = np.unique(subj)
    boots = []
    for _ in range(1000):
        pick = rng.choice(us, len(us))
        idx = np.concatenate([np.flatnonzero(subj == s) for s in pick])
        if o2.y.values[idx].min() == o2.y.values[idx].max():
            continue
        boots.append(metrics(o2.y.values[idx], p4[idx]))
    boots = pd.DataFrame(boots)
    ci = {k: [float(boots[k].quantile(0.025)), float(boots[k].quantile(0.975))] for k in boots}

    sub = []
    groups = {"Sex": np.where(o2.female == 1, "female", "male"),
              "Age": pd.cut(o2.age, [0, 75, 85, 200], labels=["<75", "75-84", "85+"]).astype(str),
              "Education": np.where(o2.educ_cat <= 2, "high school or less", "beyond high school")}
    for (proto, name), p in oof_store.items():
        if not proto.startswith(("P3", "P4")):
            continue
        for gname, g in groups.items():
            for level in sorted(set(g)):
                mask = g == level
                yy = o2.y.values[mask]
                if yy.min() == yy.max():
                    continue
                b = []
                idx_all = np.flatnonzero(mask)
                for _ in range(500):
                    pick = rng.choice(np.unique(subj[idx_all]), len(np.unique(subj[idx_all])))
                    idx = np.concatenate([idx_all[subj[idx_all] == s] for s in pick])
                    if o2.y.values[idx].min() != o2.y.values[idx].max():
                        b.append(roc_auc_score(o2.y.values[idx], p[idx]))
                sub.append({"protocol": proto, "model": name, "group": gname, "level": level, "n_visits": int(mask.sum()),
                            "n_subjects": int(len(np.unique(subj[mask]))), "pos_share": float(yy.mean()),
                            "auc": roc_auc_score(yy, p[mask]), "auc_lo": float(np.quantile(b, 0.025)),
                            "auc_hi": float(np.quantile(b, 0.975))})
    pd.DataFrame(sub).to_csv(RES / "subgroups.csv", index=False)
    meta["p4_lr_ci"] = ci
    (RES / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(res.groupby(["protocol", "model"])[["auc", "bal_acc", "brier", "ece"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
