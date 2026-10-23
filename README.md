# From 1.00 to 0.77: how much dementia-prediction accuracy on OASIS is leakage?

📄 **Paper (PDF): [oasis-leakage.pdf](oasis-leakage.pdf)** · [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23593046.svg)](https://doi.org/10.5281/zenodo.23593046)

Target: CDR > 0 per visit, OASIS-2 (373 visits, 150 participants),
external test on OASIS-1 (aged 60+).

| Protocol | Logistic regression AUC | Random forest AUC |
|---|---|---|
| Visit split, CDR as a feature | 1.000 | 1.000 |
| Visit split, MMSE, no CDR | 0.907 | 0.904 |
| Participant split, MMSE | 0.900 | 0.842 |
| Participant split, demographics + MRI only | 0.766 | 0.671 |
| External OASIS-1, demographics + MRI | 0.742 | 0.877 |

Subgroup AUCs (sex, age, education) have 95% intervals 0.18–0.29 wide: the data cannot settle fairness either way.

## Reproduce

The OASIS summary files are available from the OASIS project (oasis-brains.org) under its data use terms. Place
`oasis_longitudinal.csv` and `oasis_cross-sectional.csv` in `data/`, then:

```bash
pip install -r requirements.txt
python src/oasis.py
python src/analyse.py
cd paper && latexmk -pdf main.tex
```

## AI assistance

An AI assistant (Claude, Anthropic) helped write analysis code and draft text. The author designed the study,
reviewed the code and results, and is responsible for the content.

## Licence

MIT (code).
