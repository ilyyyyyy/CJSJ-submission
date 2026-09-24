# Consistency and ethical integrity in Claude Opus 5

Code and data for a paper submitted to the *Columbia Junior Science Journal*.

## Running it

```
python3 -m pip install -r requirements.txt
python3 run.py
```

Tables 2–5 and every figure in the paper is written to `output/`.

## Data

- `pair1_ethical_integrity.csv`, `pair2_consensus_deferral.csv`, `pair3_foundations.csv`: model ratings, one row per rating on the 1–6 scale. "Endorsement" is the agreement question.
- `baseline.csv`: the ethicality question asked twice, alone.
- `human_group_means.csv`: human survey means per question, text sample and version, made by `make_human_means.py`. Individual responses are not published.
- `text_samples.csv`: page, justification type and fable for each text sample. Each moral quotes the original passage from Kamandaki, *The Essence of Politics*, trans. J. R. Knutson (Harvard University Press, 2021).