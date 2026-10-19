Human spot-check of the AI-judged results (backend/scripts/spotcheck.py), 2026-10-19.

- human_labels.csv: one person (the project author, not a lawyer), labelling blind to the AI labels;
  column human_label is exactly as entered (the blind measurement); human_label_after_review is the
  same person's labels after comparing with a second reading by Claude (an AI): 4 changed (S15, S40,
  S45, S47), so that column is not independent of the AI reading. Items: 38 citations from the faithfulness evaluation
  and 12 machine-translated Hindi questions. AI labels added afterwards for comparison.
- The item texts are in data/reports/spotcheck/spotcheck.xlsx (regenerate with `spotcheck.py make`).
