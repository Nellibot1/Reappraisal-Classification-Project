# Data availability

Raw research data is intentionally excluded from Git by default.

The local `data/` directory contains reappraisal text, annotation matrices, annotator metadata, and study material. Some records include free-text responses and demographic fields such as age and gender. These files should be added to a public repository only after confirming participant consent, ethics requirements, licensing, and the intended data-sharing policy.

The literature and project-planning notes in `data/notes/` are included in the repository because they do not contain the raw annotation tables.

For reproducible public releases, prefer one of the following:

1. Deposit an approved, de-identified dataset in an appropriate research repository and link its DOI here.
2. Provide a data-preparation script plus instructions for obtaining the source data.
3. Keep restricted data outside Git and document the access procedure.

Do not use `git add -f data/...` until the sharing review is complete.

Row-level `*_test_predictions.csv` files are also excluded because they repeat the original situation and reappraisal text. Public notebook copies should have saved cell outputs cleared so displayed data samples are not published accidentally.
