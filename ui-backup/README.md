# Earlier interfaces

Exact copies of the frontend (`src/`, `public/`, `index.html`) from before each redesign of 5 November 2026:

- `old-ui-2026-11-05/`: the original interface.
- `law-report-2026-11-05/`: the black-on-paper version that came next.
- `chapters-2026-11-05/`: colour chapters, before Case, Verify, Label and Explain were rebuilt around legal objects.
- `legal-tools-2026-11-05/`: the legal tools, before chat answers were simplified.
- `chat-record-2026-11-05/`: simplified chat answers, before the Verify and Case results were redesigned.
- `results-2026-11-05/`: the redesigned results, before the Label page became a shelf and a marking desk.

To go back to one:

    bash ui-backup/restore-old-ui.sh
    bash ui-backup/restore-old-ui.sh law-report
    bash ui-backup/restore-old-ui.sh chapters
    bash ui-backup/restore-old-ui.sh legal-tools
    bash ui-backup/restore-old-ui.sh chat-record
    bash ui-backup/restore-old-ui.sh results

The current interface is saved first, so restoring can be undone. Only the frontend changed in the
redesigns. Restoring changes the interface only: fixes made to the backend stay in place.
