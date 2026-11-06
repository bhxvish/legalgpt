#!/bin/bash
# Puts an earlier interface back. Run from anywhere:
#   bash ui-backup/restore-old-ui.sh              the original interface
#   bash ui-backup/restore-old-ui.sh law-report   the black-on-paper version
#   bash ui-backup/restore-old-ui.sh chapters     colour chapters, before the legal redesign of the tools
#   bash ui-backup/restore-old-ui.sh legal-tools  the legal tools, before chat answers were simplified
# The current interface is saved first, in ui-backup/replaced-<date-time>, so this can be undone too.
set -e
cd "$(dirname "$0")/.."
case "${1:-old}" in
  old) OLD=ui-backup/old-ui-2026-11-05/frontend ;;
  law-report) OLD=ui-backup/law-report-2026-11-05/frontend ;;
  chapters) OLD=ui-backup/chapters-2026-11-05/frontend ;;
  legal-tools) OLD=ui-backup/legal-tools-2026-11-05/frontend ;;
  chat-record) OLD=ui-backup/chat-record-2026-11-05/frontend ;;
  results) OLD=ui-backup/results-2026-11-05/frontend ;;
  *) echo "Choose: old, law-report, chapters, legal-tools, chat-record or results"; exit 1 ;;
esac
SAVED="ui-backup/replaced-$(date +%Y%m%d-%H%M%S)/frontend"
mkdir -p "$SAVED"
cp -r frontend/src frontend/public frontend/index.html "$SAVED/"
rm -rf frontend/src frontend/public
cp -r "$OLD/src" "$OLD/public" frontend/
cp "$OLD/index.html" frontend/index.html
echo "Interface restored from $OLD. The one it replaced is in $SAVED"
