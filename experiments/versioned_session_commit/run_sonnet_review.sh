#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
prompt_file="$repo_root/experiments/versioned_session_commit/sonnet_review_prompt.md"
review_file="$repo_root/dossier/REVIEW_E-31_PRERUN_2026-10-17.md"

if test -e "$review_file"; then
  printf '%s\n' "Review file already exists: $review_file" >&2
  exit 1
fi

temporary_file=$(mktemp /tmp/e31-sonnet-review.XXXXXX)
trap 'rm -f "$temporary_file"' EXIT HUP INT TERM

claude -p \
  --model sonnet \
  --effort high \
  --permission-mode dontAsk \
  --allowedTools "Read,Grep,Glob" \
  --no-session-persistence \
  --output-format text \
  "$(sed -n '1,$p' "$prompt_file")" > "$temporary_file"

if ! test -s "$temporary_file"; then
  printf '%s\n' "Sonnet returned an empty review." >&2
  exit 1
fi

mv "$temporary_file" "$review_file"
trap - EXIT HUP INT TERM
printf '%s\n' "$review_file"
