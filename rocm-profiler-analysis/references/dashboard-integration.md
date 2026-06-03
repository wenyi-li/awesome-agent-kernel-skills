# Dashboard Integration

This skill is meant to feed the dashboard, not just produce local notes.

## Minimum Dashboard Surfaces

### Experiment Detail Page

Show:

- profile summary
- kernel table
- overlap opportunities
- fuse opportunities
- link to raw trace / fixed Perfetto trace

### Trial Detail

Show:

- which trial produced the profile
- exact GPU IDs used
- exact base image
- server flags
- whether preflight passed

## Summary-Level Uses

These artifacts are also useful outside the detail page:

- experiment overview can show whether a run has profiling artifacts
- review summaries can link directly to structured optimization evidence
- future SFT/data-flywheel pipelines can pull tables instead of scraping free-form markdown

## One Important Rule

Do not let the dashboard invent profiling conclusions. It should render the artifacts produced by
the skill. If the skill did not emit a table, the UI should show that the data is missing rather
than synthesizing a guess.
