# Contributing

This repo grows in the open. If you've drilled a question that isn't written up yet, or you spot something wrong or unclear in an existing one, a PR is very welcome.

## Adding a new case study

1. Check the [README index](README.md#the-28-questions) for the question's number and category — pick an unwritten one, or propose a new question via an issue if it's not on the list.
2. Copy [`TEMPLATE.md`](TEMPLATE.md) to `case-studies/NN-topic-name.md` (two-digit zero-padded number matching the README index, dash-separated words) and fill it in.
3. Update the README index table to link to your new file.
4. Open a PR. Short, incremental PRs (even just one section of a case study) are fine — they don't need to be perfect on the first pass.

## Fixing or improving an existing case study

Typos, broken links, factual corrections, clearer explanations, better diagrams — all welcome as PRs, no need to ask first.

## Style guide

- **Write for a stranger.** No personal anecdotes tied to a specific person's employer or projects. If a concrete example helps, make it a generic one or use a named persona, the way existing case studies use recurring example users.
- **Cite real tools and papers, with links.** Put them in the "Further reading & tools" section rather than as unlinked name-drops in prose.
- **Analogies over jargon.** Define every acronym on first use. Assume the reader is smart but new to this specific topic.
- **Concise over exhaustive.** If something is already covered well in another case study, link to it instead of repeating it.
- **Match the shape.** Follow [`TEMPLATE.md`](TEMPLATE.md)'s section order so the series stays navigable as it grows.

## Reporting issues

Found an error, a stale reference, or something confusing? Open an issue — it doesn't need to come with a fix.
