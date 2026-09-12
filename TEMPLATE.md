# Case Study Template

Copy this file to `case-studies/NN-topic-name.md` (two-digit, zero-padded question number from the [README index](README.md#the-28-questions), dash-separated) and fill it in. Delete these instructional notes as you go — they're guidance, not headings to keep.

Every case study in this repo follows the same shape, so a reader who's internalized one can navigate any other cold. See `case-studies/02-news-feed-ranking.md` for a fully worked example of this skeleton.

**Style rules:**
- Write for a stranger. No "your resume," "your wheelhouse," or references to a specific person's projects/employers. If an example helps, use a generic one or a named persona (the existing case studies use recurring example users like "Maya" and "Tom" — reuse or invent your own).
- Real tools, papers, and libraries are welcome and encouraged — link them in **Further Reading & Tools**, not as unlinked name-drops in prose.
- Favor analogies and worked examples over jargon. Define every acronym on first use.
- Concise beats exhaustive. If a section would just repeat another case study's content verbatim, link to it instead of restating it.
- Use `mermaid` code blocks for diagrams — they render natively on GitHub.

---

```markdown
# Case Study: <Title> — <example companies/products>

> **Q<N>** · Tag: **[Core / Domain / Emerging]** · Family: <category from the README index>
>
> **Prompt:** *"<how this gets phrased in an actual interview>"*
>
> **Teaches:** <the one or two transferable patterns this question exists to test>

<1-3 sentence framing: why this question matters, what makes it hard, how it connects to other questions in the series.>

---

## The 60-second answer

<A single dense paragraph a candidate could say from memory that compresses the entire design. Everything else in the document unpacks this paragraph.>

---

## Mental model

<The 2-4 core ideas that, if internalized, let a reader improvise the rest of the answer. Use analogies. This is what separates "memorized the steps" from "actually understands it.">

---

## The 9-step walkthrough

Apply the [standard framework](README.md#the-9-step-framework) to this problem. Spend real airtime on steps 1-3 in an actual interview; steps 4-6 are usually where the deepest technical discussion happens.

**Step 1 — Clarify & scope.** ...

**Step 2 — Frame as an ML problem.** ...

**Step 3 — Metrics.** ...

**Step 4 — Data & labels.** ...

**Step 5 — Features & leakage.** ...

**Step 6 — Model.** ...

**Step 7 — Training.** ...

**Step 8 — Evaluation & serving.** ...

**Step 9 — Monitoring & iteration.** ...

---

## Deep dives

<2-5 subsections on the specific topics that separate a senior answer from a mid-level one for this question. Go genuinely deep on each — this is the section interviewers actually probe.>

---

## Diagrams

<At least one mermaid diagram of the system's data/request flow.>

---

## Interview delivery guide

<A timed script (assume a 45-minute loop) showing how to pace through the sections above out loud, plus 3-5 meta-tips specific to this question (what to volunteer unprompted, common traps in delivery, etc.).>

---

## Curveballs

<4-6 "now the interviewer changes a constraint" follow-ups, each resolved in 1-3 sentences by mapping back to the framework.>

---

## Common failure modes

<A bulleted list of how candidates actually lose this question — not knowledge gaps, but process/delivery mistakes.>

---

## Flashcards

| Cue | What you must be able to say |
|---|---|
| ... | ... |

---

## One-page cheat sheet

<A dense, compressed version of the whole document — the thing to glance at 10 minutes before the interview.>

---

## Glossary

<Short definitions of every acronym/term of art introduced above.>

---

## Further reading & tools

<Real papers and tools, each with a link. Group as "Papers" and "Tools" if there are enough of both.>

---

*Part of the [ML System Design Case Studies](README.md) series. Connects to: <2-4 other Q-numbers and one line each on the shared pattern.>*
```
