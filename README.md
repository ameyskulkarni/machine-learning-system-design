# ML System Design Case Studies

**A Blind-75-style prep list for machine learning system design interviews — coverage through patterns, not volume.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

There are 28 questions here, organized around 9 recurring archetypes (retrieval + ranking, calibration, imbalance, embeddings, drift, and so on). Master those archetypes and almost any prompt an interviewer invents turns out to be a remix of something you've already reasoned through — the same way knowing a few dozen patterns gets you through most coding interviews.

Each case study is a deep, self-contained teaching document: a 60-second answer, a step-by-step design walkthrough, worked examples and analogies, deep dives on the parts interviewers actually probe, curveball follow-ups, common failure modes, and a one-page cheat sheet for last-minute review.

## Status

**11 of 28 written so far.** This is a work in progress, built and published incrementally as each case study gets drilled and polished — not a finished product held back until complete. See the [index below](#the-28-questions) for what exists today, and [CONTRIBUTING.md](CONTRIBUTING.md) if you want to help fill in the rest.

## How to use this repo

1. Read the [9-step framework](#the-9-step-framework) once — it's the skeleton every case study hangs on.
2. Pick a case study and read it end-to-end to build the full picture.
3. Do a timed 30-minute whiteboard pass using its framework section, out loud, without looking. Note exactly which step you stumbled on — that's your gap, not the topic.
4. Use the flashcards and one-page cheat sheet in each file for spaced-repetition-style review afterward.
5. Once you're comfortable with a few, do a mock combining a random question with a curveball follow-up ("now it has to run on the edge," "now latency drops to 50ms," "now labels are delayed 3 weeks").

## The 9-step framework

Adapted from [alirezadir/AIMLInterviews](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md). This is the skeleton every case study in this repo applies. Interviewers grade *process* as much as they grade the final answer — a candidate who jumps straight to model architecture without scoping the problem first is the single most common way senior loops get failed.

1. **Clarify & scope.** Functional vs. non-functional requirements. Scale (QPS, users, items), latency budget, freshness, hardware/edge constraints, privacy. Turn a vague prompt into a written spec before modeling anything.
2. **Frame as an ML problem.** Define input → output precisely. What exactly is predicted? Is ML even the right tool, or does a heuristic baseline win? Pick the learning paradigm (classification / ranking / regression / retrieval / generation).
3. **Metrics.** Offline proxy metric(s), an online business/north-star metric, and *guardrail* metrics. Name the offline-online gap explicitly — this is where senior signal lives.
4. **Data.** Sources, how labels are actually obtained (often the hardest part — talk about it before you're asked), volume, class imbalance, freshness, privacy, sampling strategy.
5. **Features.** Feature engineering, embeddings, feature stores, and **leakage / point-in-time correctness**. Mention leakage before they ask.
6. **Model.** Start with a baseline, then justify each step up in complexity against the latency/cost budget. For large-scale retrieval, articulate the **candidate-generation → ranking** two-stage pattern.
7. **Training.** Pipeline, retraining cadence, distributed strategy, and reproducibility (data + label + code versioning).
8. **Evaluation & serving.** Offline eval → shadow → canary → A/B. Batch vs. real-time inference, latency budget decomposition, model versioning/rollback.
9. **Monitoring & iteration.** Data drift, concept drift, feedback loops, degradation alerts, retraining triggers. Close the loop back to training data.

> Practice tip: for each question, do a timed 30-minute whiteboard pass using this skeleton. If you can't fill all 9 steps out loud, that's your weak spot for that archetype — not a reason to move to a different question.

## The 28 questions

**Tags:** **[Core]** — asked in nearly every ML system design loop, regardless of your background. **[Domain]** — common when the role touches that specific area (CV, infra, etc.), and a strong signal when it comes up. **[Emerging]** — increasingly asked at senior level, especially with GenAI in the loop.

### Part 1 — Recommendation & Ranking
*The single most-asked family. The two-stage retrieval-then-rank pattern here transfers to search, ads, and even RAG.*

| # | Question | Tag | Status |
|---|---|---|---|
| 1 | [Video recommendation — YouTube / TikTok home feed](case-studies/01-video-recommendation.md) | Core | ✅ |
| 2 | [News feed ranking — Facebook / LinkedIn / X](case-studies/02-news-feed-ranking.md) | Core | ✅ |
| 3 | [Ad click-through-rate prediction](case-studies/03-ad-ctr-prediction.md) | Core | ✅ |
| 4 | "Similar items" / e-commerce product recommendation | Core | coming soon |
| 5 | People-You-May-Know / connection recommendation | Core | coming soon |

### Part 2 — Search & Retrieval
*Retrieval is the backbone of modern ML systems, RAG included.*

| # | Question | Tag | Status |
|---|---|---|---|
| 6 | Search ranking — query → results | Core | coming soon |
| 7 | [Visual search — image-to-image (Pinterest Lens / Google Lens)](case-studies/07-visual-search-image-to-image.md) | Domain | ✅ |
| 8 | Semantic document retrieval / the retrieval half of RAG | Emerging | coming soon |

### Part 3 — Classification, Detection & Trust/Safety
*Imbalance, adversarial drift, and human-in-the-loop — patterns that show up everywhere.*

| # | Question | Tag | Status |
|---|---|---|---|
| 9 | [Spam / abuse detection](case-studies/09-spam-abuse-detection.md) | Core | ✅ |
| 10 | Harmful content / content moderation | Core | coming soon |
| 11 | [Fraud / anomaly detection](case-studies/11-real-time-fraud-detection.md) | Core | ✅ |

### Part 4 — Computer Vision Systems

| # | Question | Tag | Status |
|---|---|---|---|
| 12 | [Object detection for autonomous driving / robotics](case-studies/12-object-detection-av-robotics.md) | Domain | ✅ |
| 13 | [Large-scale image classification / auto-tagging](case-studies/13-large-scale-image-auto-tagging.md) | Domain | ✅ |
| 14 | OCR / document understanding pipeline | Domain | coming soon |
| 15 | Video understanding / action recognition | Domain | coming soon |

### Part 5 — NLP & LLM Systems
*The fastest-growing category in loops.*

| # | Question | Tag | Status |
|---|---|---|---|
| 16 | RAG — LLM Q&A over a private corpus | Emerging | coming soon |
| 17 | LLM assistant / agent with tools | Emerging | coming soon |
| 18 | LLM fine-tuning / customization pipeline | Emerging | coming soon |
| 19 | Machine translation / sequence-to-sequence | Core | coming soon |
| 20 | Image generation / diffusion serving | Domain | coming soon |

### Part 6 — Prediction & Marketplace

| # | Question | Tag | Status |
|---|---|---|---|
| 21 | ETA / travel-time prediction | Core | coming soon |
| 22 | Dynamic pricing / surge | Emerging | coming soon |

### Part 7 — ML Infrastructure & Platform
*Senior/staff loops love these — most candidates hand-wave infra, so going deep here stands out.*

| # | Question | Tag | Status |
|---|---|---|---|
| 23 | Feature store | Domain | coming soon |
| 24 | Distributed model training platform | Domain | coming soon |
| 25 | [Model serving / inference system](case-studies/25-model-serving-inference-system.md) | Domain | ✅ |
| 26 | [A/B testing / experimentation platform](case-studies/26-experimentation-platform.md) | Core | ✅ |
| 27 | [Data labeling / annotation pipeline with active learning](case-studies/27-data-labeling-active-learning.md) | Domain | ✅ |
| 28 | ML monitoring / drift detection | Domain | coming soon |

## Contributing

More case studies are coming as they get written and drilled — PRs filling in the "coming soon" rows, or improving existing ones, are very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the style guide and [TEMPLATE.md](TEMPLATE.md) for the skeleton every case study follows.

## License

[MIT](LICENSE)

---

*The written content in this repo was drafted with the help of AI and then reviewed and edited for accuracy.*
