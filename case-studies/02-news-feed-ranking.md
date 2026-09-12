# Case Study: News Feed Ranking — Facebook / LinkedIn / X

> **Q2** · Tag: **[Core]** · Family: Recommendation & Ranking
>
> **Prompt:** *"Rank the home feed to maximize meaningful engagement."*
>
> **Teaches:** Multi-objective ranking with real-time features, position bias, and freshness/recency blending.

This is the first full case study in the series, and it's a deliberate choice to start here. News feed ranking is the *Rosetta Stone* of ML system design: it contains the two-stage funnel, multi-task learning, calibration, position bias, feedback loops, real-time features, and the offline-online gap all in one problem. Master this one deeply and you will recognize its pieces inside a dozen other prompts.

If you're new to system design and feel intimidated: that's normal, and this document is written for exactly that person. We move slowly, use analogies, and build the design one layer at a time. Read it once end-to-end to see the whole shape, then use the [Interview Delivery Guide](#part-c--the-interview-delivery-guide) and [Cheat Sheet](#appendix-1--one-page-cheat-sheet) to drill.

---

## Table of Contents

- [Part A — Understanding the Problem](#part-a--understanding-the-problem)
  - [What is a "feed," really?](#what-is-a-feed-really)
  - [The 60-second answer (memorize this spine)](#the-60-second-answer-memorize-this-spine)
  - [Mental model](#mental-model)
- [Part B — The Design, Step by Step](#part-b--the-design-step-by-step)
  - [Step 1 — Clarify & Scope](#step-1--clarify--scope)
  - [Step 2 — Frame as an ML Problem](#step-2--frame-as-an-ml-problem)
  - [Step 3 — Metrics](#step-3--metrics)
  - [Step 4 — Data & Labels](#step-4--data--labels)
  - [Step 5 — Features](#step-5--features)
  - [Step 6 — Model (the heart)](#step-6--model-the-heart)
  - [Step 7 — Training](#step-7--training)
  - [Step 8 — Evaluation & Serving](#step-8--evaluation--serving)
  - [Step 9 — Monitoring & Iteration](#step-9--monitoring--iteration)
- [Part C — The Interview Delivery Guide](#part-c--the-interview-delivery-guide)
- [Part D — Deep Dives (the senior-signal topics)](#part-d--deep-dives-the-senior-signal-topics)
  - [Deep Dive 1 — Combining many objectives into one score](#deep-dive-1--combining-many-objectives-into-one-score)
  - [Deep Dive 2 — Position & presentation bias](#deep-dive-2--position--presentation-bias)
  - [Deep Dive 3 — Freshness & recency blending](#deep-dive-3--freshness--recency-blending)
  - [Deep Dive 4 — Engagement-bait & the guardrail problem](#deep-dive-4--engagement-bait--the-guardrail-problem)
  - [Deep Dive 5 — Feedback loops & why your data lies to you](#deep-dive-5--feedback-loops--why-your-data-lies-to-you)
- [Part E — Curveball Follow-ups](#part-e--curveball-follow-ups)
- [Part F — Common Failure Modes](#part-f--common-failure-modes)
- [Flashcards](#flashcards)
- [Appendix 1 — One-Page Cheat Sheet](#appendix-1--one-page-cheat-sheet)
- [Appendix 2 — Glossary](#appendix-2--glossary)
- [Further Reading & Tools](#further-reading--tools)
- [Appendix 3 — How this connects to the rest of the 28](#appendix-3--how-this-connects-to-the-rest-of-the-28)

---

# Part A — Understanding the Problem

## What is a "feed," really?

Strip away the app and a feed is one question asked millions of times per second:

> *"This person just opened the app. Of the hundreds or thousands of things I could show them right now, in what **order** should I show them?"*

That's it. The output is an **ordered list**. Everything in this document is machinery for producing a good order.

**Analogy — the newspaper editor.** Imagine you're the editor of a personalized newspaper that must be re-typeset from scratch every single time each of two billion readers opens it, in under half a second, using only what's happened in the world up to this instant. You can't run every story past every reader. You have to *predict* which stories each reader will care about, lay out the front page accordingly, and do it before they get impatient and close the paper. News feed ranking is that editor, automated.

**Why this is harder than it looks.** A naive answer — "show newest first" — is a real, respectable baseline (X still offers "Latest" mode; feeds *used* to work this way). But it fails in a specific way: your loud, prolific acquaintance who posts 20 times a day drowns out the one important post from a close friend. The ML system exists to fix exactly this: **to surface the relevant needle in the fresh haystack.**

**The subtle word: "meaningful."** The prompt doesn't say "maximize engagement" — it says *meaningful* engagement. That single word is the whole interview. Naively maximizing clicks and time-spent is a solved, easy problem that produces a *terrible* product: it amplifies outrage, clickbait, and doom-scrolling. The senior-level challenge is defining and optimizing a target that correlates with long-term user value, not short-term compulsion. Circle that word out loud in your interview. It signals maturity immediately.

### How feed ranking differs from video recommendation ([Q1](./01-video-recommendation.md))

They share the two-stage funnel, but the differences shape your whole design. Naming these differences early shows you're not just pattern-matching.

| | **Video Rec (YouTube/TikTok)** | **News Feed (FB/LinkedIn/X)** |
|---|---|---|
| **Inventory** | Billions of videos, mostly *out-of-network* | Mostly *in-network* (your friends/follows) + some recommended |
| **Candidate set size** | Massive; retrieval is the hard part (ANN over billions) | Hundreds to low-thousands; retrieval is often a graph fetch |
| **Freshness** | Matters, but evergreen content is fine | **Critical** — a post from 3 min ago vs 3 days ago is night and day |
| **Dominant signal** | Content-to-interest match | **Social affinity** (who posted it) often dominates |
| **Objective feel** | "What will they *watch*?" | "What will they *meaningfully interact with*, without regretting it?" |

The takeaway to say aloud: *"Because feed inventory is mostly in-network and small per request, candidate generation is lighter than in video rec — the intellectual weight moves to the multi-task ranker and the re-ranking / integrity layer."*

## The 60-second answer (memorize this spine)

Before the detail, here is the entire system in one breath. If you can say this fluently, you have a spine to hang everything on — and if you freeze in an interview, you can always fall back to this and then expand.

> *"I'll treat this as a two-stage ranking problem. **First, candidate generation** pulls a few thousand eligible posts per request — mostly in-network posts since the user's last visit, plus some recommended out-of-network content via embedding retrieval. **Second, a multi-task ranking model** scores each candidate by predicting several engagement probabilities — likes, comments, shares, dwell, and importantly negative actions like hides and reports. I **combine those predictions into a single value score** with business-tuned weights, so a meaningful comment counts more than a passive like and a hide counts strongly negative. Then a **re-ranking layer** applies diversity, freshness, and integrity rules before returning the ordered feed. I'll pay special attention to **position bias** in the training data, **point-in-time correctness** of real-time features to avoid leakage, and the **feedback loop** where ranking shapes the very data I train on. Success is measured by an online meaningful-engagement metric with retention and integrity guardrails, validated through A/B tests, not just offline AUC."*

Everything below is that paragraph, unpacked.

---

## Mental model

Three ideas carry most of the answer. Internalize these and you can improvise the rest.

### Idea 1 — "Meaningful" is a different target than "engaging," and conflating them is the trap

The prompt says *meaningful* engagement, not raw engagement. Raw engagement is easy to maximize and easy to get wrong: outrage, clickbait, and doom-scrolling are all *highly engaging*. The newspaper editor from above wouldn't fill the front page with tabloid shock headlines just because they sell — a good editor weighs a thoughtful op-ed a reader will value over a headline that only bait-clicks. That's why the ranker predicts several distinct actions (like, comment, share, hide, report) and combines them with **business-chosen weights** instead of optimizing one raw "did they engage" label — the weighting *is* where "meaningful" gets defined.

### Idea 2 — Engagement is a biased, noisy proxy, not a clean satisfaction signal

There's no column in the data that says "the user was satisfied." You only observe engagement on the posts the *old ranker chose to show*, in the *slot it chose to show them in* — so the signal is contaminated by two biases at once: **position bias** (the same post gets more engagement at slot 1 than slot 10, purely from placement) and **selection bias** (you have no data on what would have happened with posts you never showed). Same shape as the eye-level-shelves problem: rank grocery items by sales and you'll conclude eye-level items are "better," when you've really just measured shelf position. Take this seriously and most of the senior-level machinery (position-as-a-feature, IPW, exploration) falls out naturally.

### Idea 3 — It's a closed loop: today's ranking becomes tomorrow's training data

The model decides what's shown → that shapes engagement → those engagements become labels → which train the next model. Left naive, this loop amplifies whatever gets rewarded (rich-get-richer, filter bubbles) and never learns about the content it buried. The fix is the same family of ideas as Idea 2's debiasing plus deliberate **exploration** and **long-term holdback groups**, so the system doesn't just optimize itself into a corner.

**The chain to say out loud in an interview:** *raw engagement ≠ meaningful engagement → the labels you'd naively train on are biased by position and selection → and the ranker's own output becomes tomorrow's biased training data. Every guardrail in this design exists to break one link in that chain.*

---

# Part B — The Design, Step by Step

We now walk the 9-step framework from `Part 0`. In a real interview you spend the **first ~10 minutes on steps 1–3** (scope, framing, metrics) — this is where senior candidates separate themselves. Weak candidates sprint to model architecture. Don't be that candidate.

---

## Step 1 — Clarify & Scope

**The goal of this step:** turn a one-line vague prompt into a written spec. Interviewers deliberately keep the prompt ambiguous to see if you'll impose structure. *Asking good clarifying questions is itself a graded skill.*

### Questions to ask the interviewer (and why each matters)

- **"Which platform / what's the inventory?"** Facebook feed = friends + groups + followed pages. LinkedIn = professional network + follows. X = follows + heavy algorithmic out-of-network injection. This changes how much candidate generation you need. *Pick one and commit — I'll assume Facebook-style for concreteness.*
- **"Is the feed purely in-network, or blended with recommended content?"** Blended feeds need a real retrieval stage; pure in-network feeds are mostly a graph fetch. Modern feeds are blended — assume that.
- **"What does the business mean by *meaningful* engagement?"** This is the most important question in the whole loop. It determines your objective. If they push it back to you, propose a definition (see Step 3) — that's a *feature*, not a cop-out.
- **"Are ads in scope?"** Almost always say: *"I'll treat ad insertion as a separate interleaving system downstream and keep this design focused on organic ranking."* Scoping *out* is as important as scoping in.
- **"Can we re-show content the user has already seen?"** Usually no / heavily downranked. Affects the candidate filter.

### Requirements you write on the board

**Functional**
- Given `(user, context)` on feed-open, return an ordered list of posts.
- Support infinite scroll (pagination — rank the next page as the user scrolls).
- Incorporate very recent posts (freshness).

**Non-functional (put rough numbers on these — interviewers love back-of-envelope)**
- **Scale:** ~2B daily active users, opening several times a day → order of **10s of billions of feed requests/day**, peak QPS in the **hundreds of thousands to low millions**.
- **Latency:** the ranking call should return in roughly **≤ few hundred ms**; total page render budget ~1–2s. Ranking a few thousand candidates through a deep net must fit inside that.
- **Freshness:** a post created seconds ago should be *eligible* this request.
- **Availability:** a broken ranker must degrade gracefully to reverse-chronological, never a blank feed.

> **Teaching note — the "degrade gracefully" line is free senior signal.** Saying *"if the ranker is unavailable, I fall back to a simple recency-sorted feed rather than showing nothing"* tells the interviewer you think about production reality, not just the happy path.

---

## Step 2 — Frame as an ML Problem

**The goal of this step:** define input → output precisely, and justify that ML is even the right tool.

### Input → Output

- **Input:** a triple `(user, candidate post, context)`.
- **Output:** a single scalar **value score** for that post — "how valuable is showing *this* post to *this* user *right now*?" — used to sort.

### Is ML the right tool? (Always ask this — it's a maturity check.)

The **baseline is reverse-chronological** (newest first). It's simple, cheap, transparent, and *not bad*. ML earns its cost by:
- surfacing an important older post over a trivial fresh one,
- learning that you engage far more with your sister's posts than a distant acquaintance's,
- demoting content you'll likely hide or report.

**Say this explicitly:** *"My first 'model' is reverse-chron. Everything fancier must beat it in an A/B test on the north-star metric, or it doesn't ship."* Establishing a baseline before architecture is one of the framework's five commandments — and one of the [most common things weak candidates forget](#part-f--common-failure-modes).

### The learning paradigm

This is **multi-task learning feeding a ranking objective**:

1. Predict several engagement **probabilities** per post: `P(like)`, `P(comment)`, `P(share)`, `P(dwell > t)`, `P(click)`, and crucially **negative** actions `P(hide)`, `P(report)`.
2. **Combine** them into one value score (Step 6 / Deep Dive 1).
3. **Sort** by value score, then re-rank for diversity/integrity.

Why predict *multiple* actions instead of one "will they engage" label? Because "engagement" isn't one thing — a like, a thoughtful comment, and a rage-report are wildly different in value, and collapsing them loses exactly the information you need to optimize for *meaningful* interaction. Keeping them separate lets you *price* each one.

---

## Step 3 — Metrics

**The goal of this step:** define how you'll know the system is good — offline, online, and what must *not* get worse. Naming the **offline-online gap** here is where the senior signal lives.

### The three tiers of metrics

**1. Offline (proxy) metrics** — computed on logged data, before shipping.
- **Per-task quality:** AUC / ROC-AUC and **calibration** for each predicted action. (Calibration matters if you literally add probabilities together — a predicted `P(share)=0.1` should mean it actually happens ~10% of the time. See [Deep Dive 1](#deep-dive-1--combining-many-objectives-into-one-score).)
- **Ranking quality:** **NDCG** (Normalized Discounted Cumulative Gain) over an engagement-weighted list — does the model put high-value posts near the top?

**2. Online (north-star) metric** — what you actually A/B test on.
- A **weighted "meaningful engagement" score.** Facebook's public version of this was **"Meaningful Social Interactions (MSI)"** — a weighted sum that valued comments, reshares-with-text, and back-and-forth conversation over passive likes.
- Broader health signals: **sessions, DAU, time-spent** — used carefully (see the guardrail warning below).

**3. Guardrail metrics** — things that must not regress even if engagement goes up.
- **Negative feedback rate:** hides, reports, "see less," "unfollow."
- **Integrity metrics:** prevalence of misinformation / borderline / hateful content served.
- **Diversity:** are we showing a variety of sources, or 8 posts from one person?
- **Long-term retention:** the hardest and most important — measured via long-running holdback groups (Deep Dive 5).

### The offline–online gap (say this out loud)

> *"Offline, I measure whether the model ranks logged impressions well. But the moment I deploy, the model **changes what users see**, which changes what they engage with, which changes my future training data. Offline AUC can go up while the online metric goes flat or down. So offline metrics are a *filter to decide what's worth A/B testing*, never the final word. The A/B test on the north-star metric is the source of truth."*

**Analogy — the driving test vs. the road.** Offline metrics are the written driving exam: cheap, fast, necessary to weed out the unqualified. But passing the written test doesn't mean you can drive in traffic. The A/B test is the actual road. You use the cheap test to decide who *gets* on the road, never to certify them.

**The engagement trap (foreshadowing Deep Dive 4).** If your north-star is naive time-spent, you will build a machine that maximizes doom-scrolling and outrage, because outrage is engaging. This is why "meaningful" and the guardrails exist. Mention this tension in Step 3 and the interviewer will know you've thought about the product, not just the math.

---

## Step 4 — Data & Labels

**The goal of this step:** where does training data come from, and — the hard part — **how are labels obtained?** Talk about labeling *before* the interviewer asks. It's the most under-discussed part of most candidates' answers.

### Data sources

- **Interaction logs:** every impression and every action on it — like, comment, share, click, dwell time, video watch time, **hide, report, "see less."** This is your primary well.
- **Content store:** the posts themselves — text, media, author, type, timestamp.
- **Social graph:** friendships, follows, group memberships, historical interaction affinity.
- **User profile:** account age, declared interests, historical engagement patterns.

### How labels are obtained (implicit feedback)

You don't have humans hand-labeling "this post is good for this user." You **infer labels from behavior**:
- A `like` → positive label for the `P(like)` head.
- An impression with no action → negative (a "0").
- Dwell time → a graded/continuous signal.
- A `hide`/`report` → strong *negative* label — gold, because explicit dislike is rare and precious.

### The five ways your labels lie (name these — this is senior territory)

1. **Position bias.** A post at slot 1 gets more engagement than the *same post* at slot 10, purely from position. If you train naively, the model "learns" that slot 1 is good — circular. → [Deep Dive 2](#deep-dive-2--position--presentation-bias).
2. **Presentation bias.** Big autoplay videos and large images grab clicks regardless of relevance.
3. **Absence ≠ dislike.** No engagement might mean "didn't like it" or "never scrolled that far." A no-action label is noisy.
4. **Selection bias (the counterfactual problem).** You only observe engagement on posts you *chose to show*. You have no data on the posts the old model buried. Your training data is a biased sample of your own past decisions. → [Deep Dive 5](#deep-dive-5--feedback-loops--why-your-data-lies-to-you).
5. **Delayed / rare labels.** Shares and reports are rare (heavy imbalance); some signals (a comment thread developing) arrive minutes to hours later.

### Class imbalance

Impressions ≫ clicks > likes > comments > shares > reports. Some heads (like `P(report)`) train on tiny positive rates. Handle with negative subsampling, class weighting, or focal-style losses, and **always calibrate afterward** if you downsampled (downsampling shifts the predicted probabilities — you must correct them back).

---

## Step 5 — Features

**The goal of this step:** what goes into the model, and — the trap — **leakage / point-in-time correctness.** Mention leakage before they ask.

### Feature families (with the intuition for each)

- **User features:** account age, historical like/comment/share rates, topic affinities, a learned user embedding. *Intuition: how does this person usually behave?*
- **Author features:** how much engagement this author's posts usually earn, posting frequency, a learned author embedding. *Intuition: is this a source people generally engage with?*
- **User–author affinity:** how often *this user* engages with *this author* — messages, past likes, tags, being family. **Often the single most predictive feature in a feed.** *Intuition: you care far more about your best friend's lunch than a stranger's think-piece.*
- **Content features:** text embedding, image/video embedding, post type (photo/link/status/video), length, detected topic, language, has-link. *Intuition: what is this post about and in what form?*
- **Context features:** time of day, day of week, device, connection quality, **session position** (how far down the scroll), time since the user's last visit. *Intuition: the same post is worth more to a bored commuter than a busy multitasker.*
- **Real-time engagement counters:** how many likes/comments this post has gathered in the last N minutes — its **velocity**. *Intuition: is this going viral right now?*
- **Cross features:** `user_topic × post_topic`, `user_country × author_country`. *Intuition: interactions between fields carry signal a linear model misses.*

### The leakage trap — point-in-time correctness (say this unprompted)

Suppose you use "number of likes this post has" as a feature. When you build a **training example** for an impression that happened at 2:00 PM, you must use the like count **as it was at 2:00 PM** — *not* the count today, after the post went viral. Using the final count leaks the future into the past: the model learns "posts that ended up with 10k likes are good," which it can't possibly know at serving time.

> **Analogy:** it's like backtesting a stock strategy using tomorrow's closing price. You'll look like a genius offline and lose money live.

The fix is a **feature store with point-in-time-correct joins**: features are time-stamped, and training joins pull the value *as of the label's timestamp.* This is precisely why Q23 — Feature Store (coming soon) would exist as its own case study. Drop the phrase "point-in-time correctness" and you've earned real points.

---

## Step 6 — Model (the heart)

**The goal of this step:** the two-stage architecture, and inside stage two, the multi-task ranker and the scoring function. This is the longest part — but remember, you should reach it only *after* nailing steps 1–3.

### The two-stage funnel

**Analogy — hiring.** You can't deeply interview all 10,000 applicants. First a cheap **résumé screen** (candidate generation) cuts 10,000 → 100. Then an expensive **panel interview** (ranking) carefully scores the 100 to pick an order. Same logic here: cheap-and-broad, then expensive-and-precise.

```
                         ┌─────────────────────────────────────────────┐
   user opens feed  ───► │  1. CANDIDATE GENERATION  (cheap, recall)   │
                         │  ─ in-network posts since last visit         │
                         │  ─ groups / pages you follow                 │
                         │  ─ out-of-network recommended (two-tower+ANN)│
                         │       thousands  ─►  reduced to ~hundreds    │
                         └───────────────────────┬─────────────────────┘
                                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │  2. RANKING  (expensive, precision)          │
                         │  multi-task deep net → P(like), P(comment),  │
                         │     P(share), P(dwell), P(hide), P(report)…  │
                         │  → combine into ONE value score              │
                         └───────────────────────┬─────────────────────┘
                                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │  3. RE-RANKING / POLICY  (rules on top)      │
                         │  diversity · freshness boost · integrity     │
                         │  demotion · dedup · ad interleaving          │
                         └───────────────────────┬─────────────────────┘
                                                 ▼
                                        ordered feed returned
```

### Stage 1 — Candidate generation

For a Facebook-style feed, candidates come from several **sources**:
- **In-network:** posts from friends, groups, and pages since your last visit — a graph/database fetch, typically hundreds to low-thousands.
- **Out-of-network recommended:** for blended feeds, retrieve relevant posts you don't follow via **two-tower embedding retrieval + approximate nearest neighbor (ANN)** search — the same pattern as video rec ([Q1](./01-video-recommendation.md)). A user tower and a content tower produce embeddings; ANN (HNSW / IVF-PQ) finds close content fast.

Goal here is **recall, not precision** — don't miss anything good; the ranker sorts it out. Cheap filters apply: remove already-seen, blocked authors, deleted/ineligible content.

> If in-network volume is already small (a light user with few friends), candidate generation is trivial and you note that the interesting work is all in ranking. If it's huge (someone in 200 groups), you pre-trim with a lightweight model. Mentioning *both* regimes shows range.

### Stage 2 — The multi-task ranker

Now the core. We want, per candidate, a set of predicted probabilities for different actions. Architecture options, in increasing sophistication:

**(a) Shared-bottom multi-task network.** Shared lower layers learn a common representation; separate "heads" (small MLPs) each predict one action. Cheap, simple, a fine baseline.

```
        sparse & dense features
                 │
        ┌────────▼────────┐
        │  shared bottom   │   (embeddings + shared MLP layers)
        └───┬───┬───┬───┬──┘
            │   │   │   │
          head head head head
            │   │   │   │
         P(like) P(cmt) P(share) P(hide) …
```

**(b) Mixture-of-Experts (MMoE).** The weakness of shared-bottom: when tasks *conflict* (what drives likes may hurt "hide-avoidance"), forcing them to share a representation hurts all of them. **MMoE** uses several expert sub-networks plus per-task **gating** that learns how much each task should draw on each expert. Tasks that align share; tasks that conflict diverge. This is a well-known industrial choice for exactly this multi-objective feed setting.

> **Analogy — the shared kitchen.** Shared-bottom is one chef cooking every dish; when the dessert and the steak need opposite techniques, quality suffers. MMoE is a kitchen of specialist stations (experts) where each dish (task) is routed (gated) to the stations it needs. Shared where helpful, separate where not.

**(c) Handling sparse features.** IDs (user, author, topic) are turned into **learned embeddings** (DLRM-style). Big categorical spaces + embeddings + feature crosses are how these models digest sparse data — a "Wide & Deep" or DLRM lineage.

**Two-stage *within* ranking.** When candidate sets are large, split ranking itself into a **light ranker** (trims hundreds → tens cheaply) and a **heavy ranker** (full multi-task net on the survivors). This keeps you inside the latency budget.

### The scoring function — turning many predictions into one order

This is the "multi-objective → single score" skill the prompt explicitly tests. The classic form is a **weighted sum**:

```
value = w_like·P(like) + w_comment·P(comment) + w_share·P(share)
        + w_dwell·P(dwell>t) + w_meaningful_comment·P(long_comment)
        − w_hide·P(hide) − w_report·P(report)
```

- **Positive weights** on good actions, **negative weights** on bad ones (hides, reports).
- Weights **encode the business's definition of "meaningful."** A thoughtful comment or a reshare-with-text is weighted far above a passive like. This is literally how Facebook's MSI worked.
- **Where do weights come from?** Two answers, and you should give both:
  - **Hand-tuned via A/B testing:** interpretable, controllable, but manual and slow. You run experiments varying weights and keep what improves the north-star without tripping guardrails.
  - **Learned:** a higher-level "value model" predicts a long-term outcome (e.g., next-week retention) and effectively learns the weights. More powerful, captures interactions, but harder to train and harder to keep safe/interpretable.

Full treatment (calibration, Pareto framing, why simply summing raw probabilities is dangerous) is in **[Deep Dive 1](#deep-dive-1--combining-many-objectives-into-one-score).**

### Stage 3 — Re-ranking / policy layer (do not skip this)

The single most-forgotten stage, and an easy way to look senior. After scoring, you **don't** just sort and return. You apply rules:
- **Diversity / author fatigue:** don't show 8 posts from your one chatty friend in a row — cap or decay repeated authors/topics.
- **Freshness boost:** nudge very recent posts up so timely content isn't buried (Deep Dive 3).
- **Integrity demotion:** downrank borderline/misinformation-flagged content even if it's "engaging" (Deep Dive 4).
- **Dedup:** collapse near-duplicate reshares.
- **Ad interleaving:** slot ads per a separate policy (scoped out earlier, but acknowledge the seam).

> **Analogy — the DJ.** The ranker tells you which songs the crowd will love most. The re-ranker is the DJ's judgment: don't play the same artist five times running, keep energy varied, don't play the track that'll clear the floor even if a few people would've loved it. Raw scores → a *set that works as a whole*.

---

## Step 7 — Training

**The goal of this step:** pipeline, cadence, distributed strategy, reproducibility.

- **Retraining cadence:** feeds shift *fast* (news cycles, trends, new content). Retrain **daily**, and for the most time-sensitive signals move toward **near-online / continual learning** on streaming logs. Model freshness is itself a lever — a model trained on last week's world underperforms.
- **Distributed training:** massive log volume needs data-parallel training across many GPUs/hosts, with big embedding tables often **sharded across parameter servers** (embedding tables for billions of IDs don't fit on one machine). This is where the Q24 — distributed training platform (coming soon) muscles matter.
- **Negative sampling:** you can't train on every impression at web scale — subsample negatives, then **recalibrate** so probabilities stay meaningful (critical because the scoring function adds probabilities).
- **Reproducibility:** version **data + features + labels + code + model** together, so any ranked feed can be reproduced and any regression diagnosed. (This is the MLOps discipline that separates "it worked on my machine" from a real platform.)

---

## Step 8 — Evaluation & Serving

**The goal of this step:** the safe path from a trained model to live traffic, and the serving architecture.

### The rollout ladder (never skip a rung)

```
offline eval  ─►  shadow  ─►  canary  ─►  A/B test  ─►  full ramp
(logged data)   (run live,   (1% real   (measure       (with instant
                 log only,    traffic)   north-star +    rollback ready)
                 don't serve)             guardrails)
```

- **Offline:** does it beat the current model on AUC/NDCG/calibration? Gate to decide *what's worth testing.*
- **Shadow:** run the new model on live requests but **don't show its output** — just log it. Catches latency blowups and crazy predictions with zero user risk.
- **Canary:** serve to a tiny slice; watch for errors and metric cliffs.
- **A/B test:** the real verdict — north-star up, guardrails not down, statistically significant. (This is [Q26 — Experimentation Platform](./26-experimentation-platform.md).)
- **Full ramp** with **instant rollback** wired up.

### Serving path (real-time) and the latency budget

Feed ranking is **real-time inference** (you can't precompute a feed that depends on posts from seconds ago). Per request:

```
fetch candidates → fetch features → run multi-task model → score & sort → re-rank → return
   (graph +          (feature store:      (batched          (weighted     (policy)
    ANN sources)      user/author/        inference          value score)
                      content + real-      on GPU/CPU)
                      time counters)
```

Decompose the budget out loud: *"If I have ~300ms, maybe 50ms candidate fetch, 100ms feature fetch, 100ms model inference on a few hundred candidates via dynamic batching, 50ms re-ranking."* Precomputing what you can (user/author embeddings offline; only content-side and real-time counters fetched live) is how you hit it. This connects to [Q25 — Model Serving](./25-model-serving-inference-system.md).

- **Model versioning / rollback:** every served ranking is tagged with a model version; a bad model is one config flip away from reverting.

---

## Step 9 — Monitoring & Iteration

**The goal of this step:** keep it healthy in production and close the loop back to training.

- **Data drift:** feature distributions move (a new device, a viral format). Monitor input distributions and alert on shift.
- **Concept drift:** the *relationship* between features and engagement changes (an election, a pandemic, a new meme format). Same features, different meaning. Detected via degradation in live metrics and via monitoring prediction-vs-outcome over time.
- **Monitor three things separately:** inputs (features), outputs (predictions/score distributions), and **delayed outcomes** (did the engagement we predicted actually happen?). A model can have healthy inputs and outputs but rotting *outcomes* — that's concept drift.
- **Guardrail dashboards:** negative-feedback rate, integrity prevalence, diversity — watched continuously, not just at ship time.
- **Retraining triggers:** scheduled (daily) *plus* event-driven (a drift alert or metric regression fires an earlier retrain).
- **Close the loop:** field failures and heavy negative feedback flow back into the training signal — the system learns from its own mistakes. (This is Q28 — ML Monitoring / Drift (coming soon).)

> **Analogy — the garden, not the statue.** A naive candidate treats the deployed model like a statue: build it, unveil it, done. A senior engineer treats it like a garden: it drifts, weeds grow (engagement-bait, feedback loops), seasons change (concept drift). The monitoring-and-retraining loop is the gardening. **Treating the system as static is a top failure mode** — say the word "loop" often.

---

# Part C — The Interview Delivery Guide

Knowing the design isn't the same as *performing* it in 45 minutes under pressure. Here's how to actually run the room.

### Timeboxing a 45-minute loop

| Time | What you're doing | The trap to avoid |
|---|---|---|
| **0–8 min** | Clarify & scope, write requirements, state scale/latency numbers | Don't skip to models. Interviewers *start* grading here. |
| **8–14 min** | Frame as ML, state the baseline, define metrics + guardrails, name the offline-online gap | Don't forget guardrails and the word "meaningful." |
| **14–22 min** | Data & labels (talk labeling + biases), features (say "point-in-time"), | Don't hand-wave where labels come from. |
| **22–35 min** | Model: two-stage funnel, multi-task ranker, the scoring function, re-ranking layer | Don't forget re-ranking. Don't over-index on architecture minutiae. |
| **35–42 min** | Serving + latency budget, rollout ladder, monitoring + feedback loop | Don't run out of time before monitoring — it's senior signal. |
| **42–45 min** | Handle a curveball follow-up (Part E) | Stay calm; map it to the framework. |

### How to sound senior (the meta-skills)

- **Narrate your process.** *"I'll start by scoping, then framing, then metrics, before I touch models — because the metric decides the model."* You're being graded on structure.
- **State trade-offs, don't hide them.** Every choice has a cost. *"MMoE handles task conflict better but adds serving cost and complexity — I'd start shared-bottom and reach for MMoE if tasks conflict."* Naming the trade-off > picking the fancy thing.
- **Drive with a baseline.** Always: heuristic → simple model → complex model, each justified against latency/cost.
- **Use your own war stories** (this is *your* differentiator). Where the topic touches **data curation, active learning, synthetic data, dedup, or MLOps/reproducibility**, anchor in real work. Most candidates over-index on architecture; you can talk credibly about *how labels are obtained and how the thing stays healthy in production* — which is exactly where senior signal lives. Bring the **data-centric** and **reproducibility** angles into even this recsys question.
- **Think out loud when stuck.** Silence reads as being lost. *"Let me think about how position bias affects this label…"* keeps you graded on reasoning.
- **Manage the whiteboard.** Draw the funnel early and point back to it. A shared visual anchor keeps you and the interviewer aligned.

---

# Part D — Deep Dives (the senior-signal topics)

These are the four "Nail these" items plus the feedback-loop topic, expanded. Interviewers *probe* these to separate senior from mid-level. Being able to go two levels deep on even one of them wins the loop.

## Deep Dive 1 — Combining many objectives into one score

**The problem.** You have `P(like)`, `P(comment)`, `P(share)`, `P(hide)`… and one slot. You need a single number to sort by.

**Weighted linear combination (the workhorse).**
```
value = Σ wᵢ · P(actionᵢ)      (with negative wᵢ for bad actions)
```
- **Pros:** interpretable, tunable, easy to reason about and A/B.
- **Cons:** weights are manual; the "right" weight for a share vs a like is a business/values judgment, not a math fact.

**Why calibration matters here (the subtle bit most people miss).** If you literally add probabilities, they must be **calibrated** — a `0.1` from the share head and a `0.1` from the like head must mean the *same real-world likelihood*, or your weights are secretly lying. If you downsampled negatives during training, raw outputs are *not* calibrated and must be corrected (e.g., Platt scaling / isotonic regression, or a closed-form correction for the downsampling rate). *A calibrated 0.1 must actually happen ~10% of the time.* (This is the same discipline as Ad CTR, [Q3](./03-ad-ctr-prediction.md).)

**Learned combination.** Instead of hand-set weights, train a **value model** that predicts a downstream long-term outcome (retention, next-day return) and let it learn how much each action is "worth." More powerful, captures cross-effects (a comment that leads to a reply thread is worth more than a lone comment), but harder to train, harder to keep safe, and less interpretable.

**Pareto / constrained framing (bonus).** You can frame this as multi-objective optimization: maximize engagement *subject to* negative-feedback ≤ threshold and diversity ≥ threshold. This ties the scoring function to the guardrails directly and sounds sophisticated because it is.

> **Analogy — the hiring committee.** Four interviewers each score a different dimension (coding, design, communication, culture). A "hire" decision is a weighted vote. Weight coding too high and you hire brilliant jerks; weight culture too high and you hire nice people who can't code. Choosing the weights *is* choosing what kind of feed (or team) you're building.

## Deep Dive 2 — Position & presentation bias

**The problem, precisely.** The same post earns more engagement at slot 1 than slot 10 *purely because of position*. Users examine top items more. If you train on raw engagement, the model concludes "top-ranked posts are good posts" — but they were top-ranked *by the old model*, so you're just teaching it to imitate the old model's mistakes. It's a snake eating its tail.

**Analogy — eye-level shelves.** Products at eye level in a store sell more than products on the bottom shelf. If you rank products by sales, you'll conclude eye-level products are "better" and give them even more eye-level space — when really you just measured shelf position, not product quality.

**Fixes (know at least two):**

1. **Position as a feature, neutralized at serving.** Feed the training model the position the item was shown at. It learns to *attribute* some engagement to position. At **serving time**, set the position feature to a constant (e.g., position 1, or a "missing" value) for every candidate, so ranking reflects *relevance only*, with position's effect stripped out. Simple and widely used.

2. **Inverse Propensity Weighting (IPW).** Estimate the probability an item at position *p* was *examined*, then weight each training example by `1 / P(examine)`. Down-weights easy top-of-feed engagements, up-weights the rarer deep-scroll ones, de-biasing the learned relevance.

3. **Exploration / randomization data.** Occasionally shuffle a small slice of traffic to collect *unbiased* examination data, which powers honest propensity estimates. Costs a little short-term engagement to buy long-term correctness.

4. **Factorized (two-tower) models** that explicitly separate a *relevance* term from a *position/examination* term, so you can drop the position term at serving.

> Presentation bias (autoplay video, big images grabbing attention) is the same disease in a different outfit — handle it the same way: treat format as a bias to model and neutralize, not a relevance signal to reward.

## Deep Dive 3 — Freshness & recency blending

**The tension.** Relevance says "show the highly-relevant post from yesterday." Timeliness says "a friend just announced a baby 2 minutes ago — that's *now*." A great feed blends both.

**Techniques:**
- **Freshness as a feature:** feed "age of post" and "time since user's last visit" into the model; it learns freshness matters differently per content type (breaking news decays fast; an evergreen photo doesn't).
- **Time-decay on the score:** multiply value by a decay like `value · e^(−λ·age)` so identical posts lose rank as they age. Tune `λ` per content type.
- **Freshness boost in re-ranking:** explicitly float very recent posts up so timely content isn't buried under an optimized-but-stale ranking.
- **Content cold-start (crucial):** a brand-new post has **zero engagement history**, so velocity/count features are blank. Lean on **content and author features** (what's it about, does this author usually get engagement) plus an **exploration boost** to give new posts a fair shot at gathering signal. Without this, new content can never "get off the ground" — a self-fulfilling death spiral.

> **Analogy — the newspaper front page again.** The editor balances breaking news (fresh, maybe less deeply relevant to you) against the important investigative piece (older, highly relevant). An all-fresh front page is chaotic; an all-relevance page misses the fire down the street. The blend *is* the editorial skill.

## Deep Dive 4 — Engagement-bait & the guardrail problem

**Why this exists.** If you optimize *raw* engagement, you will build a machine that learns the most reliable engagement-drivers are **outrage, clickbait, and rage-bait** — because provocation is engaging. Left unchecked, the optimizer amplifies exactly the content that makes the product worse and users unhappy. This is the central cautionary tale of modern feed ranking.

**The real-world lesson (cite it, carefully).** Facebook's shift to "Meaningful Social Interactions" heavily weighted reshares and comments to move away from passive consumption — but weighting reshares/reactions strongly was later found to amplify divisive, outrage-driven content, and the weighting (notably on "angry" reactions) was walked back. The lesson for your design: **your objective is a powerful optimizer aimed straight at whatever you tell it to want — so be extremely careful what you tell it to want.** Second-order effects of the objective are the whole ballgame.

**Countermeasures (say several):**
- **Value the right actions.** Weight thoughtful comments and shares-with-text over passive clicks; that's what "meaningful" operationalizes.
- **Penalize negative feedback directly.** Put `P(hide)` and `P(report)` in the scoring function with strong negative weights — the model learns to avoid content people regret seeing.
- **Integrity classifiers as demotions.** Separate models flag clickbait, rage-bait, misinformation, borderline content; the re-ranking layer **demotes** them regardless of predicted engagement.
- **Ask users directly.** Surveys ("Was this post worth your time?", "See less of this") give *explicit* value labels that pure behavior can't, and can be trained toward.
- **Guardrail metrics with teeth.** A model that lifts engagement but raises the negative-feedback or integrity-prevalence guardrail **does not ship**, full stop.

> **Analogy — the vending machine that only stocks candy.** Ask it to maximize "units sold" and it'll fill every slot with sugar, because sugar sells. Sales go up; the customers get sick and stop coming. "Meaningful engagement" + guardrails is how you make the machine stock food people are glad they ate.

## Deep Dive 5 — Feedback loops & why your data lies to you

**The loop.** Your model decides what users see → that shapes what they engage with → those engagements become your next training labels → which train the next model. The system **learns from data it generated itself.** Three consequences:

1. **Selection bias / the counterfactual gap.** You only observe outcomes for posts you *showed*. You have no idea how the posts you buried would have done. Your training set is a biased echo of your own past policy — the model can get very good at ranking the kind of content it already ranks highly, and blind to everything else.

2. **Filter bubbles / rich-get-richer.** Content and authors that got shown get more engagement data, so they get shown more, so they get more data… Popular items entrench; niche-but-good items starve. Same mechanism creates ideological bubbles.

3. **Amplification of whatever the objective rewards** (ties to Deep Dive 4) — the loop *accelerates* second-order effects, good or bad.

**How to fight it (this is the senior payoff):**
- **Exploration.** Deliberately show some content the model is *uncertain* about to gather unbiased data (an explore/exploit trade-off — ε-greedy, Thompson sampling, or dedicated exploration traffic). Costs a little engagement now to keep the model honest long-term.
- **Inverse propensity weighting** to statistically correct the selection bias (same tool as Deep Dive 2).
- **Long-term holdback groups.** Keep a small population on an old/neutral experience for weeks to measure *true* long-term effects (retention, well-being) free of the short-term loop — the only honest read on whether you're building something good or just addictive.
- **Diversity injection** in re-ranking to counter bubble formation directly.

> **Analogy — training a chef only on dishes customers finished.** If you conclude "people love what they finished," you'll narrow the menu to five dishes forever, never discovering the sixth dish they'd have loved — because you never served it. Exploration is deliberately putting new dishes on the menu to *find out.*

---

# Part E — Curveball Follow-ups

Interviewers pressure-test by changing a constraint mid-stream. The skill isn't having memorized answers — it's **mapping the curveball back to the framework** and reasoning live. Some common ones:

- **"Now it has to run with a 50ms budget."** → Shrink model (distillation/quantization), precompute more offline (user & author embeddings), add a cheaper light-ranker before the heavy one, cut candidate count, cache aggressively. Trade some accuracy for latency and *say so*. (Serving, [Q25](./25-model-serving-inference-system.md).)

- **"A user just signed up — cold start."** → No history → lean on demographics, declared onboarding interests, popular/quality content, and *exploration* to learn fast. Distinguish **user** cold-start from **content** cold-start (Deep Dive 3).

- **"Engagement is up but users report the feed feels worse."** → Textbook engagement-bait / objective-mismatch (Deep Dive 4). You optimized a proxy that diverged from real value. Add negative-feedback penalties, integrity demotions, "worth your time" surveys, and check guardrails/long-term holdbacks.

- **"Labels are delayed — a comment thread develops over hours."** → Some heads have delayed labels. Train with delayed-feedback handling (wait windows, importance weighting for not-yet-observed positives), and separate fast signals (click/dwell) from slow ones (thread depth). (Same shape as fraud chargebacks, [Q11](./11-real-time-fraud-detection.md).)

- **"Now blend in lots of out-of-network recommended content."** → The candidate-generation stage grows a real two-tower + ANN retrieval leg (like video rec, [Q1](./01-video-recommendation.md)); the ranker now must weigh social affinity (absent for strangers) against pure content relevance, and integrity/quality bars rise for out-of-network.

- **"How do you handle a user with 5,000 friends posting 10,000 candidates?"** → Pre-trim in candidate generation with a lightweight model, cap per-author, then heavy-rank the survivors. Note the funnel is *elastic* — you widen or narrow stages to fit the load.

- **"How would you A/B test this without market interference?"** → Careful randomization; watch for network effects (your friend's better feed changes what they post, affecting you); use cluster/ego-network randomization; long holdbacks. (Experimentation, [Q26](./26-experimentation-platform.md).)

**The meta-move:** whatever they throw, respond with *"That changes [which framework step], so here's how I'd adapt…"* — you never leave the skeleton.

---

# Part F — Common Failure Modes

Self-audit against these before every mock. Most rejections are one of these, not a knowledge gap:

- ❌ **Jumping to architecture before scoping.** Racing to "I'll use MMoE" in minute two. Scope and metrics first, always.
- ❌ **No baseline.** Fancy model with nothing to beat. Always start reverse-chron → simple → complex.
- ❌ **Ignoring how labels are obtained.** Assuming clean labels appear. Talk implicit feedback, position bias, delayed labels.
- ❌ **Forgetting the offline–online gap.** Treating offline AUC as truth. The A/B test is truth; offline is a filter.
- ❌ **Forgetting the re-ranking layer.** Sorting by raw score and stopping. Diversity/integrity/freshness live here.
- ❌ **Treating the system as static.** No monitoring, no drift, no retraining, no feedback loop. Say "loop" a lot.
- ❌ **Optimizing naive engagement.** Missing the "meaningful" trap and building an outrage machine.
- ❌ **Hand-waving leakage.** Not mentioning point-in-time correctness for real-time count features.
- ❌ **No numbers.** Never quantifying scale or the latency budget. Ballpark them out loud.
- ❌ **Silent problem-solving.** Thinking without narrating. You're graded on visible reasoning.

---

# Flashcards

| Cue | What you must be able to say |
|---|---|
| What is the "meaningful" trap? | The prompt says *meaningful* engagement, not raw engagement. Optimizing raw engagement (clicks, time-spent) builds an outrage/clickbait machine; "meaningful" forces you to define and weight the actions that actually correlate with long-term user value. |
| Why is user–author affinity often the single most predictive feature? | It captures *who posted it*, which in an in-network feed dominates content quality — you care more about your best friend's lunch photo than a stranger's think-piece. |
| What is position bias, precisely? | The same post earns more engagement at slot 1 than slot 10 purely from position, not relevance. Training naively on raw engagement teaches the model to imitate the old model's slot ordering. |
| Name two fixes for position bias. | (1) Feed position as a training feature, then neutralize it to a constant at serving time. (2) Inverse Propensity Weighting — weight each example by 1/P(examine) to de-bias rarer deep-scroll engagements. |
| What is the offline–online gap? | Offline metrics (AUC, NDCG, calibration) measure ranking quality on logged, already-biased impressions. Deploying a model changes what users see, which changes what they engage with — so offline AUC can rise while the online north-star metric goes flat or down. Offline is a filter for what's worth A/B testing; the A/B test is the source of truth. |
| Why must predicted probabilities be calibrated in this system? | The scoring function *adds* probabilities together (`Σ wᵢ·P(actionᵢ)`). If a 0.1 from the share head and a 0.1 from the like head don't both mean "happens ~10% of the time," the weights are secretly lying about their relative importance. |
| Why does negative subsampling require recalibration? | Downsampling negatives shifts the class balance the model was trained on, so its raw output probabilities no longer reflect real-world frequencies — they must be corrected back (e.g., Platt scaling, isotonic regression, or a closed-form downsampling correction). |
| What problem does MMoE solve that shared-bottom doesn't? | When tasks conflict (what drives likes may increase hides), forcing them through one shared representation hurts all tasks. MMoE routes each task through per-task gates over a set of expert sub-networks, so conflicting tasks can diverge while aligned tasks still share. |
| What is the feedback loop, and why is it dangerous? | The model decides what users see → that shapes engagement → those engagements become tomorrow's training labels → which train the next model. The system trains on data it generated itself, causing selection bias (no data on buried content), filter bubbles (rich-get-richer), and amplification of whatever the objective rewards. |
| Name two countermeasures to the feedback loop. | Exploration (deliberately show uncertain content to gather unbiased data) and long-term holdback groups (a population kept on an old/neutral experience to measure true long-term effects, free of the short-term loop). |
| What is point-in-time correctness and why does it matter here? | A feature (e.g., "likes on this post") must be joined at its value *as of the training example's timestamp*, not its current value — otherwise you leak the future (final like count) into a past prediction, which the model can never know at serving time. |
| Why is content cold-start a self-fulfilling death spiral if unaddressed? | A brand-new post has zero engagement history, so velocity/count features are blank; if the model only trusts historical engagement signal, new posts never accumulate enough exposure to gather that signal in the first place. Fix with content/author features plus an exploration boost. |
| What is the real-world engagement-bait lesson (MSI)? | Facebook's Meaningful Social Interactions (MSI) weighted comments/reshares over passive likes to reduce passive consumption — but weighting reactions like "angry" heavily was later found to amplify divisive, outrage-driving content, and had to be walked back. Lesson: the objective is a powerful optimizer aimed at exactly what you tell it to want. |
| What are the three tiers of metrics, in one line each? | Offline (proxy quality: AUC, calibration, NDCG — a filter for what to test), Online (the north-star, e.g. weighted meaningful engagement — the source of truth via A/B test), Guardrails (must-not-regress: negative feedback, integrity, diversity, retention). |
| Why is reverse-chronological still worth mentioning as a baseline? | It's simple, cheap, transparent, and not bad — every fancier model must beat it (or a later ranked baseline) in an A/B test on the north-star metric, or it doesn't ship. It's also the graceful-degradation fallback if the ranker goes down. |
| What's the difference between data drift and concept drift here? | Data drift: input feature distributions shift (a new device type, a viral format). Concept drift: the relationship between features and engagement itself changes (an election, a new meme format) — same features, different meaning. |
| Why does the re-ranking / policy layer exist after scoring? | Raw value-score order alone can show 8 posts from one chatty friend, bury breaking news, or surface engaging-but-borderline content. Re-ranking applies diversity/author-fatigue caps, a freshness boost, integrity demotions, dedup, and ad interleaving on top of the score. |
| How do you decompose the feed-ranking latency budget out loud? | E.g., ~300ms total: ~50ms candidate fetch, ~100ms feature fetch, ~100ms model inference (dynamic batching), ~50ms re-ranking — precomputing user/author embeddings offline is what makes this budget achievable. |

---

# Appendix 1 — One-Page Cheat Sheet

**Problem:** order N candidate posts per feed-open to maximize *meaningful* (not raw) engagement, at ~B-user scale, in a few hundred ms.

**Spine:** candidate generation → multi-task ranking → scoring function → re-ranking → serve.

```
1. SCOPE      platform? blended? "meaningful"=? ads out. B DAU, 100k+ QPS, ≤~300ms,
              fresh-critical, degrade→reverse-chron.
2. FRAME      (user,post,ctx)→value score. Baseline=reverse-chron. Multi-task→rank.
3. METRICS    offline: per-task AUC+calibration, NDCG. online: weighted meaningful
              engagement (MSI). guardrails: neg-feedback, integrity, diversity,
              retention. NAME the offline-online gap.
4. DATA       impression/action logs; labels = implicit feedback. Biases: position,
              presentation, absence≠dislike, selection, delay. Imbalance + calibrate.
5. FEATURES   user / author / USER-AUTHOR AFFINITY / content / context / real-time
              counters / crosses. SAY: point-in-time correctness (no leakage).
6. MODEL      2-stage funnel. Cand-gen: in-network fetch + two-tower/ANN OON.
              Rank: multi-task net (shared-bottom→MMoE), heads→P(action).
              Score = Σ wᵢP(actionᵢ) − penalties. Re-rank: diversity/fresh/integrity.
7. TRAIN      retrain daily→near-online; distributed + sharded embeddings; neg-sample
              +recalibrate; version data/feature/label/code/model.
8. SERVE      real-time. budget: fetch cand/feat + infer + rerank. rollout: offline→
              shadow→canary→A/B→ramp, instant rollback.
9. MONITOR    data drift vs concept drift; watch inputs/preds/outcomes separately;
              guardrail dashboards; retrain triggers; CLOSE THE LOOP.
```

**Four deep dives to nail:** (1) weighted vs learned scoring + calibration · (2) position bias → position-as-feature / IPW / exploration · (3) freshness → decay + boost + content cold-start · (4) engagement-bait → guardrails + integrity demotion + "worth your time."

**Your edge:** data-centric (labeling, active learning, dedup, synthetic data) + MLOps/reproducibility. Bring both to every answer.

---

# Appendix 2 — Glossary

- **Candidate generation:** cheap first stage that fetches a manageable set of eligible items (favor recall).
- **Ranking:** expensive second stage that precisely scores/orders candidates (favor precision).
- **Two-tower model:** separate user and item encoders producing embeddings; enables offline item embedding + fast ANN retrieval.
- **ANN (Approximate Nearest Neighbor):** fast "find similar embeddings" search (HNSW, IVF-PQ) trading a little recall for big speed.
- **Multi-task learning:** one model with multiple heads predicting different targets (like/comment/share/hide…).
- **MMoE (Multi-gate Mixture of Experts):** multi-task architecture where per-task gates decide how much to share each expert — handles task conflict.
- **Calibration:** predicted probabilities match real frequencies (a 0.1 happens ~10% of the time). Required when you *add* probabilities.
- **NDCG:** ranking-quality metric rewarding relevant items placed near the top.
- **Position bias:** engagement inflated by screen position, not relevance.
- **IPW (Inverse Propensity Weighting):** re-weighting examples by 1/P(observed) to de-bias.
- **Point-in-time correctness:** using each feature's value *as of the label's timestamp* to prevent future-leakage.
- **Feature store:** system serving consistent, point-in-time-correct features for both training (offline) and serving (online).
- **Data drift:** input feature distributions change over time.
- **Concept drift:** the feature→label relationship itself changes.
- **Feedback loop:** the model's outputs shape the data used to train its successor.
- **MSI (Meaningful Social Interactions):** engagement objective weighting conversation/reshares over passive consumption.
- **Guardrail metric:** a metric that must not regress even if the primary metric improves.
- **Shadow / canary / A/B:** progressively riskier rollout stages (log-only → tiny slice → controlled experiment).

---

# Further Reading & Tools

**Papers**

- [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://scholar.google.com/scholar?q=Modeling+Task+Relationships+in+Multi-task+Learning+with+Multi-gate+Mixture-of-Experts) — the MMoE architecture used in [Step 6](#stage-2--the-multi-task-ranker) to resolve task conflict between engagement heads.
- [Wide & Deep Learning for Recommender Systems](https://scholar.google.com/scholar?q=Wide+%26+Deep+Learning+for+Recommender+Systems) — the wide-plus-deep lineage behind how sparse ID features and crosses are digested (Step 6).
- [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://scholar.google.com/scholar?q=Deep+Learning+Recommendation+Model+for+Personalization+and+Recommendation+Systems) — the DLRM approach to large embedding tables for sparse categorical features (Step 6, Step 7).
- [Bringing People Closer Together](https://scholar.google.com/scholar?q=Bringing+People+Closer+Together) — Meta's public writeup on the shift toward "Meaningful Social Interactions" (Step 3, Deep Dive 4); read alongside the later reporting on why the reaction-weighting had to be walked back.

**Tools**

- [hnswlib](https://github.com/nmslib/hnswlib) — a widely used HNSW implementation for the approximate nearest neighbor search behind out-of-network candidate retrieval.
- [FAISS](https://github.com/facebookresearch/faiss) — Meta's library for ANN search including IVF-PQ, used at the same stage.

---

# Appendix 3 — How this connects to the rest of the 28

News feed ranking is a hub. Recognizing shared machinery is how you make the Blind-28 pay off — each pattern learned here is reusable:

- **[Q1](./01-video-recommendation.md) Video recommendation** — same two-stage funnel; feed leans in-network + social affinity, video leans out-of-network + ANN retrieval.
- **[Q3](./03-ad-ctr-prediction.md) Ad CTR prediction** — the **calibration** discipline here is the whole point there (calibrated P(click) feeds the auction).
- **Q6 Search ranking (coming soon)** — learning-to-rank and NDCG; the retrieval→ranking split is identical.
- **[Q11](./11-real-time-fraud-detection.md) Fraud detection** — extreme imbalance + **delayed labels** (chargebacks ≈ slow-developing comment threads).
- **Q23 Feature store (coming soon)** — the platform that makes **point-in-time correctness** real.
- **Q24 Distributed training (coming soon)** — how you train these models with **sharded embedding tables** across many GPUs.
- **[Q25](./25-model-serving-inference-system.md) Model serving** — the **dynamic batching / latency budget / rollback** machinery behind Step 8.
- **[Q26](./26-experimentation-platform.md) A/B testing platform** — the "how do you *know* it shipped an improvement" muscle; network-effect randomization.
- **[Q27](./27-data-labeling-active-learning.md) Data labeling + active learning** — where labels *come from*; your differentiated territory.
- **Q28 ML monitoring / drift (coming soon)** — Step 9 as its own case study.

Master this one and you've pre-loaded pieces of ten others. That's the Blind-28 thesis working as intended.

---

*Part of the [ML System Design Case Studies](../README.md) series. Contributions and corrections welcome — see [CONTRIBUTING.md](../CONTRIBUTING.md).*
