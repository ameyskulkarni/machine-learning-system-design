# Case Study: Real-Time Fraud / Anomaly Detection

> **Q11** · Tag: **[Core]** · Family: Classification, Detection & Trust/Safety
>
> **Prompt:** *"Detect fraudulent transactions in real time."*
>
> **Teaches:** Extreme imbalance (<0.1% positives), cost-sensitive learning, real-time low-latency scoring, and graph/velocity features. Master this one and you've also half-answered spam detection, abuse detection, and content moderation — they share the same skeleton.

---

## How to use this document

This is a teaching case study, not a cheat sheet. Read it top to bottom once to build intuition, then use the **Cheat Sheet** at the end to rehearse. Every section maps to a step of the Part 0 framework so you learn the *process*, not just the answer.

The single most important mindset shift for this question:

> **Fraud detection is not a classification problem. It is a cost-minimization problem that happens to use a classifier.**

If you internalize only one thing, make it that. Almost every "senior signal" moment in this loop comes from reasoning about *costs and consequences* rather than *model accuracy*.

---

## The 60-second answer

If you had to compress the whole design into one breath:

> Fraud detection is a **cost-minimization problem that happens to use a classifier**, not a classification problem, and it's shaped by three forces: a **microscopic base rate** (well under 1%), an **adaptive adversary** that evolves to evade whatever you ship, and **labels that arrive late and biased** — chargebacks land 30–90+ days out, and you only ever learn outcomes for transactions you *approved*, so blocked traffic vanishes without a verdict (rejection inference). That kills accuracy and even ROC-AUC as metrics, so I'd walk **PR-AUC** for ranking, then set the actual decision threshold on a **cost curve** — comparing `P(fraud) × amount` against the cost of declining, subject to a max false-decline guardrail — which only works if the model is **calibrated**, especially since downsampling negatives during training distorts the raw probabilities and needs an explicit correction (analytic β-shift, then Platt/isotonic). Features lean on **velocity** (bursty counts per card/device over sliding windows) and **graph/entity-linking** signals (one device fanning out across many cards) to catch rings a single transaction can't reveal. The system itself is a **rules engine plus a model cascade** — instant rules and allow-lists as the first line of defense, then a low-latency GBDT as the default scorer, escalating to heavier sequence/graph models or human review only where risk justifies the cost — served **near-real-time** with precomputed features in an online store and live streaming velocity counters, since feature fetch (not inference) is the real latency bottleneck. Finally, because approved-only feedback quietly blinds the model to its own mistakes, I'd close the loop with a dollar-bounded control group that approves some flagged transactions anyway, feed confirmed outcomes back into frequent retraining, and monitor leading indicators (score drift, review hit rate) since true labels lag too far behind to catch decay in real time.

Every clause of that paragraph is unpacked below.

---

## Interview delivery guide

A timed script for a **45-minute loop**, paced against this file's own sections. Adjust on the fly if the interviewer lingers somewhere — that's a signal about what they care about, so follow it.

| Time | Section | What to actually do |
|---|---|---|
| 0:00–1:30 | **0. First, read the room** | Name the three forces up front, unprompted: microscopic base rate, an adaptive adversary, and late/biased labels. This single move reframes the whole conversation before you've been asked a single question. |
| 1:30–8:00 | **1. Clarify & scope** | Ask the functional and non-functional questions; do not let yourself sketch a model yet. Land on the one-line spec (fraud type, TPS, latency budget, base rate, decision space, objective) and say it out loud as a checkpoint. |
| 8:00–11:30 | **2. Frame as ML problem** | State input → output, and immediately volunteer "rules AND a model, not a model that replaces rules." Rank supervised as primary, unsupervised anomaly detection as the novel-fraud safety net. |
| 11:30–20:00 | **3. Metrics** | The heart of the interview — protect this time block. Walk accuracy → ROC-AUC → PR-AUC → cost curve → calibration, in that order, so the interviewer sees you *arrive* at cost-based thinking rather than being led to it. |
| 20:00–28:00 | **4. Data & labels** | Cover label delay (chargebacks), label noise (friendly fraud), and the feedback/selection-bias trap (rejection inference) — then the train/val/test split rules. This is the second area where seniority is judged; don't rush it to save time for the model. |
| 28:00–32:00 | **5. Features** | Group into transaction / velocity / graph / behavioral, and name leakage (point-in-time correctness) and freshness unprompted. |
| 32:00–36:00 | **6. Model** | Baseline (rules + logistic regression) → GBDT as the default, justified by latency + tabular data + explainability → sequence/GNN only if justified. Mention the cascade. |
| 36:00–38:00 | **7. Training** | Retraining cadence tied to concept drift, reproducibility, time-based validation split — keep this tight, most of it was already earned in §3/§4. |
| 38:00–42:00 | **8. Serving & evaluation** | Draw the data-flow diagram. State explicitly that feature fetch, not inference, is the latency bottleneck. Mention step-up as a middle option between approve/decline. Close with shadow → canary → A/B → rollback. |
| 42:00–45:00 | **9. Monitoring** | Data vs. concept drift, leading indicators (labels lag!), retraining triggers, close the loop back to training data. |
| (buffer) | **Curveballs** | Whatever's left goes to follow-ups — expect at least one on labels or rejection inference. |

**Meta-tips — volunteer these before you're asked, not in response to a prompt:**

- **Say "accuracy is meaningless here" in the first few minutes**, before the interviewer can steer you toward it. It's the fastest way to signal you're not going to default to a leaderboard mindset.
- **Bring up calibration the moment you mention a threshold or `P(fraud) × amount`.** Don't wait for "is your model calibrated?" — state the formula (`logit(p_true) = logit(p_sampled) + log(β)`) as soon as downsampling comes up, since the two are inseparable.
- **Raise label delay and chargebacks before being asked "where do your labels come from?"** — mentioning the 30–90+ day lag and the maturation window unprompted, in §1 or §4, shows you already know the punchline before the interviewer sets it up.
- **Volunteer the feedback-loop / rejection-inference problem even if nobody asks "how do you know your blocks were right?"** — this is explicitly the question that most reveals seniority, so don't wait for it to be asked; work it into §4 naturally.
- **Frame rules and ML as coexisting, not competing, the first time modeling comes up** — it pre-empts the trap of describing the model as if it's the only line of defense.

---

## 0. First, read the room: what makes this question different

Before touching the framework, understand the three forces that make fraud unlike a normal classification task. Naming these early in the interview instantly signals experience.

**1. The needle is microscopic.** Fewer than 1 in 1,000 transactions is fraud — sometimes 1 in 10,000. This breaks every metric your instincts reach for and forces cost-based thinking.

**2. The adversary is intelligent and adaptive.** A cat classifier competes against cats, and cats don't read your model to evade it. Fraudsters do. The moment your model closes a hole, they probe for the next one. This is closer to cybersecurity or an immune system than to standard ML: the distribution *fights back*. Consequence: your model has a shelf life measured in weeks, and "retrain frequently" is a design requirement, not an afterthought.

**3. The truth arrives late — and biased.** You rarely know at transaction time whether something was fraud. The label often shows up weeks later as a chargeback, or never shows up at all. Worse, you only learn the outcome of transactions you *approved* — the ones you blocked vanish without a verdict. This "you only see what you let through" problem is the deepest part of the question and where the best candidates separate themselves.

> **Analogy — the smoke detector.** A smoke detector that never beeps is "right" 99.9% of the time, because your house almost never burns. It is also completely worthless, and its worthlessness is invisible in the accuracy number. Fraud detection lives entirely in that 0.1%. Any metric or design that lets the 99.9% dominate is lying to you.

---

## 1. Clarify & scope (spend real time here — do not skip to the model)

The fastest way to fail a senior loop is to start drawing a neural network in minute two. Turn the vague prompt into a written spec. Ask questions like:

**Functional questions**
- What are we protecting? Card-not-present e-commerce payments? Card-present? Account takeover? New-account (synthetic identity) fraud? *These are genuinely different problems.* Assume **card-not-present transaction fraud** unless told otherwise, and say so.
- What's the decision space? Just `approve / decline`? Or the richer, more realistic `approve / decline / step-up challenge (3-D Secure, OTP) / send to manual review`? The richer space is a better answer — it lets you act on uncertainty instead of guessing.
- Who consumes the output — an automated block, or a human analyst queue?

**Non-functional questions (the ones that shape the architecture)**
- **Scale:** How many transactions per second? A big processor sees tens of thousands of TPS at peak. This decides your infra.
- **Latency budget:** This is the crux. The fraud check happens *inline* during payment authorization. The entire auth round-trip (merchant → network → issuer → back) typically must finish in a few hundred milliseconds, and fraud scoring gets a **slice of that — often 20–100 ms end to end**, including feature fetch. State this budget explicitly; it eliminates half of all model choices.
- **Base rate:** What fraction is actually fraud? Get them to give you a number (or assume ~0.1–0.5%). This drives your metrics.
- **Cost asymmetry:** What does a false negative cost (fraud loss + liability + investigation)? What does a false positive cost (a declined legitimate customer — lost sale, support cost, and *churn*)? You need both to design anything.

**Turn it into a one-line spec you write on the board:**

> *"Inline scorer for card-not-present transactions, ~10k TPS peak, ≤50 ms latency budget, ~0.2% base fraud rate, outputting a calibrated risk score that maps to approve / step-up / decline / review, optimized to minimize expected dollar loss subject to a bounded false-decline rate."*

Notice how much that single sentence already communicates. Every subsequent decision references it.

---

## 2. Frame it as an ML problem

**Input → output, precisely.** Input: a transaction event (amount, merchant, timestamp, card token, device fingerprint, IP, billing/shipping, plus a rich set of *derived* features about the card/device/merchant's recent history). Output: `P(fraud | transaction)` — a **calibrated probability**, not just a rank.

**Is ML even the right tool?** Partly no — and saying this is a plus. A **rules engine** (velocity thresholds, blocklists of known-bad cards/devices, hard geo rules) should be your first line of defense and will *stay* in production forever, sitting alongside the model. Rules are instant, interpretable, and let you react to a new attack in minutes without retraining. ML handles the fuzzy, high-dimensional patterns rules can't express. **The mature answer is "rules AND a model," not "a model that replaces rules."**

> **Analogy — bouncer and detective.** The rules engine is the bouncer who instantly turns away anyone on the banned list or without ID — fast, dumb, reliable. The model is the detective who reads subtle cues the bouncer can't articulate. You want both at the door.

**Supervised or anomaly detection?** Both, layered:
- **Primary: supervised classification.** You have labels (imperfect ones — see §4), and supervised learning beats unsupervised whenever labels exist. This catches *known* fraud patterns.
- **Complement: unsupervised anomaly detection** (isolation forest, autoencoder reconstruction error). This is your safety net for *novel* attacks that have no labels yet — the cold-start problem for a brand-new fraud pattern. Position it as "catches things the supervised model has never seen," not as the main workhorse.

Stating both, and correctly ranking supervised as primary, is a clean senior signal.

---

## 3. Metrics — the heart of this question

This is where most candidates quietly fail, and where you can shine. Walk it in three moves.

### 3a. Why accuracy is meaningless

With a 0.1% base rate, a model that predicts "never fraud" scores **99.9% accuracy** and catches zero fraud. Accuracy is dominated by the trivial majority class. Say this out loud immediately — it's the smoke-detector point.

### 3b. Why even ROC-AUC is misleading here

ROC-AUC is the reflexive next answer, but under extreme imbalance it's still deceptive. ROC plots true-positive rate vs. false-positive rate, and FPR has that gigantic true-negative count in its denominator. You can add thousands of false positives and the FPR barely moves, so ROC-AUC stays flatteringly high while your review queue drowns in false alarms.

**Use Precision-Recall AUC (PR-AUC) instead.** Precision (of the transactions I flagged, how many were truly fraud) has no giant true-negative term hiding its problems, so PR-AUC actually reflects performance on the rare class you care about.

> **Analogy — grading a lifeguard.** ROC-AUC asks "what fraction of the ocean did you correctly leave alone?" — trivially near-perfect. PR asks "of the people you pulled out, how many were actually drowning (precision), and of the people drowning, how many did you save (recall)?" That's the question that matters.

| Metric | What it measures | Why it fails / works for fraud |
|---|---|---|
| Accuracy | Overall correctness | **Useless** — 99.9% by predicting "never fraud" |
| ROC-AUC | Ranking across all thresholds | **Misleading** — true-negative flood masks false positives |
| **PR-AUC** | Precision vs. recall across thresholds | **Good** — focuses on the rare positive class |
| **Recall @ fixed precision** | Fraud caught while keeping false alarms tolerable | **Operational** — matches a real precision constraint |
| **Precision @ fixed recall** | False-alarm rate while catching a target % of fraud | **Operational** — matches a "catch 80% of fraud" mandate |
| **Detection rate @ review budget** | Fraud caught in the top-N cases a team can review | **Realistic** — analysts have finite capacity |

### 3c. The metric that actually matters: expected cost

Ranking metrics tell you the model is *good*; they don't tell you where to set the threshold. The threshold is a **business decision driven by costs.** Build the cost matrix explicitly:

|  | Predicted legit | Predicted fraud |
|---|---|---|
| **Actually legit** | ✅ $0 (correct approve) | ❌ **False positive** — declined a good customer: lost margin on the sale + support cost + risk of churn |
| **Actually fraud** | ❌ **False negative** — fraud loss (transaction amount) + chargeback fee + liability | ✅ Fraud prevented |

Two subtleties that mark you as experienced:

1. **False-negative cost scales with the transaction amount.** Missing a \$5 fraud ≠ missing a \$5,000 fraud. So you don't pick one global threshold — you pick the decision that **minimizes expected dollar loss per transaction:** compare `P(fraud) × amount` (expected loss if approved) against the expected cost of declining. A cheap transaction can clear a higher risk bar than an expensive one. This is why you need a **calibrated probability**, not just a rank.

2. **False-positive cost is real but hard to quantify — and asymmetric in reputation.** A wrongly declined customer at the grocery checkout may never come back, and they'll tell their friends. Over-aggressive fraud models routinely destroy more value than the fraud they stop. Always name a **guardrail: a maximum acceptable false-decline rate**, and treat it as a hard constraint the optimizer can't violate.

**Cost curves / the operating point:** plot expected total cost as you sweep the threshold, and pick the minimum subject to the false-decline guardrail. That single sentence — "I'd choose the operating point by minimizing expected cost on a cost curve, constrained by a max false-positive rate" — is the whole metrics answer distilled.

### 3d. Calibration — making the probability honest

Everything in §3c multiplies `P(fraud)` by dollars, so the probability has to *mean* what it says. A model is **calibrated** when its numbers are honest: among all transactions it scores 0.02, about 2% truly turn out to be fraud. Calibration is a *separate axis* from ranking — a model can rank perfectly (every fraud above every legit) yet be wildly miscalibrated (all its scores bunched up near 0.9). PR-AUC and ROC-AUC measure only ranking, so they can look great while the probabilities are lies. That's fine for a leaderboard and fatal for a cost-based decision.

> **Analogy — the weather forecaster.** "70% chance of rain" is calibrated if, across all the days the forecaster said 70%, it actually rained about 70% of the time. Someone can be excellent at *sorting* rainy days from dry ones and still useless if they always scream "90%." Discrimination = "can you sort?"; calibration = "do you know how sure you are?"

**Why it's load-bearing here specifically.** Your decision rule compares `P(fraud) × amount` against the cost of declining — you are literally doing arithmetic on the probability. The moment you downsample negatives (§4d/§4e), the model thinks fraud is far more common than it is and inflates every score; feed that into `P × amount` and every threshold is wrong. Calibration is what makes a "0.02" safe to put into a decision.

**What breaks calibration:** downsampling / rebalancing (the big one for fraud), class weighting, and certain model families (deep nets are typically over-confident; SVMs and naive Bayes are natively off). Plain GBDTs trained with log-loss are *fairly* calibrated out of the box — until you resample.

**How to measure it** (you can't fix what you don't measure):
- **Reliability diagram** — bucket the predictions, then plot *mean predicted probability* (x-axis) against *actual observed fraud fraction* (y-axis) per bucket. Perfect calibration sits on the 45° diagonal; sagging below it means over-prediction (your downsampled model), above it means under-confidence.
- **Expected Calibration Error (ECE)** — the sample-weighted average gap between predicted and observed across buckets. One number that summarizes the diagram.
- **Brier score** — mean squared error between the predicted probability and the 0/1 outcome. Handy, but at a 0.2% base rate it's dominated by the easy negatives, so trust the reliability diagram *in the high-score region* where your few positives actually live. (Practical headache: most buckets contain almost no fraud, so estimating calibration exactly where it matters needs a lot of data.)

**How to fix it** — from most-specific-to-fraud to most-general:

**(a) Analytic prior correction — the direct antidote to downsampling.** Because you downsampled *only* the negatives by a *known* factor, there's an exact closed-form fix, and it's just a shift of the intercept in log-odds space:

```
logit(p_true) = logit(p_sampled) + log(β)

  β        = fraction of negatives you kept
  logit(p) = log( p / (1 − p) )
```

Equivalently, in odds space: `odds_true = odds_sampled × β`. Using the §4e example (you kept ~4% of the negatives, so β ≈ 0.04) you add `log(0.04) ≈ −3.22` to every predicted logit, pulling the inflated scores back down to the true ~0.2% scale. It's exact and needs no extra data — reach for this first whenever downsampling is involved.

**(b) Platt scaling — parametric cleanup.** Fit a tiny logistic regression, `p_cal = sigmoid(a · s + b)`, mapping the raw score `s` to the label, on a held-out set *at the true base rate*. Only two parameters, so it's robust even when positives are scarce. Best when the residual distortion is roughly sigmoidal.

**(c) Isotonic regression — non-parametric.** Fits a flexible monotonic step function from score → probability. More powerful than Platt (it can fix non-sigmoidal shapes) but data-hungry and prone to overfitting when positives are few — use it only when you have plenty of fraud examples.

**(d) Temperature scaling — the deep-net default.** Divide the logits by a single learned scalar `T`. One parameter, very data-efficient, and it leaves ranking completely untouched.

**Three rules to state out loud:**
- **Calibrate on a separate holdout at the true base rate**, never on training data — the model has memorized training, so it will look perfectly calibrated there and lie to you. Your natural-rate validation set (§4e) is the right place.
- **Monotonic calibration never changes ranking.** All four methods above are monotone, so AUC / PR-AUC are unchanged — you're only relabeling the score axis with honest numbers. Recalibrating therefore *cannot* hurt discrimination, which is reassuring.
- **Recalibrate in production periodically.** Drift can shift calibration even while ranking holds, so calibration monitoring feeds your retraining triggers (§9).

**The end-to-end recipe** (this ties §4d, §4e, and this section together): train on downsampled data → apply the analytic β-correction → fit Platt (or isotonic) on a natural-base-rate holdout to mop up whatever's left → verify with a reliability diagram + ECE → *then* set your cost-based threshold on the now-trustworthy probabilities.

---

## 4. Data & labels — where the depth is

The interviewer's favorite follow-up is "where do your labels come from?" Have a real answer.

### 4a. The label-delay problem (chargebacks arrive weeks later)

The ground-truth "this was fraud" signal usually comes from a **chargeback** — the cardholder disputes the charge with their bank — which can land **30–90+ days** after the transaction. Consequences:

- **You cannot fully trust recent data.** Last week's transactions haven't had time to "mature"; a transaction currently labeled "legit" might just be fraud that hasn't been disputed *yet*. Training on immature labels teaches the model that fraud is legit.
- **The fix: a label maturation window.** Only treat labels as final after enough time has passed for chargebacks to arrive. This forces a lag between "data collected" and "data usable for training."

> **Analogy — planting and harvesting.** You can't judge the harvest the day after you plant. Labels need to ripen. Training on this-week's data is harvesting green fruit.

### 4b. Labels are noisy, not just late

- **Chargebacks ≠ fraud perfectly.** Some are *first-party fraud* ("friendly fraud"): the real customer made the purchase, then lies to get their money back. Some real fraud is never disputed (small amounts nobody bothers with).
- **Use faster, cheaper proxy labels for early signal:** manual-review analyst decisions, customer "I didn't make this" reports, and confirmed-fraud feeds from the card networks. These arrive in hours/days and let you react quickly, while mature chargeback labels arrive later for final ground truth. A good pipeline blends fast-but-noisy and slow-but-reliable labels.

### 4c. The feedback / selection-bias trap (the deepest point in this question)

Your model **blocks** transactions. Blocked transactions never get a chargeback outcome — so **you never learn whether your blocks were correct.** Over time your training data only contains outcomes for what you *approved*, and the model becomes blind to the region it's already policing. This is **rejection inference** / selection bias.

> **Analogy — the doctor who never follows up on patients sent home.** If you only track outcomes for patients you admitted, you'll never learn that some you turned away were sick. Your "admit" model looks great and is quietly biased.

How to counter it (name at least one):
- **Hold out a tiny, dollar-bounded control group** that gets approved even when flagged (only for low-amount transactions, so the cost of learning is capped). This gives you *unbiased* labels on traffic you'd normally block — the only way to measure the model on its own blind spot.
- **Rejection inference techniques** to model the likely outcome of rejected cases.
- **Log the score of everything, including blocks**, so you can at least monitor distribution shift there.

### 4d. Class imbalance in the training data

- **Keep every positive; downsample the negatives** (you have millions to spare). This is standard and effective — but it changes the base rate the model sees, so you **must recalibrate probabilities afterward** (prior correction / Platt / isotonic — see §3d for the how) so a "0.02" still means a real 2% chance. Ties directly to §3c (you need calibrated probabilities for cost-based decisions).
- Alternatives: class weights / cost-sensitive loss; focal loss for deep models. Be skeptical of SMOTE-style synthetic oversampling for real fraud — it invents unrealistic transactions in a space where fraudsters are very specific.
- **Never rebalance the test set.** Evaluate on the true base rate, or your PR-AUC is fiction.

### 4e. Building the train / val / test sets

Two separate decisions hide inside "how do I build the datasets": *how to split* (along what axis) and *what positive-to-negative ratio* each split gets. Conflating them is a classic mistake, so treat them one at a time.

**Decision 1 — split by time, never randomly.** Fraud has strong temporal structure and an adversary that evolves week to week, so you use an **out-of-time split**: train on the oldest window, validate on the next, test on the most recent. Random shuffling leaks *future* fraud patterns into training and turns your test score into fantasy — you'd be grading the model on patterns it has already seen. A time split mimics reality: learn from the past, predict the future.

Two fraud-specific wrinkles that mark you as experienced:
- **Label maturity caps how recent your test set can be.** If chargebacks take ~90 days (§4a), the test window must be old enough for its labels to have matured. So "most recent" really means "most recent data with *trustworthy* labels" — often data that's already a few months old. The very latest transactions can't be in the test set because you don't yet know their truth.
- **Leave a gap (embargo / purge) between splits.** Because labels ripen over time and some aggregate features span windows, a buffer between train and val/test prevents subtle leakage across the boundary. (This is walk-forward validation with a purge gap.)
- For hyperparameter tuning, prefer **forward-chaining CV** (expanding window: train Jan–Mar → validate Apr; train Jan–Apr → validate May; …) over ordinary k-fold, for the same temporal reason.

**Decision 2 — the ratio, and the rule everyone gets wrong:**

> **Validation and test keep the TRUE base rate. Only the training set may be rebalanced.**

Your val/test sets must look like production (~0.2% fraud). Rebalance them to 50/50 and your precision, recall, PR-AUC, and expected-cost numbers all describe a world that doesn't exist — a "92% precision" that collapses to 4% the moment it hits real traffic. Never touch the evaluation distribution.

The **training** set is where you're free to rebalance (this is the §4d machinery):
- **Positives: use every single one.** Each confirmed fraud is gold, and the *absolute count of positives* — not the ratio — is what actually caps how much the model can learn. With only a few thousand positives total, lean on class weights rather than aggressive downsampling.
- **Negatives: downsample to a target ratio that is itself a hyperparameter** you tune on the natural-rate validation set. Common range is 1:10 to 1:50. There's no magic number: 1:1 gives the sharpest gradient signal but throws away enormous negative diversity and demands heavy recalibration, while 1:50 keeps more variety. Sweep it.
- **Sample negatives thoughtfully.** Over-represent the *hard* negatives — legit transactions that look fraud-ish — and cover the space across merchants, geographies, and amounts. (If you've done hard-negative mining in detection, it's the same instinct.)

**Worked example.** Say 1 year, 500M transactions, 0.2% fraud (≈1M fraud, 499M legit):

| Split | Window | Transactions | Fraud | Legit | Ratio used |
|---|---|---|---|---|---|
| Train | Months 1–8 | ~333M raw | keep all ~666k | downsample to ~13.3M | **1:20 (rebalanced)** |
| Validation | Months 9–10 | ~83M | ~166k | ~83M | **natural 0.2%** |
| Test | Months 11–12 | ~83M | ~166k | ~83M | **natural 0.2%** |

In the train row you kept all 666k frauds and discarded ~96% of the negatives to reach 1:20 — so the fraction of negatives *retained* is β ≈ 13.3M / 333M ≈ **0.04**. Hold onto that β: it's the exact number that feeds the calibration correction in §3d.

And here's the consequence to say out loud: **the moment you downsample negatives, the model's output probabilities become wrong** — it now believes fraud is ~4.8% common (1 in 21) instead of 0.2%, so it systematically over-predicts. That's not a bug to hide; it's a known distortion you deliberately undo with calibration (§3d).

---

## 5. Features — velocity and graph are the stars

Feature engineering matters *more* than model choice here. Group your features and, crucially, tie each group to a fraud behavior.

**Transaction-level (the raw event):** amount, currency, merchant category, hour of day, card-present vs. not, billing-shipping mismatch, is-new-device, etc.

**Velocity features (the workhorse — "how fast?").** Fraud comes in bursts: a stolen card gets tested and drained quickly. Count/sum over sliding windows:
- transactions per card in the last 1 min / 1 hr / 24 hr
- distinct merchants per card per hour; distinct countries per card per day (impossible-travel)
- amount spent per card per window vs. that card's historical norm

> **Analogy — a heartbeat monitor.** A single beat tells you little; the *rate and rhythm* reveal distress. Velocity features are the transaction heartbeat.

**Graph / entity-linking features (catches organized rings — the "many cards, one device" point).** A single transaction can look clean while the *network* screams fraud:
- one **device or IP** touching many different cards
- one **card** fanning out across many merchants
- shared **shipping addresses**, **emails**, or **device fingerprints** across accounts
- graph structure features: node degree, connected-component size, and learned **GNN/node2vec embeddings** of the card-device-merchant graph

> **Analogy — the getaway van.** Any single break-in looks like an isolated event. Notice the same van parked outside ten burgled houses and the *pattern* convicts them. Graph features see the van.

**Behavioral / historical:** deviation from the cardholder's own normal pattern (this user never shops at 3 a.m. in another country), device reputation, merchant risk score.

**Two feature warnings to raise unprompted:**
- **Leakage / point-in-time correctness.** Every feature must be computed *as of transaction time* — never with information from the future. The classic bug: a feature like "total chargebacks on this card" that silently includes chargebacks that happened *after* the transaction you're scoring. A **feature store with point-in-time-correct joins** is how you prevent this at scale. (This is exactly the feature-store question, Q23 (coming soon) — reuse the answer.)
- **Freshness.** Velocity features are worthless if they're an hour stale. This drives the streaming architecture in §8.

---

## 6. Model — baseline first, then justify each step

Interviewers reward a baseline before the fancy model. Climb the ladder and justify each rung against the latency/cost budget.

1. **Rules + logistic regression baseline.** Rules for known patterns; logistic regression for a calibrated, interpretable, millisecond-fast score. Ship this first. It's a real system, not a strawman.
2. **Gradient-boosted trees (XGBoost / LightGBM) — the industry workhorse.** Fraud data is tabular, mixed-type, with nonlinear interactions and missing values. GBDTs handle all of that, train fast, calibrate well, serve in single-digit milliseconds (and can be compiled with something like Treelite), and are far easier to explain to a fraud analyst than a deep net. **This is your default answer for the core scorer.** Say why: latency budget + tabular data + explainability.
3. **Deep / sequence models** when transaction *history as a sequence* carries signal (an RNN/transformer over a card's recent transactions captures "this doesn't fit the pattern"). Justify the extra latency and complexity before reaching for it.
4. **Graph neural networks** for the ring-detection features in §5 — often run **offline/near-real-time** to produce entity risk embeddings that the inline model reads as features, rather than running the whole GNN in the 50 ms budget.
5. **Anomaly detectors** (isolation forest / autoencoder) as the unsupervised safety net for novel fraud (§2).

> **The cascade — your cost-sensitivity at the system level (borrow the airport-security mental model).** Everyone walks through the cheap metal detector (the fast model on 100% of traffic). A few beep and go to secondary screening (a heavier model, or a step-up challenge). A tiny number go to a full search (human review). You spend expensive compute and expensive human attention only where the risk justifies it. If you have a CV background, this is the same "cheap-then-expensive" cascade you use in detection pipelines — say so.

---

## 7. Training

- **Pipeline:** assemble training data with point-in-time-correct feature joins (§5), respecting the label maturation window (§4a). Downsample negatives, keep all positives, then recalibrate (§4d).
- **Retraining cadence:** *frequent*, because the adversary adapts. Daily-to-weekly is common. This is a **design requirement driven by concept drift**, not a nice-to-have. Contrast with a cat classifier you might retrain yearly.
- **Reproducibility:** version data, labels (which mature over time — the *same* transaction's label changes!), features, and code. Being able to reconstruct exactly what a model saw is essential when you need to diagnose a bad decision or an audit. (Ties to the monitoring and feature-store questions.)
- **Validation split by time, not randomly.** Fraud has strong temporal structure and adversarial drift; a random split leaks future patterns into training and massively overstates performance. **Always use a time-based (out-of-time) validation split** (see §4e for how to build the splits and choose the positive/negative ratio). Raising this unprompted is a strong signal.

---

## 8. Serving & evaluation

### Serving architecture (walk this as a data flow)

```
Transaction event
      │
      ▼
[Rules engine]  ── hard block / allow-list (instant, no ML) ──► decision (fast path)
      │ (passes rules)
      ▼
[Feature assembly]
   ├─ Online feature store (Redis/DynamoDB): precomputed slow aggregates, entity/graph embeddings
   └─ Streaming layer (Kafka + Flink): live velocity counters (last 1m/1h/24h)
      │
      ▼
[Model server]  low-latency GBDT (compiled), calibrated P(fraud)
      │
      ▼
[Decision policy]  cost-based threshold + guardrails
      │
      ├─► approve
      ├─► step-up challenge (3-D Secure / OTP)   ← act on uncertainty instead of guessing
      ├─► decline
      └─► queue for human review
      │
      ▼
[Async logging] every transaction + score + features (point-in-time snapshot) → training store
                                                        │
                              (weeks later) chargebacks / reports → labels → retrain
```

**Latency budget decomposition (say this out loud):** the model inference is usually *not* the bottleneck — a compiled GBDT scores in a few ms. **The feature fetch is.** That's precisely why slow features are precomputed into a fast online store and only cheap velocity counters are computed live. Budget it roughly: network + feature fetch (the big one) + inference + rules + decision, all inside ~50 ms.

**The step-up option is a senior move.** Instead of a binary approve/decline on an uncertain score, issue a challenge (one-time passcode, 3-D Secure). It converts a risky guess into evidence — a real customer passes, a fraudster usually can't — at the cost of a little friction. It also softens the false-positive problem: you inconvenience rather than reject.

### Rollout & evaluation

- **Offline:** PR-AUC and cost curves on an out-of-time test set at the real base rate.
- **Shadow:** run the new model alongside production, logging its would-be decisions without acting on them. Compare.
- **Canary → A/B:** ramp traffic gradually. The north-star online metrics are **dollars of fraud lost** and **false-decline rate**, with guardrails on customer complaints and approval rate.
- **Instant rollback** on a versioned model registry — a bad fraud model bleeds money by the minute.

---

## 9. Monitoring & iteration (closing the loop)

Fraud is the domain where monitoring is *most* critical, because the adversary guarantees your model decays.

- **Data drift** (inputs shift): new merchant onboarded, holiday spending spike, new geography. Monitor input feature distributions.
- **Concept drift** (the relationship shifts): a *new fraud pattern* emerges — same-looking inputs, different truth. This is fast and adversarial here. Distinguishing the two matters because they need different responses (data drift may just need recalibration; concept drift needs new features/retraining).

> **Analogy — the immune system.** Pathogens mutate to evade antibodies, so your body keeps generating new ones. Your fraud model is an immune system against an adversary that mutates on purpose. Static defenses lose.

**The monitoring twist: your best metric is late.** Precision/recall on mature labels lags by weeks (§4a). So you need **leading indicators** that move *now*:
- sudden shift in the **score distribution** (the model suddenly seeing lots of high-risk scores)
- spike or drop in **decline / approval rates**
- **manual-review hit rate** (analysts confirming fraud faster than chargebacks return)
- **customer complaint volume** (a proxy for false positives)

**Retraining triggers:** scheduled frequent retrains *plus* drift-triggered retrains, always respecting label maturity. Close the loop: confirmed-fraud cases and analyst decisions flow straight back into the label queue and the next training set.

---

## Curveball follow-ups (rehearse these — they *will* come)

Interviewers test depth by mutating the problem. Prepare a crisp move for each.

- **"Now it has to run at the edge / on-device."** → Push velocity counters and a tiny model to the edge for a fast first pass; keep the heavy graph model and review in the cloud. Emphasize the cascade.
- **"Latency drops to 10 ms."** → Drop live feature fetches you can't afford; precompute more; shrink/compile the model; lean harder on the rules engine for the fast path; move graph scoring fully offline into cached entity embeddings.
- **"Labels are delayed 3 weeks — how do you train?"** → Maturation window + fast proxy labels (analyst decisions, customer reports) for early signal, mature chargeback labels for final truth. Mention the label of a given transaction *changing* over time.
- **"A new fraud pattern appears overnight."** → Rules engine reacts in minutes (no retrain); unsupervised anomaly detector flags the novelty; emergency retrain once you have a handful of confirmed cases; add the pattern's signal as a feature.
- **"How do you know your blocks were right?"** → The rejection-inference / selection-bias answer (§4c): dollar-bounded approve-anyway control group. This is the question that most reveals seniority — nail it.
- **"Isn't blocking real customers worse than some fraud?"** → Yes — cost asymmetry, the false-positive guardrail, and the step-up challenge as a middle path. Mention fairness: monitor for disparate decline rates across regions/demographics (regulatory and ethical exposure).
- **"How do you explain a decline to a customer or a regulator?"** → GBDTs + SHAP-style feature attributions give per-decision reasons; some jurisdictions legally require adverse-action reasons. This is also why you don't reach for an opaque deep net without cause.

---

## Common failure modes

How candidates actually lose this question — these are delivery and process mistakes, not knowledge gaps:

- **Leading with accuracy (or even ROC-AUC) instead of cost-based metrics.** Reciting accuracy, or stopping at ROC-AUC, without noticing that a 99.9% score means nothing at a 0.1% base rate — or that ROC-AUC's true-negative-heavy denominator hides a flooded review queue (§3a–§3b). The interviewer is listening for whether you *arrive* at PR-AUC and a cost curve on your own.
- **Never getting to a threshold at all.** Talking only about ranking metrics (PR-AUC, recall@precision) and never closing the loop to "here's how I'd actually set the decision threshold" via expected dollar cost (§3c). A ranking metric tells you the model is good; it doesn't tell you what to *do*.
- **Forgetting calibration once dollars enter the picture.** Proposing `P(fraud) × amount` as the decision rule and then never mentioning that downsampling negatives (§4d) inflates every score, or skipping the analytic β-correction / Platt / isotonic fix (§3d). This is the single easiest way to look like you don't understand your own proposal.
- **Ignoring label delay and treating recent data as ground truth.** Assuming labels are available immediately and building training/validation splits on the most recent data without accounting for the 30–90+ day chargeback lag or the maturation window (§4a) — this quietly invalidates everything downstream.
- **No exploration / rejection-inference discussion.** Never raising that the model only observes outcomes for transactions it *approved*, and that blocked transactions vanish without a verdict (§4c). Interviewers explicitly use "how do you know your blocks were right?" as a seniority filter — silence here is a strong negative signal.
- **Treating the system as static.** Designing a one-shot model and stopping, without addressing that the adversary adapts, that retraining must be frequent (§7, §9), or that concept drift needs different handling than data drift. Fraud is the one domain where "ship it and move on" is visibly wrong.
- **Rebalancing the validation or test set along with training.** Applying the same downsampling to val/test as to train (§4e), which makes precision/recall/PR-AUC describe a world that doesn't exist and collapses the moment the model hits real traffic.
- **Jumping to the model before scoping.** Sketching an architecture in the first few minutes instead of pinning down the fraud type, latency budget, base rate, decision space, and cost asymmetry (§1) — the classic way to fail a senior loop early.
- **Presenting ML as a replacement for rules rather than a complement.** Missing the "rules AND a model" framing (§2) and implying the rules engine is a legacy component to be phased out, rather than a permanent fast, interpretable first line of defense.
- **Skipping point-in-time correctness.** Proposing features like "total chargebacks on this card" without flagging that they must be computed as-of transaction time, not with future information (§5) — a classic leakage bug that inflates offline metrics and fails silently in production.
- **Only counting the cost of missed fraud.** Optimizing purely for catching fraud and forgetting that false positives have a real, if harder-to-quantify, cost — lost sales, support burden, and customer churn (§3c) — and that a false-decline guardrail is a hard constraint, not a nice-to-have.

---

## Flashcards

| Cue | What you must be able to say |
|---|---|
| Why is accuracy useless for fraud? | At a 0.1% base rate, "always predict legit" scores 99.9% accuracy while catching zero fraud — the smoke-detector problem. |
| Why is ROC-AUC misleading here? | False-positive rate's denominator (true negatives) is enormous, so thousands of false positives barely move it; ROC-AUC stays high while the review queue drowns in false alarms. |
| What metric replaces ROC-AUC for this problem? | **PR-AUC** — precision has no giant true-negative term, so it actually reflects performance on the rare positive class. |
| What ultimately sets the decision threshold? | **Expected dollar cost**, not a ranking metric: compare `P(fraud) × amount` against the cost of declining, on a cost curve, subject to a max false-decline guardrail. |
| What does it mean for a model to be "calibrated"? | Its probabilities are honest — among all transactions scored 0.02, about 2% are truly fraud. Calibration is separate from ranking quality. |
| Why does downsampling negatives break calibration? | It inflates how common the model believes fraud is, so raw scores overstate the true probability — dangerous once you multiply by dollar amount. |
| What's the exact fix for downsampling-induced miscalibration? | The analytic prior correction: `logit(p_true) = logit(p_sampled) + log(β)`, where β is the fraction of negatives kept — an exact, data-free intercept shift. |
| Platt scaling vs. isotonic regression? | Platt fits a 2-parameter logistic mapping (robust with few positives); isotonic fits a flexible monotonic step function (more powerful, but data-hungry). |
| Why do fraud labels arrive late? | Ground truth usually comes from a **chargeback**, which can take 30–90+ days to land — recent "legit" labels may just be undisputed fraud that hasn't matured yet. |
| What is the feedback / selection-bias trap? | You only observe outcomes for **approved** transactions — blocked ones never get a verdict — so the model goes blind to the region it already polices (rejection inference). |
| How do you counter the rejection-inference trap? | Hold out a small, dollar-bounded control group that gets approved even when flagged, to get unbiased labels on traffic you'd normally block. |
| How should train/val/test be split for fraud? | **By time** (out-of-time split) with a purge/embargo gap — never randomly, since fraud has strong temporal structure and an adaptive adversary. |
| Which split(s) may be rebalanced? | Only the **training** set. Validation and test must keep the true base rate, or every metric describes a world that doesn't exist. |
| What are velocity features? | Counts/sums over sliding time windows (transactions per card per minute/hour/day, distinct merchants or countries) — fraud arrives in fast bursts. |
| What do graph/entity-linking features catch that a single transaction can't? | Organized rings: one device touching many cards, one card fanning out across merchants, shared addresses/emails/fingerprints — via graph structure and GNN/node2vec embeddings. |
| Why are GBDTs the default model here? | Tabular, mixed-type, nonlinear data; they train fast, calibrate reasonably well out of the box, serve in single-digit ms when compiled, and are explainable to analysts — unlike a deep net. |
| What is the "step-up" decision option? | Instead of a binary approve/decline on an uncertain score, issue a challenge (OTP, 3-D Secure) — converts a risky guess into evidence at the cost of a little friction. |
| Data drift vs. concept drift in fraud? | Data drift = inputs shift (new merchant, holiday spike). Concept drift = same-looking inputs, new truth (a new fraud pattern) — it needs new features/retraining, not just recalibration. |
| Why must retraining be frequent? | The adversary actively probes and evades the current model, so its effectiveness has a shelf life measured in weeks, not months. |

---

## Cheat Sheet — the 3-minute verbal skeleton

Rehearse until you can say this cold:

1. **Scope:** card-not-present, inline during auth, ~50 ms budget, ~0.2% base rate. Decision space = approve / step-up / decline / review. Objective = minimize expected dollar loss under a max false-decline guardrail.
2. **Frame:** rules engine **AND** a supervised classifier (primary) + unsupervised anomaly detector (novel-fraud net). Output a **calibrated** P(fraud).
3. **Metrics:** accuracy is meaningless (smoke detector); ROC-AUC misleads under imbalance; use **PR-AUC**, then set the operating point on a **cost curve** subject to a false-decline cap. Calibration matters because decisions are `P × amount`.
4. **Data/labels:** labels arrive weeks late via **chargebacks**, are **noisy** (friendly fraud), and are **biased** (you only see approved outcomes → rejection inference → dollar-bounded control group). Maturation window + fast proxy labels. **Splits: time-based (out-of-time) + purge gap; val/test keep the true base rate, only the train set is rebalanced.**
5. **Features:** **velocity** (bursts) + **graph/entity linking** (many cards, one device) are the stars. Watch **leakage / point-in-time correctness**; keep features **fresh** via streaming.
6. **Model:** rules + logistic baseline → **GBDT workhorse** (tabular, fast, explainable) → sequence/GNN when justified. **Cascade** (airport security) for cost control.
7. **Training:** downsample negatives, then **recalibrate** (analytic β-correction → Platt/isotonic on a natural-rate holdout; reliability diagram + ECE to verify); retrain **frequently** (adversarial); **time-based split**; version everything.
8. **Serving:** streaming velocity + online feature store + compiled GBDT; **feature fetch is the latency bottleneck**, not inference; **step-up** to act on uncertainty; shadow → canary → A/B; instant rollback.
9. **Monitoring:** data vs. concept drift; labels lag so use **leading indicators** (score dist, decline rate, review hit rate, complaints); drift-triggered + scheduled retrains; close the loop.

---

## Glossary

- **PR-AUC** — Area under the Precision-Recall curve; a ranking metric that stays informative under extreme class imbalance because precision has no large true-negative term to hide behind.
- **ROC-AUC** — Area under the Receiver Operating Characteristic curve (true-positive rate vs. false-positive rate); misleading under extreme imbalance because the false-positive rate's huge true-negative denominator masks real error.
- **GBDT (Gradient-Boosted Decision Trees)** — An ensemble of shallow trees trained sequentially, each correcting the previous ones' errors; the default model family for tabular fraud data.
- **XGBoost / LightGBM** — Popular open-source, industry-standard GBDT implementations.
- **Treelite** — A model compiler that converts trained tree ensembles (e.g., GBDTs) into optimized native code for low-latency inference.
- **SHAP (SHapley Additive exPlanations)** — A game-theoretic method for attributing a model's prediction to its input features; used to explain individual fraud decisions to analysts or regulators.
- **GNN (Graph Neural Network)** — A neural network architecture that learns representations directly over graph-structured data (here, the card-device-merchant graph), used to surface organized fraud rings.
- **node2vec** — An algorithm that learns low-dimensional embeddings of graph nodes via biased random walks; one way to featurize entities in the fraud graph.
- **Chargeback** — A cardholder-initiated dispute of a transaction with their bank that reverses the payment; the primary (but slow and noisy) source of fraud ground-truth labels.
- **Velocity features** — Features that count or sum transaction activity over sliding time windows (e.g., transactions per card per hour), capturing the "burstiness" characteristic of fraud.
- **Calibration** — The property that a model's predicted probabilities match observed outcome frequencies (a 0.02 score should be right ~2% of the time); a separate axis from ranking quality.
- **ECE (Expected Calibration Error)** — A single summary statistic for miscalibration: the sample-weighted average gap between predicted probability and observed outcome frequency across score buckets.
- **Brier score** — The mean squared error between a predicted probability and the binary outcome; can be dominated by easy negatives under extreme imbalance.
- **Platt scaling** — A calibration method that fits a 2-parameter logistic regression mapping raw model scores to calibrated probabilities.
- **Isotonic regression** — A non-parametric calibration method that fits a flexible monotonic step function from score to probability; more powerful than Platt scaling but more data-hungry.
- **Temperature scaling** — A calibration method (common for deep nets) that divides logits by a single learned scalar, correcting over/under-confidence without changing ranking.
- **TPS** — Transactions per second; the throughput unit used to size the serving system.
- **OTP (One-Time Passcode)** — A short-lived code sent to a user for step-up authentication when a transaction looks risky but not certainly fraudulent.
- **3-D Secure** — A card-network authentication protocol (e.g., Verified by Visa) used as a step-up challenge for uncertain transactions.
- **Rejection inference** — Techniques for estimating the likely outcome of transactions that were rejected and therefore never observed, correcting for the selection bias the decision system itself creates.
- **Out-of-time split** — A train/validation/test split made along the time axis (train on older data, test on newer) rather than randomly, respecting temporal and adversarial structure.
- **Embargo / purge gap** — A deliberate time buffer left between train and validation/test windows to prevent label-maturation or feature-window leakage across the split boundary.
- **Forward-chaining CV** — A cross-validation scheme for time-series data: train on an expanding historical window, validate on the next chronological slice, and repeat forward in time.
- **Friendly fraud (first-party fraud)** — A chargeback filed by the genuine cardholder who made the purchase but disputes it anyway to get a refund; a source of label noise.
- **Synthetic identity fraud** — Fraud committed using a fabricated identity built from a mix of real and fake information, typically targeting new-account opening rather than existing cards.
- **Card-not-present (CNP)** — A transaction (e.g., e-commerce) where the physical card isn't presented to the merchant, as opposed to a card-present (swiped/inserted) transaction.
- **Isolation forest / autoencoder** — Unsupervised anomaly-detection models (isolation forest isolates outliers via random partitioning; an autoencoder flags high reconstruction error) used as a safety net for novel, as-yet-unlabeled fraud patterns.
- **Feature store** — A system that computes and serves features consistently — and point-in-time-correctly — for both offline training and online serving.
- **Point-in-time correctness** — The guarantee that a feature used to score a historical event reflects only information available as of that event's timestamp, preventing feature/label leakage from the future.

---

## Further reading & tools

**Tools**

- [XGBoost](https://xgboost.ai) — the industry-standard GBDT library.
- [LightGBM](https://lightgbm.readthedocs.io/) — a faster, histogram-based GBDT library, often preferred at very large scale.
- [Treelite](https://treelite.readthedocs.io/) — compiles trained tree ensembles into optimized native code for low-latency serving.
- [SHAP](https://shap.readthedocs.io/) — feature-attribution library for explaining individual model predictions.

**Papers**

- Friedman, ["Greedy Function Approximation: A Gradient Boosting Machine"](https://scholar.google.com/scholar?q=Greedy+Function+Approximation%3A+A+Gradient+Boosting+Machine) (2001) — the foundational gradient-boosting paper behind GBDTs.
- Grover & Leskovec, ["node2vec: Scalable Feature Learning for Networks"](https://scholar.google.com/scholar?q=node2vec%3A+Scalable+Feature+Learning+for+Networks) (2016) — the random-walk graph-embedding method referenced in the graph/entity-linking features.
- Wu et al., ["A Comprehensive Survey on Graph Neural Networks"](https://scholar.google.com/scholar?q=A+Comprehensive+Survey+on+Graph+Neural+Networks) (2020) — a broad survey of the GNN architectures used to score the card-device-merchant graph.

---

## The transferable patterns (why this question is worth 5 questions)

This archetype hands you reusable machinery for the rest of your prep:

- **Extreme imbalance + cost-based metrics** → reuse verbatim for **spam/abuse ([Q9](./09-spam-abuse-detection.md))** and **content moderation (Q10, coming soon)**.
- **Cascade of cheap-then-expensive models + human-in-the-loop** → **moderation (Q10, coming soon)**, and structurally identical to detection cascades in vision.
- **Calibrated probabilities feeding a downstream decision** → **ad CTR prediction ([Q3](./03-ad-ctr-prediction.md))** (there it feeds the auction; here it feeds `P × amount`).
- **Real-time features / streaming / online-offline consistency** → **feature store (Q23, coming soon)** and **model serving ([Q25](./25-model-serving-inference-system.md))**.
- **Adversarial drift + retraining triggers + monitoring with lagging labels** → **ML monitoring (Q28, coming soon)**.
- **Graph features / link prediction** → **People-You-May-Know (Q5, coming soon)**.

Two habits to carry into *every* answer, not just this one: (1) reason about **costs and consequences**, not just accuracy — that's where senior signal lives; (2) always account for **how labels are actually obtained and how the system feeds back on itself**, because most candidates forget the loop and treat the system as static.

---

*Part of the [ML System Design Case Studies](../README.md) series. Framework based on a 9-step scoping skeleton: Clarify → Frame → Metrics → Data → Features → Model → Training → Serving → Monitoring.*
