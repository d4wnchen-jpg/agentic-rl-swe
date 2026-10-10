# Pitfalls

Six pitfalls hit while getting the agentic RL loop running on a single GPU.
Each one has measurements behind it.

**Full derivations are in [RESULTS.md](RESULTS.md); how it is built is in
[IMPLEMENTATION.md](IMPLEMENTATION.md). This file only states conclusions.**

[中文](LESSONS.md) | **English**

| # | Pitfall | Cost |
|---|---|---|
| 1 | [Template renders the two sites differently](#1-chat-template-the-two-sites-disagree) | 90% of rollouts echo after training |
| 2 | [Grading tests visible to the model](#2-grading-tests-visible-to-the-model) | 99% hit the turn limit, 1/113 submits |
| 3 | [Eval set composition hides the effect](#3-eval-set-composition-hides-the-effect) | Real gain diluted to +2 tasks |
| 4 | [Zero-variance groups are 36%](#4-zero-variance-groups-are-36) | Training-volume estimate low by 1.6× |
| 5 | [Evaluation noise is ±6pp](#5-evaluation-noise-is-6pp) | Two runs point opposite ways |
| 6 | [The model stops touching things](#6-the-model-stops-touching-things) | "Less verification" tracks lower success |

---

## 1. chat template: the two sites disagree

Qwen3.5's native template inserts `<think>\n\n</think>\n\n` at the **generation
point** but not in **history rendering** — two code paths, so consecutive turns
do not form a token prefix chain, and the prompt tail keeps a complete empty
thinking block that can be copied wholesale.

| | Base | step020 |
|---|---|---|
| Rollouts that echoed | 0 / 113 | **108 / 113 (96%)** |
| What it echoed | — | Always `<think>\n\n</think>\n\n`, longest 77,824 chars |

**Fix:** use [`tools/tpl_chat.jinja`](../tools/tpl_chat.jinja), so the generation
prompt is the history rendering with empty content.
Measured repetition **90.3% → 0%**.

**⚠️ Swapping the template requires retraining.** Train and inference must match:
changing only the inference configuration without retraining took the same-task
pass rate from 25.4% to 15.3%.

→ Full derivation: [RESULTS.md §7](RESULTS.md#7-生成退化一个观察到的现象)

---

## 2. Grading tests visible to the model

When the test files are open to the model, it runs the suite. pytest emits tens
of thousands of characters, `obs_cap` truncates them, and **only the last ~66
tests survive** — so the model never gets a signal that it is done.

| | Base | step020 |
|---|---|---|
| Hit the turn limit | 112 / 113 (99%) | 85 / 113 (75%) |
| Submitted voluntarily | 1 | 19 |
| F2P nodes its runs covered | 34.8% | 30.2% |

**Of the 27 tasks that ended up passing, only 7 (26%) ever saw a green run.**

**Fix (following the reference implementation):** remove the F2P test files from
the sandbox (restore them before grading), and have the prompt tell the model to
"write a minimal reproduction script → edit → re-run to confirm".
This project has restored the workflow (on by default); removing the test files
is still to do.

---

## 3. Eval set composition hides the effect

74% of the 113-task eval set is beyond the base model.

| Tier | Tasks | Base | step020 | Change |
|---|---|---|---|---|
| mixed (learning signal) | 17 | 5/17 | **9/17** | **+4** |
| allpass | 12 | 10/12 | 10/12 | 0 |
| allfail | 84 | 12/84 | 10/84 | −2 |
| **Total** | **113** | **23.9%** | **25.7%** | **+2** |

**Training improved the mixed tier by 80% relative, and the allfail tier (74% of
the set) diluted that into +2 tasks overall.**

**Fix:** report the eval set by tier. Without that, no experiment can show an effect.

---

## 4. Zero-variance groups are 36%

```
80 groups total (20 steps x 4 tasks)   29 zero-variance (36%)   64% usable signal
```

A group whose 8 rollouts all pass or all fail has advantage identically zero and
produces no gradient.

**Impact:** "detecting 12 pp needs 15× the training volume" becomes **24×** once
corrected for usable signal.

**Fix:** make the task pool dynamic — drop tasks the model has stopped failing on
(DAPO's Dynamic Sampling).

---

## 5. Evaluation noise is ±6pp

The same configuration, evaluated twice, points in opposite directions:

| Run | Base | step020 | Change |
|---|---|---|---|
| First | 30.1% | 27.4% | −2.7 pp |
| Second | 23.9% | 25.7% | +1.8 pp |

The base model alone differs by 6.2 pp between runs. Per task: 13 fail→pass,
11 pass→fail, net +2, but 24 flips in total — **the flip count is 12× the net change**.

**Fix:** a larger eval set, and averaging over k>1 samples. Until then, a
single-run difference is not a conclusion.

---

## 6. The model stops touching things

| | Base | step020 |
|---|---|---|
| Never edited any file | 36% | **68%** |
| Never ran pytest | 26% | **55%** |
| Edits per rollout | 1.9 | 0.5 |

And within step020, **the successful group ran pytest 86% of the time versus 44%
for the failing group** — less verification tracks directly with lower success.

This is adaptation to a bad environment, not capability loss: if running the tests
never shows you the result, running them less and submitting earlier is rational.

**Fix:** retrain after fixing pitfalls 1 and 2.

→ Full derivation: [RESULTS.md §4.1](RESULTS.md#41-附带观察编辑率下降)

---

## What to change

| # | Change | Evidence | Cost |
|---|---|---|---|
| **1** | **Report the eval set by tier** (mixed / allpass / allfail) | Pitfall 3 | Free |
| **2** | Grow the eval set to ~150 tasks | Pitfall 5: SE 4.3 pp → 3.7 pp | Free |
| **3** | Make the task pool dynamic | Pitfall 4: 36% zero-variance | Small |
| **4** | Take larger policy steps | `clip_frac` stays 0.0–0.2%, so PPO degenerates to REINFORCE | Tuning |
| **5** | Average over k>1 samples | Pitfall 5 | Eval time ×k |
| **6** | Do an SFT cold start first | Base is 23.9%, so ~27 successful trajectories are already available | Data + short run |
| **7** | Raise training volume to ≥10,000 rollouts | Detecting 12 pp needs ~24× | ~140 hours on one GPU |
| **8** | Grow the data to thousands of tasks | The reference implementation uses 6,000 | Data work |

**Priority: 1 > 2 > 3 > 4 > 6 > 7 > 8.** The first is a precondition — without
per-tier reporting, none of the later experiments can show an effect.

---

## Method

**One: "aligned with the reference implementation" has to be checked word by word.**

Of the four places this project claimed alignment, three did not hold:

| Claim | Reality |
|---|---|
| chat template aligned with the reference | two sites differ |
| prompts aligned with SWE-agent | five-step workflow was dropped |
| `obs_cap` aligned with the reference | 8000 vs 10000 |
| `_OVERLONG_WARNING` aligned with the reference | ✅ identical |

**Two: "I did not see it" is not "it is not there".**

Before concluding X does not exist, search for X itself. This project got that
wrong three times.

**Three: work back from the symptom; do not take the cheapest explanation.**

```
Symptom: 99% hit the turn limit, 1/113 submits
  Cheapest explanation → "not enough training" → add 24× (140 hours) → nothing changes
  Actual cause         → a copyable block at the prompt tail plus truncated output,
                         so the model never gets a completion signal
```

List every cause the symptom could have first, then rule them out with data one by one.
