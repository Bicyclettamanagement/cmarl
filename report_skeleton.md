# Report skeleton

Working title: When explicit context helps multi-agent RL — and when the test is invalid

One-sentence claim: Explicit physics context helps only if the context changes behavior, the hidden baseline can learn the task, and the feature stays inside the training range.

Story order: proposal questions → diagnostic hypotheses → Pistonball (H1) → Multiwalker (H1 rejected, H2 rejected) → clip probe → H3 left open → N=10 as a blocked precondition (appendix).

Hypothesis-driven strategy: the proposal questions stay the questions. H1–H3 are explanations checked after the Pistonball comparison came out null. Each verdict decides the next experiment. A rejected hypothesis is a result. An untested one stays in scope.

## Abstract

- Keywords: contextual MARL, MATD3, concatenation vs hidden context, in-distribution, out-of-distribution, feature extrapolation
- Gap: single-agent contextual RL is established; how to feed context into MARL, and whether it generalizes, is not
- Question kept from the proposal: does explicit context help relative to a hidden-context baseline?
- Answer in three conditions, not a yes/no
- Pistonball: physics slices flat or cloned → context not identifiable; varying training helps both methods
- Multiwalker: concat in-distribution IQM 61 vs hidden 29; out-of-distribution IQM −16 vs hidden 16
- Clip probe: same concat policies, true out-of-distribution physics, vector magnitude clamped to the training window → out-of-distribution IQM −16 → 42; in-distribution stays ~62
- Mass and friction: mostly vector extrapolation. Gravity: extrapolation plus a real dynamics failure
- N=10 Pistonball: base skill not learned → not a context comparison

## 1. Introduction

- MARL: shared environment, coordination, brittle policies under changed dynamics
- Context: metadata for dynamics or task (here: physics factors)
- Single-agent contextual RL as motivation; multi-agent integration still open (from `introduction.md`)
- This project: one integration, concatenation, against a hidden-context baseline
- Contribution is the test and the conditions, including null and harmful results
- Roadmap: proposal questions → diagnostic hypotheses and the decision rule → Pistonball → Multiwalker → clip → what was left open

## 2. Related work

Half a page. One sentence per cluster, plus the citation. No definitions of CTDE, TD3, or in-distribution versus out-of-distribution. Those conventions live in the method.

- Contextual MDPs and explicit context features
  - Single-agent line this study extends: a context changes the dynamics, and the agent may be given that context or left to infer it
  - Cite the definitional cMDP paper you used, and CARL (Benjamins et al.) as the close predecessor: context-aware versus context-blind agents, with concatenation as the simple injection
  - Role in this report: hidden versus concat is that comparison, moved to cooperative MARL
- Domain randomization
  - Training on a range of dynamics rather than one nominal setting (Tobin et al.)
  - Role: the Pistonball “varying ≫ static” result sits here. It is shared by hidden and concat, so it is not evidence for explicit context
- Zero-shot generalization
  - Policies trained in one set of environments fail on held-out variations; extrapolation past the training support is a known failure mode (Kirk et al., survey)
  - Role: justifies the hard factor grid and the clip probe. The probe asks whether the failure is the out-of-range feature
- The algorithm and the score, one clause each
  - MATD3 (Ackermann et al.) as the continuous-action CTDE baseline. No derivation
  - IQM and bootstrap intervals (Agarwal et al., rliable) as the reported aggregate
- Other ways to inject context, one clause, then stop
  - Learned embeddings, FiLM, and latent context exist. This study tests concatenation only, which is why H3 stays open
- Leave out
  - How twin critics, target networks, or CTDE work
  - A tour of QMIX, MAPPO, and other MARL algorithms
  - PettingZoo and the two environments: name them in the method, not here

## 3. Questions

Keep the proposal questions here. H1–H3 are not these questions. They are the explanations checked in the Discussion after the first comparison came out null.

- RQ1 (integration), from `research_questions.md`
  - Form used: global physics parameters, identical for every agent
  - Mechanism tested: direct concatenation
  - Comparison: explicit concat vs hidden
  - Left open, and named later as H3: latent context, FiLM, actor-only vs critic-only, agent-local context
- RQ2 (generalization)
  - In-distribution vs out-of-distribution on a fixed hard factor grid
  - Zero-shot to unseen factors in the same environment
  - Sample efficiency recorded (steps to threshold); secondary to the return gap
  - Transfer across a suite of environments: not run, because the clip result replaced that step
- Diagnostic hypotheses (details and verdicts in `hypotheses.md`; argued in the Discussion)
  - H1 Environment insensitive: the factors barely change return, so context has nothing to condition on
  - H2 Context redundant: the factors matter, but observations already reveal them, so concat ≈ hidden
  - H3 Wrong context kind: global physics is the wrong signal
  - Follow-up, after H2: the out-of-distribution loss is extrapolation of the concatenated vector

## 4. Method

### 4.1 Plan vs what was run

- Proposal (`methodology.md`): pre-study on one environment (concat vs hidden) → pick a winner → transfer study on further environments
- What changed: a null comparison is not a winner to carry into transfer. The next experiment is the alternative explanation that the null left open
- Stopping rule, in order
  - H1 supported → the environment cannot identify context. Switch environment. Do not add a new encoder
  - H1 rejected and H2 supported (concat ≈ hidden) → observations already carry the factor. Stop
  - H1 rejected and H2 rejected → context is used, and the in-distribution / out-of-distribution split is the result. Next test is the mechanism of the out-of-distribution half (clip), not a new architecture and not a transfer suite
  - Clip recovers return → the feature was extrapolated. Transfer of that feature is not supported
  - Clip leaves a drop → that axis is also a dynamics failure (gravity)
  - H3 stays open: a different context kind was not required to explain the measured failure
  - Separate precondition, not one of H1–H3: if the hidden baseline never learns the skill, the comparison is blocked (N=10)
- Executed path
  - Pistonball N=5, static vs varying, hidden vs concat (the 2×2) → H1 supported
  - Multiwalker, same protocol → H1 rejected, H2 rejected
  - Clip probe on the existing concat checkpoints → extrapolation mostly supported
  - N=10 curriculum: attempt to make Pistonball harder; stopped after seed 1 because the skill was not learned

### 4.2 Algorithm and context

- MATD3, shared actor, one-hot id
- Context keys concatenated when `context_mode=concat`; omitted when hidden
- Normalization, this project’s convention: feature = value / |reference|, so gravity is a negative factor. The clip keeps that sign and clamps the magnitude to [0.75, 1.5]
- Same networks, same training budget, paired seeds
- Training regimes
  - Static: one nominal physics setting
  - Varying: factors resampled inside [0.75, 1.5]
- Eval grid (hard): one axis at a time, factors from 0.25× to 4×, plus the nominal point
- In-distribution: factors inside [0.75, 1.5], including nominal
- Out-of-distribution: factors outside that window
- Scores: IQM over seeds × contexts; gap = in-distribution IQM − out-of-distribution IQM

### 4.3 Environments

- Pistonball (PettingZoo): N=5 pistons, image observations, ball physics (mass, friction, elasticity, gravity). Goal: move the ball to the left wall
- Multiwalker: N=3 walkers, vector observations, package on top. Keys: gravity, package mass, terrain friction. Forward reward scaled ×5
- Why the second environment exists: H1 was supported on Pistonball, so the same protocol was repeated where a sensitive axis could reject it

### 4.4 Clip probe

- Eval only, concat checkpoints, hard grid, 5 seeds
- Simulator stays at the true factor
- Only the vector fed to the policy is changed: magnitude clamped to [0.75, 1.5], sign kept
- In-distribution points are unchanged by the clamp
- This is the follow-up hypothesis, not part of the original proposal: if the out-of-distribution drop is extrapolation of the feature, clamping recovers return; if it is the dynamics, clamping does not

## 5. Results

### 5.1 Pistonball: the comparison is not identifiable

- Figure/table: `reports/diagnostic_hard_dashboard.csv`, `reports/diagnostic_hard_slices.csv`, `reports/figures/diagnostic_day/`
- 2×2 IQM (hard grid): hidden static 67.1, hidden varying 88.8, concat static 70.8, concat varying 85.9
- Regime effect: varying ≫ static for both methods
- Context effect: concat does not beat hidden
- Every axis flat. Hidden mass and elasticity cloned (identical returns across factors)
- Reading, for the Discussion: H1 supported. The factors do not move return, so concatenation has nothing to condition on
- Phrase: null result with a diagnosed cause, not a failed run
- H2 is not testable here. Concat ≈ hidden is what H1 predicts, so it is not evidence that observations already reveal the dynamics

### 5.2 Multiwalker: concat helps inside the window and hurts outside it

- Figure/table: `reports/diagnostic_multiwalker_dashboard.csv`, `reports/rliable_multiwalker.json`, `reports/figures/multiwalker_matrix/`
- Hidden: IQM 21.4, in-distribution 29.4, out-of-distribution 15.5, gap 13.9
- Concat: IQM 11.9, in-distribution 61.3, out-of-distribution −15.5, gap 76.8
- All three axes sensitive (gravity, package mass, terrain friction), strong negative factor–return correlation
- Reading, for the Discussion: H1 rejected (all three axes sensitive). H2 rejected (concat is not about the same as hidden)
- The split is the result: in-distribution gain, out-of-distribution loss. Overall IQM hides it. Lead with the split

### 5.3 Clip probe: most of the out-of-distribution loss is the vector

- Figure/table: `reports/diagnostic_multiwalker_clip_dashboard.csv`, `reports/diagnostic_multiwalker_clip_slices.csv`, `reports/figures/multiwalker_context_clip/`, `reports/rliable_multiwalker_context_clip.json`
- Three-way IQM: hidden 21.4, concat 11.9, clipped concat 49.7 [43.9, 55.3]
- In-distribution IQM: 61.3 → 62.1 (clamp is a no-op on training-range features)
- Out-of-distribution IQM: −15.5 → 42.1 (hidden out-of-distribution is 15.5)
- Probability hidden beats clipped concat: 0.20 [0.14, 0.26]
- Per-axis drop (in-distribution mean − out-of-distribution mean)

  | Axis | Hidden | Concat | Clipped concat |
  |---|---:|---:|---:|
  | Package mass | 6.8 | 69.1 | 14.5 |
  | Terrain friction | 11.1 | 64.9 | 8.7 |
  | Gravity | 61.2 | 87.3 | 54.0 |

- Reading, for the Discussion: the extrapolation hypothesis is mostly supported
- Mass and friction: drop mostly disappears → those failures are extrapolation of the concatenated feature
- Gravity: drop shrinks (87 → 54) and clipped out-of-distribution mean (−15.6) is still better than hidden (−36.6), but a large penalty remains → dynamics and extrapolation both matter
- Caveat, one sentence: episode returns are not bitwise repeats across evals; every seed’s out-of-distribution mean still rose by far more than the in-distribution jitter

### 5.4 Sample efficiency and coordination

- Both Multiwalker methods reached the performance threshold (median steps on the order of 2.5–2.9×10⁵). Not the contrast to feature
- Pistonball learning curves: varying regime is the sample-efficiency story, shared by hidden and concat
- Per-agent reward variance was uninformative here. Do not build a coordination claim on it

## 6. Discussion

Argue the hypotheses. The numbers stay in Results. One short paragraph per verdict: the claim, the verdict, the condition it settles. Say explicitly that H1–H3 were the explanations checked after Pistonball, not the questions in Section 3.

- H1 — environment insensitive
  - Claim: factors barely change return, so explicit context has nothing to condition on
  - Pistonball: supported. All axes flat; hidden mass and elasticity cloned; gap ≈ 0; the real effect is varying ≫ static
  - Multiwalker: rejected. Gravity, package mass, and terrain friction all move return
  - Condition settled: a context comparison is informative only when the factor changes behavior. This is why the environment switch is part of the result
- H2 — context redundant
  - Claim: the environment is sensitive, but observations already reveal the dynamics, so concat ≈ hidden
  - Pistonball: not testable, because H1 already holds
  - Multiwalker: rejected. In-distribution IQM 61 vs 29; out-of-distribution IQM −16 vs 16; overall IQM then ranks hidden ahead (21 vs 12, probability hidden beats concat ≈ 0.52)
  - Condition settled: explicit context is used. It specializes inside the training window and is brittle outside it
- Follow-up — feature extrapolation
  - Claim: the Multiwalker out-of-distribution loss is an out-of-range vector, not an inability to control the new physics
  - Verdict: mostly supported. In-distribution IQM stays ~62; out-of-distribution IQM −16 → 42
  - Mass and friction recover. Gravity only partly does (drop 87 → 54)
  - Condition settled: the concatenated feature is trustworthy inside the training support. Outside it, the same feature can mislead the policy even when the new physics is controllable
  - Two failure modes from the same probe: recovered return means the policy could control the new physics and was misled by the feature; the remaining gravity drop means the new dynamics are hard even with an in-range feature
- H3 — wrong context kind
  - One scope sentence, not a finding. Global physics is relevant on Multiwalker, and the measured failure is how that vector is used out of distribution
  - Agent-local context, FiLM, latent context, and actor-only vs critic-only were not run. The results do not say those would fail
- Precondition, separate from H1–H3
  - The hidden baseline has to be able to learn the skill. The N=10 pilot fails this. One sentence here; curves in the appendix
- What “context helps” means after these verdicts
  - Higher return inside the training window on a sensitive environment
  - A separate check that the gain survives out-of-distribution physics
  - Concat can be the better in-distribution method and still lose on overall IQM
- Link back to the proposal
  - RQ1 is answered for concatenation only
  - RQ2 is answered as a split (in-distribution gain, out-of-distribution loss), not as uniform transfer
  - The planned transfer suite was the step this sequence replaced, because extrapolation of the feature was the open failure

## 7. Scope and limitations

- H3 left open on purpose: latent, FiLM, actor-vs-critic, and agent-local context. The open question after H2 was support of the feature already in use, so a new encoder was not the next experiment
- Two environments, chosen by the validity check. No third-environment transfer suite
- Pistonball context axes were weak; that was measured and is the reason for the switch
- Multiwalker: 5 seeds, varying regime, forward reward ×5, hard one-axis grid
- N=10 curriculum: one seed per method. Advances were mostly the episode cap, not mastery. Concat peaked and then saturated. Treat as a pilot that the harder task was not learned
- Clip probe reuses the concat checkpoints. It does not retrain
- Hidden-context agents never see a vector, so there is no clip condition for them

## 8. Conclusion

- Repeat the verdicts: H1 supported on Pistonball, rejected on Multiwalker; H2 rejected on Multiwalker; extrapolation mostly supported; H3 left open
- Pistonball: regime matters, context does not, because the slices are flat
- Multiwalker: explicit context is worth a large in-distribution gain and costs a large out-of-distribution loss
- That cost is mostly extrapolation of the vector for mass and friction, and only partly so for gravity
- A harder agent count does not create a context result if the base algorithm never learns the task

## Appendix A — N=10 curriculum pilot

- Purpose: shorten the credit-assignment horizon by spawning the ball nearer the goal, then walking the spawn right
- Seed 1 only. Do not put it in the main table
- Hidden: best in-training eval ~68, offline nominal success ~0.3, post-graduation success ~0.07
- Concat: peak ~60 around 100k steps, then eval frozen; offline nominal success ~0.1
- Most stage advances hit the episode cap
- Reading: precondition for a context test was not met

## Appendix B — Figure checklist

Canonical set, with place and why: `reports/figures/report/FIGURES.md`.
Regenerate with `scripts/collect_report_series.py` then `scripts/export_report_figures.py`.

- Pistonball 2×2 and flat/clone drops: `reports/figures/report/pb_iqm_id_ood.png`, `pb_axis_drops.png`, `pb_slices_varying.png`
- Pistonball learning curves (secondary): `pb_learning.png`
- Multiwalker ID/OOD and slices: `mw_id_ood.png`, `mw_slices.png`
- Clip probe, three methods: `clip_id_ood.png`, `clip_axis_drops.png`, `clip_slices.png`
- N=10 native-task eval, appendix only: `n10_eval.png`
- Do not use `reports/figures/multiwalker_context_clip/compare_context_slices.png` (drawn on Pistonball axes) or `compare_sample_efficiency.png` there (clip is not a separate training run)

## Talk map (~10 slides)

- Proposal question, then the diagnostic hypotheses and the stopping rule
- Setup: MATD3, hidden vs concat, ID window, hard grid
- H1 on Pistonball: 2×2, flat/clone, varying ≫ static
- Why that null cannot answer “does context help,” and why the next step is another environment
- H1 rejected and H2 rejected on Multiwalker: ID gain, OOD loss
- Clip follow-up: world unchanged, vector clamped
- Clip verdict: mass/friction recover, gravity only partly
- H3 in one sentence: left open. N=10 in the same slide: skill not learned, so not a fourth hypothesis
- Conclusion: what “context helps” means after the verdicts
