# Hypotheses and findings

Keywords: contextual MARL, MATD3, concatenation vs hidden context, in-distribution, out-of-distribution, flat slices, feature extrapolation

Setup: oracle physics vector concatenated to actor and critic, vs a hidden-context baseline. Hard grid varies one factor at a time (0.25×–4×). In-distribution window is [0.75, 1.5].

## H1 — Environment insensitive

- Claim: physics factors barely change return, so explicit context has nothing to condition on
- Test: C.2 slices. Flat means ID−OOD drop < 5. Cloned means identical returns across factors
- Pistonball: **supported**
  - All axes flat, including gravity (mass, friction, elasticity, gravity)
  - Hidden mass and elasticity cloned
  - ID→OOD IQM gap ≈ 0 for every method
  - Concat does not beat hidden (varying IQM 85.9 vs 88.8; PoI hidden>concat 0.56)
  - Real effect is the training regime: varying ≫ static (IQM ~86–89 vs ~67–71)
- Multiwalker: **rejected**
  - Gravity, package mass, and terrain friction all sensitive
  - Hidden drops: gravity 61, mass 7, friction 11
  - Concat drops: gravity 87, mass 69, friction 65
- Finding: a context comparison is only informative when the factor changes behavior. Pistonball failed that check; Multiwalker passed it

## H2 — Context redundant

- Claim: the environment is sensitive, but observations already reveal the dynamics, so concat ≈ hidden
- Pistonball: **not testable** (needs non-flat slices; H1 holds)
- Multiwalker: **rejected** as “concat ≈ hidden”
  - ID IQM: concat 61.3 vs hidden 29.4
  - OOD IQM: concat −15.5 vs hidden 15.5
  - Overall IQM: hidden 21.4 vs concat 11.9 (PoI hidden>concat 0.52)
  - Median concat−hidden: about +35 on ID contexts, about −20 on OOD contexts
- Finding: explicit context is used. It specializes inside the training window and is brittle outside it. Overall IQM hides that split

## H3 — Wrong context kind

- Claim: global physics is the wrong signal; agent-local context would be what helps
- Status: **not supported / not pursued**
  - Pistonball: physics itself was uninformative, so a new context kind would not fix a flat task
  - Multiwalker: global physics is task-relevant (H1 rejected)
- Finding: the open failure was out-of-distribution use of the physics vector, not a missing local feature
- Not run: agent-kind ablation, FiLM, latent context, actor-only vs critic-only

## Follow-up — Feature extrapolation (clip probe)

- Claim: Multiwalker concat’s OOD loss is an out-of-range context vector, not an inability to control the new physics
- Test: eval only, same concat checkpoints, true OOD physics, vector magnitude clamped to [0.75, 1.5] (sign kept)
- Result: **mostly supported**
  - ID IQM unchanged: 61.3 → 62.1
  - OOD IQM: −15.5 → 42.1 (hidden OOD is 15.5)
  - Overall IQM: 11.9 → 49.7
  - Package mass drop 69 → 14; terrain friction 65 → 9 → those failures are mostly feature extrapolation
  - Gravity drop 87 → 54; a large penalty remains → dynamics and extrapolation both matter
- Finding: inside the training support the concatenated feature is useful. Outside it, the same feature can mislead the policy even when the new physics is controllable

## What “context helps” means here

- Helps in-distribution on a sensitive environment (Multiwalker concat)
- Does not help when slices are flat (Pistonball)
- Does not, by itself, give robust zero-shot transfer under hard factor extrapolation
- Domain randomization (varying training) was the large Pistonball effect, shared by hidden and concat
