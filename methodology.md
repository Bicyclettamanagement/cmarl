# 4. Methodology
The methodology is structured in two stages. First, a prestudy systematically evaluates different strategies for incorporating contextual information into a multi-agent reinforcement learning framework. This stage focuses on isolating the effects of context integration mechanisms under
controlled conditions. The pre-study may be iteratively extended to additional environments before selecting the final
integration strategy for transfer evaluation. Second, a transfer study investigates the generalization capabilities of the
most effective context-integration strategy across multiple
environments and varying types of contextual information.
For evaluation the results will be compared to a hiddencontext baseline (see subsection ”Evaluation” below).
| REVIEW: *Rewrite this opening. The two-stage plan (compare several integration strategies, pick a winner, transfer suite on further environments and further kinds of context) was not run. A null comparison did not produce a winner to transfer. Keep one short paragraph that states this original plan, then state the stopping rule: H1 supported means switch environment; H1 and H2 both rejected means the next test is the clip, not a transfer suite. Delete "different strategies", "may be iteratively extended", and "if time permits". The second environment was required because H1 was supported on Pistonball, not added as an optional replication. Write what ran in the past tense.* |
| --- |


The experiments will be based on TD3, chosen for its stability and continuous-action suitability.
| REVIEW: *Rewrite. The algorithm is MATD3, with a shared actor and a one-hot agent id, not single-agent TD3. "Chosen for its stability" is vague and is not a result you show. Cite Ackermann et al. and stop; no derivation.* |
| --- |


To address Research Question 1, the primary integration
strategy investigated in this work is direct concatenation
of contextual features to the agent observation. Depending on the outcome of the pre-study, additional integration
mechanisms may be explored. Each method will be treated
as a configurable module that can be attached to the TD3
architecture.
| REVIEW: *Keep the sentence that concatenation is the integration under test. Delete "additional integration mechanisms may be explored" and "configurable module that can be attached". Those mechanisms were not built. One scope sentence is enough: H3 (latent context, FiLM, actor-only versus critic-only, agent-local context) stays open because the open failure was support of the feature already in use.* |
| --- |



## 4.1. Pre-Study Design
The pre-study is conducted on a single benchmark environment with one fixed type of contextual variable in order to
tightly control experimental factors and limit computational
cost. Its objective is to validate direct context concatenation
as a baseline integration strategy and to compare it against a
hidden-context baseline under identical training conditions.
Methods are evaluated using cumulative return, convergence
behaviour, zero-shot generalization within the same environment, and stability across seeds. Based on these metrics, the
most robust configuration is selected for subsequent transfer
experiments. If time permits, the pre-study may be extended
to one additional environment to increase statistical confidence without substantially expanding the experimental
scope.
| REVIEW: *Keep the comparison itself: same training conditions, concatenation against hidden, return, and seed-to-seed variation. Delete "validate" and "the most robust configuration is selected for subsequent transfer experiments." The pre-study tests the comparison. It does not certify concatenation or choose a configuration. Delete the "one additional environment to increase statistical confidence" sentence. Multiwalker repeated the protocol so a sensitive axis could reject H1. Also name, in this section or the next, what is missing from this draft: Pistonball and Multiwalker, static versus varying, the training window [0.75, 1.5], the hard grid (0.25×–4×, one axis at a time), five seeds, and IQM.* |
| --- |



## 4.2. Transfer Study Design
The transfer study evaluates the selected context-integration
strategy from the pre-study under zero-shot conditions in a
small suite of additional environments with different types
of contextual variation. Training is performed on a restricted
subset of context configurations, while evaluation is carried
out exclusively on previously unseen contexts to assess
zero-shot generalization. Performance is compared against
a hidden-context baseline to quantify the benefit of explicit
context access. The number of environments and context dimensions is deliberately limited to ensure feasibility within
the available computational and time budget.
| REVIEW: *Delete this section as an executed stage. Replace it with two sentences: the transfer suite was the planned next step; the clip probe replaced it because the open failure was extrapolation of the feature you already had. Also fix the design claim that evaluation is "exclusively on previously unseen contexts." Every run is scored on in-distribution and out-of-distribution factors. That split is the result. The N=10 curriculum is not part of this stage; it is an appendix, because the base skill was not learned.* |
| --- |



## 4.3. Evaluation
Evaluation focuses on three main aspects: performance,
zero-shot generalization, and sample efficiency. Performance is measured by cumulative return, reported as mean
and variance across multiple random seeds. Training dynamics and sample efficiency are analyzed using learning curves
(reward versus environment steps) and the median number
of steps required to reach a predefined performance threshold. Zero-shot generalization is evaluated by the achieved
return on held-out, unseen context configurations and by the
corresponding generalization gap between in-distribution
and out-of-distribution performance. For multi-agent analysis, both team-level and per-agent rewards are reported,
along with reward variance across agents as an indicator of
coordination quality. Training stability is further assessed
using convergence variability across seeds and optimization
diagnostics such as gradient norms. All results are reported
with appropriate confidence intervals and statistical comparisons where applicable.
| REVIEW: *This metric list is reusable after these changes. Keep return, learning curves, steps to a threshold, the in-distribution versus out-of-distribution gap, and interval estimates. Replace "mean and variance" with IQM and bootstrap intervals (Agarwal et al., rliable); mean is not the headline aggregate. Delete gradient norms; they are not in the results. Demote per-agent reward variance to one sentence that it was recorded and did not carry a claim. Replace "where applicable" with the comparisons you report, including the probability that one method beats the other.* |
| --- |


In addition to explicit context concatenation, a hiddencontext baseline will be evaluated, where environment variations are not provided to the agents and must be inferred
implicitly from state transitions. In contextual reinforcement learning, the hidden-context baseline means that during training and evaluation the agent does not get access
to the context variables (like robot mass, friction, actuator
strength). The environment dynamics still depend on these
hidden parameters, but the agent must infer them indirectly
from experience. This baseline is used to measure how
much explicit context information helps compared to learning without it. If contextual methods outperform the hidden
baseline, it demonstrates the value of explicitly modeling context.
| REVIEW: *Keep the hidden-baseline definition (dynamics vary, the vector is withheld, the agent must infer the factor from experience). Replace "robot mass, friction, actuator strength" with the factors you used: Pistonball ball mass, friction, elasticity, and gravity; Multiwalker gravity, package mass, and terrain friction. Rewrite the last sentence. Outperforming the hidden baseline is not the definition of success. On Multiwalker, concatenation does that in distribution and fails out of distribution. The claim is the three conditions: the factor changes behavior, the hidden baseline can learn the task, and the feature stays inside the training range.* |
| --- |



## 4.4. Optional
• Implementation details
• Ablations
• expected outcomes
| REVIEW: *Delete this placeholder section. Implementation details are now the method: normalization (feature = value / reference, so gravity is a negative factor), the factor window, and the hard grid. Expected outcomes are the hypotheses in the discussion, not a leftover bullet.* |
| --- |

