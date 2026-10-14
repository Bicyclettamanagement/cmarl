# 1. Introduction
Multi-agent reinforcement learning (MARL) provides a
framework for studying sequential decision-making in systems where multiple autonomous agents interact within a
shared environment. A central challenge in Reinforcement
Learning (RL) and MARL is that learned policies often fail
to generalize beyond the specific environment configurations encountered during training. Variations in dynamics,
task parameters or reward structures can lead to brittle or
unstable behaviour. To address this limitation, contextual
information (additional meta-data describing properties of
the environment or task) has emerged as a promising mechanism for improving adaptability and robustness.
| REVIEW: *Keep the opening (shared environment, brittle policies), and cut it to about three sentences for this supervisor. Rewrite this sentence. "Promising mechanism for improving adaptability and robustness" states the benefit before the test. Context is the intervention. The result is that it helps only inside the training window on a sensitive environment.* |
| --- |


While contextual reinforcement learning has shown benefits
in single-agent settings, its role in multi-agent systems is
considerably less understood. Moreover, there is little systematic understanding of how context should be integrated
into MARL architectures effectively.
| REVIEW: *Keep this gap: single-agent contextual RL exists; the multi-agent case is open. Drop "effectively" — it has no operational meaning here. The comparison is concatenation versus a hidden baseline, scored by IQM, split into in-distribution and out-of-distribution. "Systematically" and "considerably less understood" are intensifiers; one plain sentence is enough.* |
| --- |


Motivated by these challenges, this project investigates the
role of context in MARL with a focus on representation and
incorporation strategies, generalization and transferability.
| REVIEW: *Rewrite this sentence. "Representation and incorporation strategies" (plural) and "transferability" describe a broader project than the one that ran. State the actual scope: one incorporation, concatenation of a global physics vector, against a hidden-context baseline. Representation learning and transfer across environments were not done. Add the shape of the answer: the contribution is the test, including null and harmful results, and the conditions under which the comparison is valid. Close with a one-sentence roadmap: Pistonball, then Multiwalker, then the clip. Leave H1–H3 out of the introduction; they are explanations checked after the first null, not the questions you started with.* |
| --- |

