# Attention Hypothesis

## Scientific motivation

Selective attention prioritizes behaviorally relevant information, while expectation concerns what is likely to occur. Working memory is also a limited resource. This motivates a computational mechanism that allocates a bounded processing budget rather than treating every input as equally important. citeturn492487search3turn492487search0turn492487search15

## Computational hypothesis

For each candidate item i:

`r_i = wg*G_i + ws*S_i + wn*N_i + wu*U_i + wc*C_i`

G = goal relevance, S = salience, N = novelty, U = urgency, C = uncertainty.
A softmax converts relative relevance into an allocation, then a fixed budget keeps the attended set bounded.

This is an engineering hypothesis, not a literal reconstruction of a biological attention circuit.

## Validation

Unit tests check that goal-relevant information is prioritized, the budget is bounded, allocations normalize, and invalid parameterizations are rejected.

Future evaluations should measure distractor resistance, urgent-event detection, novelty capture, and goal switching.

## Prediction connection

Prediction and attention stay separate: expected probability is not the same thing as behavioral relevance. Prediction-error signals can later provide evidence for surprise or corrective processing. citeturn492487search3turn492487search1
