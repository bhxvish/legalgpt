# What LoRA fine-tuning changed (explainability of the local models)

Method: Section 7 of the LoRA paper (Hu et al., ICLR 2022) on our adapters' weight updates, plus behavioural tests on held-out questions. Reproduce: `backend/scripts/analyze_lora.py`.

## Weights: v0.7-3b-l4 (base Qwen/Qwen2.5-3B-Instruct, r=16, alpha=32, 252 matrices)

| module | mean ||dW||/||W|| | mean amplification |
|---|---|---|
| q_proj | 0.00730 | 0.84 |
| k_proj | 0.00696 | 0.31 |
| v_proj | 0.00763 | 0.34 |
| o_proj | 0.00803 | 0.96 |
| gate_proj | 0.01342 | 2.62 |
| up_proj | 0.01291 | 3.04 |
| down_proj | 0.00461 | 1.1 |

- Layers changed most: 2, 3, 1, 4, 5.
- Effective rank of dW (of r=16): mean 13.04; directions holding 90% of the update: mean 8.72.
- Amplification factor ||dW||/||U^T W V||: mean 1.32.
- W projected on dW's directions vs random directions: 1.13x; vs W's own top directions: 0.0278.

## Weights: v0.6-scenario-refusals (base Qwen/Qwen2.5-1.5B-Instruct, r=16, alpha=32, 196 matrices)

| module | mean ||dW||/||W|| | mean amplification |
|---|---|---|
| q_proj | 0.00825 | 0.74 |
| k_proj | 0.00842 | 0.33 |
| v_proj | 0.00874 | 0.34 |
| o_proj | 0.00935 | 0.86 |
| gate_proj | 0.00988 | 2.2 |
| up_proj | 0.00921 | 1.98 |
| down_proj | 0.00494 | 1.1 |

- Layers changed most: 20, 19, 17, 25, 21.
- Effective rank of dW (of r=16): mean 13.2; directions holding 90% of the update: mean 8.96.
- Amplification factor ||dW||/||U^T W V||: mean 1.08.
- W projected on dW's directions vs random directions: 1.07x; vs W's own top directions: 0.0308.

## Subspace similarity: v0.5-scenario vs v0.6-scenario-refusals

| module | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| q_proj | 0.0047 | 0.0046 | 0.0049 | 0.0071 | 0.0116 |
| k_proj | 0.0051 | 0.0057 | 0.0058 | 0.0074 | 0.0118 |
| v_proj | 0.0069 | 0.0052 | 0.0054 | 0.0073 | 0.0121 |
| o_proj | 0.0119 | 0.0081 | 0.0067 | 0.008 | 0.0124 |
| gate_proj | 0.0555 | 0.0297 | 0.0175 | 0.0134 | 0.0148 |
| up_proj | 0.0163 | 0.0102 | 0.0083 | 0.0086 | 0.0125 |
| down_proj | 0.109 | 0.068 | 0.0401 | 0.023 | 0.0136 |
| random | 0.00056 | 0.0012 | 0.0024 | 0.0047 | 0.00927 |

## Subspace similarity: v0.4-plain vs v0.5-scenario

| module | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| q_proj | 0.0022 | 0.0028 | 0.004 | 0.0062 | 0.0111 |
| k_proj | 0.0036 | 0.0038 | 0.0042 | 0.006 | 0.011 |
| v_proj | 0.006 | 0.005 | 0.0051 | 0.0067 | 0.0116 |
| o_proj | 0.0038 | 0.0043 | 0.0047 | 0.0069 | 0.0117 |
| gate_proj | 0.0601 | 0.033 | 0.0193 | 0.0146 | 0.0155 |
| up_proj | 0.0082 | 0.006 | 0.0057 | 0.0072 | 0.0117 |
| down_proj | 0.0416 | 0.0292 | 0.0183 | 0.0115 | 0.0077 |
| random | 0.0007 | 0.00117 | 0.00236 | 0.00474 | 0.00918 |

## Behaviour: v0.7-3b-l4 on 16 held-out scenarios

| tokens | mean log-prob gain (tuned - base) | tokens |
|---|---|---|
| citation marker | +1.267 | 104 |
| section number | +0.232 | 153 |
| punishment terms | +9.490 | 7 |
| all tokens | +1.738 | 2168 |

Most raised tokens: `punishment` (+14.02), `…` (+12.46), `fit` (+12.12), `section` (+11.23), `See` (+10.52), `simple` (+8.30), `Whether` (+8.24), `P` (+7.49), `shown` (+7.27), `up` (+7.23), `causing` (+6.87), `can` (+6.81).

Inside-the-model source attribution (leave one source out, drop in the answer's mean log-probability): the most influential source is one the answer cites in 16 of 16 answers with citations; mean drop 0.5427 for cited sources vs 0.0007 for uncited ones.
