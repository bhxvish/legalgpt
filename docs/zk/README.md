# Verifiable LegalGPT: zero-knowledge proofs and signed answers

These files let anyone check LegalGPT's proofs without trusting the server. They were produced by
`backend/scripts/zk_setup.py` (code in `backend/app/zk/`).

| File | What it is |
|---|---|
| `manifest.json` | The commitments. Merkle root of the 685 IPC chunks, hash of their embeddings, hashes of the Prolog rules and the LoRA adapter, circuit parameters, the retrieval rules, the server's signing key, and measured proof times |
| `verdict/settings.json`, `verdict/vk.key` | Verification key of the **fact-check circuit** |
| `verdict/layout.json` | The fact encoding and the element matrix taken from the Prolog rules: what each output of the circuit means |
| `retrieval/settings.json`, `retrieval/vk.key` | Verification key of the **retrieval circuit** |

The proving keys (about 1.3 GB) and the server's private signing key stay in `data/zk/` and are never published.

## What is proved

**1. Fact-check verdicts (zero knowledge).** The extracted facts are encoded as 23 small integers
and kept private; the proof publishes only their Poseidon hash. The circuit multiplies them by a
fixed matrix built from `rules/*.pl`. This gives one status per element of all 7 sections:
satisfied, violated or missing. The circuit also outputs a validity check that is 0 only for a
well-formed encoding, so a dishonest prover can't use impossible facts, such as two act types at
once. A valid proof shows that **this verdict is what the published rules give for the committed
facts**. If the facts are later disclosed, anyone can check they hash to the commitment. Proving
takes about 1 s; verifying about 0.03 s; the proof is 33 KB.

**2. Retrieval (verifiable computation; the question is public).** The circuit computes the
similarity of the question's embedding to **all 685 committed chunk embeddings**. Those embeddings
are constants inside the circuit, so the verification key commits to them. The verifier re-runs the
retriever's own rules on the proven scores and checks that the cited sources are exactly what they
select: top 8, explicit section matches, role weights, the 0.55 floor and the 6-source and
6,000-character budget. This shows that no relevant section was dropped and none was slipped in.
Proving takes about 8 s, verifying 0.1 s, and the proof is 196 KB. Scores are fixed point (2^-12),
accurate to about 0.0005.

**3. Answers (signed, not proved).** Every legal answer carries an Ed25519-signed record. The
record holds hashes of the question, the exact prompt and the answer, the cited chunks with their
Merkle leaves, the model (plus the adapter hash for the tuned model) and the corpus root. It
proves the answer came from this server with these inputs and was not changed afterwards.

## What is NOT proved

- **That the facts were extracted correctly.** Extraction is done by an LLM; the proof starts
  from its output. The two misses in the fact-check test set came from extraction, not the rules.
- **That the answer is legally right**, or that the LLM computed it: a 1.5B–120B model is far
  beyond what can be proved today. Answers are signed instead.
- **The query embedding** (MiniLM). It is a public input; the verifier recomputes it and compares.
- **Follow-up questions**, whose retrieval also searches with the previous question.
- **The trusted setup.** KZG uses EZKL's public reference string; the keys were made by
  `zk_setup.py`, and anyone can rebuild them from the corpus, rules and EZKL 23.0.5.

## Check a proof yourself

In the app, the **Verify** tab has *Generate proof → Verify proof → Download proof*. Each chat
answer has a **Signed** panel with *Check signature*, *Prove the sources* and *Download record*.
Offline, from a clone of the repo:

```bash
pip install -r backend/requirements.txt -r backend/requirements-zk.txt
python backend/scripts/zk_verify.py verdict-proof-s304A.json
python backend/scripts/zk_verify.py verdict-proof-s304A.json --facts facts.json
python backend/scripts/zk_verify.py retrieval-proof.json
python backend/scripts/zk_verify.py signed-answer.json --question "..." --answer-file answer.txt
```

The verifier reads only this folder and `data/corpus/ipc_chunks.jsonl`. It checks the proof
against the published key and decodes the result from the values the proof binds, never from
fields the prover could edit. It also re-runs the retrieval rules and checks the corpus hashes to
the published root.

## API

| Endpoint | |
|---|---|
| `GET /api/zk/manifest` | Commitments, plus what this server can prove and verify |
| `GET /api/zk/inclusion/{chunk_id}` | Merkle inclusion proof of a chunk |
| `POST /api/zk/prove/verdict` `{facts, section}` | Proof bundle for a verdict |
| `POST /api/zk/prove/retrieval` `{question}` | Proof bundle for retrieval |
| `POST /api/zk/verify` `{bundle, facts?, question?, answer?}` | Checks a proof bundle or a signed answer |
| `POST /api/verify?prove=true` | The fact check, with a proof of its verdict attached |

Chat responses carry the signed record as an `attestation` server-sent event.

## Rebuilding

Run `python backend/scripts/zk_setup.py` after changing the corpus, the rules or the retrieval
settings. A test fails if the rules no longer match the manifest. Building takes about a minute and
needs SWI-Prolog, EZKL and the search index.
