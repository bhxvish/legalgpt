# LoRA adapters on held-out IPC sections

41 questions on 29 IPC sections never used in training (app/finetuning/ipc_qa.py EVAL_SECTIONS); only questions whose section the retriever returned.

| metric | v0.2-20261002-203807 | v0.3-ipcqa |
|---|---|---|
| answered (not "I don't know") | 39% | 100% |
| cites the asked section | 39% | 98% |
| all citations valid | 39% | 100% |
| mean answer length (chars) | 183 | 373 |
| mean seconds per answer | 1.33 | 2.59 |

## Sample answers

**What does Section 279 of the IPC provide?**
- *v0.2-20261002-203807*: I don't know — the retrieved sources do not cover this question.
- *v0.3-ipcqa*: Section 279 of the IPC (Rash driving or riding on a public way) provides: Whoever drives any vehicle, or rides, on any public way in a manner so rash or negligent as to endanger human life, or to be likely to cause hurt or injury to any other person, shall be punished with imprisonment of either des

**What is the punishment for murder?**
- *v0.2-20261002-203807*: Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine. [2]
- *v0.3-ipcqa*: Under Section 302 of the IPC [2], whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.

**What does Section 302 of the IPC provide?**
- *v0.2-20261002-203807*: Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine. [1]
- *v0.3-ipcqa*: Section 302 of the IPC (Punishment for murder) provides: Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine. [1]

**What is the punishment for culpable homicide not amounting to murder?**
- *v0.2-20261002-203807*: I don't know — the retrieved sources do not cover this question.
- *v0.3-ipcqa*: Under Section 304 of the IPC [1], whoever commits culpable homicide not amounting to murder, shall be punished with imprisonment for life, or imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine, if the act by which the death is caused is done

**What does Section 304 of the IPC provide?**
- *v0.2-20261002-203807*: Whoever commits culpable homicide not amounting to murder, shall be punished with imprisonment for life, or imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine, if the act by which the death is caused is done with the intention of causing dea
- *v0.3-ipcqa*: Section 304 of the IPC (Punishment for culpable homicide not amounting to murder) provides: Whoever commits culpable homicide not amounting to murder, shall be punished with imprisonment for life, or imprisonment of either description for a term which may extend to ten years, and shall also be liabl

**What does Section 304A of the IPC provide?**
- *v0.2-20261002-203807*: I don't know — the retrieved sources do not cover this question.
- *v0.3-ipcqa*: Section 304A of the IPC (Causing death by negligence) provides: Whoever causes the death of any person by doing any rash or negligent act not amounting to culpable homicide, shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both
