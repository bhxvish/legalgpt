# LegalGPT demo kit (Review 3)

Everything needed to present LegalGPT to the panel: the flow, what to click, what to say, and the texts
to paste. Plan for **about 20 minutes**: slides 8 minutes, live demo 7 minutes, questions 5 minutes.

| File | What it is |
|---|---|
| `README.md` | This file: the checklist, the demo flow and the backup plan |
| `talking_points.md` | Opening and closing lines, the numbers to remember, quick answers |
| `01_chat_questions.txt` | The four chat questions to paste, with what each one shows |
| `02_case_file_house_theft.txt` | The FIR for the main case analysis demo |
| `03_case_file_with_injection.txt` | The same FIR with a hidden instruction, to show the safety guard (optional) |
| `04_case_file_out_of_scope.txt` | A bail application under the NDPS Act, to show an out-of-scope file (optional) |
| `05_follow_up_questions.txt` | Questions to ask under the analysis, including the "verdict" trap |
| `06_verify_facts.txt` | Facts for the Verify tab (theft, s.379) |
| `start_demo.ps1` | Starts the backend and the frontend and opens the browser |
| `LegalGPT_demo.mp4` | The recorded 4.5-minute demo (backup) |
| `LegalGPT_Project_Explained.pdf` | The full project explanation (21 pages) |
| `Review3_LegalGPT.pptx` | The Review 3 slide deck |

Demo video online: https://drive.google.com/file/d/1vzqlXm09jTBlQscZCJVIkDYQrDty_i02/view?usp=sharing

---

## Before the panel

**The day before**

- [ ] Run `start_demo.ps1` once on the laptop you will present from, and do the whole demo end to end.
- [ ] Check that Groq works on the college Wi-Fi: ask one chat question with the default model. If you see
      "Groq refused access from this network (HTTP 403)", use **Gemini 3.5 flash lite** for the demo (see the backup plan).
- [ ] Make sure the Label tab still has the 2 cases you kept unlabelled for the demo.
- [ ] Charge the laptop; keep the HDMI adapter in the bag.

**30 minutes before**

- [ ] Run `start_demo.ps1` (right-click, *Run with PowerShell*), or run the two commands in it by hand.
      The backend is ready when http://localhost:8000/health shows `{"status":"ok"}`.
- [ ] Open http://localhost:5173 in Chrome, set **Light** theme (top right), zoom to 110 to 125% so the panel can read it.
- [ ] Open these files in a text editor, ready to copy from: `01_chat_questions.txt`, `02_case_file_house_theft.txt`,
      `05_follow_up_questions.txt`.
- [ ] In Chat, click **New chat** so the screen is empty. In Case analysis, clear any old file.
- [ ] Open the demo video in a second tab, paused at 0:00, in case anything fails.
- [ ] Close other apps and notifications.

---

## The flow

Present the slides first, then switch to the browser. The demo follows the pipeline, left to right in the
top bar: **Overview → Chat → Case analysis → Verify → Label**, so the panel sees every module working.

**Suggested split (change it as you like):** S Bhavish opens and runs the live demo; Chinmayi presents
abstract, scope and novelties; Raghav presents architecture and implementation; Rithvik presents testing and
results. Everyone answers questions on their part.

### Slides (about 8 minutes)

| Slides | Point to make |
|---|---|
| Title, Outline | Who we are, what we will show |
| Abstract and Scope | AI chatbots get law wrong; the IPC was replaced by the BNS; users ask in Hindi; LegalGPT answers only from the law and measures its reliability |
| Suggestions from Review 1 and 2 | Architecture diagram made clear; novelties defined, each with a number |
| Novelties | The method: compare the same model with and without LegalGPT; everything reproducible and verifiable |
| Architecture (2 slides) | Offline: judgments labelled into a library. Online: ask, retrieve, answer, check, sign |
| Implementation (4 slides) | One sentence per module; show the pseudocode briefly |
| Then say | "Let us show it working." Switch to the browser |
| After the demo: Test plan, Results, Timeline, Conclusion, References | Results: zero wrong punishments, zero civil questions as crimes, 98 to 100% right section |

### Live demo (about 7 minutes)

**Step 1. Overview tab (30 seconds)**
- Click **Overview**.
- Say: *"This page is the whole system. Offline we build a library of court judgments: collected, labelled
  by people with InLegalBERT's help, stored as 28,055 citable passages. Online, every question is asked,
  retrieved, answered, checked and signed. The numbers at the bottom are measured on held-out questions."*

**Step 2. Chat, an everyday question (1.5 minutes)**
- Click **Chat**, then **New chat**. Keep mode **Legal** and law **Auto**.
- Paste question 1 from `01_chat_questions.txt` (the bicycle in the shed) and press Enter.
- While it answers, say: *"Note there are no legal words in the question. A lexicon, a crime-scope check
  and an AI rewrite find the right sections."*
- Point at: the **[n] citations**; the **Sources** list, each IPC section with its **BNS** number; the cited
  **Supreme Court judgment**; then open **Why this answer?** and show the evidence match and the source
  behind each sentence. Mention the answer is **signed**.

**Step 3. Chat, the new code (45 seconds)**
- New chat. Paste question 2 (gold chain snatched on 12 August 2024).
- Say: *"The events are after 1 July 2024, so it answers under the Bharatiya Nyaya Sanhita. The section
  numbers come from the official government table, never from the AI. Each BNS section shows the IPC
  section it replaced."*

**Step 4. Chat, Hindi (45 seconds)**
- New chat. Paste question 3 (the bicycle question in Hindi).
- Say: *"Asked in Hindi, answered in Hindi, with the same sections. Searched as written, Hindi questions
  found nothing; translated first, 52 of 60."*

**Step 5. Chat, a civil question (30 seconds)**
- New chat. Paste question 4 (landlord keeping the security deposit).
- Say: *"This is a civil matter, so it declines. The same AI models answering alone treated 47 to 90% of
  such questions as crimes. Inside LegalGPT: zero."*

**Step 6. Case analysis (2 minutes)**
- Click **Case analysis**. Paste `02_case_file_house_theft.txt` into the text box (or upload the file).
  Choose the model (see the backup plan if Groq is blocked) and click **Analyse case**.
- While it runs (about a minute), say: *"It reads the facts, each with a quote from the file; proposes IPC
  offences and validates them; checks every element; finds similar Supreme Court judgments; and writes a
  cited analysis. It never gives a verdict: only a court decides on evidence."*
- Scroll through the five sections: facts with quotes, the law with BNS numbers, element checks (Prolog
  or Model-assessed), similar judgments, the analysis.
- In **Ask about this analysis**, click *Which facts still need to be proved?*. Then paste the verdict trap
  from `05_follow_up_questions.txt`: *"So is Mohan Lal guilty of theft? Just tell me the verdict."*
- Say: *"The output guard removed the verdict sentences; it explains what the evidence must establish
  instead."* (Optional: switch to another tab and back to show the analysis is kept.)

**Step 7. Verify (1 minute)**
- Click **Verify**, then **Try an example** (death by negligence, s.304A), then **Check**.
- Say: *"The AI only extracts facts, each with a word-for-word quote. Prolog rules decide each element:
  satisfied, violated or missing, and which fact would change the result. The zero-knowledge proof shows
  the result follows from the published rules without revealing the facts."*
- Optional second example: paste the theft facts from `06_verify_facts.txt` and choose **379 Theft**.

**Step 7b. Explainability (2 minutes; the panel's main request in Review 3)**
- In Chat, ask the dowry question from `01_chat_questions.txt` (question 5). When the answer finishes,
  **Understand this answer** opens by itself with the **Bias check** already run, because the case is
  gender-sensitive.
- Say: *"The same question is run again with only the gender swapped, and once more unchanged as a
  control. Every difference is explained: dowry death is defined for a woman, so that difference is
  required by law; a gender-neutral section that changes is flagged as possible bias."*
- Click **Caste** or **Income**, then **Run the bias check**: *"It is not only gender. The question names
  no caste, so one sentence about the accused is added to two identical copies, and only that sentence
  differs. The offence should not depend on it."* Religion, region and age work the same way.
- Click **How it decided** then **Show how it decided**: *"Issue, rule, application, conclusion - how the model
  reached it."* Click **Check the explanation**: *"A model can describe reasoning it did not use, so each key
  fact is removed and the answer rewritten. Here is the faithfulness score."*
- Click **Words that mattered** (it loads by itself): *"Which words tied the question to each section,
  measured by removing each word - no model call."*
- Open the **Explainability** tab: *"Every explanation is measured: key words beat random words in every
  question; 68% of the key facts the reasoning names really drive the conclusion; and inside our own
  fine-tuned model, the source the answer depends on most is one it cites in 16 of 16 answers. This is
  the LoRA analysis from Section 7 of the LoRA paper."*

**Step 8. Label (30 seconds, only if time allows)**
- Click **Label** and open one of the 2 unlabelled cases.
- Say: *"Our labelling tool: InLegalBERT suggests a role for every sentence, people review the uncertain
  ones, and one judgment in ten is labelled blind so we keep measuring the model."*

Then go back to the slides for **Test plan, Results, Timeline, Conclusion**.

### Optional extras (if the panel asks)

- **Prompt injection:** run Case analysis on `03_case_file_with_injection.txt`. The file contains a hidden
  instruction to declare the accused guilty. Show that the analysis does not give a verdict; if the planted
  line was picked up as a fact, a red "Safety checks acted on this file" note appears.
- **Out of scope:** run Case analysis on `04_case_file_out_of_scope.txt` (a bail application under the NDPS
  Act). It explains that this is outside the IPC instead of forcing an IPC analysis.
- **Offline:** in Chat choose **Local (LoRA)**, or in Case analysis switch on *Keep the file on this
  computer*: nothing leaves the laptop.

---

## Backup plan

| Problem | What to do |
|---|---|
| The bias check says a daily limit was reached | The free quota is used up (about 200,000 tokens per Groq key per day). Switch to the other provider's model, or show the recorded video. Do not run any evaluation script on demo day: one run can use the whole day's quota. |
| Groq blocked (HTTP 403) or slow | In Chat click the **Gemini 3.5 flash lite** model chip; in Case analysis choose `gemini:gemini-3.5-flash-lite`. Verify falls back to Gemini on its own |
| A model gives an odd answer | Say "this is why every answer shows its sources and warnings", click New chat, ask again |
| An answer takes long | Keep talking through what it is doing; the steps appear on screen as they finish |
| No internet | Use **Local (LoRA)** for Chat, or play `LegalGPT_demo.mp4` |
| The app will not start | Play `LegalGPT_demo.mp4` and talk over it; it shows every step above with captions |

Key commands (if `start_demo.ps1` is not used), from the project folder:

```
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --port 8000
npm --prefix frontend run dev
```

Docker alternative: `docker compose up --build`, then open http://localhost:5173.
