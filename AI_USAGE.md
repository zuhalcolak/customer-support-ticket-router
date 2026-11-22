# AI Usage Statement

AI coding assistants were used in this homework, as allowed by Section 6 of the assignment.

## Tools
* **Claude (Anthropic) via Claude Code** – an agentic coding assistant working in my local
  project folder.

## What the AI assistant was used for
| Task | How |
|---|---|
| Code generation | Drafted the Python modules (`src/`, `scripts/`, `app/`), the tests and the plotting code. |
| Debugging | Found and fixed problems while running the pipeline, e.g. very slow MPS training caused by dynamic padding shapes (fixed by padding to a fixed length), and a spelling-correction bug that turned "kahvaltı" into "kahvealtı" (fixed with a known-word list and a suffix whitelist). |
| Data authoring | Drafted the 100 hand-written appliance-support example sentences in `data/manual/appliance_examples.csv`, which I reviewed. They are labelled as authored examples and are not presented as real customer data. |
| Documentation | Drafted README.md, DATASET_REPORT.md, this file and the demo-video script. |
| Presentation | Drafted the slide content and generated the PDF from the measured result files. |

## What the AI did *not* do
* **No result was taken from an AI answer.** Every number in the README, the reports and the
  slides was produced by running the scripts in this repository on my machine (Apple M3,
  16 GB RAM). The raw outputs are in `reports/` (`results.csv`, `results.json`,
  `training_logs/*.json`).
* The AI assistant ran the training and evaluation commands on my computer through the coding
  tool; the models were really trained and measured. No metric was estimated or copied from a
  paper or model card.
* The demo video is recorded and narrated by me.

## Responsibility
I reviewed the code and I am responsible for every line in this repository. I can explain each
component in the architecture diagram (`docs/architecture.png`).
