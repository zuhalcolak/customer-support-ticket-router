# Compliance checklist (SWE015 homework, Use Case 2)

Status as of 2026-11-22. "Done" means the item was executed and checked, not only written.

| Requirement | Status | Evidence |
|---|---|---|
| One of the five official use cases | Done | Use Case 2 – Customer Support Ticket Router |
| R1 Working interactive application, accepts new input | Done | `streamlit run app/main.py`; new typed messages classified in the browser during development |
| B: regex normalization + model-code extraction | Done | `src/preprocessing/text.py`; tests in `tests/test_preprocessing.py` |
| B: edit-distance correction of misspelled product names | Done | `levenshtein`, `correct_token`; `dishwaser → dishwasher` in app and tests |
| B: ELIZA-style replies for small talk | Done | `src/routing/smalltalk.py`; `tests/test_routing.py` |
| B: TF-IDF + logistic regression baseline | Done | `src/classical/model.py`, `scripts/train_classical.py` |
| A: feed-forward net on averaged FastText, SGD + backprop | Done | `src/neural/model.py`, `scripts/train_neural.py` (FastText trained on train split only) |
| C: BERT-family model fine-tuned for intent classification | Done | XLM-RoBERTa base, `scripts/train_transformer.py` |
| D: knowledge distillation into a smaller student | Done | `scripts/distill_student.py` (mDistilBERT, α=0.7, T=2) |
| D: quantization (mandatory) | Done | dynamic INT8, `scripts/quantize_model.py`; reload verified: 38 INT8 Linear, 2 INT8 Embedding, 0 float layers |
| R5 Application runs the quantized Transformer | Done | `PRIMARY_MODEL = "quantized"` in `src/pipeline.py`; `test_quantized_model_inference` |
| R3 Four models on the same held-out test set | Done | `scripts/evaluate_all.py`, 1,206 test messages |
| Metrics: accuracy, out-of-scope recall | Done | `reports/results.csv` |
| R4 Results table: metric, size (MB), latency (≥100 runs), hardware | Done | 10 warm-up + 200 timed runs, batch 1, CPU; `reports/results.json`, `reports/hardware.json` |
| R2 Four technique areas + technique map slide | Done | slide 5 of `docs/presentation.pdf` |
| R6 Architecture diagram that matches the code | Done | `docs/architecture.png` (each box names its source file) |
| R7 Repository: README with setup + single run command, requirements.txt, no large weights | Done (local) | fresh-copy test: new venv + `requirements.txt` + `artifacts.zip` → 35/35 tests passed, app health OK; 90 tracked files, 3 MB |
| R8 AI usage slide | Done | slide 11, `AI_USAGE.md` |
| Presentation 8–12 slides, Intro / Development / Conclusion, first slide with name, ID, use case, links | Done except video link | `docs/presentation.pdf`, 11 slides; video link is still `[INSERT DEMO VIDEO URL]` |
| At least one failure example | Done | slide 10, `reports/failure_examples.csv` |
| Automated tests executed | Done | `python -m pytest -q` → 35 passed |
| Dataset sources and licenses documented | Done | `DATASET_REPORT.md` |
| Two-minute demo script | Done | `docs/DEMO_SCRIPT.md` |

## Still to do by the student (needs personal accounts)
1. Push the repository to https://github.com/zuhalcolak/customer-support-ticket-router (public, or shared with the instructor).
2. Upload `artifacts.zip` (≈1.5 GB) to Google Drive ("anyone with the link") or Hugging Face Hub, and put the link in README.md ("Model artifacts").
3. Record the ≤2:00 narrated demo video, upload it (YouTube unlisted or Drive), and rebuild the slides with the link:
   `python scripts/make_slides.py --video <URL>`
4. Open both links in a private browser window to check that they work.
