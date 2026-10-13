# EmergencyNLP — Real-Time Emergency Message Intelligence System

An NLP-based system that reads a free-text emergency message and automatically
identifies the **emergency type**, **priority level**, **location**, **people
affected**, and **required emergency service** — with a confidence score and
extracted keywords. Built with Python, spaCy, and scikit-learn. No paid APIs;
everything runs locally.

```
Input message
   -> Text cleaning -> Lowercase -> Tokenization -> Stop-word removal -> Lemmatization
   -> TF-IDF features -> Logistic Regression (category)
   -> spaCy NER (location / org / person / date / time)
   -> Rule-based priority scoring
   -> Rule-based required-service detection
   -> Keyword extraction
   -> Emergency response result (+ saved to SQLite)
```

---

## 1. Project structure

```
EmergencyNLP/
├── app.py                 Streamlit dashboard (the UI)
├── train_model.py         Trains & evaluates the ML classifier
├── preprocessing.py       Text cleaning / tokenizing / lemmatizing pipeline
├── nlp_utils.py           NER, location, people, service, priority, keywords, response
├── database.py            SQLite storage (auto-creates the database)
├── test_samples.py        Standalone test script with sample messages
├── emergency_dataset.csv  Training data (message, category, priority)
├── model.pkl              Trained Logistic Regression model (created by training)
├── vectorizer.pkl         Fitted TF-IDF vectorizer (created by training)
├── model_metrics.json     Saved accuracy / precision / recall / F1 / confusion matrix
├── emergency.db           SQLite database (created automatically)
├── requirements.txt
├── README.md
└── assets/
    └── logo.png            Generated automatically the first time the app runs
```

You do **not** need to create `model.pkl`, `vectorizer.pkl`, `emergency.db` or
`assets/logo.png` yourself — the app creates all of them automatically the
first time it runs.

---

## 2. Installation (Windows + VS Code)

### Step 1 — Install Python
Install Python 3.10 – 3.13 from [python.org](https://www.python.org/downloads/)
and tick **"Add Python to PATH"** during setup.

### Step 2 — Open the project in VS Code
Open VS Code → **File → Open Folder…** → select the `EmergencyNLP` folder.
Install the **Python** extension if VS Code prompts you to.

### Step 3 — Open a terminal in VS Code
`Terminal → New Terminal` (it opens inside the `EmergencyNLP` folder).

### Step 4 — Create and activate a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate
```
Your terminal prompt should now start with `(venv)`.

### Step 5 — Install the required packages
```bash
pip install -r requirements.txt
```

### Step 6 — Download the spaCy English model
```bash
python -m spacy download en_core_web_sm
```
If this fails because of a network/proxy restriction, download the wheel
manually and install it:
```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```
**The app still works without spaCy** — a simple rule-based lemmatizer and
regex fallback take over automatically, only with slightly less accurate
location/entity detection.

---

## 3. Running the project

### Option A — Run the dashboard (recommended)
```bash
streamlit run app.py
```
This opens `http://localhost:8501` in your browser. On the very first run the
app trains the model automatically (a few seconds) because `model.pkl` does
not exist yet — you do not need to run anything else first.

### Option B — Train the model manually (optional)
```bash
python train_model.py
```
Prints accuracy, precision, recall, F1-score, 5-fold cross-validation and the
confusion matrix in the terminal, and saves `model.pkl`, `vectorizer.pkl`, and
`model_metrics.json`.

### Option C — Run the automated test script
```bash
python test_samples.py
```
Analyzes a set of hand-written sample messages (not in the training data) and
prints PASS/FAIL for each expected field, plus invalid-input handling.

## 3b. Troubleshooting

| Problem | Fix |
|---|---|
| Wall of `missing ScriptRunContext` warnings | You ran `python app.py`. Use **`streamlit run app.py`** instead. |
| Accuracy is very low (for example 22%) or the classes look different (`Medical` instead of `Medical Emergency`) | You are running an old `train_model.py` / dataset from a different version. Extract this zip into a **new empty folder**, or replace *all* files, then delete old `model.pkl`, `vectorizer.pkl`, `model_metrics.json` and `emergency.db`. `emergency_dataset.csv` should have about 1,490 rows. |
| `pip install` fails building numpy / scikit-learn / spacy | Make sure you use the `requirements.txt` from this zip (it uses flexible versions). If spaCy alone fails, delete the `spacy` line and install the rest; the app still works with its built-in fallbacks. |
| "Saved model was made with another scikit-learn version" | Nothing to do: the app retrains automatically in a few seconds. |
| Database warning on the Analyzer page | Delete an old `emergency.db`; it is recreated automatically with the correct table. |

---

## 4. Using the dashboard

| Page | What it shows |
|---|---|
| **Home** | Project overview and how the pipeline works |
| **Emergency Analyzer** | Type or pick a sample message → **Analyze** → full response ticket |
| **Analytics** | Category/priority distribution, priority-per-category breakdown, messages per day |
| **Model Performance** | Accuracy, precision, recall, F1, confusion matrix, retrain button |
| **History** | Every analysed message stored in SQLite, with filters and CSV export |

### Sample test messages
```
There is a major accident near the railway station. Two people are seriously
injured. Please send an ambulance immediately.

There is a fire accident at Madurai bus stand. Three people are injured.
Please send an ambulance and fire rescue team immediately.

Flood water has entered our street in Velachery. Four people are stranded
on the first floor. Please send a rescue team.

My father has severe chest pain and is not breathing properly. Please send
an ambulance to Anna Nagar immediately.

Strong earthquake at Bhuj. A building has collapsed and several people are
trapped under the debris. Send help immediately.

A man is threatening shopkeepers with a knife near the Trichy bus stand.
Please send the police urgently.

Street light is not working near the temple. Please repair it when possible.
```
These (and a few more) are also available from the **"Try a sample message"**
dropdown in the Emergency Analyzer page.

---

## 5. Module explanations

**`preprocessing.py`** — the text pipeline: clean text (remove URLs/symbols) →
lowercase → tokenize → remove stop words → lemmatize (spaCy, with a small
rule-based fallback if spaCy is unavailable). Also validates user input
(empty, too short, too long, no readable words).

**`train_model.py`** — loads `emergency_dataset.csv`, preprocesses every
message, splits it into train/test, fits a `TfidfVectorizer` +
`LogisticRegression`, evaluates on the held-out test set (accuracy,
precision, recall, F1, confusion matrix, 5-fold cross-validation), and saves
`model.pkl` / `vectorizer.pkl` / `model_metrics.json`. `load_or_train_model()`
is what the app calls — it trains automatically if the files are missing or
were saved with an incompatible library version.

**`nlp_utils.py`** — the "brain" of the project:
- `get_entities()` — spaCy NER (falls back to regex for dates/times only).
- `extract_location()` — a layered approach: preposition + place-word rule
  (e.g. *"near the railway station"*) → highway numbers → spaCy location
  entities → capitalised-word fallback → lowercase-word fallback.
- `extract_people_affected()` — regex + number-word parsing
  (*"three people"*, *"a family of four"*, *"several"*, *"a woman"* → 1).
- `RuleBasedServiceDetector` — keyword table mapping words to services, plus
  a default service per category. It inherits from an abstract
  `ServiceDetector` class so it can later be swapped for a machine-learning
  version without changing any other file (`EmergencyAnalyzer(service_detector=MyMLDetector())`).
- `calculate_priority()` — a weighted score (category base score + critical
  /high/medium keyword hits − de-escalation words + people-affected bonus),
  turned into Critical/High/Medium/Low, with a safety rule that forces at
  least "Critical" whenever a genuine life-threat word (fire, trapped,
  unconscious, explosion…) is present and not cancelled by words like
  "minor" or "under control".
- `extract_keywords()` — lexicon words found in the message, topped up with
  high-IDF words from the TF-IDF vocabulary if fewer than three were found.
- `EmergencyAnalyzer` — ties every step together into one `.analyze(message)`
  call and never raises an exception to the caller (it returns
  `{"success": False, "error": ...}` on any problem).

**`database.py`** — SQLite storage. `init_db()` creates the
`emergency_messages` table automatically if it does not exist;
`save_message()`, `get_history()`, `count_messages()`, and
`delete_all_messages()` wrap the rest of the database access.

**`app.py`** — the Streamlit dashboard only; it imports and displays results
from the modules above.

---

## 6. Machine learning explanation

- **Features:** TF-IDF (Term Frequency – Inverse Document Frequency) turns
  each cleaned/lemmatized message into a vector of word (and 2-word phrase)
  importance scores, so common words like "the" contribute almost nothing
  while rare, meaningful words like "explosion" contribute a lot.
- **Model:** `LogisticRegression` (`scikit-learn`) — a fast, interpretable
  linear classifier well suited to sparse TF-IDF text features and small
  datasets. Multiclass classification uses one-vs-rest internally to choose
  among the 8 categories, and `predict_proba` gives the confidence score
  shown on the dashboard.
- **Evaluation:** the dataset is split into training (80%) and test (20%)
  sets. Because the sample dataset was generated from message templates, rows
  from the *same template* are kept together on one side of the split (a
  "template-grouped split") so the test set contains phrasing the model has
  genuinely never seen — a fair, non-inflated accuracy estimate. Metrics
  reported: **accuracy, precision, recall, F1-score** (weighted average
  across classes) and a **confusion matrix**, plus 5-fold cross-validation
  for a second opinion.
- **Priority detection is rule-based, not ML-based:** it combines the
  predicted category, weighted keyword hits, and the number of people
  affected into a score, then maps the score to Critical/High/Medium/Low.
  This was a deliberate design choice — priority depends on *severity
  language*, which a small labeled dataset cannot teach as reliably as an
  explicit, explainable rule set (and it makes the "why" fully visible to
  the user, which is important for an emergency-response tool).

---

## 7. NLP techniques used

| Technique | Where |
|---|---|
| Text cleaning (regex) | `preprocessing.clean_text` |
| Tokenization | `preprocessing.tokenize` |
| Stop-word removal | `preprocessing.remove_stop_words` |
| Lemmatization (spaCy + fallback) | `preprocessing.lemmatize` |
| TF-IDF vectorization (unigrams + bigrams) | `train_model.py` |
| Named Entity Recognition | `nlp_utils.get_entities` (spaCy `en_core_web_sm`) |
| Rule-based information extraction | location, people-affected, service, keywords, priority |
| Multiclass text classification | `LogisticRegression` on TF-IDF features |

---

## 8. Testing procedure

1. **Unit-style script:** `python test_samples.py` — runs 17 hand-written
   messages (not present in the training data) through the full pipeline and
   checks the emergency type, priority, location, people count and required
   service against expected values, plus 6 invalid inputs (empty, whitespace,
   too short, symbols only, digits only, `None`) to confirm the system never
   crashes.
2. **Model evaluation:** `python train_model.py` — prints accuracy,
   precision, recall, F1-score, 5-fold CV accuracy and the confusion matrix
   on a held-out test set.
3. **Manual testing:** use the sample-message dropdown in the **Emergency
   Analyzer** page and try your own messages, including messages with a
   missing location ("Someone is unconscious and bleeding...") or a missing
   people count, to see the graceful fallbacks ("Not specified" /
   "Not mentioned").
4. **Robustness:** try an empty message, a message with only symbols, or a
   very long message — the app shows a friendly validation error instead of
   crashing.

---

## 9. Model performance (example, on the sample dataset)

Numbers vary slightly each time the dataset is regenerated or retrained; the
current ones are visible on the **Model Performance** page and in
`model_metrics.json`. On this project's sample dataset (~1,490 messages, a
template-grouped 80/20 split):

| Metric | Score |
|---|---|
| Accuracy | ~0.92 |
| Precision (weighted) | ~0.92 |
| Recall (weighted) | ~0.92 |
| F1-score (weighted) | ~0.92 |
| 5-fold CV accuracy | ~0.89 |
| Priority-rule agreement with dataset labels | ~0.95 |

**Important academic honesty note:** the dataset was generated from
sentence templates (with randomized locations, people counts, and light
noise) because a real, labeled emergency-message dataset was not available.
This keeps the numbers realistic for a *student project* but they should not
be presented as production-grade accuracy on real-world emergency calls —
see "Future enhancements" below.

---

## 10. Future enhancements

- Replace the rule-based required-service detector with a trained multi-label
  classifier (the code is already structured for this — see `ServiceDetector`
  in `nlp_utils.py`).
- Fine-tune a transformer model (e.g. DistilBERT) for category classification
  instead of TF-IDF + Logistic Regression, if a larger real dataset becomes
  available.
- Add multi-language support (many emergency callers in India mix English
  with regional languages).
- Add a live map view of extracted locations using geocoding.
- Add SMS/WhatsApp integration so real messages can be piped into the system.
- Add user authentication and a dispatcher/admin view separate from the
  public analyzer.
- Replace the sample dataset with a larger, professionally annotated,
  real-world emergency-message dataset.

---

## 11. Project presentation points

1. **Problem:** emergency call/message centers must quickly triage a large
   volume of free-text messages into category, urgency, and required
   response — manual triage is slow and error-prone.
2. **Solution:** an end-to-end local NLP pipeline (cleaning → TF-IDF →
   Logistic Regression → NER → rule-based extraction) that turns raw text
   into a structured, actionable ticket in under a second.
3. **Highlights:** modular code (each concern is its own file), automatic
   model training and database creation, graceful handling of missing
   location/people-count/low-confidence cases, an explainable priority score
   ("why is this Critical?"), and a fully offline architecture (no paid
   APIs, works without internet once spaCy's model is installed).
4. **Tech stack:** Python, pandas, NumPy, scikit-learn, spaCy, SQLite,
   Streamlit, Plotly.
5. **Impact:** faster, more consistent first-pass triage that a human
   dispatcher can review and confirm — not a replacement for human judgment,
   but a decision-support tool.

---

## 12. Viva questions and answers

**Q1. Why did you use TF-IDF instead of word embeddings (Word2Vec/BERT)?**
TF-IDF is fast, needs no pretraining, works well on small datasets, and is
easy to explain in a viva. Embeddings/transformers usually need more data and
compute to outperform TF-IDF on a small, domain-specific dataset like this
one; the README lists it under "Future enhancements".

**Q2. Why Logistic Regression and not a deep learning model?**
Logistic Regression is fast to train, interpretable, works well on sparse
TF-IDF features, and does not overfit as easily as a deep model on a dataset
of this size (~1,490 messages).

**Q3. How do you avoid data leakage between train and test sets?**
The dataset was generated from sentence templates, so a plain random split
could put near-duplicate messages in both sets. `train_model.py` implements a
**template-grouped split**: all messages from the same template stay
together on one side, so the test set has never seen that phrasing.

**Q4. How is "priority" decided — is it also machine-learned?**
No — priority is a transparent, rule-based score: a category base score plus
weighted keyword hits (critical/high/medium) minus de-escalation words, plus
a bonus for the number of people affected, mapped to
Critical/High/Medium/Low. A safety rule guarantees at least "Critical" when a
genuine life-threat word is present (unless clearly cancelled, e.g. "minor
fire, already extinguished"). This was a deliberate choice for
explainability in a safety-critical context.

**Q5. What happens if spaCy is not installed?**
`preprocessing.py` and `nlp_utils.py` both detect this and fall back to a
lightweight rule-based lemmatizer and regex-based date/time extraction, so
the app keeps running (location detection then relies more on the
keyword/rule fallback instead of spaCy's entity recognizer).

**Q6. What happens if the message has no location or no people count?**
`extract_location()` returns `"Not specified"` and
`extract_people_affected()` returns `"Not mentioned"` — the analyzer never
crashes; these are treated as normal, valid outcomes.

**Q7. How is the model retrained, and does the app crash if `model.pkl` is
missing?**
`load_or_train_model()` in `train_model.py` tries to load `model.pkl` and
`vectorizer.pkl`; if either file is missing or fails to load (e.g. saved with
an incompatible scikit-learn version), it automatically calls
`train_and_save()` to retrain from `emergency_dataset.csv` before continuing.

**Q8. How would you extend this to a multi-label required-service
prediction?**
The `ServiceDetector` abstract base class in `nlp_utils.py` already separates
"how a service is decided" from "how it's used". A new class (e.g.
`MLServiceDetector`) implementing `detect()` with a trained multi-label
classifier can be passed to `EmergencyAnalyzer(service_detector=...)` without
touching any other file.

**Q9. Why is the dataset synthetic/template-based, and is that a limitation?**
Yes, it is an acknowledged limitation, stated openly in the README: no
public, labeled, real-world emergency-message dataset with category and
priority labels was available for a student project. The templates were
written to reflect realistic Indian emergency scenarios and locations, and
light noise (case changes, abbreviations like "pls") was added for
robustness, but the reported metrics should be read as a controlled
demonstration, not real-world performance.

**Q10. How does the system stay usable if the ML model is very unsure?**
`EmergencyAnalyzer.LOW_CONFIDENCE` (40%) triggers a warning message shown on
the dashboard ("Low model confidence — please review this message
manually"), and a message with zero recognizable words returns a safe
"Other" category with 0% confidence instead of guessing randomly.

---

## 13. Notes

- This is an **academic/portfolio project**. It is not connected to any real
  emergency service and must never be used to actually dispatch help — for a
  real emergency, always call your local emergency number (**112** in
  India).
- All processing is local; no message is sent to any external server or
  paid API.

Developed by Janani.
