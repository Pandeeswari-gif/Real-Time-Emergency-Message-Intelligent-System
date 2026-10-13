"""
train_model.py
==============
Trains the emergency-category classifier.

    emergency_dataset.csv  ->  preprocessing  ->  TF-IDF  ->  Logistic Regression

Outputs:
    model.pkl            trained Logistic Regression model
    vectorizer.pkl       fitted TF-IDF vectorizer
    model_metrics.json   accuracy, precision, recall, F1, confusion matrix ...

Run it manually with:      python train_model.py
The dashboard also calls load_or_train_model(), which trains automatically
when the model files are missing or cannot be loaded.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from preprocessing import preprocess_text, spacy_available

logger = logging.getLogger(__name__)

# ---- File locations (all inside the project folder) ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "emergency_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

REQUIRED_COLUMNS = ["message", "category", "priority"]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Read the CSV, check the columns and remove empty / duplicate rows."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPlace 'emergency_dataset.csv' in the project folder."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing these columns: {missing}")
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""].drop_duplicates(subset="message").reset_index(drop=True)
    if df["category"].nunique() < 2:
        raise ValueError("The dataset needs at least two different categories.")
    return df


def split_dataset(df: pd.DataFrame):
    """
    Split the data into training (80%) and testing (20%) parts.

    The sample dataset was written from message templates, so several rows are
    close paraphrases of each other. A plain random split would put near-copies
    of test messages into the training data and make the accuracy look too good.
    When a 'template_id' column exists, all rows of one template therefore stay
    on the SAME side of the split, so the test set contains phrasing the model
    has never seen. Without that column a normal stratified random split is used.
    """
    if "template_id" in df.columns and df["template_id"].notna().all():
        rng = np.random.RandomState(RANDOM_STATE)
        test_templates = set()
        for _, part in df.groupby("category"):          # 20% of templates per category
            templates = list(part["template_id"].unique())
            rng.shuffle(templates)
            test_templates.update(templates[: max(1, int(round(len(templates) * TEST_SIZE)))])
        is_test = df["template_id"].isin(test_templates)
        return df[~is_test], df[is_test], "template-grouped split"
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                   stratify=df["category"])
    return train, test, "stratified random split"


def evaluate_priority_rules(df: pd.DataFrame) -> Optional[Dict[str, object]]:
    """
    Check how well the rule-based priority scorer agrees with the 'priority'
    column of the dataset. The TRUE category is used, so this measures only the
    priority logic. (The scorer is rule based, so there is nothing to train.)
    """
    try:
        from nlp_utils import calculate_priority, extract_people_affected
        labels = ["Critical", "High", "Medium", "Low"]
        predicted = []
        for message, category in zip(df["message"], df["category"]):
            hint = int(extract_people_affected(message)["score_hint"])
            predicted.append(calculate_priority(message, category, hint)["priority"])
        matrix = confusion_matrix(df["priority"], predicted, labels=labels)
        return {
            "accuracy": float(accuracy_score(df["priority"], predicted)),
            "labels": labels,
            "confusion_matrix": matrix.tolist(),
            "samples": int(len(df)),
        }
    except Exception as exc:
        logger.warning("Priority evaluation skipped: %s", exc)
        return None


def train_and_save(verbose: bool = True) -> Dict[str, object]:
    """Train the model, evaluate it, save everything and return the metrics."""
    df = load_dataset()
    if verbose:
        print(f"Dataset loaded: {len(df)} messages, {df['category'].nunique()} categories")
        print(f"spaCy lemmatizer in use: {spacy_available()}")

    # 1. NLP preprocessing (clean -> lowercase -> tokens -> stop words -> lemmas)
    df["processed"] = df["message"].apply(preprocess_text)

    # 2. Train / test split (stratified keeps the category balance)
    train_df, test_df, split_name = split_dataset(df)
    x_train, y_train = train_df["processed"], train_df["category"]
    x_test, y_test = test_df["processed"], test_df["category"]

    # 3. TF-IDF features (unigrams + bigrams), learned from training data only
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    # 4. Logistic Regression classifier
    model = LogisticRegression(max_iter=2000, C=10, random_state=RANDOM_STATE)
    model.fit(x_train_vec, y_train)

    # 5. Evaluation on the unseen test set
    y_pred = model.predict(x_test_vec)
    labels = sorted(df["category"].unique())
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    # 5-fold cross-validation on the training data (a second opinion).
    # With template ids, whole templates are kept together in each fold.
    if "template_id" in train_df.columns:
        cv, cv_groups = GroupKFold(n_splits=5), train_df["template_id"]
    else:
        cv, cv_groups = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), None
    cv_pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10, random_state=RANDOM_STATE),
    )
    cv_scores = cross_val_score(cv_pipeline, x_train, y_train, cv=cv, groups=cv_groups,
                                scoring="accuracy")

    metrics = {
        "split_strategy": split_name,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": int(len(df)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "num_features": int(len(vectorizer.vocabulary_)),
        "spacy_lemmatizer": bool(spacy_available()),
        "labels": labels,
        "class_counts": df["category"].value_counts().to_dict(),
        "priority_counts": df["priority"].value_counts().to_dict(),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "priority_rules": evaluate_priority_rules(df),
    }

    # 6. Save model, vectorizer and metrics
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"Split: {split_name} -> {len(x_train)} training / {len(x_test)} test messages")
        print("\n===== Model performance (test set) =====")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}  (weighted)")
        print(f"Recall   : {metrics['recall']:.4f}  (weighted)")
        print(f"F1-score : {metrics['f1_score']:.4f}  (weighted)")
        print(f"5-fold CV accuracy: {metrics['cv_accuracy_mean']:.4f} +/- {metrics['cv_accuracy_std']:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
        print("Confusion matrix (rows = true, columns = predicted):")
        print(pd.DataFrame(matrix, index=labels, columns=labels))
        if metrics["priority_rules"]:
            print(f"\nRule-based priority accuracy vs dataset labels: "
                  f"{metrics['priority_rules']['accuracy']:.4f}")
        print(f"\nSaved: {os.path.basename(MODEL_PATH)}, {os.path.basename(VECTORIZER_PATH)}, "
              f"{os.path.basename(METRICS_PATH)}")
    return metrics


def load_metrics() -> Optional[Dict[str, object]]:
    """Read the saved metrics file (None if it does not exist or is broken)."""
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_or_train_model() -> Tuple[object, object]:
    """
    Load model.pkl and vectorizer.pkl. The model is trained again when
      - a file is missing,
      - the files were saved with a different scikit-learn version
        (pickles are not portable between versions), or
      - loading fails for any other reason.
    Training only takes a few seconds.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        saved = load_metrics() or {}
        if saved.get("sklearn_version") == sklearn.__version__:
            try:
                return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
            except Exception as exc:
                logger.warning("Could not load saved model (%s). Training again...", exc)
        else:
            logger.warning("Saved model was made with another scikit-learn version. Training again...")
    train_and_save(verbose=False)
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    train_and_save(verbose=True)
