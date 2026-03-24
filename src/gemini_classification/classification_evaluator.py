import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from google.cloud import storage
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    accuracy_score,
    confusion_matrix,
)

from src.gemini_classification.config import BUCKET_NAME


class ClassificationEvaluator:
    """Manages downloading, parsing, and evaluating Vertex AI batch predictions."""

    # Centralized mapping to handle different prompt versions/outputs
    LABEL_MAPPING = {
        "average performance": 1,
        "underperforming": 0,
        "overperforming": 2,
        "0": 0,
        "1": 1,
        "2": 2,
    }
    VALID_LABELS = list(LABEL_MAPPING.keys())

    def __init__(self, bucket_name: str = BUCKET_NAME):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

    def _parse_gcs_uri(self, uri: str) -> str:
        """Removes gs://bucket_name/ from a URI to get the blob path."""
        return uri.replace(f"gs://{self.bucket_name}/", "")

    def load_predictions(self, gcs_uri: str) -> pd.DataFrame:
        """Downloads jsonl files from a GCS path and returns a cleaned DataFrame."""
        prefix = self._parse_gcs_uri(gcs_uri)
        blobs = self.bucket.list_blobs(prefix=prefix)

        all_data = []
        for blob in blobs:
            if not blob.name.endswith(".jsonl"):
                continue

            content = blob.download_as_text()
            for line in content.strip().split("\n"):
                if not line:
                    continue
                record = json.loads(line)

                # Robust extraction of Gemini response
                try:
                    prediction = record["response"]["candidates"][0]["content"][
                        "parts"
                    ][0]["text"].strip()
                except (KeyError, IndexError):
                    prediction = "ERROR"

                all_data.append(
                    {
                        "ground_truth": str(record.get("ground_truth", "N/A")),
                        "prediction": prediction,
                    }
                )

        return pd.DataFrame(all_data)

    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates classification metrics and counts malformed outputs."""
        df = df.copy().dropna()
        y_true = df["ground_truth"].astype(str).str.strip().str.lower()
        y_pred = df["prediction"].astype(str).str.strip().str.lower()

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)

        # Count invalid predictions (those not in our allowed list)
        invalid_mask = ~y_pred.isin(self.VALID_LABELS)
        n_invalid = invalid_mask.sum()

        print(f"\n{'=' * 40}\n{'METRICS REPORT':^40}\n{'=' * 40}")
        print(f"Accuracy:  {accuracy:.4f}\nF1 Score:  {f1:.4f}\nInvalid:   {n_invalid}")
        print(
            f"\nDetailed Report:\n{classification_report(y_true, y_pred, zero_division=0)}"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_invalid": n_invalid,
        }

    def plot_confusion_matrix(self, df: pd.DataFrame, output_name: str):
        """Generates and saves a confusion matrix heatmap."""
        labels = sorted(df["ground_truth"].unique().tolist())
        cm = confusion_matrix(df["ground_truth"], df["prediction"], labels=labels)

        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(f"Confusion Matrix: {output_name}")

        out_path = f"./results/classification/confusion_m/{output_name}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    def list_recent_prediction_uris(
        self, base_prefix: str = "batch_predictions/"
    ) -> List[str]:
        """Finds the most recent 'predictions.jsonl' in each experiment folder."""
        top_level = self.bucket.list_blobs(prefix=base_prefix, delimiter="/")
        list(top_level)  # populate prefixes

        uris = []
        for folder in top_level.prefixes:
            sub_dirs = self.bucket.list_blobs(prefix=folder, delimiter="/")
            list(sub_dirs)
            pred_folders = sorted([p for p in sub_dirs.prefixes if "prediction-" in p])

            if pred_folders:
                latest = pred_folders[-1]
                # Check if file exists
                if self.bucket.blob(f"{latest}predictions.jsonl").exists():
                    uris.append(f"gs://{self.bucket_name}/{latest}predictions.jsonl")
        return uris

    def save_test_submission(self, df: pd.DataFrame, filename: str):
        """Formats predictions for dashboard evaluation."""
        df["output"] = (
            df["prediction"]
            .str.strip()
            .str.lower()
            .map(self.LABEL_MAPPING)
            .fillna(1)
            .astype(int)
        )

        out_path = f"./results/classification/submissions/{filename}.txt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df[["output"]].to_csv(out_path, index=False, header=False)


def evaluate_results():
    eval_mgr = ClassificationEvaluator()

    # 1. Gather URIs
    all_uris = eval_mgr.list_recent_prediction_uris()
    train_val_uris = [u for u in all_uris if "val_set" in u or "train_set" in u]
    test_uris = [u for u in all_uris if "test_set" in u]

    # 2. Process Training/Validation (Full Metrics)
    summary_results = []
    for uri in train_val_uris:
        name = uri.split("/")[-3]
        print(f"Evaluating: {name}")

        df = eval_mgr.load_predictions(uri)
        metrics = eval_mgr.compute_metrics(df)
        metrics["uri"] = uri
        metrics["experiment"] = name
        summary_results.append(metrics)

        eval_mgr.plot_confusion_matrix(df, name)

    # Save summary CSV
    if summary_results:
        pd.DataFrame(summary_results).to_csv(
            "./results/classification/gemini_eval_summary.csv", index=False
        )

    # 3. Process Test sets (Submission format)
    for uri in test_uris:
        name = uri.split("/")[-3]
        df = eval_mgr.load_predictions(uri)
        eval_mgr.save_test_submission(
            df,
            name,
        )
        print(f"Saved formatted test set results for submission for {uri}")


if __name__ == "__main__":
    evaluate_results()
