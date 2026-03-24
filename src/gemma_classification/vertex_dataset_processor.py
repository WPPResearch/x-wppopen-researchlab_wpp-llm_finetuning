from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.gemma_classification.config import DATASET_VERSION, SYSTEM_PROMPT_VERSION
from src.gemma_classification.helpers import (
    save_jsonlines,
    verify_local_datasets,
)
from src.gemma_classification.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class VertexDatasetProcessor:
    """Handles downloading, cleaning, and formatting datasets for Vertex AI."""

    # Model constants
    INT_MAP: Dict[str, int] = {"neg": 0, "avg": 1, "pos": 2}
    OUTCOME_MAP: Dict[str, str] = {
        "neg": "Underperforming",
        "avg": "Average performance",
        "pos": "Overperforming",
    }

    def __init__(self, dataset_version: str, system_prompt_v: int):
        self.version = dataset_version
        self.sys_v = system_prompt_v

        self.raw_path = Path(f"data/raw_data/{self.version}")
        self.ft_output_dir = Path(
            f"data/preprocessed_data/for_finetuning_gemma/{self.version}"
        )
        self.eval_output_dir = Path(
            f"data/preprocessed_data/for_evaluation_gemma/{self.version}"
        )
        self.test_output_dir = Path(
            f"data/preprocessed_data/for_testing_gemma/{self.version}"
        )

    def check_local_data(self) -> None:
        """Verify that Git LFS files are present."""
        print(f"--- Checking Local Data for Version: {self.version} ---")
        verify_local_datasets(self.version)

    def _load_and_clean(
        self, file_path: Path, balance_classes: bool = False
    ) -> pd.DataFrame:
        """Internal utility to handle the Pandas heavy lifting."""
        df = pd.read_json(file_path, lines=True)

        # Vectorized list extraction
        cols = ["platform", "par_brand", "par_audience", "par_creative"]
        for col in [c for c in cols if c in df.columns]:
            df[col] = df[col].str[0].fillna("")

        if "geo" in df.columns:
            df["geo"] = df["geo"].astype(str).str.strip("[]")

        if "label" in df.columns:
            df["outcome"] = df["label"].map(self.OUTCOME_MAP)
            df["integer_label"] = df["label"].map(self.INT_MAP)
            df = df.drop_duplicates().reset_index(drop=True)

            if balance_classes:
                avg_df = df[df["label"] == "avg"]
                other_df = df[df["label"] != "avg"]
                n_target = (
                    other_df["label"].value_counts().max()
                    if not other_df.empty
                    else len(avg_df)
                )
                avg_sampled = avg_df.sample(
                    n=min(int(n_target), len(avg_df)), random_state=72
                )
                df = pd.concat([other_df, avg_sampled]).sample(frac=1, random_state=42)

        return df.reset_index(drop=True)

    def _create_jsonl_row(self, row: pd.Series, for_training: bool) -> Dict[str, Any]:
        """Maps a single row to Gemma training format."""

        label_key = "integer_label" if self.sys_v == 3 else "outcome"
        label_text = str(row.get(label_key, "")).strip()

        user_text = USER_PROMPT_TEMPLATE.format(
            audience=row.get("par_audience", ""),
            brand=row.get("par_brand", ""),
            creative=row.get("par_creative", ""),
            platform=row.get("platform", ""),
            geolocation=row.get("geo", ""),
        ).strip()

        system_text = SYSTEM_PROMPT[self.sys_v].strip()

        # Combine system + user into a single prompt
        full_prompt = f"{system_text}\n\nUser:\n{user_text}\n\nAssistant:\n"

        if for_training:
            return {"prompt": full_prompt, "response": label_text}

        return {"prompt": full_prompt, "ground_truth": label_text}

    def _save_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, balanced: bool):
        """Internal helper to save both training and evaluation versions of a split."""
        suffix = "_balanced" if balanced else ""

        # Configure targets
        targets = [
            (True, self.ft_output_dir),  # Training format
            (False, self.eval_output_dir),  # Evaluation format
        ]

        for is_train, out_dir in targets:
            out_dir.mkdir(parents=True, exist_ok=True)

            t_data = train_df.apply(
                self._create_jsonl_row, axis=1, args=(is_train,)
            ).to_list()
            v_data = val_df.apply(
                self._create_jsonl_row, axis=1, args=(is_train,)
            ).to_list()

            save_jsonlines(
                out_dir
                / f"train_sys_prompt_{self.sys_v}_{DATASET_VERSION}{suffix}.jsonl",
                t_data,
            )
            save_jsonlines(
                out_dir
                / f"val_sys_prompt_{self.sys_v}_{DATASET_VERSION}{suffix}.jsonl",
                v_data,
            )

    def run_finetuning_pipeline(self):
        """Main entry point for training/val data generation."""
        input_file = self.raw_path / f"{self.version}_train.jsonl"

        for balance in [False, True]:
            print(f"Processing Finetuning Set (Balanced={balance})...")
            df = self._load_and_clean(input_file, balance_classes=balance)

            # 80/20 Split & Maximum 5K rows for validation set is a vertex requirement
            val_df = df.sample(frac=0.2, random_state=27).head(5000)
            train_df = df.drop(val_df.index)

            self._save_split(train_df, val_df, balanced=balance)

    def run_testing_pipeline(self):
        """Main entry point for test data generation."""
        print("Processing Test Set...")
        input_file = self.raw_path / f"{self.version}_test.jsonl"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        test_df = self._load_and_clean(input_file, balance_classes=False)
        test_data = test_df.apply(
            self._create_jsonl_row, axis=1, args=(False,)
        ).to_list()

        save_jsonlines(
            self.test_output_dir
            / f"test_sys_prompt_{self.sys_v}_{DATASET_VERSION}.jsonl",
            test_data,
        )

