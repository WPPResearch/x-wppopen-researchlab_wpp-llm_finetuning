import os
import json
import pandas as pd
from src.qwen_local_classification.config import Config


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def balance_df(
        self, df: pd.DataFrame, label_col: str = "label", strategy="undersample"
    ) -> pd.DataFrame:
        counts = df[label_col].value_counts()

        if strategy == "undersample":
            target = counts.min()
            balanced = df.groupby(label_col).sample(
                n=target, replace=False, random_state=self.config.seed
            )

        elif strategy == "oversample":
            max_count = counts.max()
            balanced_list = []

            for cls, cls_count in counts.items():
                cap = int(0.4 * max_count)
                if cls_count < cap:
                    n_samples = cap
                    replace = True
                else:
                    n_samples = cls_count
                    replace = False
                balanced_list.append(
                    df[df[label_col] == cls].sample(
                        n=n_samples, replace=replace, random_state=self.config.seed
                    )
                )
            balanced = pd.concat(balanced_list, ignore_index=True)
        else:
            raise ValueError("strategy must be 'undersample' or 'oversample'")

        return balanced.sample(frac=1, random_state=self.config.seed).reset_index(
            drop=True
        )

    def read_jsonl_to_df(self, filepath: str) -> pd.DataFrame:
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_number}: {e}")

        df = pd.DataFrame(records)
        cols = ["platform", "par_brand", "par_audience", "par_creative"]
        for col in [c for c in cols if c in df.columns]:
            df[col] = df[col].str[0].fillna("")

        if "geo" in df.columns:
            df["geo"] = df["geo"].astype(str).str.strip("[]").str.replace("'", "")

        if "label" in df.columns:
            df["label"] = df["label"].map(self.config.int_map)
            df = df.drop_duplicates().reset_index(drop=True)
        return df

    def format_text_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df["text"] = (
            "platform: "
            + df["platform"].str.strip()
            + "\n"
            + "brand: "
            + df["par_brand"].str.strip()
            + "\n"
            + "audience: "
            + df["par_audience"].str.strip()
            + "\n"
            + "creative: "
            + df["par_creative"].str.strip()
            + "\n"
            + "geolocation: "
            + df["geo"].str.strip()
        )
        return df

    def apply_prompt_template(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the training prompt template (with label) to the 'text' column."""
        texts = []
        for i in range(len(df)):
            text_ = df["text"].iloc[i]
            if "label" in df.columns:
                label_ = df["label"].iloc[i]
                text = self.config.prompt_template.format(text_, label_)
            else:
                text = self.config.inference_prompt_template.format(text_)
            texts.append(text)

        df_copy = df.copy()
        df_copy["text"] = texts
        return df_copy

    def prepare_and_save_data(self):
        train_path = f"data/raw_data/{self.config.dataset_version}/{self.config.dataset_version}_train.jsonl"
        test_path = f"data/raw_data/{self.config.dataset_version}/{self.config.dataset_version}_test.jsonl"

        train_out_dir = (
            f"data/preprocessed_data/for_training_qwen/{self.config.dataset_version}"
        )
        eval_out_dir = (
            f"data/preprocessed_data/for_evaluation_qwen/{self.config.dataset_version}"
        )
        test_out_dir = (
            f"data/preprocessed_data/for_testing_qwen/{self.config.dataset_version}"
        )

        os.makedirs(train_out_dir, exist_ok=True)
        os.makedirs(eval_out_dir, exist_ok=True)
        os.makedirs(test_out_dir, exist_ok=True)

        # Train & Val
        print(f"Loading training data from {train_path}...")
        df = self.read_jsonl_to_df(train_path)
        df = self.format_text_column(df)

        val_df = df.sample(frac=0.2, random_state=self.config.seed)
        train_df = df.drop(val_df.index)

        balanced_train_df = self.balance_df(train_df, "label", strategy="oversample")

        # Save
        val_df.to_csv(f"{eval_out_dir}/preprocessed_val.csv", index=False)
        train_df.to_csv(f"{train_out_dir}/preprocessed_train.csv", index=False)
        balanced_train_df.to_csv(
            f"{train_out_dir}/preprocessed_oversampled_train.csv", index=False
        )

        # Test
        print(f"Loading test data from {test_path}...")
        test_df = self.read_jsonl_to_df(test_path)
        test_df = self.format_text_column(test_df)
        test_df.to_csv(f"{test_out_dir}/preprocessed_test.csv", index=False)

        print("Data preparation complete.")
