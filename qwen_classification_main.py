import argparse
import pandas as pd
from src.qwen_local_classification.config import Config
from src.qwen_local_classification.data_processor import DataProcessor
from src.qwen_local_classification.model import LLMClassifier
from src.qwen_local_classification.trainer import ModelTrainer
from src.qwen_local_classification.evaluator import ModelEvaluator


def preprocess(config: Config):
    processor = DataProcessor(config)
    processor.prepare_and_save_data()


def train(config: Config):
    llm = LLMClassifier(config)
    trainer = ModelTrainer(config, llm)

    # Load dataset
    print("Loading prepared training data...")
    train_df = pd.read_csv(
        f"data/preprocessed_data/for_training_qwen/{config.dataset_version}/preprocessed_oversampled_train.csv"
    )

    # Format prompts
    processor = DataProcessor(config)
    train_df = processor.apply_prompt_template(train_df)

    trainer.train(train_df, is_oversampled=True)


def evaluate(config: Config):
    llm = LLMClassifier(config)
    evaluator = ModelEvaluator(config, llm)

    print("Loading validation data...")
    val_df = pd.read_csv(
        f"data/preprocessed_data/for_evaluation_qwen/{config.dataset_version}/preprocessed_val.csv"
    )

    processor = DataProcessor(config)
    # The evaluation loop only evaluates logits so we format without labels essentially
    val_df = processor.apply_prompt_template(
        val_df.drop(columns=["label"]) if "label" in val_df else val_df
    )
    # Put labels back for passing to eval loop
    val_df_labels = pd.read_csv(
        f"data/preprocessed_data/for_evaluation_qwen/{config.dataset_version}/preprocessed_val.csv"
    )["label"]
    val_df["label"] = val_df_labels

    evaluator.evaluate(val_df, batch_size=16)


def predict(config: Config, boost: bool = False):
    llm = LLMClassifier(config)
    evaluator = ModelEvaluator(config, llm)

    print("Loading testing data...")
    test_df = pd.read_csv(
        f"data/preprocessed_data/for_testing_qwen/{config.dataset_version}/preprocessed_test.csv"
    )

    processor = DataProcessor(config)
    test_df = processor.apply_prompt_template(test_df)

    # In previous test inference the num classes + 1 token ID logic required 4 boost factors
    # e.g., boost_factors = torch.tensor([2.5, 1.0, 2.5, 1.0])
    # However since we fixed the logic to only use the exact number of class tokens
    # we just need 3 factors for 3 classes
    factors = [1.0, 1.0, 1.0]
    if boost:
        factors = [2.5, 1.0, 2.5]

    import os

    out_dir = "results/classification/submissions"
    os.makedirs(out_dir, exist_ok=True)

    base_name = f"QWEN_{config.dataset_version}_lr{config.learning_rate}_bs{config.per_device_train_batch_size}_ep{config.num_train_epochs}_r{config.r}_{config.seed}"
    if boost:
        base_name += "_boosted"

    out_file = os.path.join(out_dir, f"{base_name}.txt")

    evaluator.predict(
        test_df,
        batch_size=16,
        boost_factors=factors if boost else None,
        output_file=out_file,
    )


def main():
    parser = argparse.ArgumentParser(description="Local LLM Classifier Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: Preprocess
    subparsers.add_parser("preprocess", help="Preprocess the dataset and split")

    # Command: Train
    subparsers.add_parser("train", help="Finetune the model on the training data")

    # Command: Evaluate
    subparsers.add_parser(
        "evaluate", help="Evaluate the model on the validation dataset"
    )

    # Command: Predict
    predict_parser = subparsers.add_parser(
        "predict", help="Generate predictions on the test dataset"
    )
    predict_parser.add_argument(
        "--boost",
        action="store_true",
        help="Apply confidence boosting adjustments to logits",
    )

    args = parser.parse_args()
    config = Config()

    if args.command == "preprocess":
        preprocess(config)
    elif args.command == "train":
        train(config)
    elif args.command == "evaluate":
        evaluate(config)
    elif args.command == "predict":
        predict(config, boost=args.boost)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
