from src.gemini_classification.classification_evaluator import evaluate_results
from src.gemini_classification.config import (
    ADAPTER_SIZE,
    BALANCED_DATA,
    DATASET_VERSION,
    LRM,
    N_EPOCHS,
    SYSTEM_PROMPT_VERSION,
)
from src.gemini_classification.vertex_batch_inference_manager import VertexBatchInferenceManager
from src.gemini_classification.vertex_dataset_processor import VertexDatasetProcessor
from src.gemini_classification.vertex_tuning_manager import VertexTuningManager

if __name__ == "__main__":
    print()

    processor = VertexDatasetProcessor(DATASET_VERSION, SYSTEM_PROMPT_VERSION)

    processor.check_local_data()
    processor.run_finetuning_pipeline()
    processor.run_testing_pipeline()

    print("All datasets processed successfully.")

    trainer = VertexTuningManager()

    # Upload the latest local files
    trainer.sync_data_to_gcs(overwrite=False)

    trainer.submit_sft_job(
        dataset_version=DATASET_VERSION,
        sys_prompt_v=SYSTEM_PROMPT_VERSION,
        balanced_data=BALANCED_DATA,
        adapter_size=ADAPTER_SIZE,
        learning_rate_multiplier=LRM,
        epochs=N_EPOCHS,
        wait=True,
    )

    inference_manager = VertexBatchInferenceManager()

    inference_manager.run_batch_inference(
        DATASET_VERSION,
        sys_prompt_v=SYSTEM_PROMPT_VERSION,
        balanced_data=BALANCED_DATA,
        adapter_size=ADAPTER_SIZE,
        lrm=LRM,
        n_epochs=N_EPOCHS,
        sets_to_run=["val", "test"],
        wait=True,
    )

    evaluate_results()
