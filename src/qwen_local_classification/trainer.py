import torch
import datasets
from trl import SFTTrainer, SFTConfig
from src.qwen_local_classification.config import Config
from src.qwen_local_classification.model import LLMClassifier


class ModelTrainer:
    def __init__(self, config: Config, llm_classifier: LLMClassifier):
        self.config = config
        self.llm = llm_classifier

    def train(self, train_df, is_oversampled: bool = True):
        self.llm.load_base_model()
        self.llm.trim_lm_head()
        self.llm.setup_peft()
        train_dataset = datasets.Dataset.from_pandas(
            train_df.fillna(""), preserve_index=False
        )
        collator = self.llm.get_data_collator()

        trainer = SFTTrainer(
            model=self.llm.model,
            tokenizer=self.llm.tokenizer,
            train_dataset=train_dataset,
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=1,
            packing=False,
            args=SFTConfig(
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                learning_rate=self.config.learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=self.config.logging_steps,
                optim=self.config.optim,
                weight_decay=self.config.weight_decay,
                lr_scheduler_type=self.config.lr_scheduler_type,
                seed=self.config.seed,
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                report_to="none",
            ),
            data_collator=collator,
            dataset_text_field="text",
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved before training.")

        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")

        save_name = (
            self.config.finetuned_oversampled_model_name
            if is_oversampled
            else self.config.finetuned_model_name
        )
        self.llm.reconstruct_and_save(save_name)
        print(f"Model saved to {save_name}")
