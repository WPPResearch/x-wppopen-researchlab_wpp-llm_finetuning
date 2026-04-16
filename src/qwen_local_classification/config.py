from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Config:
    # Dataset
    dataset_version: str = "V15"
    num_classes: int = 3
    int_map: Dict[str, int] = field(
        default_factory=lambda: {"neg": 0, "avg": 1, "pos": 2}
    )
    seed: int = 3407

    # Model configuration
    model_name: str = "unsloth/Qwen3-4B-Base"
    load_in_4bit: bool = False
    max_seq_length: int = 2048
    dtype: str = None  # None for auto detection

    # Training configuration
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 10
    learning_rate: float = 1e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    output_dir: str = "qwen_finetuned/checkpoints"
    num_train_epochs: int = 5

    # PEFT configuration
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_rslora: bool = True

    # Prompt Template
    prompt_template: str = """Here is a marketing campaign:
{}

Classify this marketing campaign regarding its success into one of the following:
class 0: Underperforming
class 1: Average performance
class 2: Overperforming

SOLUTION
The correct answer is: class {}"""

    inference_prompt_template: str = """Here is a marketing campaign:
{}

Classify this marketing campaign regarding its success into one of the following:
class 0: Underperforming
class 1: Average performance
class 2: Overperforming

SOLUTION
The correct answer is: class """

    @property
    def finetuned_model_name(self) -> str:
        return f"qwen_finetuned/finetuned-unsloth-{self.dataset_version}"

    @property
    def finetuned_oversampled_model_name(self) -> str:
        return f"qwen_finetuned/finetuned-unsloth-oversampled-{self.dataset_version}"
