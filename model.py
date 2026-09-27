"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    # TODO: call FastLanguageModel.from_pretrained with 4-bit loading and return (model, tokenizer)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True
    )
    return model,tokenizer
    pass

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    # TODO: sum p.numel() over every parameter tensor in the module
    total_parameters = sum(p.numel() for p in model.parameters())
    return int(total_parameters)
    pass

# Step 3 - is_model_4bit_quantized
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # TODO: walk the model's submodules and check for a bitsandbytes Linear4bit instance
    for module in model.modules():
        module_classname = module.__class__.__name__
        
        # Check if it matches the standard bitsandbytes 4-bit linear layers
        if "Linear4bit" in module_classname or "BnbQuantizedLinear" in module_classname:
            return True
            
    return False

    pass

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # TODO: if the tokenizer is missing a pad token, reuse its eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer    
    pass

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt
    return ["q_proj","k_proj","v_proj","o_proj"]
    pass

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    if target_modules is None:
        target_modules = get_lora_target_modules()

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0,  # Unsloth supports optimized dropout 0
        bias="none",     # Exclude bias parameters from training
        use_gradient_checkpointing="unsloth",  # Enables Unsloth's memory-efficient checkpointing
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model

    pass

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # TODO: sum p.numel() over model.parameters() where requires_grad is True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    pass

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # TODO: return the fraction of parameters that are trainable.
    return trainable_count/total_count
    pass

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    # TODO: return a tiny hand-written list of instruction/response example dicts.
    return [
        {
            "instruction": "Summarize the key benefit of low-rank adaptation (LoRA) in one sentence.",
            "response": "LoRA drastically reduces fine-tuning memory requirements by freezing the base model weights and training only small, rank-decomposition matrices."
        },
        {
            "instruction": "Write a Python function to check if a word is a palindrome.",
            "response": "def is_palindrome(word: str) -> bool:\n    cleaned = word.lower()\n    return cleaned == cleaned[::-1]"
        },
        {
            "instruction": "Explain the difference between a scalar and a vector in physics.",
            "response": "A scalar quantity has only magnitude (such as temperature or mass), whereas a vector quantity has both magnitude and direction (such as velocity or force)."
        },
        {
            "instruction": "Convert the temperature 25 degrees Celsius to Fahrenheit.",
            "response": "To convert Celsius to Fahrenheit, multiply by 1.8 and add 32. Thus, 25°C equals 77°F."
        },
        {
            "instruction": "square the number n",
            "response": "Square of a number n is n multiplied by n"
        }
    ]
    
    pass

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # TODO: combine example['instruction'] and example['response'] into one string
    #return f"### Instruction:\n{example['instruction']}\n\n### response:\n{example['response']}
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    pass

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # TODO: apply format_instruction_example to every example and return the list
    return [format_instruction_example(ex) for ex in examples]
    pass

# Step 12 - build_text_dataset
def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # TODO: return a datasets.Dataset with one 'text' column holding the given strings
    return Dataset.from_dict({"text":texts})
    pass

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # TODO: call the tokenizer on text and return its input_ids as a plain list
    return tokenizer.encode(text, add_special_tokens=True)
    pass

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    # TODO: return the length of the input_ids sequence
    return len(input_ids)
    pass

# Step 15 - build_training_arguments
import torch
from trl import SFTConfig

def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    # TODO: build TrainingArguments with batch size 1, given max_steps, given lr, bf16 or fp16.
    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        bf16=supports_bf16,
        fp16=not supports_bf16,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
    )
    pass

# Step 16 - build_sft_trainer
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""
    # TODO: wire model, tokenizer, dataset, and training_args into an SFTTrainer
    training_args.max_seq_length = max_seq_length

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    
    return trainer
    
    
    pass

# Step 17 - run_sft_training
def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    # TODO: drive the trainer through its short optimization run and return the final loss
    train_result = trainer.train()
    metrics = train_result.metrics()
    final_loss = metrics.get("train_loss", float("nan"))
    return float(final_loss)
    pass

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

