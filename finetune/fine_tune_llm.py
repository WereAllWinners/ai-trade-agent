#!/usr/bin/env python3
"""
Advanced LLM Fine-tuning Script – Fixed to use correct data file
"""

import os
import json
import torch
import argparse
from pathlib import Path
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from datetime import datetime

class AdvancedModelTrainer:
    def __init__(
        self,
        base_model="unsloth/qwen2.5-32b-instruct-bnb-4bit",
        existing_adapter=None,
        output_dir="finetune/finance_qwen_32b_lora",
        max_seq_length=2048
    ):
        self.base_model = base_model
        self.existing_adapter = existing_adapter
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        
        # Check mode
        if existing_adapter and Path(existing_adapter).exists():
            self.mode = "continue"
            print(f"🔄 CONTINUE TRAINING MODE from {existing_adapter}")
        else:
            self.mode = "fresh"
            print(f"🆕 FRESH TRAINING MODE")
            print(f"📥 Base model: {base_model}")

    def load_training_data(self, data_path: str):
        print(f"\n📚 Loading training data from: {data_path}")
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"   Raw entries loaded: {len(raw_data)}")
        
        # Format for Qwen 2.5 chat template
        formatted = []
        for i, ex in enumerate(raw_data):
            input_text = ex.get('input', '')
            output_text = ex.get('output', '')
            
            # Full prompt with system instruction
            full_text = f"""<|im_start|>system
You are an expert financial trading advisor with knowledge from the world's best investors including Warren Buffett, Nancy Pelosi, Cathie Wood, and Michael Burry. You analyze stocks using technical indicators, fundamental analysis, insider trading patterns, congressional trades, and proven strategies.<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output_text}<|im_end|>"""
            
            formatted.append({"text": full_text})
            
            # Print first 2 examples for sanity check
            if i < 2:
                print(f"\nExample {i+1}:")
                print(f"Input preview: {input_text[:200]}...")
                print(f"Output preview: {output_text[:200]}...")
        
        dataset = Dataset.from_list(formatted)
        print(f"✅ Formatted dataset ready with {len(dataset)} examples")
        
        return dataset

    def load_model(self):
        print(f"\n{'='*70}")
        print("📥 LOADING MODEL")
        print(f"{'='*70}")
        
        if self.mode == "continue":
            print(f"🔄 Loading existing adapter: {self.existing_adapter}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.existing_adapter,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
        else:
            print(f"📥 Downloading base model: {self.base_model}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            
            print("\n🔧 Adding LoRA adapters...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=64,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_alpha=64,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
        
        print("✅ Model and tokenizer loaded")
        return model, tokenizer

    def train(self, data_path: str, num_epochs=3, batch_size=2, learning_rate=2e-4):
        print(f"\n{'='*70}")
        print("🎓 STARTING FINE-TUNING")
        print(f"{'='*70}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size} (with grad accumulation)")
        print(f"Learning rate: {learning_rate}")
        print(f"{'='*70}\n")

        # Load model
        model, tokenizer = self.load_model()

        # Load data
        dataset = self.load_training_data(data_path)

        # Output dir with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.output_dir}_{timestamp}"

        print(f"💾 Saving to: {output_dir}")

        # Training args – optimized for your ZGX Nano
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
        )

        # Trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=training_args,
        )

        # Show GPU info
        if torch.cuda.is_available():
            print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        print(f"\n{'='*70}")
        print("🚀 TRAINING STARTED")
        print(f"{'='*70}\n")

        trainer.train()

        # Save final model
        print(f"\n💾 Saving final model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n✅ TRAINING COMPLETE")
        print(f"Model saved to: {output_dir}")
        return output_dir


def main():
    parser = argparse.ArgumentParser(description='Fine-tune trading LLM')
    
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data JSON file (required)')
    
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Path to existing adapter to continue training')
    
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Per-device batch size')
    
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='Learning rate')
    
    args = parser.parse_args()

    trainer = AdvancedModelTrainer(
        existing_adapter=args.continue_from
    )

    output_dir = trainer.train(
        data_path=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    print(f"\nModel ready at: {output_dir}")


if __name__ == "__main__":
    main()