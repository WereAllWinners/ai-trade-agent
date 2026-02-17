#!/usr/bin/env python3
"""
Advanced LLM Fine-tuning Script
Supports fresh training, continuing from adapters, and resuming from checkpoints
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

        if existing_adapter and Path(existing_adapter).exists():
            self.mode = "continue"
            print(f"🔄 CONTINUE TRAINING MODE from {existing_adapter}")
        else:
            self.mode = "fresh"
            print(f"🆕 FRESH TRAINING MODE - Base: {base_model}")

    def load_training_data(self, data_paths):
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        all_examples = []

        for data_path in data_paths:
            if not Path(data_path).exists():
                print(f"  ⚠️  Skipping missing file: {data_path}")
                continue

            print(f"\n📚 Loading: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            print(f"   Entries: {len(raw_data)}")

            for ex in raw_data:
                input_text  = ex.get('input', '')
                output_text = ex.get('output', '')
                if not input_text or not output_text:
                    continue

                full_text = f"""<|im_start|>system
You are an expert financial trading advisor with knowledge from the world's best investors including Warren Buffett, Nancy Pelosi, Cathie Wood, and Michael Burry. You analyze stocks using technical indicators, fundamental analysis, insider trading patterns, congressional trades, and proven strategies.<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output_text}<|im_end|>"""

                all_examples.append({"text": full_text})

        print(f"\n✅ Total formatted examples: {len(all_examples)}")

        if len(all_examples) == 0:
            raise ValueError("No valid training examples found!")

        return Dataset.from_list(all_examples)

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
            print(f"📥 Loading base model: {self.base_model}")
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

        print("✅ Model ready")
        return model, tokenizer

    def train(self, data_paths, num_epochs=3, batch_size=2,
              learning_rate=2e-4, checkpoint_path=None):

        print(f"\n{'='*70}")
        print("🎓 STARTING FINE-TUNING")
        print(f"{'='*70}")
        print(f"Mode:          {self.mode.upper()}")
        print(f"Epochs:        {num_epochs}")
        print(f"Batch size:    {batch_size} x 4 grad accum = {batch_size*4} effective")
        print(f"Learning rate: {learning_rate}")
        if checkpoint_path:
            print(f"Resuming from: {checkpoint_path}")
        print(f"{'='*70}\n")

        model, tokenizer = self.load_model()
        dataset = self.load_training_data(data_paths)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.output_dir}_{timestamp}"
        print(f"\n💾 Output directory: {output_dir}")

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
            save_strategy="steps",
            save_steps=500,           # Save every 500 steps (~4 hours)
            save_total_limit=5,       # Keep last 5 checkpoints
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=training_args,
        )

        if torch.cuda.is_available():
            print(f"\n🖥️  GPU:  {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        print(f"\n{'='*70}")
        print("🚀 TRAINING STARTED")
        print(f"{'='*70}\n")

        trainer.train(resume_from_checkpoint=checkpoint_path)

        print(f"\n💾 Saving final model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        metadata = {
            'mode': self.mode,
            'base_model': self.base_model,
            'data_paths': data_paths if isinstance(data_paths, list) else [data_paths],
            'num_epochs': num_epochs,
            'dataset_size': len(dataset),
            'checkpoint_resumed': checkpoint_path,
            'timestamp': timestamp,
            'output_dir': output_dir
        }
        with open(f"{output_dir}/training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*70}")
        print(f"🎉 TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"✅ Model saved to: {output_dir}")
        print(f"\n💡 Next steps:")
        print(f"   python3 scripts/model_inference_lora.py")
        print(f"   sudo systemctl restart ai-trading-bot.service")
        print(f"   sudo systemctl restart ai-options-bot.service")

        return output_dir


def main():
    parser = argparse.ArgumentParser(description='Fine-tune trading LLM')

    parser.add_argument('--data', type=str, nargs='+', required=True,
                        help='Path(s) to training data JSON file(s)')

    parser.add_argument('--continue-from', type=str, default=None,
                        help='Path to existing LoRA adapter to continue training from')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint to resume '
                             '(e.g. finetune/finance_qwen_32b_lora_xxx/checkpoint-30)')

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=2e-4)

    args = parser.parse_args()

    # Determine adapter source - checkpoint takes priority over continue-from
    adapter_source = args.checkpoint or args.continue_from

    # Auto-detect if nothing specified
    if adapter_source is None:
        default_path = '/home/zgx/personal-projects/ai-trade-agent/finetune/finance_qwen_32b_lora'
        if Path(default_path).exists():
            print(f"🔍 Found existing model at: {default_path}")
            response = input("Continue training from this model? [Y/n]: ").strip().lower()
            if response != 'n':
                adapter_source = default_path

    trainer = AdvancedModelTrainer(existing_adapter=adapter_source)

    output_dir = trainer.train(
        data_paths=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint
    )

    print(f"\n✅ All done! Model ready at: {output_dir}")


if __name__ == "__main__":
    main()
