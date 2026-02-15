#!/usr/bin/env python3
"""
Advanced LLM Fine-tuning Script
Supports both fresh training and continuing from existing models
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
    """
    Elite model fine-tuning with support for continuing training
    """
    
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
        
        # Check if continuing training
        if existing_adapter and Path(existing_adapter).exists():
            self.mode = "continue"
            print(f"🔄 CONTINUE TRAINING MODE")
            print(f"📂 Loading existing adapter from: {existing_adapter}")
        else:
            self.mode = "fresh"
            print(f"🆕 FRESH TRAINING MODE")
            print(f"📥 Will download base model: {base_model}")
    
    def load_training_data(self, data_path):
        """
        Load and format training data
        """
        print(f"\n📚 Loading training data from {data_path}...")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Format for instruction fine-tuning - create the full text here
        formatted_data = []
        for example in data:
            instruction = example.get('input', '')
            output = example.get('output', '')
            
            # Qwen 2.5 chat format - create the full text
            text = f"""<|im_start|>system
You are an expert financial trading advisor with knowledge from the world's best investors including Warren Buffett, Nancy Pelosi, Cathie Wood, and Michael Burry. You analyze stocks using technical indicators, fundamental analysis, insider trading patterns, congressional trades, and proven strategies.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
            
            formatted_data.append({
                'text': text,
                'source': example.get('metadata', {}).get('source', 'unknown')
            })
        
        print(f"✅ Loaded {len(formatted_data)} training examples")
        
        # Show data distribution
        sources = {}
        for ex in formatted_data:
            source = ex['source']
            sources[source] = sources.get(source, 0) + 1
        
        print(f"\n📊 Data distribution:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {source}: {count} examples")
        
        return Dataset.from_list(formatted_data)
    
    def load_model(self):
        """
        Load either fresh base model or existing adapter
        """
        print(f"\n{'='*70}")
        print(f"📥 LOADING MODEL")
        print(f"{'='*70}")
        
        if self.mode == "continue":
            # Load existing LoRA adapter
            print(f"🔄 Loading existing fine-tuned model...")
            print(f"   Base: {self.base_model}")
            print(f"   Adapter: {self.existing_adapter}")
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.existing_adapter,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            
            print(f"✅ Loaded existing adapter successfully")
            
        else:
            # Fresh start - download base model
            print(f"📥 Downloading base model: {self.base_model}")
            print(f"   (This may take a few minutes...)")
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            
            print(f"✅ Base model loaded")
            
            # Add LoRA adapters
            print(f"\n🔧 Configuring LoRA adapters...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
            
            print(f"✅ LoRA adapters configured")
        
        return model, tokenizer
    
    def train(self, training_data_path, num_epochs=3, batch_size=2, learning_rate=2e-4):
        """
        Main training loop
        """
        print(f"\n{'='*70}")
        print(f"🎓 STARTING FINE-TUNING")
        print(f"{'='*70}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"{'='*70}\n")
        
        # Load model
        model, tokenizer = self.load_model()
        
        # Load dataset
        dataset = self.load_training_data(training_data_path)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.mode == "continue":
            output_dir = self.existing_adapter
            backup_dir = f"{self.existing_adapter}_backup_{timestamp}"
            print(f"💾 Backing up existing model to: {backup_dir}")
            os.system(f"cp -r {self.existing_adapter} {backup_dir}")
        else:
            output_dir = f"{self.output_dir}_{timestamp}"
        
        print(f"💾 Output directory: {output_dir}")
        
        # Training arguments optimized for your ZGX nano
        print(f"\n⚙️  Configuring training arguments...")
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
        
        # Create trainer
        print(f"👨‍🏫 Creating SFT Trainer...")
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
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n🖥️  GPU: {gpu_name}")
            print(f"💾 VRAM: {gpu_memory:.1f} GB")
        
        # Train!
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING STARTED")
        print(f"{'='*70}\n")
        
        trainer.train()
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'='*70}")
        
        # Save final model
        print(f"\n💾 Saving final model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training metadata
        metadata = {
            'mode': self.mode,
            'base_model': self.base_model,
            'training_data': training_data_path,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'timestamp': timestamp,
            'dataset_size': len(dataset),
            'output_dir': output_dir
        }
        
        with open(f"{output_dir}/training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved to: {output_dir}")
        print(f"📊 Training metadata saved")
        
        # Show next steps
        print(f"\n{'='*70}")
        print(f"🎉 SUCCESS!")
        print(f"{'='*70}")
        print(f"\n💡 Next steps:")
        print(f"   1. Test inference:")
        print(f"      python3 scripts/model_inference_lora.py")
        print(f"   \n   2. Update trading bots to use new model:")
        print(f"      Edit scripts/model_inference_lora.py")
        print(f"      Change model path to: {output_dir}")
        print(f"   \n   3. Restart trading bots:")
        print(f"      sudo systemctl restart ai-trading-bot.service")
        print(f"      sudo systemctl restart ai-options-bot.service")
        print(f"{'='*70}\n")
        
        return output_dir

def main():
    parser = argparse.ArgumentParser(description='Fine-tune trading LLM')
    
    parser.add_argument('--data', type=str, 
                       required=True,
                       help='Path to training data JSON file')
    
    parser.add_argument('--continue-from', type=str,
                       default=None,
                       help='Path to existing adapter to continue training (optional)')
    
    parser.add_argument('--base-model', type=str,
                       default='unsloth/qwen2.5-32b-instruct-bnb-4bit',
                       help='Base model to use (only for fresh training)')
    
    parser.add_argument('--output', type=str,
                       default='finetune/finance_qwen_32b_lora',
                       help='Output directory for model')
    
    parser.add_argument('--epochs', type=int, 
                       default=3,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int,
                       default=2,
                       help='Per-device batch size')
    
    parser.add_argument('--learning-rate', type=float,
                       default=2e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Auto-detect existing model if not specified
    if args.continue_from is None:
        default_path = '/home/zgx/personal-projects/ai-trade-agent/finetune/finance_qwen_32b_lora'
        if Path(default_path).exists():
            print(f"🔍 Found existing model at: {default_path}")
            response = input(f"Continue training from this model? [Y/n]: ").strip().lower()
            if response != 'n':
                args.continue_from = default_path
    
    # Create trainer
    trainer = AdvancedModelTrainer(
        base_model=args.base_model,
        existing_adapter=args.continue_from,
        output_dir=args.output
    )
    
    # Train
    output_dir = trainer.train(
        training_data_path=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print(f"\n✅ All done! Model ready at: {output_dir}")

if __name__ == "__main__":
    main()