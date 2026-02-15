#!/usr/bin/env python3
"""
Model Inference with LoRA - OPTIMIZED with model caching
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re

# Global model cache - load once, use many times
_MODEL_CACHE = {
    'model': None,
    'tokenizer': None,
    'loaded': False
}

def load_model_once():
    """Load model into cache if not already loaded."""
    if _MODEL_CACHE['loaded']:
        print("✅ Using cached model (already loaded)")
        return _MODEL_CACHE['model'], _MODEL_CACHE['tokenizer']
    
    print("Loading base model and tokenizer...")
    base_model_path = "Qwen/Qwen2.5-32B-Instruct"
    lora_adapter_path = "/home/zgx/personal-projects/ai-trade-agent/finetune/finance_qwen_32b_lora"
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Loading LoRA adapter...")
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # Cache the model
    _MODEL_CACHE['model'] = model
    _MODEL_CACHE['tokenizer'] = tokenizer
    _MODEL_CACHE['loaded'] = True
    
    print("✅ Model loaded and cached successfully!")
    return model, tokenizer

def get_trading_decision(prompt, max_new_tokens=512, temperature=0.7):
    """
    Get trading decision from the model.
    Uses cached model for fast inference.
    """
    # Get cached model
    model, tokenizer = load_model_once()
    
    # Format prompt for Qwen
    messages = [
        {"role": "system", "content": "You are a professional stock trading analyst. Provide clear, actionable trading decisions with confidence scores."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def parse_decision(response):
    """Parse the model's response into structured decision."""
    response_lower = response.lower()
    
    # Extract decision (BUY, SELL, HOLD)
    decision = "hold"
    if "buy" in response_lower and "don't buy" not in response_lower and "not buy" not in response_lower:
        decision = "buy"
    elif "sell" in response_lower and "don't sell" not in response_lower and "not sell" not in response_lower:
        decision = "sell"
    
    # Extract confidence (look for numbers between 0 and 1)
    confidence = 0.5
    confidence_match = re.search(r'confidence[:\s]+(\d*\.?\d+)', response_lower)
    if confidence_match:
        try:
            conf_val = float(confidence_match.group(1))
            # Handle both 0-1 scale and 0-100 scale
            if conf_val > 1:
                conf_val = conf_val / 100
            confidence = max(0.0, min(1.0, conf_val))
        except:
            pass
    
    # Extract reasoning (first 200 chars of response)
    reasoning = response[:200].replace('\n', ' ').strip()
    
    return {
        'decision': decision,
        'confidence': confidence,
        'reasoning': reasoning,
        'raw_response': response
    }

if __name__ == "__main__":
    # Test the model
    test_prompt = """Analyze AAPL for trading:

Current Price: $225.50
RSI (14): 45.2
MACD: 1.23
Volume Ratio: 1.2x average
Price Change (100 bars): +2.5%

Discovery Signals: Momentum +15%, Near 52W high

Based on this data, should we BUY, SELL, or HOLD? Provide your decision, confidence (0-1), and reasoning."""
    
    print("Testing model inference...")
    print("="*70)
    
    response = get_trading_decision(test_prompt)
    print("Raw Response:")
    print(response)
    print("="*70)
    
    decision = parse_decision(response)
    print("\nParsed Decision:")
    print(f"Decision: {decision['decision'].upper()}")
    print(f"Confidence: {decision['confidence']:.2f}")
    print(f"Reasoning: {decision['reasoning']}")
