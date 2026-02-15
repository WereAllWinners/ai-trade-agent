import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "/home/zgx/personal-projects/ai-trade-agent/finetune/finance_qwen_gguf"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_trading_decision(prompt):
    """Get trading decision from the fine-tuned model."""
    # Simple format without chat template
    full_prompt = f"""You are a financial trading AI expert. Provide reasoned decisions based on data.

{prompt}

Response:"""
    
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            min_new_tokens=50,  # Force minimum response length
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode and remove the input prompt
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part after "Response:"
    if "Response:" in response:
        response = response.split("Response:")[-1].strip()
    
    return response

if __name__ == "__main__":
    # Test with simpler prompt
    test_prompt = """Analyze AAPL stock:
- RSI: 65.3 (neutral to overbought)
- MACD: 0.45 (bullish signal)
- Volatility: 0.023 (moderate)
- Recent return: 0.015 (positive)

Provide:
1. Decision: buy, sell, or hold
2. Predicted return: numerical value
3. Reasoning: your analysis"""
    
    print("\nTesting model...\n")
    response = get_trading_decision(test_prompt)
    print("="*60)
    print("MODEL RESPONSE:")
    print("="*60)
    print(response)
    print("="*60)
