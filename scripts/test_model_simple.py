import sys
sys.path.append('/home/zgx/personal-projects/ai-trade-agent/scripts')
from model_inference_lora import get_trading_decision, parse_decision

# Test with simpler, more direct prompts
test_cases = [
    {
        'name': 'Strong Buy Signal',
        'prompt': 'Ticker NVDA has RSI 35 (oversold), MACD positive crossover 1.2, low volatility 0.015, recent return +3.5%. Should I buy, sell, or hold? Predicted return and reasoning.'
    },
    {
        'name': 'Strong Sell Signal', 
        'prompt': 'Ticker AMD has RSI 82 (overbought), MACD negative crossover -0.8, high volatility 0.055, recent return -4.2%. Should I buy, sell, or hold? Predicted return and reasoning.'
    },
    {
        'name': 'Neutral Hold Signal',
        'prompt': 'Ticker MSFT has RSI 52 (neutral), MACD near zero 0.05, moderate volatility 0.022, recent return +0.3%. Should I buy, sell, or hold? Predicted return and reasoning.'
    }
]

for test in test_cases:
    print(f"\n{'='*70}")
    print(f"TEST: {test['name']}")
    print('='*70)
    
    response = get_trading_decision(test['prompt'])
    parsed = parse_decision(response)
    
    print(f"✓ Decision: {parsed['decision'].upper()}")
    print(f"✓ Predicted Return: {parsed['predicted_return']:.4f} ({parsed['predicted_return']*100:.2f}%)")
    print(f"✓ Confidence: {parsed['confidence']:.2f}")
    print(f"\n📝 Full Response:\n{parsed['raw_response']}")
