import ollama
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_modelfile(model_dir='finance_qwen_gguf'):
    """Create Modelfile for Ollama."""
    modelfile_content = f"""FROM {model_dir}/unsloth.Q4_K_M.gguf

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
"""
    
    modelfile_path = os.path.join(model_dir, 'Modelfile')
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    logging.info(f"Created Modelfile at {modelfile_path}")
    return modelfile_path

def load_tuned_model(model_dir='finance_qwen_gguf', model_name='finance_qwen_32b_tuned'):
    """Load the tuned GGUF model into Ollama."""
    try:
        # Check if GGUF file exists
        gguf_path = os.path.join(model_dir, 'unsloth.Q4_K_M.gguf')
        if not os.path.exists(gguf_path):
            logging.error(f"GGUF file not found at {gguf_path}")
            logging.error("Make sure to run the GGUF conversion in fine_tune_llm.py first!")
            return False
        
        # Create Modelfile
        modelfile_path = create_modelfile(model_dir)
        
        # Create model in Ollama using subprocess (more reliable)
        result = subprocess.run(
            ['ollama', 'create', model_name, '-f', modelfile_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logging.info(f"✅ Successfully loaded model '{model_name}' into Ollama")
            return True
        else:
            logging.error(f"Failed to create model: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return False

def test_inference(model_name='finance_qwen_32b_tuned'):
    """Test the loaded model with a sample prompt."""
    test_prompt = """Analyze AAPL with the following data:
    - RSI: 65.3
    - MACD: 0.45
    - Volatility: 0.023
    - Recent return: 0.015
    
    Provide decision (buy/sell/hold), predicted return, and reasoning."""
    
    try:
        response = ollama.generate(model=model_name, prompt=test_prompt)
        logging.info("=== Test Inference Result ===")
        logging.info(response['response'])
        return response['response']
    except Exception as e:
        logging.error(f"Inference test failed: {e}")
        return None

def run_inference(prompt, model='finance_qwen_32b_tuned'):
    """Run inference with the tuned model for trading logic."""
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    print("Loading tuned model into Ollama...")
    if load_tuned_model():
        print("\nTesting inference...")
        test_inference()
    else:
        print("Failed to load model. Check logs above.")