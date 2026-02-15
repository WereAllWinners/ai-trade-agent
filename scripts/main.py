import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main pipeline for data collection and preparation.
    Note: Fine-tuning is done separately via fine_tune_llm.py
    """
    
    # Check if data already exists
    data_dir = "data/finance_tuning"
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        logging.info(f"Data already exists in {data_dir}")
        user_input = input("Regenerate data? (y/n): ")
        if user_input.lower() != 'y':
            logging.info("Using existing data. Run fine_tune_llm.py to train.")
            return
    
    # Generate new data using your existing data_collection.py
    logging.info("Starting data collection pipeline...")
    from data_collection import generate_training_examples, save_dataset
    
    chunks = generate_training_examples(num_examples=50000, chunk_size=10000)
    save_dataset(chunks)
    
    logging.info(f"✅ Data collection complete! {len(chunks)} chunks saved.")
    logging.info("Next step: Run 'python fine_tune_llm.py' to start training")

if __name__ == "__main__":
    main()