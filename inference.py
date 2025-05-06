import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd
import json
from torch.utils.data import DataLoader

# Assuming your project structure is something like:
# project_root/
#  ├── model/
#  │   ├── util.py
#  │   ├── database_util.py
#  │   ├── model.py
#  │   ├── dataset_2.py (this modified version)
#  │   ├── json.py
#  │   └── trainer.py (you might need to import collator from here or database_util)
#  ├── data/
#  │   ├── test_data.json
#  │   ├── column_min_max_vals.csv
#  │   └── (your trained model, e.g., results/best_model.pt)
#  └── inference.py (this script) or add to Training_My.ipynb

# Adjust imports based on your file locations and how you've structured 'model' package
from model.util import Normalizer, load_column_stats # Assuming load_column_stats is in util.py
from model.database_util import Encoding, collator # Ensure collator is accessible
from model.model import QueryFormer
from model.dataset_2 import PlanTreeDataset # Use the modified dataset

def run_inference(args, model_path, test_data_path, output_csv_path):
    """
    Runs inference on the test data using the trained model.

    Args:
        args (Args): Configuration arguments (similar to training).
        model_path (str): Path to the trained model checkpoint (.pt file).
        test_data_path (str): Path to the test_data.json file.
        output_csv_path (str): Path to save the output CSV file.
    """
    print(f"Using device: {args.device}")

    # 1. Load Column Stats & Initialize Encoding
    # This should be the same way it was done for training
    column_stats_csv_path = './data/column_min_max_vals.csv' # Make sure this path is correct
    if not os.path.exists(column_stats_csv_path):
        raise FileNotFoundError(f"Column stats file not found: {column_stats_csv_path}. "
                                "Ensure it's in the './data/' directory relative to this script, or update path.")
    column_min_max_vals, col2idx = load_column_stats(column_stats_csv_path)
    encoding = Encoding(
        column_min_max_vals=column_min_max_vals,
        col2idx=col2idx,
        op2idx={'>':0, '=':1, '<':2, 'NA':3} # Consistent with training
    )
    print("Encoding initialized.")

    # 2. Initialize Normalizer for Cardinality
    # CRITICAL: Use the *exact same* normalization parameters as during training.
    # The notebook used card_norm = Normalizer(1, 100).
    # If training actually learned min/max, those specific values must be used here.
    # For this example, sticking to the notebook's fixed values:
    card_norm = Normalizer(mini=1.0, maxi=100.0)
    print(f"Cardinality Normalizer initialized with min_log={card_norm.mini}, max_log={card_norm.maxi}")

    # (Dummy cost normalizer, not strictly needed if only predicting card and model returns 1 output)
    cost_norm = Normalizer(0,1)

    # 3. Initialize Model
    model = QueryFormer(
        emb_size=args.embed_size,
        ffn_dim=args.ffn_dim,
        head_size=args.head_size,
        dropout=args.dropout,
        n_layers=args.n_layers,
        use_sample=args.use_sample_in_model, # Use args for this
        use_hist=args.use_hist_in_model,   # Use args for this
        pred_hid=args.pred_hid,
        bin_number=50 # Default from model.py, ensure consistency
    )
    print("Model initialized.")

    # 4. Load Trained Model Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=torch.device(args.device))
    
    # Adjust key if model was saved directly or within a dict with 'model_state_dict' or 'model'
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint) # Assuming checkpoint is the state_dict itself

    model.to(args.device)
    model.eval() # Set model to evaluation mode
    print(f"Trained model weights loaded from {model_path}")

    # 5. Load Test Data
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_input_data = json.load(f)
    print(f"Loaded {len(test_input_data)} queries from {test_data_path}")

    # 6. Create Test Dataset
    # hist_file and table_sample are None because use_hist/use_sample are False
    test_ds = PlanTreeDataset(
        input_data=test_input_data,
        encoding=encoding,
        hist_file=None,
        card_norm=card_norm,
        cost_norm=cost_norm, # Dummy
        to_predict='card',   # Important for how labels might be structured if used
        table_sample=None,
        use_hist_in_model=args.use_hist_in_model, # Pass these to dataset
        use_sample_in_model=args.use_sample_in_model
    )
    print("Test dataset created.")

    # 7. Create DataLoader
    # The collator function expects a list of tuples, where each tuple is (data_item, label_item)
    # dataset.__getitem__ returns (collated_dict, (cost_label, card_label))
    # list(zip(*batch_from_loader)) will give [(collated_dict1, collated_dict2,...), ((cost_l1,card_l1), (cost_l2,card_l2)...)]
    # The collator in database_util.py is: collator(small_set) where small_set = list(zip(*batch_items))
    # It then takes small_set[0] for features and small_set[1] for labels.
    test_loader = DataLoader(
        test_ds,
        batch_size=args.bs, # Use batch size from args
        shuffle=False,      # DO NOT shuffle test data for consistent output mapping
        collate_fn=lambda b: collator(list(zip(*b)))
    )
    print("Test DataLoader created.")

    # 8. Inference Loop
    all_query_ids = test_ds.query_ids # Get all query IDs in order
    predicted_cardinalities_unnormalized = []

    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, (batch_data, batch_labels_dummy) in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            batch_data = batch_data.to(args.device)

            # Assuming model returns a single tensor for cardinality prediction
            # (as per model.py having the second pred head commented out)
            pred_normalized = model(batch_data)
            pred_normalized = pred_normalized.squeeze().cpu().numpy()

            # Unnormalize predictions
            # Handle case where pred_normalized might be a single float (if batch size was 1)
            if pred_normalized.ndim == 0:
                pred_normalized = np.array([pred_normalized])
            
            pred_unnormalized = card_norm.unnormalize_labels(pred_normalized)
            predicted_cardinalities_unnormalized.extend(pred_unnormalized)
    
    print("Inference complete.")

    # 9. Prepare and Save Output
    if len(all_query_ids) != len(predicted_cardinalities_unnormalized):
        print(f"Warning: Mismatch in number of query IDs ({len(all_query_ids)}) "
              f"and predictions ({len(predicted_cardinalities_unnormalized)}). "
              "CSV output might be affected. This could happen if DataLoader drops last batch.")
        # Adjust to the number of predictions made
        # all_query_ids = all_query_ids[:len(predicted_cardinalities_unnormalized)]


    output_df = pd.DataFrame({
        'Query ID': all_query_ids[:len(predicted_cardinalities_unnormalized)], # Ensure lists are same length for DataFrame
        'Predicted Cardinality': predicted_cardinalities_unnormalized
    })

    # Optional: Round predictions to nearest integer
    output_df['Predicted Cardinality'] = np.round(output_df['Predicted Cardinality']).astype(int)
    # Optional: Clip negative predictions to a minimum (e.g., 1 or 0)
    output_df['Predicted Cardinality'] = output_df['Predicted Cardinality'].clip(lower=1)


    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


if __name__ == '__main__':
    # Define arguments (mirroring your training Args class)
    class InferenceArgs:
        bs = 128  # Can be same as training or different for inference
        embed_size = 64
        pred_hid = 128
        ffn_dim = 128
        head_size = 12
        n_layers = 8
        dropout = 0.1 # Dropout is typically turned off by model.eval(), but params are needed for init
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # These should match the model's training configuration
        use_sample_in_model = False # As per notebook QueryFormer init
        use_hist_in_model = False   # As per notebook QueryFormer init

    inference_args = InferenceArgs()

    # --- Configuration for paths ---
    # Path to your trained model (e.g., the .pt file saved by trainer.py's logging)
    # You need to find the actual path to your best model from the training logs or args.newpath
    # For example, if your training saved it as './results/some_hash.pt'
    TRAINED_MODEL_PATH = './results/best_model.pt' # <--- !!! UPDATE THIS PATH !!!
    
    TEST_DATA_JSON_PATH = './data/test_data.json'
    OUTPUT_CSV_PATH = './predicted_cardinalities.csv'
    
    # Check if model path exists, provide a placeholder if not found for now
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"Warning: Trained model {TRAINED_MODEL_PATH} not found.")
        print("Please update TRAINED_MODEL_PATH to the correct .pt file.")
        # Example: Create a dummy model file for testing the script structure if you don't have one yet
        # This is NOT for actual inference.
        if not os.path.exists('./results'): os.makedirs('./results')
        # torch.save(QueryFormer(**vars(inference_args)).state_dict(), TRAINED_MODEL_PATH) # Remove after actual model is available
        print("Exiting. Please provide a valid model path.")
        exit()


    run_inference(inference_args, TRAINED_MODEL_PATH, TEST_DATA_JSON_PATH, OUTPUT_CSV_PATH)
