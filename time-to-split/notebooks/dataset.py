import pandas as pd  
import numpy as np  
from src.preprocess.utils import dataset_stats, calculate_sequence_stats  
import os
  
def display_detailed_stats(data, dataset_name):  
    """Format and display detailed statistics."""  
    stats = dataset_stats(data, extended=True)  
      
    # Basic stats
    print(f"\n=== {dataset_name} Basic Statistics ===")  
    print(f"Users: {stats['n_users']:,}")  
    print(f"Items: {stats['n_items']:,}")  
    print(f"Interactions: {stats['n_interactions']:,}")  
    print(f"Density: {stats['density']:.6f}")  
    print(f"Average sequence length: {stats['avg_seq_length']:.2f}")  
      
    # Sequence length stats
    print(f"\n=== Sequence Length Statistics ===")  
    for key in ['seq_len_mean', 'seq_len_std', 'seq_len_min', 'seq_len_max', 'seq_len_median']:  
        if key in stats:  
            print(f"{key}: {stats[key]:.2f}")  
      
    # Time stats
    print(f"\n=== Time Statistics ===")  
    print(f"Time range (days): {stats['timestamp_range_in_days']:.1f}")  
    print(f"Mean user duration (days): {stats['mean_user_duration']:.1f}")  
    print(f"Median user duration (days): {stats['median_user_duration']:.1f}")  
  
# Example usage
data_path = os.environ["SEQ_SPLITS_DATA_PATH"]  
data = pd.read_csv(os.path.join(data_path, "preprocessed", "Yelp.csv"))  
display_detailed_stats(data, "yelp")
