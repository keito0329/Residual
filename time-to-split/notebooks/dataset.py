import pandas as pd  
import numpy as np  
from src.preprocess.utils import dataset_stats, calculate_sequence_stats  
import os
  
def display_detailed_stats(data, dataset_name):  
    """詳細な統計情報を整形して表示"""  
    stats = dataset_stats(data, extended=True)  
      
    # 基本情報  
    print(f"\n=== {dataset_name} 基本統計 ===")  
    print(f"ユーザー数: {stats['n_users']:,}")  
    print(f"アイテム数: {stats['n_items']:,}")  
    print(f"インタラクション数: {stats['n_interactions']:,}")  
    print(f"密度: {stats['density']:.6f}")  
    print(f"平均シーケンス長: {stats['avg_seq_length']:.2f}")  
      
    # シーケンス長統計  
    print(f"\n=== シーケンス長統計 ===")  
    for key in ['seq_len_mean', 'seq_len_std', 'seq_len_min', 'seq_len_max', 'seq_len_median']:  
        if key in stats:  
            print(f"{key}: {stats[key]:.2f}")  
      
    # 時間統計  
    print(f"\n=== 時間統計 ===")  
    print(f"時間範囲（日数）: {stats['timestamp_range_in_days']:.1f}")  
    print(f"平均ユーザー期間（日数）: {stats['mean_user_duration']:.1f}")  
    print(f"中央値ユーザー期間（日数）: {stats['median_user_duration']:.1f}")  
  
# 使用例  
data_path = os.environ["SEQ_SPLITS_DATA_PATH"]  
data = pd.read_csv(os.path.join(data_path, "preprocessed", "Steam.csv"))  
display_detailed_stats(data, "Steam")