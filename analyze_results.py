# analyze_results.py ─────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_grid_results(csv_path: str = "./Results/grid_search/grid_search_results.csv"):
    """그리드 서치 결과 분석"""
    
    df = pd.read_csv(csv_path)
    
    # 1. 전체 통계
    print("=== Grid Search Results Summary ===")
    print(f"Total configurations: {df['config_id'].nunique()}")
    print(f"Tasks: {df['task'].unique()}")
    print(f"Best R²: {df['R2'].max():.4f}")
    print(f"Best RMSE: {df['RMSE'].min():.2f}")
    print(f"Total training time: {df['training_time'].sum():.2f} seconds")
    
    
    if 'CVR' in df.columns:
        z_df = df[df['task'] == 'Z']
        print(f"Best CVR (Z task): {z_df['CVR'].min():.4f}")
    
    # 2. 파라미터별 성능 분석
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    params = ['batch_size', 'lr', 'hidden', 'k', 'dropout', 'lambda_max']
    
    for i, param in enumerate(params):
        ax = axes[i//3, i%3]
        
        # 파라미터별 R² 분포
        param_performance = df.groupby(param)['R2'].agg(['mean', 'std']).reset_index()
        
        ax.errorbar(param_performance[param], param_performance['mean'], 
                   yerr=param_performance['std'], marker='o')
        ax.set_xlabel(param)
        ax.set_ylabel('R² (mean ± std)')
        ax.set_title(f'Performance vs {param}')
        
        if param == 'lr':
            ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('./Results/grid_search/parameter_analysis.png')
    plt.show()
    
    # 3. Top configurations
    print("\n=== Top 10 Configurations ===")
    top_configs = df.nlargest(10, 'R2')[['config_id', 'task', 'batch_size', 'lr', 'weight_decay', 
                                        'hidden', 'k', 'dropout', 'lambda_max', 'R2', 'RMSE']]
    print(top_configs.to_string(index=False))
    
    return df

if __name__ == "__main__":
    analyze_grid_results()
