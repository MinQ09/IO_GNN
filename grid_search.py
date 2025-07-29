# grid_search.py ─────────────────────────────────────────────────────────────
"""
Grid search runner for IO-GNN hyperparameter optimization.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from typing import Dict, Any, List
import warnings
import time
warnings.filterwarnings('ignore')

from config import Config
from run_single import run_single


def run_single_config(config_and_id):
    """단일 config 실행 (multiprocessing용)"""
    config, config_id = config_and_id
    
    try:
        print(f"\n{'='*60}")
        print(f"Running Config {config_id}: {config.get_param_string()}")
        print(f"{'='*60}")
        
        results = {}
        for kind in ["Z"]:  # VA도 원하면 ["Z", "VA"]로 변경
            try:
                start_time = time.time()
                model, hist, _, metrics = run_single(config, seed=config.seeds[0], kind=kind)
                end_time = time.time()
                
                # 결과 저장
                results[kind] = {
                    'metrics': metrics,
                    'best_val_loss': min(hist['val_tot']) if hist['val_tot'] else float('inf'),
                    'final_train_loss': hist['train_tot'][-1] if hist['train_tot'] else float('inf'),
                    'final_val_R2': hist['val_R2'][-1] if hist['val_R2'] else 0.0,
                    'final_val_CVR': hist['val_CVR'][-1] if kind == "Z" and hist.get('val_CVR') else None,
                    'training_time': end_time - start_time,
                    'epochs_trained': len(hist['train_tot']) if hist['train_tot'] else 0,
                }
                
                print(f"✅ {kind} completed - Test R²: {metrics.get('R2', 'N/A'):.4f}, "
                      f"CVR: {metrics.get('CVR', 'N/A'):.4f}, Time: {end_time-start_time:.1f}s")
                
            except Exception as e:
                print(f"❌ {kind} failed: {str(e)}")
                results[kind] = {'error': str(e)}
        
        # Config 정보와 함께 반환
        return {
            'config_id': config_id,
            'config_params': {
                'batch_size': config.batch_size,
                'lr': config.lr,
                'weight_decay': config.weight_decay,
                'hidden': config.hidden,
                'k': config.k,
                'dropout': config.dropout,
                'lambda_max': config.lambda_max,
                'seed': config.seeds[0]
            },
            'results': results
        }
        
    except Exception as e:
        print(f"❌ Config {config_id} completely failed: {str(e)}")
        return {
            'config_id': config_id,
            'config_params': {
                'batch_size': getattr(config, 'batch_size', 'N/A'),
                'lr': getattr(config, 'lr', 'N/A'),
                'weight_decay': getattr(config, 'weight_decay', 'N/A'),
                'hidden': getattr(config, 'hidden', 'N/A'),
                'k': getattr(config, 'k', 'N/A'),
                'dropout': getattr(config, 'dropout', 'N/A'),
                'lambda_max': getattr(config, 'lambda_max', 'N/A'),
                'seed': getattr(config, 'seeds', [0])[0]
            },
            'results': {'error': str(e)}
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='grid_config.yaml', 
                       help='Config file path')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel jobs')
    parser.add_argument('--kinds', nargs='+', default=['Z'],
                       choices=['Z', 'VA'], help='Tasks to run')
    args = parser.parse_args()
    
    # Config 로드 또는 생성 (안전한 fallback)
    if Path(args.config).exists():
        base_config = Config.load(args.config)
    else:
        print(f"⚠️  Config file '{args.config}' not found! Using default configuration.")
        # 기본 설정으로 Config 생성
        base_config = Config(
            batch_size=64,  # 필수 필드에 기본값 제공
            data_dir=Path("./Data"),
            out_dir=Path("./Results/grid_search"),
        )
    
    # 그리드 서치 활성화
    base_config.grid_search = True
    
    # 그리드 설정 생성
    configs = base_config.generate_grid_configs()
    print(f"\n🚀 Starting grid search with {len(configs)} configurations")
    print(f"📊 Using {args.n_jobs} parallel jobs")
    
    # 결과 저장 디렉토리
    results_dir = Path("./Results/grid_search")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 병렬 실행
    config_with_ids = [(config, i) for i, config in enumerate(configs)]
    
    start_time = time.time()
    
    if args.n_jobs == 1:
        # 순차 실행
        all_results = []
        for i, config_data in enumerate(config_with_ids):
            print(f"\n⏳ Progress: {i+1}/{len(config_with_ids)}")
            result = run_single_config(config_data)
            all_results.append(result)
    else:
        # 병렬 실행
        with mp.Pool(args.n_jobs) as pool:
            all_results = pool.map(run_single_config, config_with_ids)
    
    total_time = time.time() - start_time
    print(f"\n🎉 All configurations completed in {total_time:.1f} seconds")
    
    # 결과 정리 및 저장
    results_summary = []
    
    for result in all_results:
        if 'error' in result['results']:
            # 에러 케이스도 기록
            error_row = {
                'config_id': result['config_id'],
                'task': 'ERROR',
                **result['config_params'],
                'error': result['results']['error']
            }
            results_summary.append(error_row)
            continue
            
        for kind in ['Z', 'VA']:
            if kind not in result['results'] or 'error' in result['results'][kind]:
                continue
                
            row = {
                'config_id': result['config_id'],
                'task': kind,
                **result['config_params'],
                **result['results'][kind]['metrics'],
                'best_val_loss': result['results'][kind]['best_val_loss'],
                'final_train_loss': result['results'][kind]['final_train_loss'],
                'final_val_R2': result['results'][kind]['final_val_R2'],
                'training_time': result['results'][kind]['training_time'],
                'epochs_trained': result['results'][kind]['epochs_trained'],
            }
            
            if kind == 'Z':
                row['final_val_CVR'] = result['results'][kind]['final_val_CVR']
            
            results_summary.append(row)
    
    # DataFrame으로 저장
    df = pd.DataFrame(results_summary)
    
    if not df.empty:
        csv_path = results_dir / "grid_search_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n📄 Results saved to: {csv_path}")
        
        # 성공한 실험만 필터링
        success_df = df[df['task'] != 'ERROR'].copy()
        
        if not success_df.empty:
            print(f"\n📊 Successfully completed: {len(success_df)} runs")
            print(f"❌ Failed runs: {len(df) - len(success_df)}")
            
            # 상위 결과 출력 (R² 기준)
            print(f"\n🏆 Top 5 Results (by R²):")
            top_r2 = success_df.nlargest(5, 'R2')[['config_id', 'task', 'batch_size', 'lr', 'hidden', 'k', 'R2', 'RMSE', 'training_time']]
            print(top_r2.to_string(index=False))
            
            # CVR 기준 (Z task만)
            z_results = success_df[success_df['task'] == 'Z'].copy()
            if not z_results.empty and 'CVR' in z_results.columns:
                print(f"\n🎯 Top 5 Results (by CVR - lower is better):")
                top_cvr = z_results.nsmallest(5, 'CVR')[['config_id', 'batch_size', 'lr', 'hidden', 'k', 'R2', 'CVR', 'training_time']]
                print(top_cvr.to_string(index=False))
            
            # 최고 성능 조합 추천
            if not z_results.empty:
                best_overall = z_results.loc[z_results['R2'].idxmax()]
                print(f"\n🥇 Best Overall Configuration:")
                print(f"   Config ID: {best_overall['config_id']}")
                print(f"   Parameters: bs={best_overall['batch_size']}, lr={best_overall['lr']:.1e}, "
                      f"wd={best_overall['weight_decay']:.1e}, hidden={best_overall['hidden']}, k={best_overall['k']}")
                print(f"   Performance: R²={best_overall['R2']:.4f}, CVR={best_overall.get('CVR', 'N/A'):.4f}, "
                      f"RMSE={best_overall['RMSE']:.0f}")
        else:
            print("❌ No successful runs found!")
    
    # 상세 결과 JSON 저장
    json_path = results_dir / "grid_search_detailed.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Grid search completed! Check {results_dir} for detailed results.")


if __name__ == "__main__":
    main()
