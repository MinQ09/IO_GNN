import pandas as pd
from pathlib import Path
from config import Config
from data_io import _safe_read_csv
import argparse

def concat_csvs(years, data_dir):
    dfs = []
    for y in years:
        path = Path(data_dir) / f"X_{y}.csv"
        df = _safe_read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def safe_read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, thousands=',').replace(['–','-',''], pd.NA)
    return df.fillna(0.0)

def concat_z_values(years, data_dir: Path) -> pd.Series:
    series_list = []
    for y in years:
        df = safe_read_csv(data_dir / f"Zf_{y}.csv")
        vals = df.iloc[:, 2] if df.shape[1] > 2 else df.iloc[:, -1]
        series_list.append(vals)
    return pd.concat(series_list, ignore_index=True)

def concat_a_values(years, data_dir: Path) -> pd.Series:
    series_list = []
    for y in years:
        df = safe_read_csv(data_dir / f"Af_{y}.csv")
        vals = df.iloc[:, 2] if df.shape[1] > 2 else df.iloc[:, -1]
        series_list.append(vals)
    return pd.concat(series_list, ignore_index=True)

def check_nans(label, df_or_series):
    total_nans = df_or_series.isna().sum().sum() if isinstance(df_or_series, pd.DataFrame) else df_or_series.isna().sum()
    print(f"{label}: total NaNs = {total_nans}")
    return total_nans

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_stats.csv", help="output CSV for stats")
    args = p.parse_args()

    cfg = Config()

    years = list(range(1, 73))
    tr_y, vl_y, ts_y = years[:-8], years[-8:-4], years[-4:]

    # X matrices
    train_df = concat_csvs(tr_y, cfg.data_dir)
    val_df   = concat_csvs(vl_y, cfg.data_dir)
    test_df  = concat_csvs(ts_y, cfg.data_dir)

    # NaN 체크
    total_nans = 0
    total_nans += check_nans("train_df", train_df)
    total_nans += check_nans("val_df",   val_df)
    total_nans += check_nans("test_df",  test_df)

    # 통계량 계산
    stats = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        stats[name] = df.describe().T[["mean", "std", "min", "max"]]
    result = pd.concat(stats, axis=1)
    print(result)
    result.to_csv(args.out)

    # Z matrices
    train_z = concat_z_values(tr_y, cfg.data_dir)
    val_z   = concat_z_values(vl_y, cfg.data_dir)
    test_z  = concat_z_values(ts_y, cfg.data_dir)

    check_nans("train_z", train_z)
    check_nans("val_z",   val_z)
    check_nans("test_z",  test_z)

    check_nans("train_z", train_z)
    print(f"train_z dtype: {train_z.dtype}")

    stats_z = pd.DataFrame({
        'train': train_z.describe(),
        'val':   val_z.describe(),
        'test':  test_z.describe(),
    })[['train','val','test']]
    print(stats_z.to_string())
    stats_z.to_csv('z_matrix_stats.csv')

    # A matrices
    train_a = concat_a_values(tr_y, cfg.data_dir)
    val_a   = concat_a_values(vl_y, cfg.data_dir)
    test_a  = concat_a_values(ts_y, cfg.data_dir)

    check_nans("train_a", train_a)
    check_nans("val_a",   val_a)
    check_nans("test_a",  test_a)
    
    check_nans("train_a", train_a)
    print(f"train_a dtype: {train_a.dtype}")

    stats_a = pd.DataFrame({
        'train': train_a.describe(),
        'val':   val_a.describe(),
        'test':  test_a.describe(),
    })[['train','val','test']]
    print(stats_a.to_string())
    stats_a.to_csv('a_matrix_stats.csv')

    if total_nans > 0:
        print("⚠️ NaN 값이 발견되었습니다. 상세 확인이 필요합니다.")
    else:
        print("✅ NaN 값이 없습니다.")
