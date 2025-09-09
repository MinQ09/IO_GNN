# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data

# --- 프로젝트 모듈 임포트 ---
from model import IOGNN_Z, IOGNN_VA
from data_io import GraphWindowDataset, collate_window
from helper import inverse_transform_predictions
from run_single import is_identity_scaler


# ───────── 공통 유틸: Data 객체 추출 및 시퀀스 정규화 ─────────
def first_data(obj: Any) -> Data:
    if isinstance(obj, Data):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            try:
                d = first_data(item)
                if isinstance(d, Data):
                    return d
            except TypeError:
                continue
    raise TypeError(f"Expected Data, got {type(obj)}")

def sanitize_seq(seq_in: Any) -> List[Data]:
    if not isinstance(seq_in, (list, tuple)):
        raise TypeError(f"Sequence must be list/tuple, got {type(seq_in)}")
    return [first_data(item) for item in seq_in]

def first_tensor(obj: Any) -> torch.Tensor:
    """obj가 Tensor이면 그대로, tuple/list면 내부에서 첫 번째 Tensor를 찾아 반환."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for x in obj:
            if isinstance(x, torch.Tensor):
                return x
    raise TypeError(f"Expected a torch.Tensor (or container of Tensors), got {type(obj)}")

# ───────── 1) 시나리오 로더 ─────────
def load_arima_scenario(csv_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
        required = {'Year','industry_id'}
        if not required.issubset(df.columns):
            print(f"[Error] Missing required columns in scenario CSV: {required - set(df.columns)}")
            return None
        df['Year'] = df['Year'].astype(int)
        df['industry_id'] = df['industry_id'].astype(int)
        df = df.set_index(['Year','industry_id']).sort_index()
        print(f"[Scenario] Loaded '{csv_path}' with shape {df.shape}.")
        return df
    except Exception as e:
        print(f"[Error] Failed to load scenario: {e}")
        return None


# ───────── 2) Forecaster ─────────
@dataclass
class ForecasterConfig:
    node_feature_names: List[str]
    industry_id_order: List[int]
    batch_size: int = 1
    num_workers: int = 0
    window: int = 2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class IOGNNForecaster:
    def __init__(self, model_path: str, scalers_path: str, cfg: ForecasterConfig, kind: str = 'Z', cfg_like: Any = None) -> None:
        self.cfg = cfg
        self.cfg_like = cfg_like or cfg
        self.kind = kind.upper()
        self.device = torch.device(cfg.device)
        self.model_path = model_path
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
        self.model: Optional[torch.nn.Module] = None
        print(f"[Init] Loaded scalers from {scalers_path}.")

    def _build_model(self, nfeat: int, n_nodes: int):
        setattr(self.cfg_like, 'n_nodes', n_nodes)
        model = IOGNN_Z(nfeat=nfeat, cfg=self.cfg_like) if self.kind == 'Z' else IOGNN_VA(nfeat=nfeat, cfg=self.cfg_like)
        return model.to(self.device).eval()

    def _load_checkpoint_filtered(self):
        ckpt = torch.load(self.model_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        cur = self.model.state_dict()
        filt = {k: v for k, v in ckpt.items() if k in cur and cur[k].shape == v.shape}
        self.model.load_state_dict(filt, strict=False)
        print(f"[Init] Loaded {len(filt)}/{len(cur)} model weights from {self.model_path}.")

    def _ensure_model_built_from_sample(self, seq_any: Any):
        if self.model is not None:
            return
        seq = sanitize_seq(seq_any)
        last_g = seq[-1]
        self.model = self._build_model(nfeat=last_g.x.size(1), n_nodes=last_g.num_nodes)
        self._load_checkpoint_filtered()

    @staticmethod
    def save_single_prediction(result: Dict[str, Any], outdir: str, kind: str):
        out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)): return int(obj)
            if isinstance(obj, (np.float64, np.float32)): return float(obj)
            return obj

        # result는 {"year": int, "pred_scaled": np.ndarray, "pred_inverse": np.ndarray|None}
        fname = out / f"prediction_{kind}_{result['year']}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(result, f, default=convert_numpy, ensure_ascii=False, indent=2)
        print(f"[Save] Wrote single year prediction to {fname}")

    @staticmethod
    def save_multi_predictions(result: Dict[str, Any], outdir: str, kind: str):
        """ predict_upto_year의 반환(result) 전체(horizons 전부)를 한 파일에 저장 """
        out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)): return int(obj)
            if isinstance(obj, (np.float64, np.float32)): return float(obj)
            return obj

        years = [h["year"] for h in result["horizons"]]
        fname = out / f"predictions_{kind}_{min(years)}_{max(years)}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(result, f, default=convert_numpy, ensure_ascii=False, indent=2)
        print(f"[Save] Wrote multi-year predictions to {fname}")

    # --- 핵심: 마지막 입력 그래프에 시나리오 X를 주입하며 target_year까지 굴려서 예측 ---
    @torch.no_grad()
    def predict_upto_year(
        self,
        history_years: List[int],     # 예: [19, 20]
        target_year: int,             # 예: 28 (당신 체계에서 2030)
        scenario_df: pd.DataFrame,    # Year=21..28 포함
    ) -> Dict[str, Any]:

        # 1) 과거 윈도우 로드 (collate_window 사용)
        ds = GraphWindowDataset(
            years=history_years,
            cfg=self.cfg_like,
            scalers=self.scalers,
            fit_scalers=False,
            scale_targets=False,
        )
        if len(ds) == 0:
            raise RuntimeError("[Error] Empty dataset. Check data_dir and years contiguity.")

        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_window)
        print("[Check] len(ds) =", len(ds))
        seqs_list, tgts_list = next(iter(dl))
        print("[Check] types:", type(seqs_list), type(tgts_list))

        seq_raw = seqs_list[0]
        tgt_raw = tgts_list[0]
        print("[Check] seq_raw len =", len(seq_raw), "; elem types:",
            [type(x) for x in (seq_raw[:3] if isinstance(seq_raw, (list, tuple)) else [])])
        print("[Check] tgt_raw type =", type(tgt_raw))

        # 강제 정제
        seq = [x if isinstance(x, Data) else first_data(x) for x in seq_raw]
        tgt = tgt_raw if isinstance(tgt_raw, Data) else first_data(tgt_raw)
        print("[Check] seq elems all Data?", all(isinstance(g, Data) for g in seq))
        print("[Check] tgt is Data?", isinstance(tgt, Data))

        # 2) 모델 준비
        self._ensure_model_built_from_sample(seq)

        # 3) 시나리오 컬럼 검사
        missing_cols = [c for c in self.cfg.node_feature_names if c not in scenario_df.columns]
        if missing_cols:
            raise RuntimeError(f"[Error] Scenario is missing required feature columns: {missing_cols}")

        node_scaler = self.scalers.get('node', {}).get('node_features')
        use_identity = is_identity_scaler(node_scaler)

        # 디바이스 이동
        seq = [g.clone().to(self.device) for g in seq]
        tgt = tgt.clone().to(self.device)

        last_hist_year = max(history_years)
        if target_year <= last_hist_year:
            raise RuntimeError(f"[Error] target_year ({target_year}) must be > last history year ({last_hist_year}).")

        horizons: List[Dict[str, Any]] = []

        # 작은 헬퍼: 출력에서 Tensor 하나만 안전 추출
        def _first_tensor(obj: Any) -> torch.Tensor:
            if isinstance(obj, torch.Tensor):
                return obj
            if isinstance(obj, (list, tuple)):
                for it in obj:
                    if isinstance(it, torch.Tensor):
                        return it
            raise TypeError(f"Model output is not a Tensor: type={type(obj)}")

        # 4) last_hist_year+1 .. target_year 루프
        for year in range(last_hist_year + 1, target_year + 1):
            # (a) 해당 연도 시나리오 X 추출 + 산업 정렬
            try:
                df_y = scenario_df.xs(year, level='Year').reindex(self.cfg.industry_id_order)
            except KeyError:
                raise RuntimeError(f"[Error] Scenario for year {year} not found in CSV.")

            # (b) 학습 입력 피처 순서로 정렬
            df_y = df_y[self.cfg.node_feature_names]

            # (c) 스케일링 (학습과 동일하게 1e6으로 나눈 뒤 표준화)
            raw_vals = (df_y.values.astype(np.float32) / 1e6)
            if not use_identity:
                x_np = node_scaler.transform(raw_vals)
                scaler_name = "standard"
            else:
                x_np = raw_vals
                scaler_name = "identity"
            x_t = torch.from_numpy(x_np).float().to(self.device)

            # 디버그: 주입 전/후 확인
            before_snippet = seq[-1].x[:5, :len(self.cfg.node_feature_names)].detach().cpu().numpy()
            seq[-1].x = x_t  # (d) 마지막 입력 그래프의 X를 시나리오 값으로 교체
            after_snippet = seq[-1].x[:5, :len(self.cfg.node_feature_names)].detach().cpu().numpy()

            print(f"[Debug][inject year={year}] scaler={scaler_name} shape={seq[-1].x.shape} (n_nodes x n_feats)")
            try:
                df_stats = df_y.describe().loc[["mean", "std", "min", "max"]]
                print("[Debug] raw(df_y) stats (first few cols):")
                print(df_stats.iloc[:, :min(3, df_stats.shape[1])])
            except Exception:
                pass
            x_stats = {
                "mean": float(seq[-1].x.mean().item()),
                "std":  float(seq[-1].x.std().item()),
                "min":  float(seq[-1].x.min().item()),
                "max":  float(seq[-1].x.max().item()),
            }
            print("[Debug] injected x stats:", x_stats)
            print("[Debug] x before (top-5 nodes, first few feats):\n",
                before_snippet[:, :min(3, before_snippet.shape[1])])
            print("[Debug] x after  (top-5 nodes, first few feats):\n",
                after_snippet[:, :min(3, after_snippet.shape[1])])

            # (e) 모델 forward (학습 시그니처: ([seq], [tgt]))
            out = self.model([seq], [tgt])
            y_pred = _first_tensor(out)

            # (f) 역변환
            try:
                y_inv = inverse_transform_predictions(y_pred, self.scalers, kind=self.kind)
            except Exception:
                y_inv = None

            horizons.append({
                "year": int(year),
                "pred_scaled": y_pred.detach().cpu().numpy(),
                "pred_inverse": None if y_inv is None else y_inv.detach().cpu().numpy(),
            })

            # (g) 롤링: window=2 유지 (마지막 그래프를 복제해 한 칸 민다)
            next_g = seq[-1].clone()
            seq = [seq[-1].clone(), next_g]

        return {"history_years": history_years, "horizons": horizons}
# ───────── 3) CLI ─────────
def main() -> None:
    p = argparse.ArgumentParser(description="IO-GNN direct scenario forecasting (roll to target year)")
    p.add_argument('--model-path', type=str, required=True)
    p.add_argument('--scalers-path', type=str, required=True)
    p.add_argument('--config-module', type=str, required=True)
    p.add_argument('--kind', type=str, default='Z', choices=['Z', 'VA'])
    p.add_argument('--history-years', type=str, required=True, help='Last window of years, e.g., 19,20')
    p.add_argument('--target-year', type=int, required=True, help='Future index to predict up to, e.g., 28')
    p.add_argument('--scenario-path', type=str, required=True)
    p.add_argument('--outdir', type=str, required=True)
    p.add_argument('--save-all', action='store_true',
                   help='Save all horizons (e.g., 21..28) to a single JSON as well')
    args = p.parse_args()

    # 1) 시나리오 파일
    scenario_df = load_arima_scenario(args.scenario_path)
    if scenario_df is None:
        return

    # 2) Config 로드
    Config = __import__(args.config_module, fromlist=['Config']).Config
    cfg_mod = Config()

    fcfg = ForecasterConfig(
        node_feature_names=list(getattr(cfg_mod, 'node_feature_names')),
        industry_id_order=list(getattr(cfg_mod, 'industry_id_order')),
        window=getattr(cfg_mod, 'window', 2),
        device=getattr(cfg_mod, 'device', ('cuda' if torch.cuda.is_available() else 'cpu')),
    )

    history_years = [int(y.strip()) for y in args.history_years.split(',')]

    # 3) 예측기
    fore = IOGNNForecaster(
        model_path=args.model_path,
        scalers_path=args.scalers_path,
        cfg=fcfg,
        kind=args.kind,
        cfg_like=cfg_mod,
    )

    # 4) 타깃 연도까지 굴려서 예측
    result = fore.predict_upto_year(
        history_years=history_years,
        target_year=args.target_year,
        scenario_df=scenario_df,
    )

    if not result.get("horizons"):
        raise RuntimeError("No horizons produced by predict_upto_year().")

    for h in result["horizons"]:
        IOGNNForecaster.save_single_prediction(
            {"year": int(h["year"]),
             "pred_scaled": h["pred_scaled"],
             "pred_inverse": h["pred_inverse"]},
            args.outdir, args.kind
        )
    print(f"\n[Info] Saved {len(result['horizons'])} yearly predictions under: {args.outdir}")

    # 5) 마지막 연도만 별도 저장
    last = result["horizons"][-1]
    IOGNNForecaster.save_single_prediction(
        {"year": int(last["year"]),
         "pred_scaled": last["pred_scaled"],
         "pred_inverse": last["pred_inverse"]},
        args.outdir, args.kind
    )

    # 6) 전체 호라이즌도 저장하고 싶다면
    if args.save_all:
        IOGNNForecaster.save_multi_predictions(result, args.outdir, args.kind)

    print(f"\n[Done] Direct forecast up to year {args.target_year} saved under: {args.outdir}")


if __name__ == '__main__':
    main()