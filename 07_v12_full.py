
import os, gc, time, warnings, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

T        = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED     = 42

t0 = time.time()
def log(m):
    print(f'[{time.time()-t0:.0f}s] {m}', flush=True)
log('환경 OK')

# ═════════════════════════════════════════════════════════════════
# 1. features.npz 로드
# ═════════════════════════════════════════════════════════════════
log('Loading features.npz ...')
d = np.load('features.npz', allow_pickle=True)
X         = d['X']
X_te      = d['X_te']
y_raw     = d['y_raw'].astype(np.float64)
y_log     = d['y_log']
y_sqrt    = d['y_sqrt']
groups    = d['groups']
test_ids  = d['test_ids']
v4_feat_names     = d['feature_cols'].tolist() if 'feature_cols' in d.files else []
layout_cluster_tr = d['layout_cluster_tr']
layout_cluster_te = d['layout_cluster_te']
timeslot_tr       = d['timeslot_tr']
timeslot_te       = d['timeslot_te']

N_tr, N_f = X.shape
N_te      = X_te.shape[0]
Y_MAX     = float(y_raw.max())
Y_MEAN    = float(y_raw.mean())
log(f'  X: {X.shape}   X_te: {X_te.shape}   y_mean: {Y_MEAN:.4f}')

# ═════════════════════════════════════════════════════════════════
# 2. CV 분할 — 3가지 fold-seed
# ═════════════════════════════════════════════════════════════════
def make_group_folds(groups_arr, n_splits=N_SPLITS, seed=None):
    if seed is None:
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(np.arange(len(groups_arr)), groups=groups_arr))
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups_arr).copy()
    rng.shuffle(uniq)
    fold_map = {g: i % n_splits for i, g in enumerate(uniq)}
    fold_ids = np.array([fold_map[g] for g in groups_arr])
    folds = []
    for k in range(n_splits):
        vai = np.where(fold_ids == k)[0]
        tri = np.where(fold_ids != k)[0]
        folds.append((tri, vai))
    return folds

folds_main = make_group_folds(groups, N_SPLITS, seed=None)
folds_alt  = make_group_folds(groups, N_SPLITS, seed=2026)
folds_alt2 = make_group_folds(groups, N_SPLITS, seed=2027)
ALL_FOLDS  = [folds_main, folds_alt, folds_alt2]

log(f'folds_main: {[len(v) for _,v in folds_main]}')
log(f'folds_alt:  {[len(v) for _,v in folds_alt]}')
log(f'folds_alt2: {[len(v) for _,v in folds_alt2]}')

# ═════════════════════════════════════════════════════════════════
# 3. 트레이너 헬퍼 (v8 동일)
# ═════════════════════════════════════════════════════════════════
def train_lgb(X_tr, X_ts, y, params, name, nr, folds=None, pp=np.expm1):
    folds = folds or folds_main
    oof  = np.zeros(N_tr, dtype=np.float32)
    pred = np.zeros(N_te, dtype=np.float32)
    n_folds = len(folds)
    for fold,(tri,vai) in enumerate(folds):
        dt = lgb.Dataset(X_tr[tri], y[tri])
        dv = lgb.Dataset(X_tr[vai], y[vai])
        m  = lgb.train(params, dt, num_boost_round=nr, valid_sets=[dv],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[vai] = pp(m.predict(X_tr[vai])).astype(np.float32)
        pred    += pp(m.predict(X_ts)).astype(np.float32) / n_folds
        del m; gc.collect()
    log(f'  [{name}] OOF: {mean_absolute_error(y_raw, oof):.5f}')
    return oof, pred

def train_lgb_seed(X_tr, X_ts, y, params, name, nr, folds, seed, pp=np.expm1):
    p = {**params, 'seed': seed, 'bagging_seed': seed, 'feature_fraction_seed': seed}
    return train_lgb(X_tr, X_ts, y, p, f'{name}_s{seed}', nr, folds, pp)

def train_xgb(X_tr, X_ts, y, params, name, nr, folds=None, pp=np.expm1):
    folds = folds or folds_main
    oof  = np.zeros(N_tr, dtype=np.float32)
    pred = np.zeros(N_te, dtype=np.float32)
    n_folds = len(folds)
    for fold,(tri,vai) in enumerate(folds):
        dt  = xgb.DMatrix(X_tr[tri], label=y[tri])
        dv  = xgb.DMatrix(X_tr[vai], label=y[vai])
        dts = xgb.DMatrix(X_ts)
        m   = xgb.train(params, dt, num_boost_round=nr,
                        evals=[(dv,'va')], early_stopping_rounds=50, verbose_eval=0)
        oof[vai] = pp(m.predict(dv)).astype(np.float32)
        pred    += pp(m.predict(dts)).astype(np.float32) / n_folds
        del m,dt,dv,dts; gc.collect()
    log(f'  [{name}] OOF: {mean_absolute_error(y_raw, oof):.5f}')
    return oof, pred

def train_xgb_seed(X_tr, X_ts, y, params, name, nr, folds, seed, pp=np.expm1):
    p = {**params, 'seed': seed}
    return train_xgb(X_tr, X_ts, y, p, f'{name}_s{seed}', nr, folds, pp)

def train_cat(X_tr, X_ts, y, params, name, folds=None, pp=np.expm1):
    folds = folds or folds_main
    oof  = np.zeros(N_tr, dtype=np.float32)
    pred = np.zeros(N_te, dtype=np.float32)
    n_folds = len(folds)
    for fold,(tri,vai) in enumerate(folds):
        m = CatBoostRegressor(**params)
        m.fit(Pool(X_tr[tri], y[tri]), eval_set=Pool(X_tr[vai], y[vai]),
              early_stopping_rounds=50, verbose=0)
        oof[vai] = pp(m.predict(X_tr[vai])).astype(np.float32)
        pred    += pp(m.predict(X_ts)).astype(np.float32) / n_folds
        del m; gc.collect()
    log(f'  [{name}] OOF: {mean_absolute_error(y_raw, oof):.5f}')
    return oof, pred

def train_cat_seed(X_tr, X_ts, y, params, name, folds, seed, pp=np.expm1):
    p = {**params, 'random_seed': seed}
    return train_cat(X_tr, X_ts, y, p, f'{name}_s{seed}', folds, pp)

def train_hgb(X_tr, X_ts, y, name, folds=None, pp=np.expm1):
    folds = folds or folds_main
    oof  = np.zeros(N_tr, dtype=np.float32)
    pred = np.zeros(N_te, dtype=np.float32)
    n_folds = len(folds)
    for fold,(tri,vai) in enumerate(folds):
        m = HistGradientBoostingRegressor(
            loss='absolute_error', learning_rate=0.05, max_iter=600,
            max_depth=None, max_leaf_nodes=63, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=None, n_iter_no_change=30, tol=1e-4,
            random_state=42)
        m.fit(X_tr[tri], y[tri])
        oof[vai] = pp(m.predict(X_tr[vai])).astype(np.float32)
        pred    += pp(m.predict(X_ts)).astype(np.float32) / n_folds
        del m; gc.collect()
    log(f'  [{name}] OOF: {mean_absolute_error(y_raw, oof):.5f}')
    return oof, pred

def train_dart(X_tr, X_ts, y, params, name, nr, folds=None, pp=np.expm1):
    folds = folds or folds_main
    oof  = np.zeros(N_tr, dtype=np.float32)
    pred = np.zeros(N_te, dtype=np.float32)
    n_folds = len(folds)
    for fold,(tri,vai) in enumerate(folds):
        dt = lgb.Dataset(X_tr[tri], y[tri])
        dv = lgb.Dataset(X_tr[vai], y[vai])
        m  = lgb.train(params, dt, num_boost_round=nr, valid_sets=[dv],
                       callbacks=[lgb.log_evaluation(0)])
        oof[vai] = pp(m.predict(X_tr[vai])).astype(np.float32)
        pred    += pp(m.predict(X_ts)).astype(np.float32) / n_folds
        del m; gc.collect()
    log(f'  [{name}] OOF: {mean_absolute_error(y_raw, oof):.5f}')
    return oof, pred

# ── Multi-fold-seed 헬퍼 ──────────────────────────────────────
# ★ CRITICAL: 다른 fold-seed 분할은 서로 다른 validation 인덱스를 가짐.
#   naive np.mean은 0이 채워진 인덱스를 포함하므로 WRONG.
#   대신 인덱스 기반 누적 평균을 사용.
def multi_fold_avg(train_fn, *args, fold_list=None, **kwargs):
    """여러 fold-seed 분할로 학습한 뒤 OOF/pred 평균. 인덱스 기반 안전 평균."""
    fold_list = fold_list or ALL_FOLDS
    oof_sum   = np.zeros(N_tr, dtype=np.float64)
    oof_count = np.zeros(N_tr, dtype=np.int32)
    pred_sum  = np.zeros(N_te, dtype=np.float64)
    for i, folds in enumerate(fold_list):
        o, p = train_fn(*args, folds=folds, **kwargs)
        # OOF: fold의 validation 인덱스에만 유효한 값이 있음
        for _, vai in folds:
            oof_sum[vai]   += o[vai].astype(np.float64)
            oof_count[vai] += 1
        pred_sum += p.astype(np.float64)
    oof_avg  = (oof_sum / np.maximum(oof_count, 1)).astype(np.float32)
    pred_avg = (pred_sum / len(fold_list)).astype(np.float32)
    log(f'  [multi-fold-avg] {len(fold_list)} folds → OOF: {mean_absolute_error(y_raw, oof_avg):.5f}')
    return oof_avg, pred_avg

log('헬퍼 정의 완료')

# ═════════════════════════════════════════════════════════════════
# 4. 하이퍼파라미터 (v6/v8 100% 동일)
# ═════════════════════════════════════════════════════════════════
P_Q50 = dict(objective='quantile', alpha=0.5, metric='mae', learning_rate=0.03,
    num_leaves=127, min_child_samples=30, feature_fraction=0.65, bagging_fraction=0.85,
    bagging_freq=1, reg_alpha=0.1, reg_lambda=0.5, verbose=-1, n_jobs=-1, seed=SEED)
P_MAE = dict(objective='regression_l1', metric='mae', learning_rate=0.03,
    num_leaves=95, min_child_samples=30, feature_fraction=0.70, bagging_fraction=0.85,
    bagging_freq=1, reg_alpha=0.1, reg_lambda=0.5, verbose=-1, n_jobs=-1)
P_HUB = dict(objective='huber', huber_delta=1.0, metric='mae', learning_rate=0.05,
    num_leaves=63, min_child_samples=60, feature_fraction=0.55, bagging_fraction=0.7,
    bagging_freq=5, reg_alpha=0.3, reg_lambda=3.0, verbose=-1, n_jobs=-1, seed=SEED)
P_XGB = dict(objective='reg:absoluteerror', eval_metric='mae', learning_rate=0.05,
    max_depth=8, min_child_weight=50, subsample=0.7, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=5.0, tree_method='hist', n_jobs=-1, verbosity=0, seed=SEED)
P_SQ  = dict(objective='quantile', alpha=0.5, metric='mae', learning_rate=0.05,
    num_leaves=63, min_child_samples=50, feature_fraction=0.6, bagging_fraction=0.7,
    bagging_freq=5, reg_alpha=0.5, reg_lambda=5.0, verbose=-1, n_jobs=-1)
P_TW  = dict(objective='tweedie', tweedie_variance_power=1.5, metric='mae', learning_rate=0.03,
    num_leaves=95, min_child_samples=40, feature_fraction=0.65, bagging_fraction=0.8,
    bagging_freq=3, reg_alpha=0.2, reg_lambda=2.0, verbose=-1, n_jobs=-1)
P_Q55 = dict(objective='quantile', alpha=0.55, metric='mae', learning_rate=0.03,
    num_leaves=127, min_child_samples=30, feature_fraction=0.65, bagging_fraction=0.85,
    bagging_freq=1, reg_alpha=0.1, reg_lambda=0.5, verbose=-1, n_jobs=-1)
P_FAIR = dict(objective='fair', fair_c=1.0, metric='mae', learning_rate=0.03,
    num_leaves=95, min_child_samples=40, feature_fraction=0.65, bagging_fraction=0.8,
    bagging_freq=3, reg_alpha=0.2, reg_lambda=2.0, verbose=-1, n_jobs=-1)
P_CAT = dict(loss_function='MAE', iterations=3000, learning_rate=0.03,
    depth=8, l2_leaf_reg=5.0, random_seed=SEED, thread_count=-1,
    bootstrap_type='Bernoulli', subsample=0.8, rsm=0.65)
P_DART = dict(objective='quantile', alpha=0.5, metric='mae',
    boosting_type='dart', learning_rate=0.05, num_leaves=95, min_child_samples=40,
    feature_fraction=0.7, bagging_fraction=0.8, drop_rate=0.1, skip_drop=0.5, max_drop=50,
    reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=-1, seed=SEED)

P_META_Q = dict(objective='quantile', alpha=0.5, metric='mae', learning_rate=0.03,
    num_leaves=63, min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=3, reg_alpha=0.3, reg_lambda=3.0, verbose=-1, n_jobs=-1, seed=SEED)
P_META_M = dict(objective='regression_l1', metric='mae', learning_rate=0.03,
    num_leaves=47, min_child_samples=60, feature_fraction=0.6, bagging_fraction=0.75,
    bagging_freq=5, reg_alpha=0.5, reg_lambda=5.0, verbose=-1, n_jobs=-1, seed=SEED)
P_META_X = dict(objective='reg:absoluteerror', eval_metric='mae', learning_rate=0.03,
    max_depth=6, min_child_weight=60, subsample=0.7, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=5.0, tree_method='hist', n_jobs=-1, verbosity=0, seed=SEED)
log('하이퍼파라미터 OK')

# ═════════════════════════════════════════════════════════════════
# 5. Feature filter (v6/v8 동일)
# ═════════════════════════════════════════════════════════════════
log('Predictive-importance feature filter ...')
imp = np.zeros(X.shape[1])
for fold,(tri,vai) in enumerate(folds_main):
    dt = lgb.Dataset(X[tri], y_log[tri])
    dv = lgb.Dataset(X[vai], y_log[vai])
    p  = dict(objective='quantile', alpha=0.5, metric='mae', learning_rate=0.05,
              num_leaves=63, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=5, verbose=-1, n_jobs=-1)
    m = lgb.train(p, dt, num_boost_round=500, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    imp += m.feature_importance(importance_type='gain')
imp /= N_SPLITS
thr = np.quantile(imp, 0.30)
keep_mask = imp > thr
if keep_mask.sum() < int(0.5 * X.shape[1]):
    keep_mask = imp > 0
log(f'  kept {keep_mask.sum()}/{X.shape[1]} features')
X_f, X_te_f = X[:, keep_mask], X_te[:, keep_mask]

# ═════════════════════════════════════════════════════════════════
# 6. G1 베이스 — 3-fold-seed 평균 (v12 변경점 1)
# ═════════════════════════════════════════════════════════════════
log('\n========== G1 (3-fold-seed avg) ==========')
oof_g1q, pred_g1q = multi_fold_avg(
    train_lgb, X, X_te, y_log, P_Q50, 'g1_q50', 4000)
oof_g1m, pred_g1m = multi_fold_avg(
    train_lgb, X, X_te, y_log, P_MAE, 'g1_mae', 4000)
oof_g1x, pred_g1x = multi_fold_avg(
    train_xgb, X, X_te, y_log, P_XGB, 'g1_xgb', 500)

# ═════════════════════════════════════════════════════════════════
# 7. G2 기본 모델 — 3-fold-seed 평균
# ═════════════════════════════════════════════════════════════════
log('\n========== G2 (3-fold-seed avg) ==========')
# g2_mae_dup, g2_sqrt 은 fold-seed avg 적용
oof_g2m, pred_g2m = multi_fold_avg(
    train_lgb, X, X_te, y_log, P_MAE, 'g2_mae_dup', 4000)
oof_g2sq, pred_g2sq = multi_fold_avg(
    train_lgb, X, X_te, y_sqrt, P_SQ, 'g2_sqrt', 500,
    pp=lambda x: np.square(np.clip(x, 0, None)))

# ═════════════════════════════════════════════════════════════════
# 8. SEED AVG — 5 seeds × 3 fold-seeds (v12 극대화)
# ═════════════════════════════════════════════════════════════════
log('\n========== SEED AVG (5 seeds × 3 fold-seeds) ==========')
SEEDS = [42, 2024, 7, 11, 99]

def seed_x_fold_avg_lgb(X_tr, X_ts, y, params, name, nr, pp=np.expm1):
    """5-seed × 3-fold-seed = 15 학습 → 인덱스 기반 안전 평균"""
    oof_sum   = np.zeros(N_tr, dtype=np.float64)
    oof_count = np.zeros(N_tr, dtype=np.int32)
    pred_sum  = np.zeros(N_te, dtype=np.float64)
    n_runs = 0
    for folds in ALL_FOLDS:
        for seed in SEEDS:
            o, p = train_lgb_seed(X_tr, X_ts, y, params, name, nr, folds, seed, pp)
            for _, vai in folds:
                oof_sum[vai]   += o[vai].astype(np.float64)
                oof_count[vai] += 1
            pred_sum += p.astype(np.float64)
            n_runs += 1
    oof_avg  = (oof_sum / np.maximum(oof_count, 1)).astype(np.float32)
    pred_avg = (pred_sum / n_runs).astype(np.float32)
    log(f'  [{name}_sa5x3] OOF: {mean_absolute_error(y_raw, oof_avg):.5f}  ({n_runs} runs)')
    return oof_avg, pred_avg

def seed_x_fold_avg_xgb(X_tr, X_ts, y, params, name, nr, pp=np.expm1):
    """5-seed × 3-fold-seed = 15 학습 → 인덱스 기반 안전 평균"""
    oof_sum   = np.zeros(N_tr, dtype=np.float64)
    oof_count = np.zeros(N_tr, dtype=np.int32)
    pred_sum  = np.zeros(N_te, dtype=np.float64)
    n_runs = 0
    for folds in ALL_FOLDS:
        for seed in SEEDS:
            o, p = train_xgb_seed(X_tr, X_ts, y, params, name, nr, folds, seed, pp)
            for _, vai in folds:
                oof_sum[vai]   += o[vai].astype(np.float64)
                oof_count[vai] += 1
            pred_sum += p.astype(np.float64)
            n_runs += 1
    oof_avg  = (oof_sum / np.maximum(oof_count, 1)).astype(np.float32)
    pred_avg = (pred_sum / n_runs).astype(np.float32)
    log(f'  [{name}_sa5x3] OOF: {mean_absolute_error(y_raw, oof_avg):.5f}  ({n_runs} runs)')
    return oof_avg, pred_avg

# q50 seed avg (핵심 모델)
oof_g2q_sa, pred_g2q_sa = seed_x_fold_avg_lgb(X, X_te, y_log, P_Q50, 'g2_q50', 4000)
# hub seed avg
oof_g2h_sa, pred_g2h_sa = seed_x_fold_avg_lgb(X, X_te, y_log, P_HUB, 'g2_hub', 500)
# xgb seed avg
oof_g2x_sa, pred_g2x_sa = seed_x_fold_avg_xgb(X, X_te, y_log, P_XGB, 'g2_xgb', 500)

# ═════════════════════════════════════════════════════════════════
# 9. DIVERSITY — seed avg 확장 (v12 변경점 2)
# ═════════════════════════════════════════════════════════════════
log('\n========== DIVERSITY (3-seed × 3-fold-seed avg) ==========')
SEEDS_DIV = [42, 2024, 7]  # diversity 모델은 3-seed로 충분

def seed_x_fold_avg_cat(X_tr, X_ts, y, params, name, pp=np.expm1):
    """CatBoost: 3-seed × 3-fold-seed = 9 학습 → 인덱스 기반 안전 평균"""
    oof_sum   = np.zeros(N_tr, dtype=np.float64)
    oof_count = np.zeros(N_tr, dtype=np.int32)
    pred_sum  = np.zeros(N_te, dtype=np.float64)
    n_runs = 0
    for folds in ALL_FOLDS:
        for seed in SEEDS_DIV:
            o, p = train_cat_seed(X_tr, X_ts, y, params, name, folds, seed, pp)
            for _, vai in folds:
                oof_sum[vai]   += o[vai].astype(np.float64)
                oof_count[vai] += 1
            pred_sum += p.astype(np.float64)
            n_runs += 1
    oof_avg  = (oof_sum / np.maximum(oof_count, 1)).astype(np.float32)
    pred_avg = (pred_sum / n_runs).astype(np.float32)
    log(f'  [{name}_sa3x3] OOF: {mean_absolute_error(y_raw, oof_avg):.5f}  ({n_runs} runs)')
    return oof_avg, pred_avg

# cat_mae: 3-seed × 3-fold
oof_cat, pred_cat = seed_x_fold_avg_cat(X, X_te, y_log, P_CAT, 'cat_mae')

# ── lgb diversity: 인덱스 기반 안전 평균 적용 ──
def _safe_seed_fold_avg_lgb(X_tr, X_ts, y, base_params, name, nr, folds_list, seeds, pp=np.expm1):
    """LGB: seed × fold-seed 학습 → 인덱스 기반 안전 평균"""
    oof_sum   = np.zeros(N_tr, dtype=np.float64)
    oof_count = np.zeros(N_tr, dtype=np.int32)
    pred_sum  = np.zeros(N_te, dtype=np.float64)
    n_runs = 0
    for folds in folds_list:
        for seed in seeds:
            p = {**base_params, 'seed': seed, 'bagging_seed': seed, 'feature_fraction_seed': seed}
            o, pr = train_lgb(X_tr, X_ts, y, p, f'{name}_s{seed}', nr, folds, pp)
            for _, vai in folds:
                oof_sum[vai]   += o[vai].astype(np.float64)
                oof_count[vai] += 1
            pred_sum += pr.astype(np.float64)
            n_runs += 1
    oof_avg  = (oof_sum / np.maximum(oof_count, 1)).astype(np.float32)
    pred_avg = (pred_sum / n_runs).astype(np.float32)
    log(f'  [{name}_sa{len(seeds)}x{len(folds_list)}] OOF: {mean_absolute_error(y_raw, oof_avg):.5f}  ({n_runs} runs)')
    return oof_avg, pred_avg

# lgb_twee: 3-seed × 3-fold (tweedie uses y_raw, pp=identity)
oof_tw, pred_tw = _safe_seed_fold_avg_lgb(
    X, X_te, y_raw.astype(np.float32), P_TW, 'lgb_twee', 3000, ALL_FOLDS, SEEDS_DIV, pp=lambda x: x)

# lgb_q55: 3-seed × 3-fold
oof_q55, pred_q55 = _safe_seed_fold_avg_lgb(
    X, X_te, y_log, P_Q55, 'lgb_q55', 4000, ALL_FOLDS, SEEDS_DIV)

# lgb_fair: 3-seed × 3-fold
oof_fair, pred_fair = _safe_seed_fold_avg_lgb(
    X, X_te, y_log, P_FAIR, 'lgb_fair', 2000, ALL_FOLDS, SEEDS_DIV)

# ═════════════════════════════════════════════════════════════════
# 10. STABLE (filtered features) — 3-fold-seed avg
# ═════════════════════════════════════════════════════════════════
log('\n========== STABLE (3-fold-seed avg) ==========')
oof_sq_f, pred_sq_f = multi_fold_avg(
    train_lgb, X_f, X_te_f, y_log, P_Q50, 'q50_stable', 4000)
oof_sh_f, pred_sh_f = multi_fold_avg(
    train_lgb, X_f, X_te_f, y_log, P_HUB, 'hub_stable', 500)
oof_sx_f, pred_sx_f = multi_fold_avg(
    train_xgb, X_f, X_te_f, y_log, P_XGB, 'xgb_stable', 500)

# ═════════════════════════════════════════════════════════════════
# 11. HGB / DART / MLP — 3-fold-seed avg
# ═════════════════════════════════════════════════════════════════
log('\n========== HGB / DART (3-fold-seed avg) ==========')
oof_hgb, pred_hgb = multi_fold_avg(
    train_hgb, X, X_te, y_log, 'hgb_v4')
oof_dart, pred_dart = multi_fold_avg(
    train_dart, X, X_te, y_log, P_DART, 'dart_v4', 1500)

# ── MLP ──
log('\n========== MLP (3-fold-seed avg) ==========')
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f'  torch device: {DEVICE}')
SQRT_MAX = float(np.sqrt(Y_MAX) * 1.2)

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, 128),     nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(128, 64),      nn.LayerNorm(64),  nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def train_mlp_robust(X_tr, X_ts, y, name, folds=None, epochs=100, batch=1024, lr=1e-3, patience=10):
    folds = folds or folds_main
    Xc  = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    Xtc = np.nan_to_num(X_ts, nan=0.0, posinf=0.0, neginf=0.0)
    oof  = np.zeros(N_tr, dtype=np.float32)
    pred = np.zeros(N_te, dtype=np.float32)
    n_folds = len(folds)
    for fold,(tri,vai) in enumerate(folds):
        q_lo = np.quantile(Xc[tri], 0.01, axis=0); q_hi = np.quantile(Xc[tri], 0.99, axis=0)
        Xt = np.clip(Xc[tri], q_lo, q_hi); Xv = np.clip(Xc[vai], q_lo, q_hi)
        Xs = np.clip(Xtc, q_lo, q_hi)
        sc = StandardScaler().fit(Xt)
        Xt = sc.transform(Xt).astype(np.float32)
        Xv = sc.transform(Xv).astype(np.float32)
        Xs = sc.transform(Xs).astype(np.float32)
        Xt_t = torch.tensor(Xt); yt_t = torch.tensor(y[tri].astype(np.float32))
        Xv_t = torch.tensor(Xv).to(DEVICE); Xs_t = torch.tensor(Xs).to(DEVICE)
        ds = TensorDataset(Xt_t, yt_t); dl = DataLoader(ds, batch_size=batch, shuffle=True)
        model = MLP(X_tr.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.L1Loss()
        best_mae = float('inf'); best_v = None; best_s = None; bad = 0
        for ep in range(epochs):
            model.train()
            for xb,yb in dl:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); p = model(xb); ls = loss_fn(p,yb); ls.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pv = model(Xv_t).cpu().numpy()
                pv_raw = np.clip(pv, 0, SQRT_MAX) ** 2
                cur = mean_absolute_error(y_raw[vai], pv_raw)
                if cur < best_mae - 1e-5:
                    best_mae = cur; best_v = pv_raw.copy()
                    ps = model(Xs_t).cpu().numpy()
                    best_s = (np.clip(ps, 0, SQRT_MAX) ** 2).copy(); bad = 0
                else:
                    bad += 1
                    if bad >= patience: break
        oof[vai] = best_v.astype(np.float32)
        pred    += best_s.astype(np.float32) / n_folds
        del model, opt, sched, Xt_t, Xv_t, Xs_t, dl, ds; gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    log(f'  [{name}] OOF: {mean_absolute_error(y_raw, oof):.5f}')
    return oof, pred

oof_mlp, pred_mlp = multi_fold_avg(
    train_mlp_robust, X, X_te, y_sqrt, 'mlp_sqrt_v4')

# ═════════════════════════════════════════════════════════════════
# 12. 베이스 18 모델 조합
# ═════════════════════════════════════════════════════════════════
base_names = ['g1_q50','g1_mae','g1_xgb',
              'g2_q50_sa','g2_mae_dup','g2_hub_sa','g2_xgb_sa','g2_sqrt',
              'cat_mae','lgb_twee','lgb_q55','lgb_fair',
              'q50_stable','hub_stable','xgb_stable']

base_oof = np.column_stack([
    oof_g1q, oof_g1m, oof_g1x,
    oof_g2q_sa, oof_g2m, oof_g2h_sa, oof_g2x_sa, oof_g2sq,
    oof_cat, oof_tw, oof_q55, oof_fair,
    oof_sq_f, oof_sh_f, oof_sx_f]).astype(np.float64)
base_pred = np.column_stack([
    pred_g1q, pred_g1m, pred_g1x,
    pred_g2q_sa, pred_g2m, pred_g2h_sa, pred_g2x_sa, pred_g2sq,
    pred_cat, pred_tw, pred_q55, pred_fair,
    pred_sq_f, pred_sh_f, pred_sx_f]).astype(np.float64)

# Sanity check + HGB/DART/MLP 추가
THRESH = Y_MAX * 3
log('\nSanity check (HGB/DART/MLP):')
extra = []
for nm,oo,pp in [('hgb_v4', oof_hgb, pred_hgb),
                  ('lgb_dart_v4', oof_dart, pred_dart),
                  ('mlp_sqrt_v4', oof_mlp, pred_mlp)]:
    mx = float(pp.max()); ok = mx <= THRESH
    log(f'  {nm:14s}  mean={pp.mean():.3f}  max={mx:.1f}  '
        f'OOF={mean_absolute_error(y_raw, oo):.5f}{"" if ok else "  ✗ EXCLUDED"}')
    if ok: extra.append((nm, oo, pp))

for n,oo,pp in extra:
    base_names.append(n)
    base_oof  = np.hstack([base_oof,  oo.reshape(-1,1)])
    base_pred = np.hstack([base_pred, pp.reshape(-1,1)])

N_BASE = len(base_names)
log(f'\nTotal base ensemble: {N_BASE} models')
for i,n in enumerate(base_names):
    log(f'  {n:18s}  OOF={mean_absolute_error(y_raw, base_oof[:,i]):.5f}')

# ═════════════════════════════════════════════════════════════════
# 13. Linear SLSQP 블렌드
# ═════════════════════════════════════════════════════════════════
def mae_obj(w, M, y):
    w = np.clip(w, 0, None); s = w.sum()
    if s < 1e-9: return 1e9
    return mean_absolute_error(y, M @ (w/s))

log('\n=== Linear SLSQP base blend ===')
nm = N_BASE
w0 = np.ones(nm)/nm
bds = [(0,1)]*nm; cns = [{'type':'eq','fun': lambda w: w.sum()-1}]
res_g = minimize(mae_obj, w0, args=(base_oof, y_raw),
                 method='SLSQP', bounds=bds, constraints=cns,
                 options=dict(maxiter=3000, ftol=1e-10))
w_lin = np.clip(res_g.x, 0, None); w_lin /= w_lin.sum()
linear_oof_mae = mean_absolute_error(y_raw, base_oof @ w_lin)
linear_oof  = (base_oof  @ w_lin).astype(np.float32)
linear_pred = (base_pred @ w_lin).astype(np.float32)
log(f'  Linear SLSQP OOF MAE: {linear_oof_mae:.5f}')
log('  weights (>0.001):')
for n, w in zip(base_names, w_lin):
    if w > 0.001: log(f'    {n:20s}  w={w:.4f}')

# ═════════════════════════════════════════════════════════════════
# 14. 메타 피처 (v8 100% 동일)
# ═════════════════════════════════════════════════════════════════
def build_meta_features(base_preds, layout_cluster, timeslot, X_orig):
    feats = {}
    for i, name in enumerate(base_names):
        feats[f'pred_{name}'] = base_preds[:, i]
    feats['ens_mean']   = base_preds.mean(axis=1)
    feats['ens_std']    = base_preds.std(axis=1)
    feats['ens_median'] = np.median(base_preds, axis=1)
    feats['ens_min']    = base_preds.min(axis=1)
    feats['ens_max']    = base_preds.max(axis=1)
    feats['ens_range']  = feats['ens_max'] - feats['ens_min']
    ranks = np.argsort(np.argsort(base_preds, axis=1), axis=1)
    feats['rank_of_median_model'] = ranks[:, N_BASE // 2].astype(np.float32)
    feats['argmax_model'] = np.argmax(base_preds, axis=1).astype(np.float32)
    feats['argmin_model'] = np.argmin(base_preds, axis=1).astype(np.float32)
    feats['ens_q25'] = np.quantile(base_preds, 0.25, axis=1)
    feats['ens_q75'] = np.quantile(base_preds, 0.75, axis=1)
    feats['ens_iqr'] = feats['ens_q75'] - feats['ens_q25']
    feats['ens_skew'] = (feats['ens_mean'] - feats['ens_median']) / (feats['ens_std'] + 1e-9)
    feats['disagreement_x_level'] = feats['ens_std'] * feats['ens_mean']
    feats['spread_indicator']     = feats['ens_max'] - feats['ens_min']
    feats['cv'] = feats['ens_std'] / (feats['ens_mean'] + 1e-9)
    feats['layout_cluster'] = layout_cluster.astype(np.float32)
    feats['timeslot']       = timeslot.astype(np.float32)
    n_keep = min(30, X_orig.shape[1])
    for j in range(n_keep):
        fname = v4_feat_names[j] if j < len(v4_feat_names) else f'v4_f{j}'
        feats[f'ctx_{fname}'] = X_orig[:, j].astype(np.float32)
    sorted_idx = np.argsort(w_lin)[::-1]
    for ii in range(min(5, N_BASE)):
        for jj in range(ii + 1, min(5, N_BASE)):
            a, b = sorted_idx[ii], sorted_idx[jj]
            feats[f'diff_{base_names[a]}__{base_names[b]}'] = base_preds[:, a] - base_preds[:, b]
    df = pd.DataFrame(feats)
    return df.values.astype(np.float32), list(df.columns)

X_meta_tr, meta_names = build_meta_features(base_oof,  layout_cluster_tr, timeslot_tr, X)
X_meta_te, _          = build_meta_features(base_pred, layout_cluster_te, timeslot_te, X_te)
log(f'  X_meta_tr: {X_meta_tr.shape}   X_meta_te: {X_meta_te.shape}')

np.savez_compressed(
    'v12_meta_checkpoint.npz',
    X_meta_tr=X_meta_tr.astype(np.float32),
    X_meta_te=X_meta_te.astype(np.float32),
    meta_names=np.array(meta_names, dtype=object),
    base_names=np.array(base_names, dtype=object),
    base_oof=base_oof.astype(np.float32),
    base_pred=base_pred.astype(np.float32),
    linear_oof=linear_oof.astype(np.float32),
    linear_pred=linear_pred.astype(np.float32),
    linear_oof_mae=np.float64(linear_oof_mae),
    w_linear=w_lin.astype(np.float64),
    simple_mean_oof=base_oof.mean(axis=1).astype(np.float32),
    simple_mean_pred=base_pred.mean(axis=1).astype(np.float32),
    y_raw=y_raw.astype(np.float32),
)
log('  saved -> v12_meta_checkpoint.npz')

# ═════════════════════════════════════════════════════════════════
# 15. 메타 러너 3개 — 3-fold-seed avg 적용
# ═════════════════════════════════════════════════════════════════
def train_meta_lgb(X_tr, X_ts, y, params, name, nr=2000, folds=None):
    folds = folds or folds_main
    oof  = np.zeros(N_tr, dtype=np.float32)
    pred = np.zeros(N_te, dtype=np.float32)
    n_folds = len(folds)
    for fold,(tri,vai) in enumerate(folds):
        dt = lgb.Dataset(X_tr[tri], y[tri])
        dv = lgb.Dataset(X_tr[vai], y[vai])
        m  = lgb.train(params, dt, num_boost_round=nr, valid_sets=[dv],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[vai] = m.predict(X_tr[vai]).astype(np.float32)
        pred    += m.predict(X_ts).astype(np.float32) / n_folds
        del m; gc.collect()
    log(f'  [{name}] OOF: {mean_absolute_error(y, oof):.5f}')
    return oof, pred

def train_meta_xgb(X_tr, X_ts, y, params, name, nr=1000, folds=None):
    folds = folds or folds_main
    oof  = np.zeros(N_tr, dtype=np.float32)
    pred = np.zeros(N_te, dtype=np.float32)
    n_folds = len(folds)
    for fold,(tri,vai) in enumerate(folds):
        dt  = xgb.DMatrix(X_tr[tri], label=y[tri])
        dv  = xgb.DMatrix(X_tr[vai], label=y[vai])
        dts = xgb.DMatrix(X_ts)
        m   = xgb.train(params, dt, num_boost_round=nr,
                        evals=[(dv,'va')], early_stopping_rounds=50, verbose_eval=0)
        oof[vai] = m.predict(dv).astype(np.float32)
        pred    += m.predict(dts).astype(np.float32) / n_folds
        del m,dt,dv,dts; gc.collect()
    log(f'  [{name}] OOF: {mean_absolute_error(y, oof):.5f}')
    return oof, pred

log('\n=== META LEARNERS (3-fold-seed avg — 인덱스 기반 안전 평균) ===')

def _safe_meta_fold_avg(train_fn, X_tr, X_ts, y, params, name, **kwargs):
    """메타 러너도 인덱스 기반 안전 평균 적용"""
    oof_sum   = np.zeros(N_tr, dtype=np.float64)
    oof_count = np.zeros(N_tr, dtype=np.int32)
    pred_sum  = np.zeros(N_te, dtype=np.float64)
    for folds in ALL_FOLDS:
        o, p = train_fn(X_tr, X_ts, y, params, name, folds=folds, **kwargs)
        for _, vai in folds:
            oof_sum[vai]   += o[vai].astype(np.float64)
            oof_count[vai] += 1
        pred_sum += p.astype(np.float64)
    oof_avg  = (oof_sum / np.maximum(oof_count, 1)).astype(np.float32)
    pred_avg = (pred_sum / len(ALL_FOLDS)).astype(np.float32)
    log(f'  [{name}_3fold] OOF: {mean_absolute_error(y, oof_avg):.5f}')
    return oof_avg, pred_avg

oof_mq, pred_mq = _safe_meta_fold_avg(train_meta_lgb, X_meta_tr, X_meta_te, y_raw.astype(np.float64), P_META_Q, 'meta_q50')
oof_mm, pred_mm = _safe_meta_fold_avg(train_meta_lgb, X_meta_tr, X_meta_te, y_raw.astype(np.float64), P_META_M, 'meta_mae')
oof_mx, pred_mx = _safe_meta_fold_avg(train_meta_xgb, X_meta_tr, X_meta_te, y_raw.astype(np.float64), P_META_X, 'meta_xgb')

# ═════════════════════════════════════════════════════════════════
# 16. 메타 SLSQP 블렌드
# ═════════════════════════════════════════════════════════════════
meta_oofs  = np.column_stack([oof_mq,  oof_mm,  oof_mx ]).astype(np.float64)
meta_preds = np.column_stack([pred_mq, pred_mm, pred_mx]).astype(np.float64)

w0_m = np.ones(3) / 3
bds_m = [(0,1)] * 3
cns_m = [{'type':'eq','fun': lambda w: w.sum()-1}]
res_m = minimize(mae_obj, w0_m, args=(meta_oofs, y_raw),
                 method='SLSQP', bounds=bds_m, constraints=cns_m,
                 options=dict(maxiter=2000, ftol=1e-10))
w_meta = np.clip(res_m.x, 0, None); w_meta /= w_meta.sum()
meta_blend_oof  = (meta_oofs @ w_meta).astype(np.float32)
meta_blend_pred = (meta_preds @ w_meta).astype(np.float32)
meta_blend_mae  = mean_absolute_error(y_raw, meta_blend_oof)
log(f'\nMeta SLSQP blend OOF: {meta_blend_mae:.5f}')
for n,w in zip(['meta_q50','meta_mae','meta_xgb'], w_meta):
    log(f'  {n:14s}  w={w:.4f}')

# ═════════════════════════════════════════════════════════════════
# 17. v12 변경점 3: 3-candidate SLSQP hedge
# ═════════════════════════════════════════════════════════════════
log('\n=== HEDGE: 3-candidate SLSQP (v12 핵심) ===')

# Simple mean (균등 평균 — 가장 robust한 baseline)
simple_mean_oof  = base_oof.mean(axis=1).astype(np.float32)
simple_mean_pred = base_pred.mean(axis=1).astype(np.float32)

# (a) v6 50/50 대조군
hedged50_oof  = (0.5 * meta_blend_oof + 0.5 * linear_oof).astype(np.float32)
hedged50_pred = (0.5 * meta_blend_pred + 0.5 * linear_pred).astype(np.float32)
hedged50_mae  = mean_absolute_error(y_raw, hedged50_oof)

# (b) v8 style 2-candidate SLSQP
two_oof  = np.column_stack([linear_oof, meta_blend_oof]).astype(np.float64)
two_pred = np.column_stack([linear_pred, meta_blend_pred]).astype(np.float64)
res2 = minimize(mae_obj, np.array([0.5, 0.5]), args=(two_oof, y_raw),
                method='SLSQP', bounds=[(0,1),(0,1)],
                constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                options=dict(maxiter=2000, ftol=1e-11))
w_h2 = np.clip(res2.x, 0, None); w_h2 /= w_h2.sum()
hedged2_oof  = (two_oof @ w_h2).astype(np.float32)
hedged2_pred = (two_pred @ w_h2).astype(np.float32)
hedged2_mae  = mean_absolute_error(y_raw, hedged2_oof)

# (c) v12 3-candidate SLSQP: linear + meta + simple_mean
three_oof  = np.column_stack([linear_oof, meta_blend_oof, simple_mean_oof]).astype(np.float64)
three_pred = np.column_stack([linear_pred, meta_blend_pred, simple_mean_pred]).astype(np.float64)
res3 = minimize(mae_obj, np.array([0.4, 0.3, 0.3]), args=(three_oof, y_raw),
                method='SLSQP', bounds=[(0,1),(0,1),(0,1)],
                constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                options=dict(maxiter=3000, ftol=1e-11))
w_h3 = np.clip(res3.x, 0, None); w_h3 /= w_h3.sum()
hedged3_oof  = (three_oof @ w_h3).astype(np.float32)
hedged3_pred = (three_pred @ w_h3).astype(np.float32)
hedged3_mae  = mean_absolute_error(y_raw, hedged3_oof)

log(f'  Linear SLSQP only   OOF: {linear_oof_mae:.5f}')
log(f'  Meta blend only     OOF: {meta_blend_mae:.5f}')
log(f'  Simple mean only    OOF: {mean_absolute_error(y_raw, simple_mean_oof):.5f}')
log(f'  v6 50/50 hedged     OOF: {hedged50_mae:.5f}')
log(f'  v8 2-cand SLSQP     OOF: {hedged2_mae:.5f}  (linear={w_h2[0]:.3f}, meta={w_h2[1]:.3f})')
log(f'  v12 3-cand SLSQP    OOF: {hedged3_mae:.5f}  (linear={w_h3[0]:.3f}, meta={w_h3[1]:.3f}, mean={w_h3[2]:.3f})')

# 최적 hedge 선택
best_hedge_name = 'v12_3cand'
best_hedge_oof  = hedged3_oof
best_hedge_pred = hedged3_pred
best_hedge_mae  = hedged3_mae

# 안전장치: 3-cand 가 2-cand 보다 OOF 나쁘면 2-cand 사용
if hedged3_mae > hedged2_mae + 0.001:
    log(f'  ⚠ 3-cand({hedged3_mae:.5f}) > 2-cand({hedged2_mae:.5f}) → 2-cand fallback')
    best_hedge_name = 'v8_2cand'
    best_hedge_oof  = hedged2_oof
    best_hedge_pred = hedged2_pred
    best_hedge_mae  = hedged2_mae

# ═════════════════════════════════════════════════════════════════
# 18. 캘리브레이션 + 제출
# ═════════════════════════════════════════════════════════════════
def calibrate_mean_scale(pred, target_mean, clip_max):
    pp = np.clip(pred, 0, clip_max).astype(np.float32)
    scale = float(target_mean / max(pp.mean(), 1e-9))
    return np.clip(pp * scale, 0, None).astype(np.float32), scale

CLIP_MAX = Y_MAX * 1.5
ss = pd.read_csv('data/sample_submission.csv')

def save_sub(pred, fname, tag=''):
    out = ss.copy(); out[T] = pred
    out.to_csv(fname, index=False)
    log(f'  {fname:35s}  mean={pred.mean():.4f}  std={pred.std():.4f}  max={pred.max():.1f}  | {tag}')

log('\n=== SUBMISSION ===')

# (1) MAIN: v12 3-candidate SLSQP hedge
p_main, s_main = calibrate_mean_scale(best_hedge_pred, Y_MEAN, CLIP_MAX)
save_sub(p_main, 'sub_v12_main.csv',
         f'★ MAIN ({best_hedge_name}, scale={s_main:.4f}, OOF={best_hedge_mae:.5f})')

# (2) 50/50 hedge 대조군
p_50, s_50 = calibrate_mean_scale(hedged50_pred, Y_MEAN, CLIP_MAX)
save_sub(p_50, 'sub_v12_50hedge.csv',
         f'50/50 hedge (scale={s_50:.4f}, OOF={hedged50_mae:.5f})')

# (3) 2-candidate hedge (v8 방식)
p_2h, s_2h = calibrate_mean_scale(hedged2_pred, Y_MEAN, CLIP_MAX)
save_sub(p_2h, 'sub_v12_2hedge.csv',
         f'2-cand hedge (scale={s_2h:.4f}, OOF={hedged2_mae:.5f})')

np.savez_compressed(
    'v12_full_oof_bundle.npz',
    best_hedge_name=np.array([best_hedge_name]),
    best_hedge_oof=best_hedge_oof.astype(np.float32),
    best_hedge_pred=best_hedge_pred.astype(np.float32),
    best_hedge_mae=np.float64(best_hedge_mae),
    linear_oof=linear_oof.astype(np.float32),
    linear_pred=linear_pred.astype(np.float32),
    linear_oof_mae=np.float64(linear_oof_mae),
    meta_blend_oof=meta_blend_oof.astype(np.float32),
    meta_blend_pred=meta_blend_pred.astype(np.float32),
    meta_blend_mae=np.float64(meta_blend_mae),
    simple_mean_oof=simple_mean_oof.astype(np.float32),
    simple_mean_pred=simple_mean_pred.astype(np.float32),
    hedged50_oof=hedged50_oof.astype(np.float32),
    hedged50_pred=hedged50_pred.astype(np.float32),
    hedged50_mae=np.float64(hedged50_mae),
    hedged2_oof=hedged2_oof.astype(np.float32),
    hedged2_pred=hedged2_pred.astype(np.float32),
    hedged2_mae=np.float64(hedged2_mae),
    w_h2=w_h2.astype(np.float64),
    w_h3=w_h3.astype(np.float64),
    w_meta=w_meta.astype(np.float64),
    w_linear=w_lin.astype(np.float64),
    y_raw=y_raw.astype(np.float32),
)
log('  saved -> v12_full_oof_bundle.npz')

log(f'\n{"="*60}')
log(f'DONE in {time.time()-t0:.0f}s ({(time.time()-t0)/3600:.1f} hrs)')
log(f'{"="*60}')
log(f'★ v12 main OOF:  {best_hedge_mae:.5f}')
log(f'  v8  best  LB:  9.7186')
log(f'  hedge weights: {best_hedge_name}')
log(f'{"="*60}')
log(f'\n제출 우선순위:')
log(f'  1) sub_v12_main.csv      ← v12 3-cand SLSQP (가장 추천)')
log(f'  2) sub_v12_2hedge.csv    ← v8 style 2-cand (검증된 방식)')
log(f'  3) sub_v12_50hedge.csv   ← 50/50 대조군')
