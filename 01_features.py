import os, gc, warnings, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
T = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
RANDOM_SEED = 42
N_CLUSTERS = 8

t0 = time.time()
def log(msg): print(f'[{time.time()-t0:.0f}s] {msg}', flush=True)
np.random.seed(RANDOM_SEED)

# ═════════════════════════════════════════════════════════════════
# PART A — V2 FEATURES (from step1_v2_clean.py)
# ═════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────
log('[1] Loading data...')
train_raw = pd.read_csv('data/train.csv')
test_raw  = pd.read_csv('data/test.csv')
layout    = pd.read_csv('data/layout_info.csv')
ss        = pd.read_csv('data/sample_submission.csv')
n_tr, n_te = len(train_raw), len(test_raw)
assert np.all(test_raw['ID'].values == ss['ID'].values)
train_raw['row_idx'] = np.arange(n_tr)
test_raw['row_idx']  = np.arange(n_te)
log(f'  Train: {n_tr} rows, Test: {n_te} rows')

# ─────────────────────────────────────────────────────────────────
# 2. LAYOUT FEATURES + K-MEANS (TRAIN-only fit)
# ─────────────────────────────────────────────────────────────────
log('[2] Layout features + K-Means (fit on TRAIN layouts only)...')
le = LabelEncoder()
layout['layout_type_enc'] = le.fit_transform(layout['layout_type'])
num_layout_cols = [c for c in layout.columns
                   if c not in ['layout_id', 'layout_type', 'layout_type_enc']
                   and layout[c].dtype != 'object']

layout['robot_density']     = (layout['robot_total'] / (layout['floor_area_sqm'] + 1)).astype(np.float32)
layout['charger_ratio']     = (layout['charger_count'] / (layout['robot_total'] + 1)).astype(np.float32)
layout['pack_per_area']     = (layout['pack_station_count'] / (layout['floor_area_sqm'] + 1)).astype(np.float32)
layout['robot_per_pack']    = (layout['robot_total'] / (layout['pack_station_count'] + 1)).astype(np.float32)
layout['robot_per_charger'] = (layout['robot_total'] / (layout['charger_count'] + 1)).astype(np.float32)

train_lids = sorted(train_raw['layout_id'].unique())
layout_train_mask = layout['layout_id'].isin(train_lids).values

X_cl_all   = layout[num_layout_cols].fillna(0).values.astype(np.float32)
sc_layout  = StandardScaler().fit(X_cl_all[layout_train_mask])
X_cl_scaled = sc_layout.transform(X_cl_all)

km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
km.fit(X_cl_scaled[layout_train_mask])
layout['layout_cluster'] = km.predict(X_cl_scaled)

train_raw = train_raw.sort_values(['scenario_id', 'row_idx']).reset_index(drop=True)
test_raw  = test_raw.sort_values(['scenario_id', 'row_idx']).reset_index(drop=True)
train_raw['timeslot'] = train_raw.groupby('scenario_id').cumcount().astype(np.int8)
test_raw['timeslot']  = test_raw.groupby('scenario_id').cumcount().astype(np.int8)

train = train_raw.merge(layout, on='layout_id', how='left')
test  = test_raw.merge(layout, on='layout_id', how='left')
y_raw_sorted = train[T].values.astype(np.float32)

# ─────────────────────────────────────────────────────────────────
# 3. KNN TARGET ENCODING
# ─────────────────────────────────────────────────────────────────
log('[3] KNN Target Encoding (5-NN in TRAIN metadata; TRAIN targets only)...')
train_lids_set = set(train['layout_id'].unique())
test_lids      = set(test['layout_id'].unique())
seen_lids   = sorted(train_lids_set & test_lids)
unseen_lids = sorted(test_lids - train_lids_set)
log(f'  Seen layouts: {len(seen_lids)}, Unseen: {len(unseen_lids)}')

gkf_te = GroupKFold(n_splits=N_SPLITS)
train_lid_te = np.zeros((n_tr, 3), dtype=np.float32)
for fold, (tri, vai) in enumerate(gkf_te.split(train, y_raw_sorted, train['layout_id'].values)):
    fs = train.iloc[tri].groupby('layout_id')[T].agg(['mean', 'median', 'std'])
    for j, stat in enumerate(['mean', 'median', 'std']):
        train_lid_te[vai, j] = train.iloc[vai]['layout_id'].map(
            fs[stat]).fillna(fs[stat].mean()).values

full_lt_stats = train.groupby('layout_id')[T].agg(['mean', 'median', 'std']).reset_index()
full_lt_stats.columns = ['layout_id', 'tgt_mean', 'tgt_median', 'tgt_std']

lid_arr   = layout['layout_id'].values
feat_arr  = layout[num_layout_cols].fillna(0).values.astype(float)
train_lid_idx_in_layout = [np.where(lid_arr == l)[0][0] for l in train_lids]
sc_knn  = StandardScaler().fit(feat_arr[train_lid_idx_in_layout])
train_sc_knn = sc_knn.transform(feat_arr[train_lid_idx_in_layout])
knn = NearestNeighbors(n_neighbors=5).fit(train_sc_knn)

unseen_rows = []
for lid in unseen_lids:
    i = np.where(lid_arr == lid)[0][0]
    d, ix = knn.kneighbors(sc_knn.transform(feat_arr[i:i+1]))
    w = 1.0 / (d[0] + 1e-6); w /= w.sum()
    nlids = [train_lids[j] for j in ix[0]]
    row = {'layout_id': lid}
    sub = full_lt_stats[full_lt_stats['layout_id'].isin(nlids)]
    for c in ['tgt_mean', 'tgt_median', 'tgt_std']:
        vals = [sub.loc[sub['layout_id'] == nl, c].values[0] * ww
                for nl, ww in zip(nlids, w) if len(sub.loc[sub['layout_id'] == nl]) > 0]
        row[c] = float(sum(vals)) if vals else float(full_lt_stats[c].mean())
    unseen_rows.append(row)
all_lt_stats = pd.concat([full_lt_stats, pd.DataFrame(unseen_rows)], ignore_index=True)

train['tgt_mean']   = train_lid_te[:, 0]
train['tgt_median'] = train_lid_te[:, 1]
train['tgt_std']    = train_lid_te[:, 2]
test_te = test[['layout_id']].merge(all_lt_stats, on='layout_id', how='left')
gm = full_lt_stats['tgt_mean'].mean()
test['tgt_mean']   = test_te['tgt_mean'].fillna(gm).astype(np.float32).values
test['tgt_median'] = test_te['tgt_median'].fillna(gm).astype(np.float32).values
test['tgt_std']    = test_te['tgt_std'].fillna(0).astype(np.float32).values

for df in [train, test]:
    df['order_vs_layout'] = (df['order_inflow_15m'] / (df['tgt_mean'] + 1)).astype(np.float32)
    df['cong_vs_layout']  = (df['congestion_score'] / (df['tgt_mean'] + 1)).astype(np.float32)

# ─────────────────────────────────────────────────────────────────
# 4. ROW-WISE FEATURES
# ─────────────────────────────────────────────────────────────────
log('[4] Row-wise feature engineering...')

def safe_div(a, b, fill=0):
    return np.where(np.abs(b) < 1e-8, fill, a / b)

def add_row_features(df):
    rt   = df['robot_total'].fillna(1)
    pc   = df['pack_station_count'].fillna(1) + 1e-3
    chrg = df['charger_count'].fillna(1) + 1e-3
    rsum = (df['robot_active'].fillna(0) + df['robot_idle'].fillna(0)
            + df['robot_charging'].fillna(0))
    df['robot_sum_gap']       = (rt - rsum).astype(np.float32)
    df['active_ratio']        = safe_div(df['robot_active'].fillna(0), rt).astype(np.float32)
    df['idle_ratio']          = safe_div(df['robot_idle'].fillna(0), rt).astype(np.float32)
    df['charging_ratio']      = safe_div(df['robot_charging'].fillna(0), rt).astype(np.float32)
    df['orders_per_pack']     = (df['order_inflow_15m'] / pc).astype(np.float32)
    df['orders_per_robot']    = (df['order_inflow_15m'] / (rt + 1e-3)).astype(np.float32)
    df['orders_per_charger']  = (df['order_inflow_15m'] / chrg).astype(np.float32)
    df['queue_per_charger']   = (df['charge_queue_length'] / chrg).astype(np.float32)
    df['dock_pressure']       = (df['loading_dock_util'] * df['outbound_truck_wait_min']).astype(np.float32)
    df['pack_pressure']       = (df['pack_utilization'] * df['orders_per_pack']).astype(np.float32)
    df['battery_stress']      = (df['low_battery_ratio'] * df['charge_queue_length']).astype(np.float32)
    df['cong_x_density']      = (df['congestion_score'] * df['max_zone_density']).astype(np.float32)
    df['urgent_x_inflow']     = (df['urgent_order_ratio'] * df['order_inflow_15m']).astype(np.float32)
    df['fault_pressure']      = (df['fault_count_15m'] * df['avg_recovery_time']).astype(np.float32)
    df['order_complexity']    = (df['unique_sku_15m'] * df['avg_items_per_order']).astype(np.float32)
    df['cong_per_robot']      = safe_div(df['congestion_score'],
                                         df['robot_active'].fillna(1) + 1).astype(np.float32)
    df['task_reassign_rate']  = safe_div(df['task_reassign_15m'],
                                         df['order_inflow_15m'] + 1).astype(np.float32)
    ts = df['timeslot'].astype(np.float32)
    df['time_x_orders_per_pack'] = (ts * df['orders_per_pack']).astype(np.float32)
    df['time_x_dock_pressure']   = (ts * df['dock_pressure']).astype(np.float32)
    df['time_x_battery_stress']  = (ts * df['battery_stress']).astype(np.float32)
    df['time_x_congestion']      = (ts * df['congestion_score']).astype(np.float32)
    df['battery_cv']             = (df['battery_std'] / (df['battery_mean'] + 1)).astype(np.float32)
    df['charge_wait_per_queue']  = (df['avg_charge_wait'] / (df['charge_queue_length'] + 1)).astype(np.float32)
    df['cold_x_heavy']           = (df['cold_chain_ratio'] * df['heavy_item_ratio']).astype(np.float32)
    df['forklift_per_area']      = (df['forklift_active_count'] / (df['floor_area_sqm'] + 1) * 1000).astype(np.float32)
    df['staff_per_order']        = (df['staff_on_floor'] / (df['order_inflow_15m'] + 1)).astype(np.float32)
    df['idle_battery_risk']      = (df['robot_idle'] * df['low_battery_ratio']).astype(np.float32)
    df['charge_stress']          = (df['charge_queue_length'] * df['low_battery_ratio']
                                    * df['robot_charging']).astype(np.float32)
    df['path_x_cong']            = (df['path_optimization_score'] * df['congestion_score']).astype(np.float32)
    df['network_stress']         = (df['wms_response_time_ms'] * df['network_latency_ms']).astype(np.float32)
    df['env_stress']             = (df['warehouse_temp_avg'] * df['humidity_pct'] / 100).astype(np.float32)
    df['conveyor_load']          = (df['conveyor_speed_mps'] * df['order_inflow_15m']).astype(np.float32)
    df['scanner_x_order']        = (df['scanner_error_rate'] * df['order_inflow_15m']).astype(np.float32)
    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7).astype(np.float32)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7).astype(np.float32)
    df['hour_sin'] = np.sin(2 * np.pi * df['shift_hour'] / 24).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * df['shift_hour'] / 24).astype(np.float32)
    df['ts_pct']   = (df['timeslot'] / 24.0).astype(np.float32)
    return df

train = add_row_features(train)
test  = add_row_features(test)

# ─────────────────────────────────────────────────────────────────
# 5. TEMPORAL FEATURES (EXPANDED)
# ─────────────────────────────────────────────────────────────────
log('[5] Temporal features (lag/lead/diff/rolling/EWM/slope) — EXPANDED...')
dyn_cols = [
    'order_inflow_15m', 'urgent_order_ratio', 'robot_active', 'robot_charging',
    'robot_utilization', 'battery_mean', 'low_battery_ratio', 'charge_queue_length',
    'congestion_score', 'max_zone_density', 'blocked_path_15m',
    'fault_count_15m', 'pack_utilization', 'loading_dock_util',
    'outbound_truck_wait_min', 'orders_per_pack', 'orders_per_robot',
    'dock_pressure', 'pack_pressure', 'battery_stress',
    'battery_std', 'avg_charge_wait', 'unique_sku_15m', 'sku_concentration',
    'robot_idle', 'near_collision_15m', 'task_reassign_15m'
]
dyn_cols = [c for c in dyn_cols if c in train.columns]
key_dyn = [
    'order_inflow_15m', 'congestion_score', 'loading_dock_util',
    'pack_utilization', 'charge_queue_length', 'low_battery_ratio',
    'orders_per_pack', 'dock_pressure', 'battery_stress',
    'max_zone_density', 'robot_utilization'
]
bidir_cols = [
    'congestion_score', 'order_inflow_15m', 'pack_utilization',
    'loading_dock_util', 'charge_queue_length', 'low_battery_ratio',
    'max_zone_density', 'robot_utilization', 'dock_pressure',
    'battery_stress', 'urgent_order_ratio'
]
bidir_cols = [c for c in bidir_cols if c in train.columns]

def add_temporal(df, cols, key_cols, bd_cols):
    g = df.groupby('scenario_id', sort=False)
    for c in cols:
        lag1 = g[c].shift(1)
        df[f'{c}_lag1']  = lag1.astype(np.float32)
        df[f'{c}_diff1'] = (df[c] - lag1).astype(np.float32)
        df[f'{c}_rm3']   = g[c].rolling(3, min_periods=1).mean().reset_index(
            level=0, drop=True).astype(np.float32)
    for c in key_cols:
        df[f'{c}_lag2']    = g[c].shift(2).astype(np.float32)
        df[f'{c}_lag3']    = g[c].shift(3).astype(np.float32)
        sh = g[c].shift(1)
        df[f'{c}_expmean'] = sh.groupby(df['scenario_id'], sort=False).transform(
            lambda s: s.expanding(min_periods=1).mean()).astype(np.float32)
    for c in bd_cols:
        df[f'{c}_lead1'] = g[c].shift(-1).astype(np.float32)
        df[f'{c}_lead2'] = g[c].shift(-2).astype(np.float32)
        df[f'{c}_lead3'] = g[c].shift(-3).astype(np.float32)
        df[f'{c}_cwin3'] = g[c].transform(
            lambda s: s.rolling(3, min_periods=1, center=True).mean()).astype(np.float32)
        df[f'{c}_cwin5'] = g[c].transform(
            lambda s: s.rolling(5, min_periods=1, center=True).mean()).astype(np.float32)
        df[f'{c}_fwd3'] = g[c].transform(
            lambda s: s[::-1].rolling(3, min_periods=1).mean()[::-1]).astype(np.float32)
        df[f'{c}_fwd5'] = g[c].transform(
            lambda s: s[::-1].rolling(5, min_periods=1).mean()[::-1]).astype(np.float32)
    for c in ['congestion_score', 'order_inflow_15m', 'robot_utilization']:
        if c in df.columns:
            df[f'{c}_slope'] = g[c].transform(
                lambda x: np.polyfit(range(len(x)), x.fillna(x.mean()), 1)[0]
                if len(x) > 1 else 0
            ).astype(np.float32)
    for c in ['congestion_score', 'order_inflow_15m', 'charge_queue_length']:
        if c in df.columns:
            df[f'{c}_cummax'] = g[c].cummax().astype(np.float32)
            df[f'{c}_cummin'] = g[c].cummin().astype(np.float32)
    for c in ['congestion_score', 'order_inflow_15m', 'low_battery_ratio',
              'robot_utilization', 'charge_queue_length', 'max_zone_density']:
        df[f'{c}_ewm3'] = g[c].transform(
            lambda x: x.ewm(span=3, min_periods=1).mean()).astype(np.float32)
    for c in ['congestion_score', 'order_inflow_15m', 'low_battery_ratio', 'robot_utilization']:
        df[f'{c}_sc_rank'] = g[c].rank(pct=True).astype(np.float32)
    for c in ['congestion_score', 'order_inflow_15m', 'robot_utilization']:
        df[f'{c}_diff2'] = g[f'{c}_diff1'].diff(1).astype(np.float32)
    return df

train = add_temporal(train, dyn_cols, key_dyn, bidir_cols)
test  = add_temporal(test, dyn_cols, key_dyn, bidir_cols)

# ─────────────────────────────────────────────────────────────────
# 6. FULL-CONTEXT SCENARIO FEATURES (EXPANDED)
# ─────────────────────────────────────────────────────────────────
log('[6] Full-context scenario features (EXPANDED)...')
ctx_cols = [
    'order_inflow_15m', 'orders_per_pack', 'congestion_score',
    'pack_utilization', 'loading_dock_util', 'low_battery_ratio',
    'charge_queue_length', 'dock_pressure', 'battery_stress',
    'robot_utilization', 'max_zone_density', 'urgent_order_ratio'
]
ctx_cols = [c for c in ctx_cols if c in train.columns]

def add_full_context(df, cols):
    agg = df.groupby('scenario_id')[cols].agg(
        ['mean', 'std', 'min', 'max', 'median']).reset_index()
    agg.columns = (['scenario_id'] +
                   [f'{c}_sc_{s}' for c in cols for s in ['mean', 'std', 'min', 'max', 'median']])
    df = df.merge(agg, on='scenario_id', how='left')
    g = df.groupby('scenario_id', sort=False)
    q10 = df.groupby('scenario_id')[cols].quantile(0.10).reset_index()
    q90 = df.groupby('scenario_id')[cols].quantile(0.90).reset_index()
    q10.columns = ['scenario_id'] + [f'{c}_sc_q10' for c in cols]
    q90.columns = ['scenario_id'] + [f'{c}_sc_q90' for c in cols]
    df = df.merge(q10, on='scenario_id', how='left').merge(q90, on='scenario_id', how='left')
    for c in cols:
        df[f'{c}_sc_range'] = (df[f'{c}_sc_max'] - df[f'{c}_sc_min']).astype(np.float32)
        df[f'{c}_sc_iqr']   = (df[f'{c}_sc_q90'] - df[f'{c}_sc_q10']).astype(np.float32)
        df[f'{c}_rankpct']  = g[c].rank(pct=True).astype(np.float32)
        df[f'{c}_zctx']     = ((df[c] - df[f'{c}_sc_mean'])
                               / (df[f'{c}_sc_std'].fillna(0) + 1e-6)).astype(np.float32)
        idx_peak   = g[c].transform('idxmax')
        idx_trough = g[c].transform('idxmin')
        df[f'{c}_dist_peak']   = (df.index.values - idx_peak.values).astype(np.float32)
        df[f'{c}_dist_trough'] = (df.index.values - idx_trough.values).astype(np.float32)
        df[f'{c}_vs_scmax'] = (df[c] / (df[f'{c}_sc_max'] + 1e-6)).astype(np.float32)
    return df

train = add_full_context(train, ctx_cols)
test  = add_full_context(test, ctx_cols)

# ─────────────────────────────────────────────────────────────────
# 7. MULTI-GRANULAR OOF TARGET ENCODING (TRAIN-only)
# ─────────────────────────────────────────────────────────────────
log('[7] Multi-granular OOF target encoding...')
for df in [train, test]:
    df['lc_str']  = df['layout_cluster'].astype(str)
    df['ts_str']  = df['timeslot'].astype(str)
    df['hr_str']  = df['shift_hour'].astype(str)
    df['dow_str'] = df['day_of_week'].astype(str)
    df['lt_str']  = df['layout_type_enc'].astype(str)

def oof_te(tr, te, gcols, fname, alpha=20):
    gmean = tr[T].mean()
    oof   = np.full(len(tr), gmean, dtype=np.float32)
    gkf   = GroupKFold(n_splits=N_SPLITS)
    for fold, (tri, vai) in enumerate(gkf.split(tr, tr[T], tr['scenario_id'])):
        st = tr.iloc[tri].groupby(gcols)[T].agg(['mean', 'count']).reset_index()
        st[fname] = ((st['mean'] * st['count']) + alpha * gmean) / (st['count'] + alpha)
        m = tr.iloc[vai][gcols].merge(st[gcols + [fname]], on=gcols, how='left')[fname].fillna(gmean)
        oof[vai] = m.values.astype(np.float32)
    full_st = tr.groupby(gcols)[T].agg(['mean', 'count']).reset_index()
    full_st[fname] = ((full_st['mean'] * full_st['count']) + alpha * gmean) / (full_st['count'] + alpha)
    te_m = te[gcols].merge(full_st[gcols + [fname]], on=gcols, how='left')[fname].fillna(gmean)
    tr[fname] = oof
    te[fname] = te_m.values.astype(np.float32)

oof_te(train, test, ['lc_str', 'ts_str'],  'te_lc_ts')
oof_te(train, test, ['lc_str', 'hr_str'],  'te_lc_hr')
oof_te(train, test, ['lc_str', 'dow_str'], 'te_lc_dow')
oof_te(train, test, ['lt_str', 'ts_str'],  'te_lt_ts')
oof_te(train, test, ['ts_str'],            'te_ts')

# ─────────────────────────────────────────────────────────────────
# 8. BUILD V2 FEATURE ARRAYS
# ─────────────────────────────────────────────────────────────────
log('[8] Building V2 feature arrays...')
drop_cols = ['ID', 'scenario_id', 'layout_id', T, 'row_idx', 'layout_type',
             'lc_str', 'ts_str', 'hr_str', 'dow_str', 'lt_str']
feature_cols_v2 = [c for c in train.columns
                   if c not in drop_cols
                   and train[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int8']]
log(f'  V2 features: {len(feature_cols_v2)}')

# Re-sort to row_idx order
train = train.sort_values('row_idx').reset_index(drop=True)
test  = test.sort_values('row_idx').reset_index(drop=True)
y_raw = train[T].values.astype(np.float32)
X_v2    = train[feature_cols_v2].values.astype(np.float32)
X_te_v2 = test[feature_cols_v2].values.astype(np.float32)
groups   = train['layout_id'].values
y_log    = np.log1p(y_raw).astype(np.float32)
y_sqrt   = np.sqrt(y_raw).astype(np.float32)
test_ids = test['ID'].values if 'ID' in test.columns else test_raw['ID'].values
layout_cluster_tr = train['layout_cluster'].values.astype(np.int16)
layout_cluster_te = test['layout_cluster'].values.astype(np.int16)
timeslot_tr       = train['timeslot'].values.astype(np.int16)
timeslot_te       = test['timeslot'].values.astype(np.int16)

log(f'  X_v2: {X_v2.shape}   X_te_v2: {X_te_v2.shape}')

# ═════════════════════════════════════════════════════════════════
# PART B — V4 SHAPE FEATURES (from step1_v4_shapes.py)
# ═════════════════════════════════════════════════════════════════
log('\n[9] Computing V4 shape features (per-scenario trajectory)...')

# Re-sort raw data by scenario for shape computation
train_sc = train_raw.sort_values(['scenario_id', 'row_idx']).reset_index(drop=True)
test_sc  = test_raw.sort_values(['scenario_id', 'row_idx']).reset_index(drop=True)
train_sc['orders_per_pack'] = (train_sc['order_inflow_15m'] /
                               (train_sc['pack_utilization'].fillna(0.5) + 1e-3))
test_sc['orders_per_pack']  = (test_sc['order_inflow_15m'] /
                               (test_sc['pack_utilization'].fillna(0.5) + 1e-3))

KEY_COLS = [
    'order_inflow_15m', 'congestion_score',
    'pack_utilization', 'charge_queue_length', 'low_battery_ratio',
]
CROSS_PAIRS = [
    ('congestion_score',    'order_inflow_15m'),
    ('charge_queue_length', 'low_battery_ratio'),
    ('pack_utilization',    'loading_dock_util'),
    ('orders_per_pack',     'pack_utilization'),
]

def shape_features_from_series(s):
    n = len(s)
    x = np.asarray(s, dtype=np.float64)
    x = np.where(np.isnan(x), np.nanmean(x) if np.any(~np.isnan(x)) else 0.0, x)
    if n <= 2:
        return dict(fft1_mag=0, fft1_phase=0, fft2_mag=0,
                    peak_count=0, argmax_frac=0, spearman_t=0,
                    curvature=0, linfit_resid=0, autocorr_lag1=0,
                    range_ratio=0)
    f   = np.fft.rfft(x - x.mean())
    mag = np.abs(f)
    phs = np.angle(f)
    idx = np.argsort(mag[1:])[::-1] + 1
    i1 = idx[0] if len(idx) > 0 else 0
    i2 = idx[1] if len(idx) > 1 else 0
    norm = (np.linalg.norm(x - x.mean()) + 1e-9)
    fft1_mag   = mag[i1] / norm if i1 > 0 else 0
    fft1_phase = phs[i1] / np.pi if i1 > 0 else 0
    fft2_mag   = mag[i2] / norm if i2 > 0 else 0
    peak_count = int(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])))
    argmax_frac = float(np.argmax(x)) / max(n-1, 1)
    rx = rankdata(x); rt = rankdata(np.arange(n))
    spearman_t = float(np.corrcoef(rx, rt)[0,1]) if n > 1 else 0.0
    if not np.isfinite(spearman_t): spearman_t = 0.0
    t_arr = np.arange(n)
    try:
        a2, a1, a0 = np.polyfit(t_arr, x, 2)
    except Exception:
        a2 = a1 = a0 = 0.0
    curvature = float(a2)
    try:
        b1, b0 = np.polyfit(t_arr, x, 1)
        resid  = x - (b1*t_arr + b0)
        linfit_resid = float(np.linalg.norm(resid) / np.sqrt(n))
    except Exception:
        linfit_resid = 0.0
    xm = x - x.mean()
    denom = float((xm*xm).sum()) + 1e-9
    autocorr_lag1 = float((xm[:-1]*xm[1:]).sum() / denom) if n > 1 else 0.0
    rng = float(x.max() - x.min())
    rng_ratio = rng / (abs(x.mean()) + 1.0)
    return dict(fft1_mag=fft1_mag, fft1_phase=fft1_phase, fft2_mag=fft2_mag,
                peak_count=peak_count, argmax_frac=argmax_frac,
                spearman_t=spearman_t, curvature=curvature,
                linfit_resid=linfit_resid, autocorr_lag1=autocorr_lag1,
                range_ratio=rng_ratio)

def pair_corr_per_scenario(df, a, b):
    def _c(g):
        if len(g) < 2: return 0.0
        xa = g[a].values; xb = g[b].values
        sa = xa.std(); sb = xb.std()
        if sa < 1e-9 or sb < 1e-9: return 0.0
        v = float(np.corrcoef(xa, xb)[0,1])
        return 0.0 if not np.isfinite(v) else v
    corrs = df.groupby('scenario_id').apply(_c)
    return df['scenario_id'].map(corrs).astype(np.float32).values

def build_shape_block(df, tag):
    log(f'  [{tag}] computing shape features (per-scenario)...')
    feats = {}
    for c in KEY_COLS:
        log(f'    {c}')
        res = df.groupby('scenario_id')[c].apply(shape_features_from_series)
        rdf = pd.DataFrame(res.tolist(), index=res.index)
        rdf.columns = [f'{c}_{k}' for k in rdf.columns]
        for col in rdf.columns:
            feats[col] = df['scenario_id'].map(rdf[col]).astype(np.float32).values
    for a, b in CROSS_PAIRS:
        fname = f'corr_{a}__{b}'
        log(f'    pair-corr {a} × {b}')
        feats[fname] = pair_corr_per_scenario(df, a, b)
    feat_df = pd.DataFrame(feats, index=df.index)
    return feat_df

shape_tr_df = build_shape_block(train_sc, 'train')
shape_te_df = build_shape_block(test_sc,  'test')

shape_tr_df['row_idx'] = train_sc['row_idx'].values
shape_te_df['row_idx'] = test_sc['row_idx'].values
shape_tr_df = shape_tr_df.sort_values('row_idx').reset_index(drop=True)
shape_te_df = shape_te_df.sort_values('row_idx').reset_index(drop=True)

shape_cols = [c for c in shape_tr_df.columns if c != 'row_idx']
log(f'  shape feature count: {len(shape_cols)}')

X_shape    = shape_tr_df[shape_cols].values.astype(np.float32)
X_te_shape = shape_te_df[shape_cols].values.astype(np.float32)
X_shape    = np.nan_to_num(X_shape,    nan=0.0, posinf=0.0, neginf=0.0)
X_te_shape = np.nan_to_num(X_te_shape, nan=0.0, posinf=0.0, neginf=0.0)

# ═════════════════════════════════════════════════════════════════
# PART C — CONCATENATE & SAVE
# ═════════════════════════════════════════════════════════════════
log('\n[10] Concatenating V2 + Shape → features.npz ...')
X    = np.hstack([X_v2,    X_shape]).astype(np.float32)
X_te = np.hstack([X_te_v2, X_te_shape]).astype(np.float32)
feature_cols = list(feature_cols_v2) + shape_cols
log(f'  Final X: {X.shape}   X_te: {X_te.shape}   features: {len(feature_cols)}')

np.savez_compressed('features.npz',
    X=X, X_te=X_te,
    y_raw=y_raw, y_log=y_log, y_sqrt=y_sqrt,
    groups=groups,
    test_ids=test_ids,
    feature_cols=np.array(feature_cols),
    layout_cluster_tr=layout_cluster_tr,
    layout_cluster_te=layout_cluster_te,
    timeslot_tr=timeslot_tr,
    timeslot_te=timeslot_te,
    n_shape_features=np.int64(len(shape_cols)))

log(f'\n=== 01_features.py DONE === {time.time()-t0:.0f}s')
log(f'  Output: features.npz  ({X.shape[1]} features)')
