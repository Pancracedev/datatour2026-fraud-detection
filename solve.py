#!/usr/bin/env python3
"""
DataTour 2026 — Mobile Money Fraud Detection
v17: VALIDATION TEMPORELLE + INTERACTIONS
  - Validation temporelle (train 0-84, val 85-105) pour estimer
    le score leaderboard avant soumission
  - Features d'interaction (orig_drained × dest_is_new, etc.)
  - Entrainement sur op03 uniquement (v16 = 0.349793 LB)
  - Meilleur: 0.34820035 (v12) → 0.349793 (v16)
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import time

np.random.seed(42)
DATA = '/home/pancrace/Bureau/competition'
EPS  = 1e-6

t0 = time.time()
print("=" * 60)
print("DATATOUR 2026  —  FRAUD PIPELINE v17")
print("=" * 60)

# ─── CHARGEMENT ───────────────────────────────────────────────────
print("\n[1/6] Chargement...")
train  = pd.read_csv(f'{DATA}/train.csv')
test   = pd.read_csv(f'{DATA}/test.csv')
sample = pd.read_csv(f'{DATA}/sample_submission (1).csv')

TARGET, ID = 'fraud_flag', 'id'
g_rate = train[TARGET].mean()
train03 = train[train['operation'] == 'op_03'].copy()
test03  = test[test['operation']   == 'op_03'].copy()

print(f"  train op03 : {len(train03):,}  fraude: {train03[TARGET].mean():.2%}")
print(f"  test  op03 : {len(test03):,}")
print(f"  periods train : {train['period'].min()}–{train['period'].max()}")

# ─── FONCTIONS DE STATS (réutilisables par split temporel) ────────
def compute_stats(ref_train):
    """Calcule toutes les stats comportementales depuis ref_train."""
    non03 = ref_train[ref_train['operation'] != 'op_03']
    med = ref_train['amount'].median()

    # Origin depuis non-op03
    oc = non03.groupby('origin_account').agg(
        orig_clean_n       =('amount','count'),
        orig_clean_mean    =('amount','mean'),
        orig_clean_std     =('amount','std'),
        orig_clean_max     =('amount','max'),
        orig_clean_bal_mean=('origin_balance_before','mean'),
        orig_clean_bal_std =('origin_balance_before','std'),
    ).reset_index()
    for c in ['orig_clean_std','orig_clean_bal_std']:
        oc[c] = oc[c].fillna(0)

    # Dest depuis non-op03
    dc = non03.groupby('destination_account').agg(
        dest_clean_n       =('amount','count'),
        dest_clean_mean    =('amount','mean'),
        dest_clean_std     =('amount','std'),
        dest_clean_bal_mean=('destination_balance_before','mean'),
    ).reset_index()
    dc['dest_clean_std'] = dc['dest_clean_std'].fillna(0)

    # Stats toutes ops
    oa = ref_train.groupby('origin_account').agg(
        orig_all_n   =('amount','count'),
        orig_all_mean=('amount','mean'),
        orig_all_std =('amount','std'),
        orig_all_max =('amount','max'),
        orig_tenure  =('period', lambda x: x.max()-x.min()),
    ).reset_index()
    oa['orig_all_std'] = oa['orig_all_std'].fillna(0)

    da = ref_train.groupby('destination_account').agg(
        dest_all_n        =('amount','count'),
        dest_all_mean     =('amount','mean'),
        dest_all_std      =('amount','std'),
        dest_all_bal_mean =('destination_balance_before','mean'),
        dest_all_bal_std  =('destination_balance_before','std'),
    ).reset_index()
    for c in ['dest_all_std','dest_all_bal_std']:
        da[c] = da[c].fillna(0)

    # Réseau
    on = ref_train.groupby('origin_account')['destination_account'].nunique().reset_index()
    on.columns = ['origin_account','orig_n_unique_dests']
    dn = ref_train.groupby('destination_account')['origin_account'].nunique().reset_index()
    dn.columns = ['destination_account','dest_n_unique_origs']

    # Stats op03
    ref03 = ref_train[ref_train['operation']=='op_03']
    oo = ref03.groupby('origin_account').agg(
        orig_op03_n   =('amount','count'),
        orig_op03_mean=('amount','mean'),
        orig_op03_std =('amount','std'),
        orig_op03_max =('amount','max'),
    ).reset_index()
    oo['orig_op03_std'] = oo['orig_op03_std'].fillna(0)

    do = ref03.groupby('destination_account').agg(
        dest_op03_n       =('amount','count'),
        dest_op03_mean    =('amount','mean'),
        dest_op03_std     =('amount','std'),
        dest_op03_bal_mean=('destination_balance_before','mean'),
    ).reset_index()
    do['dest_op03_std'] = do['dest_op03_std'].fillna(0)

    # Paires op03
    po = ref03.groupby(['origin_account','destination_account']).agg(
        pair_n       =('amount','count'),
        pair_amt_mean=('amount','mean'),
        pair_amt_std =('amount','std'),
        pair_amt_max =('amount','max'),
    ).reset_index()
    po['pair_amt_std'] = po['pair_amt_std'].fillna(0)

    known_orig       = set(ref_train['origin_account'])
    known_dest       = set(ref_train['destination_account'])
    known_orig_clean = set(non03['origin_account'])
    known_dest_clean = set(non03['destination_account'])

    return dict(oc=oc, dc=dc, oa=oa, da=da, on=on, dn=dn,
                oo=oo, do=do, po=po, med=med,
                known_orig=known_orig, known_dest=known_dest,
                known_orig_clean=known_orig_clean,
                known_dest_clean=known_dest_clean)

def build_feats(df, S):
    d = df.copy().reset_index(drop=True)
    amt = d['amount']
    ob  = d['origin_balance_before']
    oa_ = d['origin_balance_after']
    db  = d['destination_balance_before']
    da_ = d['destination_balance_after']
    med = S['med']

    # Transaction pures
    d['mismatch_orig']     = (ob - oa_ - amt).abs()
    d['mismatch_orig_log'] = np.log1p(d['mismatch_orig'])
    d['mismatch_dest']     = (da_ - db - amt).abs()
    d['orig_neg_after']    = (oa_ < 0).astype('int8')
    d['dest_zero_before']  = (db == 0).astype('int8')
    d['orig_zero_after']   = (oa_ == 0).astype('int8')
    d['orig_drained']      = (oa_ < 0.01 * ob.abs() + EPS).astype('int8')
    d['amount_log']        = np.log1p(amt)
    d['orig_bal_log']      = np.sign(ob) * np.log1p(ob.abs())
    d['dest_bal_log']      = np.log1p(db.abs())
    d['amt_ratio_orig']    = (amt / (ob.abs() + EPS)).clip(0, 10)
    d['amt_ratio_dest']    = (amt / (db.abs() + EPS)).clip(0, 10)
    d['exceeds_orig']      = (amt > ob).astype('int8')
    d['net_flow_norm']     = ((oa_ - ob) + (da_ - db)) / (amt + EPS)
    d['balance_ratio']     = (db / (ob.abs() + EPS)).clip(-10, 10)

    # Nouveauté
    d['orig_is_new']        = (~d['origin_account'].isin(S['known_orig'])).astype('int8')
    d['dest_is_new']        = (~d['destination_account'].isin(S['known_dest'])).astype('int8')
    d['orig_no_clean_hist'] = (~d['origin_account'].isin(S['known_orig_clean'])).astype('int8')
    d['dest_no_clean_hist'] = (~d['destination_account'].isin(S['known_dest_clean'])).astype('int8')

    # Joins
    d = (d.merge(S['oc'], on='origin_account',      how='left')
          .merge(S['dc'], on='destination_account',  how='left')
          .merge(S['oa'], on='origin_account',        how='left')
          .merge(S['da'], on='destination_account',   how='left')
          .merge(S['on'], on='origin_account',        how='left')
          .merge(S['dn'], on='destination_account',   how='left')
          .merge(S['oo'], on='origin_account',        how='left')
          .merge(S['do'], on='destination_account',   how='left')
          .merge(S['po'], on=['origin_account','destination_account'], how='left'))

    fills = {
        'orig_clean_n':0,'orig_clean_mean':med,'orig_clean_std':0,
        'orig_clean_max':med,'orig_clean_bal_mean':0,'orig_clean_bal_std':0,
        'dest_clean_n':0,'dest_clean_mean':med,'dest_clean_std':0,'dest_clean_bal_mean':0,
        'orig_all_n':0,'orig_all_mean':med,'orig_all_std':0,'orig_all_max':med,'orig_tenure':0,
        'dest_all_n':0,'dest_all_mean':med,'dest_all_std':0,'dest_all_bal_mean':0,'dest_all_bal_std':0,
        'orig_n_unique_dests':0,'dest_n_unique_origs':0,
        'orig_op03_n':0,'orig_op03_mean':med,'orig_op03_std':0,'orig_op03_max':0,
        'dest_op03_n':0,'dest_op03_mean':med,'dest_op03_std':0,'dest_op03_bal_mean':0,
        'pair_n':0,'pair_amt_mean':med,'pair_amt_std':0,'pair_amt_max':0,
    }
    for col, val in fills.items():
        if col in d.columns:
            d[col] = d[col].fillna(val)

    # Aliases après merge
    amt = d['amount']
    ob  = d['origin_balance_before']

    # Dérivées
    d['amt_z_clean_orig']   = ((amt-d['orig_clean_mean'])/(d['orig_clean_std']+EPS)).clip(-5,5)
    d['amt_z_all_orig']     = ((amt-d['orig_all_mean'])  /(d['orig_all_std']  +EPS)).clip(-5,5)
    d['amt_z_all_dest']     = ((amt-d['dest_all_mean'])  /(d['dest_all_std']  +EPS)).clip(-5,5)
    d['amt_z_op03_orig']    = ((amt-d['orig_op03_mean']) /(d['orig_op03_std'] +EPS)).clip(-5,5)
    d['amt_z_op03_dest']    = ((amt-d['dest_op03_mean']) /(d['dest_op03_std'] +EPS)).clip(-5,5)
    d['bal_z_clean_orig']   = ((ob -d['orig_clean_bal_mean'])/(d['orig_clean_bal_std']+EPS)).clip(-5,5)
    d['amt_pct_clean_max']  = (amt/(d['orig_clean_max']+EPS)).clip(0,5)
    d['amt_pct_op03_max']   = (amt/(d['orig_op03_max'] +EPS)).clip(0,5)
    d['orig_vol_clean_log'] = np.log1p(d['orig_clean_n'])
    d['dest_vol_clean_log'] = np.log1p(d['dest_clean_n'])
    d['orig_vol_op03_log']  = np.log1p(d['orig_op03_n'])
    d['dest_vol_op03_log']  = np.log1p(d['dest_op03_n'])
    d['orig_vol_all_log']   = np.log1p(d['orig_all_n'])
    d['orig_tenure_log']    = np.log1p(d['orig_tenure'])
    d['orig_net_log']       = np.log1p(d['orig_n_unique_dests'])
    d['dest_net_log']       = np.log1p(d['dest_n_unique_origs'])
    d['pair_is_new']        = (d['pair_n']==0).astype('int8')
    d['pair_n_log']         = np.log1p(d['pair_n'])
    d['pair_amt_z']         = ((amt-d['pair_amt_mean'])/(d['pair_amt_std']+EPS)).clip(-5,5)
    d['pair_exceeds_max']   = (amt>d['pair_amt_max']).astype('int8')

    # ── Interactions clés ──
    d['drain_to_new']       = d['orig_drained'] * d['dest_is_new']
    d['drain_to_newpair']   = d['orig_drained'] * d['pair_is_new']
    d['high_ratio_newdest'] = (d['amt_ratio_orig'] > 0.5).astype('int8') * d['dest_is_new']
    d['exceed_and_newpair'] = d['pair_exceeds_max'] * d['pair_is_new']
    d['no_hist_drain']      = d['dest_no_clean_hist'] * d['orig_drained']
    d['bal_z_x_ratio']      = (d['bal_z_clean_orig'] * d['amt_ratio_orig']).clip(-10, 10)

    return d

# ─── VALIDATION TEMPORELLE ────────────────────────────────────────
print("\n[2/6] Validation temporelle (train 0-84, val 85-105)...")
VAL_PERIOD = 85

tr_mask  = train['period'] <  VAL_PERIOD
va_mask  = train['period'] >= VAL_PERIOD

train_tr = train[tr_mask]
train_va = train[va_mask]

print(f"  Train: periods 0-{VAL_PERIOD-1}  ({len(train_tr):,} lignes, {train_tr[TARGET].mean():.2%} fraude)")
print(f"  Val  : periods {VAL_PERIOD}-105  ({len(train_va):,} lignes, {train_va[TARGET].mean():.2%} fraude)")

# Stats depuis la partie train temporelle seulement
S_tr = compute_stats(train_tr)

# Op03 uniquement pour validation
tr03 = train_tr[train_tr['operation']=='op_03']
va03 = train_va[train_va['operation']=='op_03']

print(f"  Train op03: {len(tr03):,}  fraude: {tr03[TARGET].mean():.2%}")
print(f"  Val   op03: {len(va03):,}  fraude: {va03[TARGET].mean():.2%}")

DROP_COLS = {'id','fraud_flag','operation','origin_account','destination_account',
             'period','op_encoded','orig_period_min','orig_period_max'}

tr03_fe = build_feats(tr03, S_tr)
va03_fe = build_feats(va03, S_tr)

feat_cols = [c for c in tr03_fe.columns if c not in DROP_COLS and c in va03_fe.columns]
Xtr = tr03_fe[feat_cols].astype('float32').fillna(-999)
Xva = va03_fe[feat_cols].astype('float32').fillna(-999)
ytr = tr03[TARGET].values
yva = va03[TARGET].values

N_TREES = 400
lgb_params = dict(
    objective='binary', metric='binary_logloss',
    boosting_type='gbdt', n_estimators=N_TREES, learning_rate=0.03,
    num_leaves=63, min_child_samples=30,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=(1-ytr.mean())/ytr.mean(),
    n_jobs=2, random_state=42, verbose=-1,
)

m_val = lgb.LGBMClassifier(**lgb_params)
m_val.fit(Xtr, ytr, callbacks=[lgb.log_evaluation(False)])
val_pred = m_val.predict_proba(Xva)[:, 1]

# AP global (avec 0 pour non-op03 en validation)
va_all_pred = np.zeros(len(train_va))
va03_local_idx = np.where(train_va['operation'].values == 'op_03')[0]
va_all_pred[va03_local_idx] = val_pred
sc_temporal = average_precision_score(train_va[TARGET].values, va_all_pred)
sc_op03_only = average_precision_score(yva, val_pred)

print(f"\n  ► Score temporel global  : {sc_temporal:.4f}  (proxy leaderboard)")
print(f"  ► Score temporel op03    : {sc_op03_only:.4f}  (baseline: {yva.mean():.4f})")
print(f"  (Réf. leaderboard actuel : 0.3498)")

# Feature importances
imp = pd.Series(m_val.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(f"\n  Top 15 features (val temporelle):")
for f, v in imp.head(15).items():
    print(f"    {f:<38} {v:>6.0f}")

# ─── MODÈLE FINAL SUR TOUT LE TRAIN ──────────────────────────────
print(f"\n[3/6] Stats finales sur tout le train...")
S_full = compute_stats(train)

print(f"\n[4/6] Modèle final (op03, {N_TREES} arbres)...")
train03_fe = build_feats(train03, S_full)
test03_fe  = build_feats(test03,  S_full)

feat_cols_f = [c for c in train03_fe.columns if c not in DROP_COLS and c in test03_fe.columns]
Xfull = train03_fe[feat_cols_f].astype('float32').fillna(-999)
Xtest = test03_fe.reindex(columns=feat_cols_f, fill_value=-999).astype('float32').fillna(-999)
yfull = train03[TARGET].values

lgb_final = lgb.LGBMClassifier(**{
    **lgb_params,
    'scale_pos_weight': (1-yfull.mean())/yfull.mean(),
})
lgb_final.fit(Xfull, yfull, callbacks=[lgb.log_evaluation(False)])
pred_op03 = lgb_final.predict_proba(Xtest)[:, 1]

print(f"  op03 : mean={pred_op03.mean():.4f}  max={pred_op03.max():.4f}  >0.5: {(pred_op03>0.5).mean():.2%}")

# ─── SOUMISSION ───────────────────────────────────────────────────
print(f"\n[5/6] Génération submission.csv...")
pred_full = pd.Series(0.0, index=test.index)
pred_full.loc[test03.index] = pred_op03

submission = pd.DataFrame({ID: test[ID], 'target': pred_full.clip(0,1).values})
assert submission.shape[0] == sample.shape[0]
assert list(submission.columns) == [ID, 'target']
assert submission[ID].is_unique
assert submission['target'].between(0,1).all()
assert set(submission[ID]) == set(sample[ID])
submission.to_csv(f'{DATA}/submission.csv', index=False)

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"  Score temporel proxy : {sc_temporal:.4f}  (LB actuel: 0.3498)")
print(f"  submission.csv : {submission.shape[0]:,} lignes  ({elapsed/60:.1f} min)")
print(f"{'='*60}")
print(submission.head(6).to_string())
