#!/usr/bin/env python3
"""
DataTour 2026 — Mobile Money Fraud Detection
v24: EARLY STOPPING TEMPOREL + SIGNAL FRAUDE-DESTINATION (anti-overfitting)
  Diagnostic v22: gap train/holdout AP de 0.50 (train=0.86 vs holdout=0.36)
  avec num_leaves=95 + 600 arbres fixes. Le proxy de validation utilisait
  aussi des hyperparamètres différents (leaves=95/depth=7) du modèle final
  réellement soumis (leaves=63/depth=6) — proxy non représentatif corrigé.
  Nouveautés v23:
    - Split fit/stop temporel (STOP_WINDOW périodes) par fold de validation
    - Early stopping LGB (average_precision) et XGB (aucpr) au lieu de
      n_estimators fixe → nombre d'arbres choisi par les données, pas deviné
    - Nombre d'arbres du modèle final dérivé de la médiane des early-stops
    - Retour aux hyperparamètres stables leaves=63/depth=6 partout
  Nouveautés v24:
    - dest_fraud_rate_op03 / dest_fraud_count_op03 enfin implémentées
      (mentionnées dans le changelog v22 mais jamais codées)
    - Lissage bayésien (k=10) pour éviter les taux 0%/100% sur comptes rares
    - self_exclude: retire la contribution de la ligne à sa propre stat
      quand on construit les features d'entraînement (évite la fuite
      d'étiquette — sans ça un compte à 1 transaction fraude aurait
      dest_fraud_rate=1.0 exactement égal à son propre label)
  Historique LB: 0.312→0.332→0.348→0.352→0.351 (v22, avant ce fix)
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import average_precision_score
import time

np.random.seed(42)
DATA = '/home/pancrace/Bureau/competition'
EPS  = 1e-6

t0 = time.time()
print("=" * 60)
print("DATATOUR 2026  —  FRAUD PIPELINE v22")
print("=" * 60)

# ─── CHARGEMENT ───────────────────────────────────────────────────
print("\n[1/5] Chargement...")
train  = pd.read_csv(f'{DATA}/train.csv')
test   = pd.read_csv(f'{DATA}/test.csv')
sample = pd.read_csv(f'{DATA}/sample_submission (1).csv')

TARGET, ID = 'fraud_flag', 'id'
train03 = train[train['operation'] == 'op_03'].copy()
test03  = test[test['operation']   == 'op_03'].copy()
test03_orig_index = test03.index
BASE_RATE = train03[TARGET].mean()

print(f"  train op03 : {len(train03):,}  fraude: {BASE_RATE:.2%}")
print(f"  test  op03 : {len(test03):,}")
print(f"  periodes   : train 0-{train['period'].max()}  test {test['period'].min()}-{test['period'].max()}")


# ─── PILIER 1 : COMPUTE STATS + VELOCITE ─────────────────────────
def compute_stats(ref_train, vel_long=20, vel_short=5):
    non03    = ref_train[ref_train['operation'] != 'op_03']
    ref_op03 = ref_train[ref_train['operation'] == 'op_03']
    med      = ref_train['amount'].median()
    p_max    = ref_train['period'].max()

    # ─ Baseline comportementale NON-op03 (100% légit) ─
    oc = non03.groupby('origin_account').agg(
        orig_clean_n       =('amount','count'),
        orig_clean_mean    =('amount','mean'),
        orig_clean_std     =('amount','std'),
        orig_clean_max     =('amount','max'),
        orig_clean_q75     =('amount', lambda x: np.percentile(x, 75)),
        orig_clean_bal_mean=('origin_balance_before','mean'),
        orig_clean_bal_std =('origin_balance_before','std'),
    ).reset_index()
    for c in ['orig_clean_std','orig_clean_bal_std']:
        oc[c] = oc[c].fillna(0)

    dc = non03.groupby('destination_account').agg(
        dest_clean_n       =('amount','count'),
        dest_clean_mean    =('amount','mean'),
        dest_clean_std     =('amount','std'),
        dest_clean_bal_mean=('destination_balance_before','mean'),
    ).reset_index()
    dc['dest_clean_std'] = dc['dest_clean_std'].fillna(0)

    # ─ Stats toutes opérations ─
    # sum/sumsq conservés pour permettre un ajustement leave-one-out exact
    # (retirer la contribution d'une ligne à sa propre stat, cf self_exclude)
    oa = ref_train.groupby('origin_account').agg(
        orig_all_n   =('amount','count'),
        orig_all_mean=('amount','mean'),
        orig_all_std =('amount','std'),
        orig_all_sum =('amount','sum'),
        orig_all_sumsq=('amount', lambda x: (x**2).sum()),
        orig_tenure  =('period', lambda x: x.max()-x.min()),
    ).reset_index()
    oa['orig_all_std'] = oa['orig_all_std'].fillna(0)

    da = ref_train.groupby('destination_account').agg(
        dest_all_n       =('amount','count'),
        dest_all_mean    =('amount','mean'),
        dest_all_std     =('amount','std'),
        dest_all_sum     =('amount','sum'),
        dest_all_sumsq   =('amount', lambda x: (x**2).sum()),
        dest_all_bal_mean=('destination_balance_before','mean'),
        dest_all_bal_std =('destination_balance_before','std'),
        dest_all_bal_sum =('destination_balance_before','sum'),
        dest_all_bal_sumsq=('destination_balance_before', lambda x: (x**2).sum()),
    ).reset_index()
    for c in ['dest_all_std','dest_all_bal_std']:
        da[c] = da[c].fillna(0)

    # ─ Réseau ─
    on_net = ref_train.groupby('origin_account')['destination_account'].nunique().reset_index()
    on_net.columns = ['origin_account','orig_n_unique_dests']
    dn_net = ref_train.groupby('destination_account')['origin_account'].nunique().reset_index()
    dn_net.columns = ['destination_account','dest_n_unique_origs']

    # ─ Stats op03 ─
    oo = ref_op03.groupby('origin_account').agg(
        orig_op03_n            =('amount','count'),
        orig_op03_mean         =('amount','mean'),
        orig_op03_std          =('amount','std'),
        orig_op03_sum          =('amount','sum'),
        orig_op03_sumsq        =('amount', lambda x: (x**2).sum()),
        orig_op03_max          =('amount','max'),
        orig_op03_q75          =('amount', lambda x: np.percentile(x, 75)),
        orig_op03_active_periods=('period', 'nunique'),   # NEW: densité temporelle
    ).reset_index()
    oo['orig_op03_std'] = oo['orig_op03_std'].fillna(0)

    do = ref_op03.groupby('destination_account').agg(
        dest_op03_n       =('amount','count'),
        dest_op03_mean    =('amount','mean'),
        dest_op03_std     =('amount','std'),
        dest_op03_sum     =('amount','sum'),
        dest_op03_sumsq   =('amount', lambda x: (x**2).sum()),
        dest_op03_bal_mean=('destination_balance_before','mean'),
    ).reset_index()
    do['dest_op03_std'] = do['dest_op03_std'].fillna(0)

    # ─ NOUVEAU: Diversité op03 de la destination (combien de victimes différentes) ─
    dn_op03 = ref_op03.groupby('destination_account')['origin_account'].nunique().reset_index()
    dn_op03.columns = ['destination_account','dest_n_unique_origs_op03']


    # ─ NOUVEAU: Money mule — destination qui apparaît aussi comme origine ─
    dest_as_orig = ref_train.groupby('origin_account').agg(
        dest_as_orig_n   =('amount','count'),
        dest_as_orig_sum =('amount','sum'),
    ).reset_index()
    dest_as_orig.columns = ['destination_account','dest_as_orig_n','dest_as_orig_sum']

    # ─ VELOCITE 3 fenêtres: 20 / 10 / 5 périodes ─
    recent_l   = ref_train[ref_train['period'] >= p_max - vel_long + 1]
    recent_l03 = recent_l[recent_l['operation'] == 'op_03']

    vel_orig = recent_l.groupby('origin_account').agg(
        orig_vel_n      =('amount','count'),
        orig_vel_n_ops  =('operation','nunique'),
        orig_vel_amt_sum=('amount','sum'),
    ).reset_index()
    vel_orig03 = recent_l03.groupby('origin_account').agg(
        orig_vel_op03_n  =('amount','count'),
        orig_vel_op03_sum=('amount','sum'),
    ).reset_index()
    vel_dest = recent_l.groupby('destination_account').agg(
        dest_vel_n      =('amount','count'),
        dest_vel_amt_sum=('amount','sum'),
    ).reset_index()
    vel_dest03 = recent_l03.groupby('destination_account').agg(
        dest_vel_op03_n=('amount','count'),
    ).reset_index()

    # fenêtre 10 périodes
    vel_mid = 10
    recent_m   = ref_train[ref_train['period'] >= p_max - vel_mid + 1]
    recent_m03 = recent_m[recent_m['operation'] == 'op_03']
    vel10_dest = recent_m.groupby('destination_account').agg(
        dest_vel10_amt_sum=('amount','sum'),
    ).reset_index()
    vel10_dest03 = recent_m03.groupby('destination_account').agg(
        dest_vel10_op03_n=('amount','count'),
    ).reset_index()

    # fenêtre 5 périodes (burst très récent)
    recent_s   = ref_train[ref_train['period'] >= p_max - vel_short + 1]
    recent_s03 = recent_s[recent_s['operation'] == 'op_03']
    vel5_orig03 = recent_s03.groupby('origin_account').agg(
        orig_vel5_op03_n=('amount','count'),
    ).reset_index()
    vel5_dest = recent_s.groupby('destination_account').agg(
        dest_vel5_n      =('amount','count'),
        dest_vel5_amt_sum=('amount','sum'),
    ).reset_index()
    vel5_dest03 = recent_s03.groupby('destination_account').agg(
        dest_vel5_op03_n=('amount','count'),
    ).reset_index()

    # ─ Paires op03 ─
    po = ref_op03.groupby(['origin_account','destination_account']).agg(
        pair_n       =('amount','count'),
        pair_amt_mean=('amount','mean'),
        pair_amt_std =('amount','std'),
        pair_amt_sum =('amount','sum'),
        pair_amt_sumsq=('amount', lambda x: (x**2).sum()),
        pair_amt_max =('amount','max'),
    ).reset_index()
    po['pair_amt_std'] = po['pair_amt_std'].fillna(0)

    # ─ Taux de fraude par destination (mules réutilisées) ─
    # Calculé uniquement à partir de ref_op03 (passé strict pour va/test ;
    # pour les lignes de ref_op03 elles-mêmes, build_feats retire leur
    # propre contribution via self_exclude pour éviter la fuite d'étiquette).
    dest_fraud = ref_op03.groupby('destination_account').agg(
        dest_fraud_count_op03=('fraud_flag','sum'),
        dest_fraud_n_op03    =('fraud_flag','count'),
    ).reset_index()
    op03_fraud_rate = ref_op03['fraud_flag'].mean()

    known_orig       = set(ref_train['origin_account'])
    known_dest       = set(ref_train['destination_account'])
    known_orig_clean = set(non03['origin_account'])
    known_dest_clean = set(non03['destination_account'])

    return dict(
        oc=oc, dc=dc, oa=oa, da=da, on=on_net, dn=dn_net,
        oo=oo, do=do, po=po,
        dn_op03=dn_op03, dest_as_orig=dest_as_orig,
        vel_orig=vel_orig, vel_orig03=vel_orig03,
        vel_dest=vel_dest, vel_dest03=vel_dest03,
        vel10_dest=vel10_dest, vel10_dest03=vel10_dest03,
        vel5_orig03=vel5_orig03,
        vel5_dest=vel5_dest, vel5_dest03=vel5_dest03,
        dest_fraud=dest_fraud, op03_fraud_rate=op03_fraud_rate,
        med=med, known_orig=known_orig, known_dest=known_dest,
        known_orig_clean=known_orig_clean, known_dest_clean=known_dest_clean,
    )


def build_feats(df, S, self_exclude=False):
    d = df.copy().reset_index(drop=True)
    amt = d['amount']
    ob  = d['origin_balance_before']
    oa_ = d['origin_balance_after']
    db  = d['destination_balance_before']
    da_ = d['destination_balance_after']
    med = S['med']

    # ─ Transaction pures ─
    d['mismatch_orig']     = (ob - oa_ - amt).abs()
    d['mismatch_orig_log'] = np.log1p(d['mismatch_orig'])
    d['mismatch_dest']     = (da_ - db - amt).abs()
    d['orig_neg_after']    = (oa_ < 0).astype('int8')
    d['dest_zero_before']  = (db == 0).astype('int8')
    d['orig_zero_after']   = (oa_ == 0).astype('int8')
    d['dest_zero_after']   = (da_ < 1).astype('int8')
    d['orig_drained']      = (oa_ < 0.01 * ob.abs() + EPS).astype('int8')
    d['dest_bal_decrease'] = (da_ < db).astype('int8')  # NEW: anomalie comptable
    d['amount_log']        = np.log1p(amt)
    d['orig_bal_log']      = np.sign(ob) * np.log1p(ob.abs())
    d['dest_bal_log']      = np.log1p(db.abs())
    d['amt_ratio_orig']    = (amt / (ob.abs() + EPS)).clip(0, 10)
    d['amt_ratio_dest']    = (amt / (db.abs() + EPS)).clip(0, 10)
    d['exceeds_orig']      = (amt > ob).astype('int8')
    d['net_flow_norm']     = ((oa_ - ob) + (da_ - db)) / (amt + EPS)
    d['balance_ratio']     = (db / (ob.abs() + EPS)).clip(-10, 10)

    # ─ Nouveauté ─
    d['orig_is_new']        = (~d['origin_account'].isin(S['known_orig'])).astype('int8')
    d['dest_is_new']        = (~d['destination_account'].isin(S['known_dest'])).astype('int8')
    d['orig_no_clean_hist'] = (~d['origin_account'].isin(S['known_orig_clean'])).astype('int8')
    d['dest_no_clean_hist'] = (~d['destination_account'].isin(S['known_dest_clean'])).astype('int8')

    # ─ Joins ─
    d = (d.merge(S['oc'],           on='origin_account',                        how='left')
          .merge(S['dc'],           on='destination_account',                   how='left')
          .merge(S['oa'],           on='origin_account',                        how='left')
          .merge(S['da'],           on='destination_account',                   how='left')
          .merge(S['on'],           on='origin_account',                        how='left')
          .merge(S['dn'],           on='destination_account',                   how='left')
          .merge(S['oo'],           on='origin_account',                        how='left')
          .merge(S['do'],           on='destination_account',                   how='left')
          .merge(S['dn_op03'],          on='destination_account',                   how='left')
          .merge(S['dest_as_orig'],     on='destination_account',                   how='left')
          .merge(S['vel_orig'],         on='origin_account',                        how='left')
          .merge(S['vel_orig03'],   on='origin_account',                        how='left')
          .merge(S['vel_dest'],      on='destination_account',                   how='left')
          .merge(S['vel_dest03'],   on='destination_account',                   how='left')
          .merge(S['vel10_dest'],   on='destination_account',                   how='left')
          .merge(S['vel10_dest03'], on='destination_account',                   how='left')
          .merge(S['vel5_orig03'],  on='origin_account',                        how='left')
          .merge(S['vel5_dest'],    on='destination_account',                   how='left')
          .merge(S['vel5_dest03'],  on='destination_account',                   how='left')
          .merge(S['dest_fraud'],   on='destination_account',                   how='left')
          .merge(S['po'],           on=['origin_account','destination_account'], how='left'))

    fills = {
        'orig_clean_n':0,'orig_clean_mean':med,'orig_clean_std':0,
        'orig_clean_max':med,'orig_clean_q75':med,'orig_clean_bal_mean':0,'orig_clean_bal_std':0,
        'dest_clean_n':0,'dest_clean_mean':med,'dest_clean_std':0,'dest_clean_bal_mean':0,
        'orig_all_n':0,'orig_all_mean':med,'orig_all_std':0,'orig_tenure':0,
        'orig_all_sum':0,'orig_all_sumsq':0,
        'dest_all_n':0,'dest_all_mean':med,'dest_all_std':0,'dest_all_bal_mean':0,'dest_all_bal_std':0,
        'dest_all_sum':0,'dest_all_sumsq':0,'dest_all_bal_sum':0,'dest_all_bal_sumsq':0,
        'orig_n_unique_dests':0,'dest_n_unique_origs':0,
        'orig_op03_n':0,'orig_op03_mean':med,'orig_op03_std':0,'orig_op03_max':0,
        'orig_op03_q75':0,'orig_op03_active_periods':0,
        'orig_op03_sum':0,'orig_op03_sumsq':0,
        'dest_op03_n':0,'dest_op03_mean':med,'dest_op03_std':0,'dest_op03_bal_mean':0,
        'dest_op03_sum':0,'dest_op03_sumsq':0,
        'dest_n_unique_origs_op03':0,
        'dest_as_orig_n':0,'dest_as_orig_sum':0,
        'dest_vel10_amt_sum':0,'dest_vel10_op03_n':0,
        'orig_vel_n':0,'orig_vel_n_ops':0,'orig_vel_amt_sum':0,
        'orig_vel_op03_n':0,'orig_vel_op03_sum':0,
        'dest_vel_n':0,'dest_vel_amt_sum':0,'dest_vel_op03_n':0,
        'orig_vel5_op03_n':0,
        'dest_vel5_n':0,'dest_vel5_amt_sum':0,'dest_vel5_op03_n':0,
        'pair_n':0,'pair_amt_mean':med,'pair_amt_std':0,'pair_amt_max':0,
        'pair_amt_sum':0,'pair_amt_sumsq':0,
        'dest_fraud_count_op03':0,'dest_fraud_n_op03':0,
    }
    for col, val in fills.items():
        if col in d.columns:
            d[col] = d[col].fillna(val)

    # ─ Leave-one-out: retire la contribution de la ligne à ses propres
    # agrégats quand df EST la source de ces stats (fit du train), pour que
    # la distribution des features soit la même au train qu'au test (où la
    # ligne n'est jamais dans ses propres stats). Sans ça, train voit une
    # version "je me connais déjà" de chaque feature que test ne voit jamais.
    def _loo(n, sum_, sumsq_, own, fallback_mean):
        n = n.to_numpy(dtype='float64'); sum_ = sum_.to_numpy(dtype='float64')
        sumsq_ = sumsq_.to_numpy(dtype='float64'); own = np.asarray(own, dtype='float64')
        n_adj   = n - 1
        sum_adj = sum_ - own
        safe_n  = np.where(n_adj > 0, n_adj, 1.0)
        mean_adj = np.where(n_adj > 0, sum_adj / safe_n, fallback_mean)
        sumsq_adj = sumsq_ - own**2
        safe_n2  = np.where(n_adj > 1, n_adj - 1, 1.0)
        var_adj  = np.where(n_adj > 1, (sumsq_adj - n_adj*mean_adj**2) / safe_n2, 0.0)
        std_adj  = np.sqrt(np.clip(var_adj, 0, None))
        return np.clip(n_adj, 0, None), mean_adj, std_adj

    if self_exclude:
        dest_all_n_orig = d['dest_all_n'].copy()
        d['orig_all_n'], d['orig_all_mean'], d['orig_all_std'] = \
            _loo(d['orig_all_n'], d['orig_all_sum'], d['orig_all_sumsq'], amt, med)
        d['dest_all_n'], d['dest_all_mean'], d['dest_all_std'] = \
            _loo(d['dest_all_n'], d['dest_all_sum'], d['dest_all_sumsq'], amt, med)
        _, d['dest_all_bal_mean'], d['dest_all_bal_std'] = \
            _loo(dest_all_n_orig, d['dest_all_bal_sum'], d['dest_all_bal_sumsq'], db, 0)
        d['orig_op03_n'], d['orig_op03_mean'], d['orig_op03_std'] = \
            _loo(d['orig_op03_n'], d['orig_op03_sum'], d['orig_op03_sumsq'], amt, med)
        d['dest_op03_n'], d['dest_op03_mean'], d['dest_op03_std'] = \
            _loo(d['dest_op03_n'], d['dest_op03_sum'], d['dest_op03_sumsq'], amt, med)
        d['pair_n'], d['pair_amt_mean'], d['pair_amt_std'] = \
            _loo(d['pair_n'], d['pair_amt_sum'], d['pair_amt_sumsq'], amt, med)

    # ─ Taux de fraude par destination (lissage bayésien anti-bruit) ─
    # self_exclude=True uniquement quand df EST la source de S['dest_fraud']
    # (fit du train) : on retire la contribution de la ligne à sa propre
    # stat pour ne pas laisser l'étiquette fuiter dans sa propre feature.
    if self_exclude:
        d['dest_fraud_count_op03'] -= d['fraud_flag'].values
        d['dest_fraud_n_op03']     -= 1
    FRAUD_SMOOTH_K = 10
    prior = S['op03_fraud_rate']
    d['dest_fraud_rate_op03'] = ((d['dest_fraud_count_op03'] + FRAUD_SMOOTH_K*prior) /
                                  (d['dest_fraud_n_op03'] + FRAUD_SMOOTH_K))
    d['dest_fraud_count_log'] = np.log1p(d['dest_fraud_count_op03'].clip(lower=0))

    # Alias post-merge
    amt = d['amount']
    ob  = d['origin_balance_before']

    # ─ Features comportementales ─
    d['amt_z_clean_orig']   = ((amt-d['orig_clean_mean'])/(d['orig_clean_std']+EPS)).clip(-5,5)
    d['amt_z_all_orig']     = ((amt-d['orig_all_mean'])  /(d['orig_all_std']  +EPS)).clip(-5,5)
    d['amt_z_all_dest']     = ((amt-d['dest_all_mean'])  /(d['dest_all_std']  +EPS)).clip(-5,5)
    d['amt_z_op03_orig']    = ((amt-d['orig_op03_mean']) /(d['orig_op03_std'] +EPS)).clip(-5,5)
    d['bal_z_clean_orig']   = ((ob -d['orig_clean_bal_mean'])/(d['orig_clean_bal_std']+EPS)).clip(-5,5)
    d['amt_pct_clean_max']  = (amt/(d['orig_clean_max']+EPS)).clip(0,5)
    d['amt_pct_op03_max']   = (amt/(d['orig_op03_max'] +EPS)).clip(0,5)
    d['amt_vs_q75_clean']   = (amt > d['orig_clean_q75']).astype('int8')
    d['amt_vs_q75_op03']    = (amt > d['orig_op03_q75']).astype('int8')
    d['orig_vol_clean_log'] = np.log1p(d['orig_clean_n'])
    d['dest_vol_clean_log'] = np.log1p(d['dest_clean_n'])
    d['orig_vol_op03_log']  = np.log1p(d['orig_op03_n'])
    d['dest_vol_op03_log']  = np.log1p(d['dest_op03_n'])
    d['orig_tenure_log']    = np.log1p(d['orig_tenure'])
    d['orig_net_log']       = np.log1p(d['orig_n_unique_dests'])
    d['dest_net_log']       = np.log1p(d['dest_n_unique_origs'])
    d['pair_is_new']        = (d['pair_n']==0).astype('int8')
    d['pair_amt_z']         = ((amt-d['pair_amt_mean'])/(d['pair_amt_std']+EPS)).clip(-5,5)
    d['pair_exceeds_max']   = (amt>d['pair_amt_max']).astype('int8')

    # ─ VELOCITE LONGUE (20 périodes) ─
    d['orig_vel_op03_log']  = np.log1p(d['orig_vel_op03_n'])
    d['dest_vel_op03_log']  = np.log1p(d['dest_vel_op03_n'])
    d['orig_vel_ratio']     = (d['orig_vel_op03_n'] / (d['orig_op03_n'] + EPS)).clip(0,1)
    d['dest_vel_ratio']     = (d['dest_vel_op03_n'] / (d['dest_op03_n'] + EPS)).clip(0,1)
    d['orig_vel_active']    = (d['orig_vel_op03_n'] > 0).astype('int8')
    d['dest_vel_active']    = (d['dest_vel_op03_n'] > 0).astype('int8')
    d['orig_vel_amt_log']   = np.log1p(d['orig_vel_amt_sum'])

    # ─ VELOCITE COURTE (5 périodes) — burst très récent ─
    d['orig_vel5_op03_log'] = np.log1p(d['orig_vel5_op03_n'])
    d['dest_vel5_amt_log']  = np.log1p(d['dest_vel5_amt_sum'])
    d['dest_vel5_op03_log'] = np.log1p(d['dest_vel5_op03_n'])
    # Burst = vel5 vs vel20: si burst5 > burst20 moyen → rafale récente
    d['orig_burst_accel']   = (d['orig_vel5_op03_n'] / (d['orig_vel_op03_n'] / 4.0 + EPS)).clip(0, 10)
    d['dest_burst_accel']   = (d['dest_vel5_op03_n'] / (d['dest_vel_op03_n'] / 4.0 + EPS)).clip(0, 10)
    d['dest_recv_accel']    = (d['dest_vel5_amt_sum'] / (d['dest_vel_amt_sum'] / 4.0 + EPS)).clip(0, 10)

    # ─ NOUVEAU: Densité op03 (activité par période active) ─
    d['orig_op03_per_period']  = (d['orig_op03_n'] / (d['orig_op03_active_periods'] + EPS)).clip(0, 20)
    # Burst actuel vs densité typique: ratio vel5/per_period (recent burst vs typical frequency)
    d['orig_op03_burst_ratio'] = (d['orig_vel5_op03_n'] / (d['orig_op03_per_period'] + EPS)).clip(0, 10)

    # ─ NOUVEAU: Money mule — destination envoie autant qu'elle reçoit ─
    d['dest_as_orig_log']   = np.log1p(d['dest_as_orig_n'])
    # Ratio: combien de transactions la destination initie / (total send + recv)
    d['dest_mule_ratio']    = (d['dest_as_orig_n'] /
                               (d['dest_as_orig_n'] + d['dest_all_n'] + EPS)).clip(0, 1)
    # Volume relatif: la destination envoie beaucoup en abs?
    d['dest_as_orig_sum_log'] = np.log1p(d['dest_as_orig_sum'])
    # Destination est-elle un relais actif? (envoie ET reçoit beaucoup)
    d['dest_relay_score']   = (d['dest_as_orig_log'] * d['dest_vol_op03_log']).clip(0, 25)


    # ─ Interactions ─
    d['drain_to_new']       = d['orig_drained'] * d['dest_is_new']
    d['drain_to_newpair']   = d['orig_drained'] * d['pair_is_new']
    d['high_ratio_newdest'] = (d['amt_ratio_orig'] > 0.5).astype('int8') * d['dest_is_new']
    d['exceed_and_newpair'] = d['pair_exceeds_max'] * d['pair_is_new']
    d['bal_z_x_ratio']      = (d['bal_z_clean_orig'] * d['amt_ratio_orig']).clip(-10,10)
    d['vel_x_drain']        = d['orig_vel_op03_n'] * d['orig_drained']
    d['new_dest_vel_burst'] = d['dest_is_new'] * d['orig_vel_op03_log']
    # NEW: large amount → mule destination
    d['large_to_mule']      = (d['amt_ratio_orig'] > 0.5).astype('int8') * d['dest_mule_ratio']
    # NEW: burst + drain = most suspicious
    d['burst_drain']        = d['orig_vel5_op03_n'] * d['orig_drained']
    d['new_dest_burst5']    = d['dest_is_new'] * d['orig_vel5_op03_log']

    # ─ NOUVEAU: Diversité op03 de la destination (combien de victimes différentes) ─
    d['dest_origs_op03_log']   = np.log1p(d['dest_n_unique_origs_op03'])
    # Ratio: chaque expéditeur op03 envoie en moyenne combien de fois?
    d['dest_avg_op03_per_orig'] = (d['dest_op03_n'] /
                                   (d['dest_n_unique_origs_op03'] + EPS)).clip(0, 20)
    # Si chaque expéditeur envoie exactement 1 fois → ratio ≈ 1 → typiquement mule collectrice

    # ─ Fenêtre 10 périodes (intermédiaire) ─
    d['dest_vel10_amt_log']    = np.log1p(d['dest_vel10_amt_sum'])
    d['dest_vel10_op03_log']   = np.log1p(d['dest_vel10_op03_n'])
    # Accélération 10→20 et 5→10
    d['dest_accel_10_20']      = (d['dest_vel10_amt_sum'] /
                                   (d['dest_vel_amt_sum'] / 2.0 + EPS)).clip(0, 4)
    d['dest_accel_5_10']       = (d['dest_vel5_amt_sum'] /
                                   (d['dest_vel10_amt_sum'] / 2.0 + EPS)).clip(0, 4)

    return d


# ─── PILIER 3 : VALIDATION TEMPORELLE MULTI-SPLIT (early stopping) ─
print("\n[2/5] Validation temporelle (3 splits, early stopping)...")

SPLITS = [(75,"0-74→75-105"), (85,"0-84→85-105"), (95,"0-94→95-105")]
DROP_COLS = {'id','fraud_flag','operation','origin_account','destination_account',
             'period','op_encoded'}

STOP_WINDOW = 15   # périodes réservées à l'early stopping, exclues du fit
MAX_TREES   = 3000 # plafond ; le nombre réel est choisi par early stopping
ES_ROUNDS   = 100

proxy_scores = {}
lgb_best_iters, xgb_best_iters = [], []

for VAL_PERIOD, lbl in SPLITS:
    tr = train[train['period'] <  VAL_PERIOD]
    va = train[train['period'] >= VAL_PERIOD]
    stop_start = VAL_PERIOD - STOP_WINDOW
    tr_fit  = tr[tr['period'] <  stop_start]
    tr_stop = tr[tr['period'] >= stop_start]

    tr_fit03  = tr_fit[tr_fit['operation']=='op_03'].reset_index(drop=True)
    tr_stop03 = tr_stop[tr_stop['operation']=='op_03'].reset_index(drop=True)
    va03      = va[va['operation']=='op_03'].reset_index(drop=True)

    # Stats calculées uniquement sur la portion fit (jamais sur stop/val)
    S_fit   = compute_stats(tr_fit)
    fit_fe  = build_feats(tr_fit03,  S_fit, self_exclude=True)
    stop_fe = build_feats(tr_stop03, S_fit)
    va_fe   = build_feats(va03,      S_fit)
    fc = [c for c in fit_fe.columns if c not in DROP_COLS and c in va_fe.columns]

    Xfit  = fit_fe[fc].astype('float32').fillna(-999)
    Xstop = stop_fe[fc].astype('float32').fillna(-999)
    Xva_  = va_fe[fc].astype('float32').fillna(-999)
    yfit  = tr_fit03[TARGET].values
    ystop = tr_stop03[TARGET].values
    yva_  = va03[TARGET].values
    spw   = (1-yfit.mean())/yfit.mean()

    # LGB — early stopping sur le PR-AUC de la fenêtre stop
    m_lgb = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        boosting_type='gbdt', n_estimators=MAX_TREES, learning_rate=0.03,
        num_leaves=63, min_child_samples=30,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
        n_jobs=8, random_state=42, verbose=-1,
    )
    m_lgb.fit(Xfit, yfit, eval_set=[(Xstop, ystop)],
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False), lgb.log_evaluation(False)])
    p_lgb = m_lgb.predict_proba(Xva_)[:,1]
    lgb_best_iters.append(m_lgb.best_iteration_)

    # XGB — early stopping sur le PR-AUC (aucpr) de la fenêtre stop
    m_xgb = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='aucpr',
        n_estimators=MAX_TREES, learning_rate=0.03,
        max_depth=6, min_child_weight=30,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
        n_jobs=8, random_state=42, tree_method='hist', verbosity=0,
        early_stopping_rounds=ES_ROUNDS,
    )
    m_xgb.fit(Xfit, yfit, eval_set=[(Xstop, ystop)], verbose=False)
    p_xgb = m_xgb.predict_proba(Xva_)[:,1]
    xgb_best_iters.append(m_xgb.best_iteration)

    # Trouver le meilleur poids LGB (grille élargie incl. 0=XGB-only)
    best_w, best_sc = 0.5, 0
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        p_ens = w*p_lgb + (1-w)*p_xgb
        va_all = np.zeros(len(va))
        va_all[np.where(va['operation'].values=='op_03')[0]] = p_ens
        sc = average_precision_score(va[TARGET].values, va_all)
        if sc > best_sc:
            best_sc, best_w = sc, w

    proxy_scores[lbl] = (best_sc, best_w)
    print(f"  {lbl} → AP={best_sc:.4f}  poids LGB={best_w:.1f}  "
          f"arbres LGB={m_lgb.best_iteration_}  XGB={m_xgb.best_iteration}")

mean_proxy  = np.mean([v[0] for v in proxy_scores.values()])
best_w_avg  = np.mean([v[1] for v in proxy_scores.values()])
N_TREES_LGB = max(50, int(np.median(lgb_best_iters) * 1.3))
N_TREES_XGB = max(50, int(np.median(xgb_best_iters) * 1.3))
print(f"\n  ► Score proxy moyen : {mean_proxy:.4f}  (ref LB v22: 0.3512 / best v17: 0.3518)")
print(f"  ► Poids LGB optimal : {best_w_avg:.2f}")
print(f"  ► Arbres finaux (médiane early-stop ×1.3): LGB={N_TREES_LGB}  XGB={N_TREES_XGB}")

# Features importantes du dernier split
imp = pd.Series(m_lgb.feature_importances_, index=fc).sort_values(ascending=False)
print(f"\n  Top 20 features:")
for f, v in imp.head(20).items():
    print(f"    {f:<42} {v:>6.0f}")

# Garde-fou: n'aller plus loin que si proxy > seuil
# NB: seuil calibré sur l'ancien proxy (hyperparams différents du modèle
# final soumis) — à recalibrer une fois le nouveau pipeline validé.
PROXY_THRESHOLD = 0.361   # au-dessus du proxy v20 (0.3609)
if mean_proxy < PROXY_THRESHOLD:
    print(f"\n  ⚠ PROXY {mean_proxy:.4f} < {PROXY_THRESHOLD} — génération submission ANNULÉE")
    print(f"  Conservation de l'ancienne submission.csv (LB 0.351159)")
    import sys; sys.exit(0)
print(f"\n  ✓ Proxy {mean_proxy:.4f} >= {PROXY_THRESHOLD} — génération submission GO")

# ─── MODELE FINAL ─────────────────────────────────────────────────
print(f"\n[3/5] Stats finales sur tout le train...")
S_full = compute_stats(train)

print(f"\n[4/5] Ensemble final (LGB={N_TREES_LGB}/XGB={N_TREES_XGB} arbres, 3 LGB + 2 XGB graines)...")
tr03_fe = build_feats(train03, S_full, self_exclude=True)
te03_fe = build_feats(test03,  S_full)

fc_f = [c for c in tr03_fe.columns if c not in DROP_COLS and c in te03_fe.columns]
Xf = tr03_fe[fc_f].astype('float32').fillna(-999)
Xt = te03_fe.reindex(columns=fc_f, fill_value=-999).astype('float32').fillna(-999)
yf = train03[TARGET].values
spw_f = (1-yf.mean())/yf.mean()

# 3 LGB seeds (num_leaves=63 stable)
final_lgb_preds = []
for seed in [42, 123, 777]:
    m = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss',
        boosting_type='gbdt', n_estimators=N_TREES_LGB, learning_rate=0.03,
        num_leaves=63, min_child_samples=30,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw_f,
        n_jobs=2, random_state=seed, verbose=-1,
    )
    m.fit(Xf, yf, callbacks=[lgb.log_evaluation(False)])
    final_lgb_preds.append(m.predict_proba(Xt)[:,1])

# 2 XGB seeds (hyperparams stables)
final_xgb_preds = []
for seed in [42, 99]:
    m_x = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        n_estimators=N_TREES_XGB, learning_rate=0.03,
        max_depth=6, min_child_weight=30,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw_f,
        n_jobs=2, random_state=seed, tree_method='hist', verbosity=0,
    )
    m_x.fit(Xf, yf)
    final_xgb_preds.append(m_x.predict_proba(Xt)[:,1])

p_lgb_final = np.mean(final_lgb_preds, axis=0)
p_xgb_final = np.mean(final_xgb_preds, axis=0)
pred_op03 = best_w_avg * p_lgb_final + (1 - best_w_avg) * p_xgb_final
print(f"  op03 mean={pred_op03.mean():.4f}  max={pred_op03.max():.4f}  >0.5:{(pred_op03>0.5).mean():.2%}")

# ─── SOUMISSION ───────────────────────────────────────────────────
print(f"\n[5/5] Generation submission.csv...")
pred_full = pd.Series(0.0, index=test.index)
pred_full.loc[test03_orig_index] = pred_op03

submission = pd.DataFrame({ID: test[ID], 'target': pred_full.clip(0,1).values})
assert submission.shape[0] == sample.shape[0]
assert list(submission.columns) == [ID, 'target']
assert submission[ID].is_unique
assert submission['target'].between(0,1).all()
assert set(submission[ID]) == set(sample[ID])
submission.to_csv(f'{DATA}/submission.csv', index=False)

non_zero_ops = test.loc[submission['target'].values > 0, 'operation'].value_counts()
print(f"\n  Vérif alignement: opérations avec score>0:")
print(f"    {dict(non_zero_ops)}")

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"  Score proxy moyen : {mean_proxy:.4f}  (LB ref: 0.3518)")
print(f"  submission.csv    : {submission.shape[0]:,} lignes  ({elapsed/60:.1f} min)")
print(f"{'='*60}")
print(submission.head(6).to_string())
