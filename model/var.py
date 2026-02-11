
def get_features(records):
    def extract(record):
        x = np.asarray(record['x'][:-1], float)
        t = np.asarray(record['t'][:-1], float)
        s = np.asarray(record['s'][:-1], int)
        
        if len(x) < 2:
            return None
        
        n_normal = (s == 1).sum()
        mu_true = np.mean(x)
        mu_true_normal = np.mean(x[s == 1])
        var_true = np.var(x)
        var_true_normal = np.var(x[s == 1])
        var_true_normal = var_true_normal if n_normal > 0 else 0
        
        n_obs = len(x)
        t_delta_last = t[-1] - t[-2]
        t_duration = t.max() - t.min()
        p_normal = n_normal / n_obs

        eps = 1e-8
        true_cv = np.sqrt(max(var_true, 0.0)) / (abs(mu_true) + eps)
        log_true_cv = np.log(true_cv + eps)
        true_cv_normal = np.sqrt(max(var_true_normal, 0.0)) / (abs(mu_true_normal) + eps)
        log_true_cv_normal = np.log(true_cv_normal + eps)
        
        return {
            'pid': record['pid'],
            'code': INVERSE_TEST_VOCAB.get(int(record['cid']), record['cid']),
            'mu_true': mu_true,
            'var_true': var_true,
            'var_true_normal': var_true_normal,
            'n_obs': n_obs,
            't_delta_last': t_delta_last,
            't_duration': t_duration,
            'n_normal': n_normal,
            'p_normal': p_normal,
            'true_cv': true_cv,
            'log_true_cv': log_true_cv,
            'true_cv_normal': true_cv_normal,
            'log_true_cv_normal': log_true_cv_normal,
        }

    rows = [r for r in (extract(rec) for rec in records) if r is not None]
    return pd.DataFrame(rows)

def _lin_reg(X, y):
    m = LinearRegression()
    m.fit(X, y)
    coef = dict(zip(X.columns, m.coef_))
    std_err = None
    if hasattr(m, "coef_"):
        pred = m.predict(X)
        n, p = X.shape
        resid = y - pred
        s2 = np.sum(resid**2) / (n - p)
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        if m.fit_intercept:
            X_mat = np.column_stack([np.ones(X_mat.shape[0]), X_mat])
        else:
            X_mat = np.array(X_mat)
        try:
            XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
            se = np.sqrt(np.diag(s2 * XtX_inv))
            if m.fit_intercept:
                se = se[1:]  # drop intercept
            std_err = dict(zip(X.columns, se))
        except Exception:
            std_err = None
    r2 = m.score(X, y)
    r2 = r2_score(y, pred)
    return r2, coef, std_err

def build_X(df, feats, code_col='code', scale=True, drop_first=True):
    X = df[feats].copy()
    log_cols = [c for c in ['n_obs', 't_delta_last', 't_duration'] if c in X.columns]
    for c in log_cols:
        X[c] = np.log1p(X[c])
    dummies = pd.get_dummies(df[code_col].astype(str), prefix='code', drop_first=drop_first)
    if scale and not X.empty:
        scaler = StandardScaler()
        X[X.columns] = scaler.fit_transform(X[X.columns])
    combos = {
        f'{f}__x__{d}': X[f].values * dummies[d].values
        for f in X.columns for d in dummies.columns
    }
    Xall = pd.concat([X, dummies, pd.DataFrame(combos, index=X.index)], axis=1)
    return Xall

def build_y(df, y_col="std", zscore=False):
    """
    Get y for regression, z-score if wanted.
    """
    y = pd.Series(df[y_col], index=df.index)
    if zscore:
        mu = y.mean(skipna=True)
        sd = y.std(ddof=0, skipna=True) or 1.0
        y = (y - mu) / (sd if (sd != 0 and np.isfinite(sd)) else 1.0)
    return y

def fit_pooled(
    df,
    model_fn, 
    feats,
    code_col='code',
    scale=True,
    y_col="log_var",
    zscore_y=True,
    drop_first=True
):
    """
    Fit model to data pooled across all codes.
    """
    X = build_X(df, feats, code_col=code_col, scale=scale, drop_first=drop_first)
    y = build_y(df, y_col=y_col, zscore=zscore_y).rename('y')
    data = pd.concat([X, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    Xc, yc = data.drop(columns=['y']), data['y']
    r2, coefs, se = model_fn(Xc, yc)
    out = {
        'model': model_fn.__name__,
        'cid': 'ALL',
        'scale': scale,
        'r2': r2,
        'std_err': se,
        'n_fit': len(yc),
        'n_dropped': len(y) - len(yc)
    }
    out.update(coefs)
    return out, yc

def fit_per_code(
    df,
    model_fn,
    feats,
    code_col='code',
    scale=True,
    y_col="log_var",
    zscore_y=True,
    drop_first=True
):
    """
    Fit separate model for each code with at least 3 rows.
    """
    res = []
    y_by_code = {}
    for code in df[code_col].unique():
        df_sub = df[df[code_col] == code]
        if len(df_sub) < 3:
            continue
        out, y_arr = fit_pooled(
            df_sub, model_fn, feats, code_col=code_col,
            scale=scale, y_col=y_col, zscore_y=zscore_y, drop_first=drop_first
        )
        out['cid'] = code
        res.append(out)
        y_by_code[code] = y_arr
    return res, y_by_code

def main():
    results, state_col = load_counterfactual_predictions([MODEL], LOG_DIR)
    results = results[results[state_col] == 1]
    pairs = results[['pid', 'cid']].drop_duplicates().values.tolist()

    # (3) Get features for those (pid, cid) pairs
    test_features_file = f'../data/processed/combined_sequences_v2_test_features.csv'
    test_pickle = f'../data/processed/combined_sequences_v2.pkl'

    if not os.path.exists(test_features_file):
        import pickle
        with open(test_pickle, 'rb') as f:
            test_seq = pickle.load(f)
        pairset = set(map(tuple, pairs))
        test_seq = [seq for seq in test_seq if (seq['pid'], seq['cid']) in pairset]
        feats_df = get_features(test_seq).sort_values('code')
        feats_df.to_csv(test_features_file, index=False)
    else:
        feats_df = pd.read_csv(test_features_file)

    # (4) Merge features + NORMA outputs
    df = feats_df.merge(results, on=['pid', 'code'], how='inner')
    df['log_var_true'] = np.log(df['var_true'] + 1e-8)
    df['log_var_pred'] = np.log(df['var_pred'] + 1e-8) if 'var_pred' in df.columns else np.nan    # in case you have predicted variance
    df['std_pred'] = np.sqrt(df['var_pred'] + 1e-8) if 'var_pred' in df.columns else np.nan
    df['std_true'] = np.sqrt(df['var_true'] + 1e-8)
    
    # (5) Regression per code to predict log variance
    # Set up your features and regression target for log(variance)
    feats = ['std_true']  # Use as a feature; you can add more, e.g. clinical features
    y_feature = 'log_var_true'    # Predicting log(var_pred) from log(var_true) or given features
    print(f"Using features: {feats} to predict {y_feature}")
    print(df[feats + [y_feature]].head())
    # If there's no predicted variance, you can swap y_feature to 'log_var_true' and set X to your features.
    # For demonstration, we'll do log_var_pred ~ log_var_true

    pc_results, pc_ys = fit_per_code(
        df,
        _lin_reg,
        feats=feats,
        code_col='code',
        scale=True,
        y_col=y_feature,
        zscore_y=True,
        drop_first=True
    )

    # Prepare results as before
    records = []
    for r in pc_results:
        rec = {
            'code': r.get('cid', '??'),
            'r2': r.get('r2', np.nan),
            'n_fit': r.get('n_fit', 0),
        }
        coefs = {k: v for k, v in r.items() if isinstance(v, (float, np.floating, int, np.integer)) and k not in ['r2', 'n_fit']}
        if r.get('std_err', None) is not None and isinstance(r['std_err'], dict):
            for k, v in r['std_err'].items():
                rec[f'std_err_{k}'] = v
        for k, v in coefs.items():
            rec[f'coef_{k}'] = v
        records.append(rec)

    reg_results_df = pd.DataFrame.from_records(records)
    cols = ['code', 'r2', 'n_fit'] \
        + sorted([c for c in reg_results_df.columns if c.startswith('coef_')]) \
        + sorted([c for c in reg_results_df.columns if c.startswith('std_err_')])

    reg_results_df = reg_results_df[cols]
    print(reg_results_df)
    print(reg_results_df.r2.mean(), reg_results_df.r2.std())
    
    output_path = f'../model/logs/{MODEL}/logvar_regression_per_code.csv'
    reg_results_df.to_csv(output_path, index=False)
    print(f"\nLog(Variance) regression metrics (per code) saved to: {output_path}")

if __name__ == "__main__":
    
    main()