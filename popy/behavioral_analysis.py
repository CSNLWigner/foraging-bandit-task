import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from statsmodels.stats.multitest import multipletests


def fit_glm(behav, target='switch', predictors=None, n_permutations=1000):
    # prepare data: get predictors, target (switch), and monkey (to handle them separately)
    df_glm = behav[predictors + [target, 'monkey']].copy()
    df_glm[target] = -1 * df_glm[target] + 1  # switch column to 0 and 1 instead of 1 and 0
    df_glm[target] = df_glm[target].astype(int)  # switch column to int and flip sign

    # Fit the model for each monkey
    res = []
    for monkey, data_unbalanced in df_glm.groupby('monkey'):
        # Split dataset
        X = data_unbalanced[predictors]
        y = data_unbalanced[target]

        # 1: Fit GLM on full data to get real coefficients
        # Define the formula
        model = LogisticRegression(fit_intercept=True, class_weight='balanced', max_iter=1000, penalty=None)
        model.fit(X, y)
        real_coeffs = model.coef_.flatten()

        permuted_coefs = np.zeros((n_permutations, X.shape[1]))
        for i in range(n_permutations):
            if i % 100 == 0:
                print(f'Permutation {i+1}/{n_permutations} for monkey {monkey}')
            y_perm = np.random.permutation(y)
            model_perm = LogisticRegression(class_weight='balanced', max_iter=1000)
            model_perm.fit(X, y_perm)
            permuted_coefs[i, :] = model_perm.coef_.flatten()
        # Compute two-sided empirical p-values
        p_values = np.mean(
            np.abs(permuted_coefs) >= np.abs(real_coeffs), axis=0
        )
        # FDR (Benjamini-Hochberg) correction across predictors, within monkey
        _, p_values_fdr, _, _ = multipletests(p_values, method='fdr_bh')

        # Results as DataFrame
        for predictor, coeff, p_value, p_value_fdr in zip(predictors, real_coeffs, p_values, p_values_fdr):
            res.append({
                'monkey': monkey,
                'variable': predictor,
                'coeffs': coeff,
                'pvalue': p_value,
                'pvalue_fdr': p_value_fdr,
                'if_significant': p_value_fdr < 0.05
            })

        # 2: Model evaluation

        # cross-validation
        cv_model = cross_validate(model, X, y, cv=10, return_estimator=True, scoring='accuracy')
        scores_cv = cv_model['test_score']

        print(f'Monkey: {monkey}, CV accuracy: {scores_cv.mean():.2f} +/- {scores_cv.std():.2f}')

    return pd.DataFrame(res)