import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import early_stopping
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier

np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
sns.set_theme(style="darkgrid")

df = pd.read_csv("cs-training.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
df.drop_duplicates(inplace=True)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.columns)
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
notnl = df[df['MonthlyIncome'].notnull()]
isnl = df[df['MonthlyIncome'].isnull()]
print(notnl['DebtRatio'].median())
print(isnl['DebtRatio'].median())
print(notnl['SeriousDlqin2yrs'].value_counts(normalize=True))
print(isnl['SeriousDlqin2yrs'].value_counts(normalize=True))
df['AgeGroup'] = pd.cut(df['age'], bins=[-1, 30, 45, 60, 75, 120], labels=['<30', '30-45', '45-60', '60-75', '75+'])

df['MonthlyIncome'] = df.groupby(['AgeGroup', 'NumberOfOpenCreditLinesAndLoans'])['MonthlyIncome'].transform(lambda x: x.fillna(x.median()))
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

ahui = df['MonthlyIncome'].quantile(0.999)
print(f'99.9% не зарабатываеют: {ahui}')

df = df[(df['MonthlyIncome'] < ahui) & (df['age']>0)]
#===========================================================
X = df.drop(columns=['SeriousDlqin2yrs', 'AgeGroup'])
y = df['SeriousDlqin2yrs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def ansamble_models(trial, model_name):
    if model_name == 'lr':
        params = {'C': trial.suggest_float('lr_C', 1e-4, 1e2, log=True), 'penalty': trial.suggest_categorical('lr_penalty', ['l1', 'l2']), 'solver':'saga', 'random_state': 42, 'max_iter':1000}
        model = LogisticRegression(**params)
    elif model_name == 'xgb':
        params = {'n_estimators': 1000, 'max_depth': trial.suggest_int('xgb_max_depth', 2, 5),'learning_rate': trial.suggest_float('xgb_learning_rate', 0.05, 0.2), 'subsample':trial.suggest_float('xgb_subsample', 0.7, 1.0), 'eval_metric':'auc', 'random_state': 42, 'n_jobs':1, 'early_stopping_rounds':50}
        model = xgb.XGBClassifier(**params)
    elif model_name == 'lgb':
        params = {'n_estimators': 1000, 'max_depth': trial.suggest_int('lgb_max_depth', 2, 5),'learning_rate': trial.suggest_float('lgb_learning_rate', 0.05, 0.2), 'subsample':trial.suggest_float('lgb_subsample', 0.7, 1.0), 'objective':'binary','eval_metric':'auc', 'random_state': 42, 'n_jobs':1, 'verbose':-1}
        model = lgb.LGBMClassifier(**params)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in cv.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[test_idx], y_train.iloc[test_idx]

        if model_name in ['lr']:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

        if model_name == 'lr':
                model.fit(X_tr, y_tr)
                preds = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, preds))
        elif model_name == 'xgb':
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, preds))
        elif model_name == 'lgb':
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
                preds = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, preds))
    return np.mean(scores)
lr_study = optuna.create_study(direction='maximize')
lr_study.optimize(lambda trial: ansamble_models(trial, 'lr'), n_trials=20, n_jobs=1)
print(f'Лучший Roc-Auc lr: {lr_study.best_value*100:.4f}%')

xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(lambda trial: ansamble_models(trial, 'xgb'), n_trials=20, n_jobs=1)
print(f'Лучший Roc-Auc xgb: {xgb_study.best_value*100:.4f}%')

lgb_study = optuna.create_study(direction='maximize')
lgb_study.optimize(lambda trial: ansamble_models(trial, 'lgb'), n_trials=20, n_jobs=1)
print(f'Лучший Roc-Auc lgb: {lgb_study.best_value*100:.4f}%')

xgb_best_params = {k.replace('xgb_', ''): v for k, v in xgb_study.best_params.items()}
lgb_best_params= {k.replace('lgb_', ''): v for k, v in lgb_study.best_params.items()}
lr_best_params =  {k.replace('lr_', ''): v for k, v in lr_study.best_params.items()}

xgb_best_model = xgb.XGBClassifier(n_estimators =1000, random_state = 42, **xgb_best_params)
lgb_best_model = lgb.LGBMClassifier(n_estimators=1000, random_state=42, **lgb_best_params)
lr_best_model = make_pipeline(StandardScaler(), LogisticRegression(**lr_best_params, solver='saga', max_iter=1000, random_state=42))




ansamble = VotingClassifier(estimators=[('lr', lr_best_model),('xgb', xgb_best_model),('lgb', lgb_best_model)], voting='soft')
ansamble.fit(X_train, y_train)
final_pred_proba = ansamble.predict_proba(X_test)[:, 1]
final_pred = ansamble.predict(X_test)
print(f'Финал работы проекта, лучшие параметры auc: {roc_auc_score(y_test, final_pred_proba)*100:.4f}%')
print(f'Лучший acc: {accuracy_score(y_test, final_pred)*100:.4f}%')






















































