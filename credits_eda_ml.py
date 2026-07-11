import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

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
print(ahui)
df = df[(df['MonthlyIncome'] < ahui) & (df['age']>0)]
#===========================================================
X = df.drop(columns=['SeriousDlqin2yrs', 'AgeGroup'])
y = df['SeriousDlqin2yrs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
lr_model = LogisticRegression()
lr_model.fit(X_train_s, y_train)
lr_pred = lr_model.predict(X_test_s)
lr_proba = lr_model.predict_proba(X_test_s)[:,1]
print(f'acc = {accuracy_score(y_test, lr_pred)*100:.2f}%')
print(f'proba = {roc_auc_score(y_test, lr_proba)*100:.2f}%')

gb_model = HistGradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_proba = gb_model.predict_proba(X_test)[:,1]

print(f'Бустинг - acc: {accuracy_score(y_test, gb_pred)*100:.2f}%')
print(f'Бустинг - roc-auc: {roc_auc_score(y_test, gb_proba)*100:.2f}%')






