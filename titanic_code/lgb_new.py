# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv('../titanic/train.csv')
df_test = pd.read_csv('../titanic/test.csv')



def get_data2(df_train):
    df_train [['Surname', 'First_Name']] = df_train['Name'].str.split(',', expand = True)
    df_train['Title'] = df_train['First_Name'].apply(lambda x: x.split('.') [0])
    df_train = df_train.drop('First_Name', axis = 1)
    # df_new = df_train[df_train.duplicated(subset=['Surname'], keep=False)]
    df_new = df_train.drop(['Cabin', 'Name', 'Ticket'], axis = 1)
    df_new['Age'] = df_new['Age'].fillna(df_new['Age'].median())
    cut_ages = ['-18', "18-30", '31-50', '50+',]
    cut_bins = [0, 18, 30, 50,100]
    df_new['Age_groups'] = pd.cut(df_new['Age'], bins=cut_bins, labels = cut_ages)
    fare_category = []
    fare = df_new['Fare']
    for f in fare:
        if f <100:
            fare_category.append('Low')
        elif f<250:
            fare_category.append('Average')
        else:
            fare_category.append('High')
    df_new['Fare Category'] = fare_category
    df_new['Family'] = (df_new['SibSp'] + df_new['Parch'])
    columns = ['Pclass', 'Sex', 'Age', 'Family', 'Embarked', 'Fare Category', 'Title']
    x=df_new[columns]
    y=df_new['Survived']
    for col in ['Sex','Embarked', 'Fare Category', 'Title']:
        x[col]=LabelEncoder().fit_transform(x[col])
    return x,y

x,y = get_data2(df_train)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
df_test['Survived'] = 11
x_test2,y_test2 = get_data2(df_test)


import lightgbm as lgb
print('Dataset loading...')
# create dataset for lightgbm
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 5,
    'max_depth': 2,
    'min_data_in_leaf': 5,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'lambda_l1': 1,
    'lambda_l2': 0.001,  # 越小l2正则程度越高
    'min_gain_to_split': 0.2,
    'verbose': 5,
    'is_unbalance': True,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0
}

# train
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                # early_stopping_rounds=500
                )
print('Start predicting...')
ID = df_test['PassengerId']
y_pred = gbm.predict(x_test2, num_iteration=gbm.best_iteration)  # 输出的是概率结果
# preds
submission = pd.DataFrame({'PassengerId':ID, 'Survived':y_pred})
print(submission.head())
submission.to_csv('tiantic_submission.csv', index=False)

# # lgb保存模型
import joblib
joblib.dump(gbm,'lgb_gbm.pkl')
lgb = joblib.load('./lgb_gbm.pkl',lgb)


import pickle
with open('lgb.pkl','wb') as f:
    pickle.dump(gbm,f)

