import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_path = "../titanic/train.csv"
test_path = "../titanic/test.csv"
df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df['train_test'] = 1
df_test['train_test'] = 0
all_data = pd.concat([df, df_test])
all_data['train_test'].value_counts()
all_data['cabin_mapped'] = all_data['Cabin'].map(lambda x: str(x)[0])
all_data['ticket_mapped'] = all_data['Ticket'].map(lambda x: x.split(" ")[-1])
all_data['Title'] = all_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
all_data['IsAlone'] = (all_data['SibSp'] + all_data['Parch']) == 0
all_data['Age*Pclass'] = all_data['Age'] * all_data['Pclass']
all_data['Age'].fillna(all_data['Age'].mean(), inplace=True)
all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
all_data['Fare'].fillna(all_data['Fare'].mean(), inplace=True)
all_data['Age*Pclass'].fillna(all_data['Age*Pclass'].median(), inplace=True)
all_data['norm_fare'] = np.log1p(all_data['Fare'])
# transform `Pclass` into a categorical
all_data['Pclass'] = all_data['Pclass'].astype(str)
all_dummies = pd.get_dummies(all_data[['Pclass', 'Sex', 'Age', 'norm_fare', 'train_test', 'Title', 'ticket_mapped', 'IsAlone', 'Age*Pclass', 'SibSp', 'Parch']])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['norm_fare', 'Age*Pclass', 'SibSp', 'Parch']] = scaler.fit_transform(all_dummies_scaled[['norm_fare', 'Age*Pclass', 'SibSp', 'Parch']])
X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)
y_train = all_data[all_data.train_test == 1].Survived
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, y_train, random_state=42, test_size=0.2)
import lightgbm as lgb
print('Dataset loading...')
# create dataset for lightgbm
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'binary_error'}, # binary_error，auc
    'num_leaves': 5, # 5转换成2反而更低了
    'max_depth': 1,
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
y_pred = gbm.predict(X_test_scaled, num_iteration=gbm.best_iteration)  # 输出的是概率结果
final_data = {'PassengerId': df_test.PassengerId, 'Survived': y_pred}
final_data['Survived'][final_data['Survived']<=0.5]=0
final_data['Survived'][final_data['Survived']>0.5]=1
final_data['Survived'] = final_data['Survived'].astype(int)
submission = pd.DataFrame(data=final_data)
submission.to_csv(f'submission_lgb.csv', index =False)