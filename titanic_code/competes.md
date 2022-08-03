1. Titanic

### Overview

- training set (train.csv)
- test set (test.csv)

**The training set** provide the outcome (also known as the “ground truth”) for each passenger. 

**The test set** should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. 

We also include **gender_submission.csv**, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

### Data Dictionary

| **Variable** | **Definition**                             | **Key**                                        |
| :----------- | :----------------------------------------- | :--------------------------------------------- |
| survival     | Survival                                   | 0 = No, 1 = Yes                                |
| pclass       | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| sex          | Sex                                        |                                                |
| Age          | Age in years                               |                                                |
| sibsp        | # of siblings / spouses aboard the Titanic |                                                |
| parch        | # of parents / children aboard the Titanic |                                                |
| ticket       | Ticket number                              |                                                |
| fare         | Passenger fare                             |                                                |
| cabin        | Cabin number                               |                                                |
| embarked     | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

### Variable Notes

**pclass**: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

**age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

**sibsp**: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

**parch**: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

使用lgb的时候出现了这个问题

Stopped training because there are no more leaves that meet the split requirements

> 上网查了一下错误信息是学习树的当前迭代停止，因为叶子节点不能够继续分裂。可能是由于'min_data_in_leaf ' 设置的过大而导致的。
>
> 这次除了出现上述的问题模型的auc 的结果始终接近于0.5。尝试减小min_data_in_leaf , 但是仍然会报错。 然而更改boost_from_average 设置True时才能开始树的学习。 值得注意的是boost_from_averagec参数在旧版本中默认为False， 新版本则为True，将其设置为True表示将初始分数调整为标签的平均值以加快收敛速度， 产生上述现象的原因可能是因为样本不平衡导致过拟合。
> ————————————————
> 版权声明：本文为CSDN博主「Script-Boy」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/regnjka/article/details/102736502

```
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 5,
    'max_depth': 6,
    'min_data_in_leaf': 450, # 重新设置成15 # 设置成15就没有问题了
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'lambda_l1': 1,
    'lambda_l2': 0.001,  # 越小l2正则程度越高
    'min_gain_to_split': 0.2,
    'verbose': 5,
    'is_unbalance': True
}
```

 No further splits with positive gain, best gain: -inf

那个参数还是得小

Trained a tree with leaves = 1 and depth = 1

之前一直都不对，仅仅做了个简单的int转换变成了0.7

```
max_depth = 2 
# 后面转换成1 有了0.003的提高
```

##### 明天争取到0.8（2022-08-04

- [lgb的matrics][https://blog.csdn.net/weixin_43440760/article/details/108843033]

