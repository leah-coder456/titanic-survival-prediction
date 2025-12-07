# 泰坦尼克号生存预测

本项目基于Kaggle Titanic数据集，使用Logistic Regression、Random Forest、KNN等机器学习模型，对乘客生存情况进行预测。项目完整涵盖数据清洗、特征工程、探索性数据分析（EDA）、模型训练、模型评估与预测提交的完整流程。

## 项目环境和运行

1.环境：

pip install numpy pandas matplotlib seaborn scikit-learn

2.运行：

jupyter notebook Titanic.ipynb

## 项目结构

```text
titanic/
├── Titanic.ipynb        # 主项目 Notebook
├── README.md            # 项目说明文档
├── submission.csv       # 最终预测结果
├── data/                # 数据集目录
│   ├── Titanic-train.csv
│   └── Titanic-test.csv
└── figures/             # 可视化结果图
    ├── gender_survival.png
    └── roc_curve.png
```

## 数据集说明

数据来源：Kaggle Titanic官方数据集

训练集：包含乘客特征以及是否存活标签（Survived）

测试集：仅包含乘客特征，用于最终预测提交

主要特征包括：

PassengerId（乘客编号）

Pclass（舱位等级）

Sex（性别）

Age（年龄）

SibSp（兄弟姐妹数量）

Parch（父母子女数量）

Fare（船票价格）

Embarked（登船港口）

Name、Cabin、Ticket等

## 数据清洗与特征工程

1.缺失值处理

Age：使用中位数填补

Fare：使用中位数填补

Embarked：使用众数填补

Cabin：缺失值统一填为"M"

2.特征构造

Title特征：
从Name中通过正则表达式提取称谓（Mr、Miss、Mrs、Master、Dr等），并合并为：

Mr/Miss/Mrs/Master

Officer（军官、医生等）

Royalty（贵族）

Other

CabinLetter特征：

从Cabin中提取首字母表示船舱区域

FamilySize特征：

FamilySize = SibSp + Parch + 1

3.One-Hot Encoding

Sex

Embarked

Title

CabinLetter

Pclass 等

所有哑变量在训练集与测试集之间通过reindex强制对齐，防止维度不一致问题。

## 探索性数据分析（EDA）

主要完成了以下可视化分析：

- 生还情况与性别：

女性生还率显著高于男性

- 生还情况与舱位等级：

一等舱生还率明显高于二、三等舱

- Age分布（KDE密度图）：

儿童及低龄乘客生还概率更高

- Fare箱线图：

幸存者整体票价分布明显高于未幸存者

### EDA示例：不同性别的生还情况

![Gender Survival](figures/gender_survival.png)


## 模型训练与评估

1.训练数据划分

训练集：80%

验证集：20%

随机种子：random_state = 42

2.训练模型

Logistic Regression

Random Forest

K-Nearest Neighbors

3.模型准确率对比

| 模型                  | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ≈ 0.806  |
| Random Forest       | ≈ 0.833  |
| KNN                 | ≈ 0.627  |

- Random Forest表现最好
- Logistic Regression其次
- KNN在未调参情况下表现较弱

4.ROC曲线与AUC评估

Logistic Regression AUC ≈ 0.879

模型具有较好的分类区分能力

### ROC示例

![ROC Curve](figures/roc_curve.png)

## 预测并生成可提交文件

submission.csv


