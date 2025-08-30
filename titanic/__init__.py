
#提供数据处理、机器学习模型、模型评估和可视化的功能
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#使用 pandas 读取 CSV 文件并查看数据的基本信息
data = pd.read_csv(r"train.csv", index_col= 0)
print("\n所有项：")
data.info()

# 输出各列中的缺失值情况
missing_data = data.isnull().sum()
print("\n各列缺失值情况：")
print(missing_data)
print("\n幸存者推断项：")

#删除冗余字段
data_new = data.drop(["Name", "Ticket", "Cabin"], axis=1)
#使用年龄字段的均值填补缺失值
data_new["Age"] = data_new["Age"].fillna(data_new["Age"].mean())
#将embarked字段中含有缺失值的行删除
data_new.dropna(axis=0)
data_new.info()

#将sex、embarked字段转换为字段属性
labels = data_new["Embarked"].unique().tolist()
data_new["Embarked"] = data_new["Embarked"].apply(lambda x: labels.index(x))
data_new["Sex"] = (data_new["Sex"] == "male").astype("int")

x = data_new.iloc[:, data_new.columns != "Survived"]
y = data_new.iloc[:, data_new.columns == "Survived"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3, random_state=25)
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])
Xtrain.head()

# 模型训练和评估
clf = DecisionTreeClassifier(random_state=25)
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)
print('score：', score)
score_mean = cross_val_score(clf, x, y, cv=10).mean()
print('score_mean：', score_mean)

#交叉验证的结果比单个的结果更低，因此要来调整参数，首先想到的是max_depth,因此绘制超参数曲线
score_test = []
score_train = []
for i in range(10):
    clf = DecisionTreeClassifier(random_state=25,
                                 max_depth=i+1)
    clf = clf.fit(Xtrain, Ytrain)
    score_tr = clf.score(Xtrain, Ytrain)
    score_te = cross_val_score(clf, x, y, cv=10).mean()
    score_train.append(score_tr)
    score_test.append(score_te)
print("\nbefore:", max(score_test))

#绘制超参数图像
plt.subplot(1, 2, 1)

plt.plot(range(1, 11), score_train, color="red", label="train")
plt.plot(range(1, 11), score_test, color="blue", label="test")
plt.legend()
plt.xticks(range(1, 11))

plt.title("Max Depth Tuning")
plt.xlabel("Max Depth")
plt.ylabel("Score")

#调整参数criterion，观察图像变化
score_test = []
score_train = []
for i in range(10):
    clf = DecisionTreeClassifier(random_state=25,
                                 max_depth=i+1,
                                 criterion="entropy")
    # 这里为什么使用“entropy”？
    # 因为我们注意到，在最大深度=3的时候，模型拟合不足，在训练集和测试集上的表现接近，
    # 但却都不是非常理想，只能够达到83%左右，所以我们要使用entropy。
    clf = clf.fit(Xtrain, Ytrain)
    score_tr = clf.score(Xtrain, Ytrain)
    score_te = cross_val_score(clf, x, y, cv=10).mean()
    score_train.append(score_tr)
    score_test.append(score_te)
print("after:", max(score_test))
print("关闭图表后，请等待运行结果······\n")

#绘制图像
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), score_train, color="red", label="train")
plt.plot(range(1, 11), score_test, color="blue", label="test")
plt.legend()
plt.xticks(range(1, 11))
plt.title("Max Depth Tuning with Entropy")
plt.xlabel("Max Depth")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

#parameters:本质是一串参数和这串参数对应的，我们希望网格搜索来搜索的参数的取值范围
gini_thresholds = np.linspace(0,0.5,20)
parameters = {
    "splitter": ["best", "random"],
    "min_samples_leaf": list(range(1, 20, 5)),
    "min_impurity_decrease": list(np.linspace(0, 0.5, 20)),
    'criterion':("gini","entropy"),
    "max_depth":list(range(1,10))}

clf = DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf, parameters, cv=10)
GS.fit(Xtrain, Ytrain)
print('准确率：', GS.best_score_)
best_params = GS.best_params_
for key, value in best_params.items():
    print(f"{key}: {value}")