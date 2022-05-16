# 导入内置数据集模块
from sklearn.datasets import load_breast_cancer
# 导入sklearn模块中的决策树费雷器类和随机森林分类器类
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# 导入sklearn模块中的模型验证类
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt

# 对随机森林分类器和决策树分类器进行比较


# 导入乳腺癌数据集
cancer = load_breast_cancer()
'''
使用train_test_split函数自动分割训练集和测试集，将训练集分为较小的训练集和验证集，根据较小的训练集来训练模型并评估
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：测试集所占样本比，如果是整数的话就是样本的数量
返回值分别是训练集的样本，训练集的目标值，测试集的样本，测试集的目标值
'''
x_train,x_test,y_train,y_test = train_test_split(cancer.data,cancer.target,test_size=0.3)
# 定义一个决策树对象用于作比较
# 决策树代表对象属性和对象值之间的一种映射关系
dt = DecisionTreeClassifier(random_state=0)
# 定义一个随机森林分类器对象
rf = RandomForestClassifier(random_state=0)
# 训练模型
dt.fit(x_train,y_train)
rf.fit(x_train,y_train)
# 返回在(x,y)上预测的准确率
score_dt = dt.score(x_test,y_test)
score_rf = rf.score(x_test,y_test)
# 输出决策树和随机森林的准确率 每次生成的不同
print('Single Tree:',score_dt)
print('Random Forest:',score_rf)
dt_scores=[]
rf_scores=[]

for i in  range(10):
    # 生成一个包含10次评估分数的数组 交叉验证不仅可以得到一个模型性能的评估值 还可以衡量评估的准确的（标准差
    rf_score = cross_val_score(
        # 随机森林分类器  n_estimators是树的数量
        RandomForestClassifier(n_estimators=25),
        cancer.data,
        cancer.target,
        # 将训练集分割成10个不同的子集，每个子集是一个折叠，对随机森林模型评估和训练--每次挑选1个折叠进行评估，另外9个折叠进行训练
        cv=10
        # 获得平均值
    ).mean()
    rf_scores.append(rf_score)
    dt_score = cross_val_score(DecisionTreeClassifier(),cancer.data,cancer.target,cv=10).mean()
    dt_scores.append(dt_score)
# 绘制评分对比曲线
plt.figure()
plt.title("Random forest VS Decision Tree")
plt.xlabel("Index")
plt.ylabel("Accuracy")
# 用于绘制折线图
plt.plot(range(10),rf_scores,label="Random Forest")
plt.plot(range(10),dt_scores,label="DecisionTree")
# 为每条线生成一个用于区分的图例
plt.legend()
# 展示绘制效果
plt.show()
# 观察弱分类器数量对分类准确度的影响
rf_scores=[]
# [1,50)
for i in range(1,50):
    rf=RandomForestClassifier(n_estimators=i)
    rf_score=cross_val_score(rf,cancer.data,cancer.target,cv=10).mean()
    rf_scores.append(rf_score)
# 生成一个新的图片
plt.figure()
plt.title("Random Forest")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.plot(range(1,50),rf_scores)
plt.show()