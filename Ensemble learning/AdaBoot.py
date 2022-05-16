# 对AdaBoost分类器和决策树分类器进行比较
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.model_selection import  train_test_split,cross_val_score
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
x_train,x_test,y_train,y_test = train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=1)
# AddBoost分类器 基于50棵决策树
abc = AdaBoostClassifier(
    # 基分类器 在分类器基础上进行boosting 默认为决策树 使用其他分类器需要指明样本权重
    DecisionTreeClassifier(),
    # 基于多类指数损失函数的逐步添加模型 SAMME:对样本集预测错误的概率进行划分
    # 模型提升准侧 数据类型为字符串型 默认SAMME.R
    algorithm='SAMME',
    # 基分类器提升(循环)次数
    n_estimators=50,
    # 学习率 表示梯度收敛速度 需要和n_estimators进行一个权衡
    learning_rate=0.1
)
dt = DecisionTreeClassifier()
abc.fit(x_train,y_train)
# 决策树分类器进行训练
dt.fit(x_train,y_train)
score_abc = abc.score(x_test,y_test)
score_dt = dt.score(x_test,y_test)
# 输出准确率
print('Ada Boost:',score_abc)
print('Desicion Tree:',score_dt)
# 测试estimators参数对分类结果的影响
abc_scores = []
for i in range(1,50):
    abc.estimators_=i
    abc.fit(x_train,y_train)
    abc_score = abc.score(x_test,y_test)
    abc_scores.append(abc_score)
plt.figure()
plt.title("AdaBoost")
plt.xlabel("n_estinmators")
plt.ylabel("Accuracy")
plt.plot(range(1,50),abc_scores)
plt.show()

