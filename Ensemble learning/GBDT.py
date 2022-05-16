from sklearn.datasets import  load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt
import numpy as np


# Gradient Boosting Decison Tree

# 对梯度提升分类器和决策树分类器进行比较
cancer = load_breast_cancer()
x_train,x_test,y_train,y_test = train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=1)
gbc = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1)
dt = DecisionTreeClassifier()
gbc.fit(x_train,y_train)
dt.fit(x_train,y_train)
score_gbc = gbc.score(x_test,y_test)
score_dt = dt.score(x_test,y_test)
print("Gradient Boosting:",score_gbc)
print("Decision Tree:",score_dt)
# 测试learning_rate对分类效果的影响
gbc_scores = []
# np.arange支持步长为小数 但range不支持
for i in np.arange(0.1,1,0.05):
    gbc.learning_rate=i
    gbc.fit(x_train,y_train)
    gbc_score = gbc.score(x_test,y_test)
    gbc_scores.append(gbc_score)
plt.figure()
plt.title("Gradient Boost")
plt.xlabel("learning_rate")
plt.ylabel("Accuracy")
plt.plot(range(len(gbc_scores)),gbc_scores)
plt.show()
gbc_scores = []
dt_scores = []
for i in range(20):
    gbc_score = cross_val_score(gbc,cancer.data,cancer.target,cv=10).mean()
    gbc_scores.append(gbc_score)
    dt_score = cross_val_score(dt,cancer.data,cancer.target,cv=10).mean()
    dt_scores.append(dt_score)
plt.figure()
plt.title("Gradient Boost VS Decision Tree")
plt.xlabel("learning_rate")
plt.ylabel("Accuracy")
plt.plot(range(20),dt_scores,label="Decision Tree")
plt.plot(range(20),gbc_scores,label="Gradient Boost")
plt.legend()
plt.show()