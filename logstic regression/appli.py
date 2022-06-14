from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import  classification_report
from sklearn.metrics import accuracy_score
cancer = load_breast_cancer()
x_train,x_test,y_train,y_test = train_test_split(
    cancer.data,
    cancer.target,
    test_size=0.2
)
# 创建实例
model = LogisticRegression()
# 拟合训练数据
model.fit(x_train,y_train)
train_score = model.score(x_train,y_train)
test_score = model.score(x_test,y_test)
print('train score:{train_score:.6f};test score:{test_score:.6f}'.format(
    train_score=train_score,test_score=test_score
))
'''
TN:真阴性 实际是负样本预测为负样本的样本数
FP:假阳性 实际是负样本预测为正样本的样本数
FN:假阴性 实际为正样本预测为负样本的样本数
TP:真阳性 实际为正样本预测为正样本的样本数
'''
y_pred = model.predict(x_test)

'''
准确率是预测正确的结果占总样本的百分比 (TP+TN)/(TP+TN+FP+FN)
当样本不平衡时，不能作为很好的衡量指标的结果
如10%的负样本 90%的正样本 将结果全部预测为正样本就会得到90%的高准确率
'''
accuracy_score_value = accuracy_score(y_test,y_pred)
'''
召回率，针对原样本而言，表示实际为正的样本中被预测为正的样本的概率 
TP/(TP+FN)
'''
recall_score_value = recall_score(y_test,y_pred)
'''
精确率，针对预测结果而言，表示在所有被预测为正的样本中实际为正的样本的概率
TP/(TP+FP)
'''
precision_score_value = precision_score(y_test,y_pred)
classification_report_value = classification_report(y_test,y_pred)
print('准确率',accuracy_score_value)
print('召回率',recall_score_value)
print("精确率：",precision_score_value)
print(classification_report_value)