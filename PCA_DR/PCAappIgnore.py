import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

#### 使用PCA算法识别手写体数字图像


digits = load_digits()
data = digits.data
target = digits.target
'''
使用train_test_split函数自动分割训练集和测试集，将训练集分为较小的训练集和验证集，根据较小的训练集来训练模型并评估
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：测试集所占样本比，如果是整数的话就是样本的数量
返回值分别是训练集的样本，训练集的目标值，测试集的样本，测试集的目标值
'''
# 1. 利用train_test_split分割样本，80%作为训练样本，20%作为测试样本
x_train,x_test,y_train,y_true = train_test_split(data,target,test_size=0.2,random_state=33)
# 标准化模块对训练和测试的特征数据进行标准化
ss=StandardScaler()
# 使用原始模型的像素特征识别
x_train=ss.fit_transform(x_train)
# 训练支持svc为基础的模型
x_test = ss.transform(x_test)

svc = SVC(kernel='rbf')
# svc.fit()得到SVM
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
#  Return the mean accuracy on the given test data and labels.
print("The Accuracy of SVA is",svc.score(x_test,y_true))
'''
构建显示主要分类指标的文本报告。
y_true  测试集的目标值
y_predict classification返回的预估目标值
target_names digits.target标签的名称
astype()转换为指定类型
'''
print("classification report of svc\n",classification_report(y_true,y_predict,target_names=digits.target_names.astype(str)))
# 当whiten为 True（默认为 False）时，“components_”向量乘以 n_samples 的平方根，然后除以平均值，以确保不相关的输出具有单位分量方差。
# 将64维降到10维
# 使用PCA降维重构后的低维特征识别图形
pca = PCA(n_components=10,whiten=True)
pca.fit(x_train,y_train)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)
svc = SVC(kernel='rbf')
svc.fit(x_train_pca,y_train)
y_pre_svc = svc.predict(x_test_pca)
print("the Accuracy of PCA_SVC  is",svc.score(x_test_pca,y_true))
print("classification report of PCA_SVC\n",classification_report(y_true,y_pre_svc,target_names=digits.target_names.astype(str)))


samples = x_test[:100]
y_pre = y_pre_svc[:100]
plt.figure(figsize=(12,38))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(samples[i].reshape(8,8),cmap='gray')
    title=str(y_pre[i])
    plt.title(title)
    plt.axis("off")
plt.show()