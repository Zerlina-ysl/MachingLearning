import matplotlib.pyplot as plt
# 导入PCA算法包
from sklearn.decomposition import PCA
# 导入鸢尾花数据集
from sklearn.datasets import load_iris
#使用PCA对鸢尾花数据进行降维
# 以字典形式导入鸢尾花数据集
data = load_iris()
y=data.target
# 降维后主成分数目为2的PCA算法
pca = PCA(n_components=2)
# 对原始数据(数据集)进行降维
reduced_X = pca.fit_transform(data.data)
red_x,red_y = [],[]
blue_x,blue_y = [],[]
green_x,green_y=[],[]
for i in range(len(reduced_X)):
    # 根据鸢尾花的类别将降维后的数据点保存在不同列表
    if y[i]==0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i]==1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
# 可视化
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()