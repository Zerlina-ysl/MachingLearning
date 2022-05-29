import numpy as np
from numpy import mat
import matplotlib.pyplot as plt

if __name__=="__main__":
    # 种子
    iter=50
    # rand随机化结果均匀分布 [0,20)
    x=np.random.rand(iter)*20
    # randn随机化结果呈正态分布
    noise = np.random.randn(iter)
    y=0.5*x+noise
    # 散点图
    plt.scatter(x,y)
    plt.show()
    # 2. 矩阵替换(x,y) y是只有y的一维矩阵 x是x为x，y是1.的二维矩阵
    # mat()是将y转换为矩阵 .T后转置
    Y_mat = mat(y).T
    # print(Y_mat)
    # iter行 2列的 1填充
    X_temp = np.ones((iter,2))
    # 将第一列全部替换为X
    X_temp[:,0]=x
    # 转换为矩阵
    X_mat=mat(X_temp)
    # print(X_mat)
    # θ=(X^T X)^-1 X^T y
    # .I求逆矩阵
    parameters = (((X_mat.T)*X_mat).I)*X_mat.T*Y_mat
    print(parameters)
    # 4. 显示
    predict_Y = X_mat*parameters

    plt.figure()
    plt.scatter(x,y,c="blue")
    plt.plot(x,predict_Y,c="red")
    plt.show()
