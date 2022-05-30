from sklearn import datasets
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 1. 将数据转换为pd的形式 在列尾添加房价PRICE
boston_dataset = datasets.load_boston()
'''
    ==============   ==============
    Samples total               506
    Dimensionality               13
    Features         real, positive
    Targets           real 5. - 50.
    ==============   ==============
    '''
# DataFrame是一个二维表格数据结构，可以看成excel表格
data= pd.DataFrame(boston_dataset.data)
# featrue_names作为DataFrame的列名
data.columns = boston_dataset.feature_names
data['PRICE']=boston_dataset.target
# 2. 取出房间数和房价 转换为矩阵x y
x = data.loc[:,'RM'].values
y = data.loc[:,'PRICE'].values

x=np.array([x]).T
y=np.array([y]).T

# 训练线性模型
l = LinearRegression()
l.fit(x,y)

# 画图显示
plt.scatter(x,y,s=10,alpha=0.3,c='green')
plt.plot(x,l.predict(x),c='blue',linewidth='1')
plt.xlabel('Number of Rooms')
plt.ylabel('House Price')
plt.show()