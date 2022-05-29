from sklearn import datasets
from sklearn.linear_model import LinearRegression
import Pandas as pd
# 1. 将数据转换为pd的形式 在列尾添加房价PRICE
boston_dataset = datasets.load_boston()
data= pd.DataFrame(boston_dataset.data)
data.columns = boston_dataset.featrue_names
data['PRICE']=boston_dataset.target
# 2. 取出房间数和房价 转换为矩阵
x = data.loc[:,'RM'].as_matrix(columns=None)
