import numpy as np

# 返回Xi-Xmean Xmean
def zeroMean(dataMet):
    #axis=0 表示按列 求取平均值 1为行
    # 1. 原始数据按行组成m行n列的矩阵dataMet
    meanVal=np.mean(dataMet,axis=0)
    # 2. 将X的每一列零均值化（减去这一列均值
    newData = dataMet-meanVal
    return newData,meanVal

def pca(dataMet,percent=0.19):
    newData,meanVal = zeroMean(dataMet)
    # rowvar=False 每列代表一个变量，而行包含观察值
    # 对Xi-Xmean求协方差矩阵 变量两两之间的协方差
    # 3. 求出协方差矩阵
    covMat = np.cov(newData,rowvar=0)
    # 4. 求出协方差矩阵的特征值eigValue以及对应的特征向量eigVector
    eigVals,eigVects = np.linalg.eig(covMat)
    # 对特征值从小到大排序
    eigVallndice=np.argsort(eigVals)
    n=5
    # 最大的n个特征值的下标
    n_eigVallndice=eigVallndice[-1:-(n+1):-1]
    # 最大的n个特征值对应的特征向量
    n_eigVect = eigVects[:n_eigVallndice]
    # 5. 计算降维后的数据矩阵
    # 低位特征空间的数据 最大的n个特征值对应的特征向量*（数据-平均值）
    lowDDataMat = newData*n_eigVect
    # 重构数据
    reconMat = (lowDDataMat*n_eigVect.T)+meanVal
    return  reconMat,lowDDataMat,n



