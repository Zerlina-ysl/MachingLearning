# 基于梯度下降树实现波士顿房价预测
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor as GBDT
from sklearn.model_selection import  train_test_split,validation_curve


# __name__是模块名 模块被运行时名称为__main__ 入口程序代码
if __name__=='__main__':
    boston = load_boston()
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target)
    # 决策树过多 会过拟合 反之欠拟合
    model = GBDT(n_estimators=50)
    model.fit(x_train,y_train)
    train_score = model.score(x_train,y_train)
    test_score = model.score(x_test,y_test)
    print(train_score,test_score)
    param_range = range(20,150,5)
    '''
    通过validation_curve 观察n_estimators的取值如何影响模型准确性
    参数1：评估器 这里是 提升决策树 max_depth是最大深度
    参数2：特征样本
    参数3：目标标签
    参数4：传入的深度参数名称
    参数5：传入的深度参数范围值
    参数6：可迭代对象 交叉验证生成器
    '''
    train_scores,val_scores = validation_curve(GBDT(max_depth=3),
                                               boston.data,boston.target,
                                               param_name = "n_estimators",
                                               param_range=param_range,
                                               # 一份作为cv集 n-1分作为training
                                               cv=5)
    # 计算均值和标准差
    train_mean =  train_scores.mean(axis=-1)
    # axis=-1是倒数第一列 是2
    train_std = train_scores.std(axis=-1)
    val_mean=val_scores.mean(axis=-1)
    val_std = val_scores.std(axis=-1)
# 建立坐标系 numrows=1指定行数 numcols=2指定列数
    # 绘制一个或多个图表 _表示整张图片 ax表示图片中各个图标
    _,ax = plt.subplots(1,2)
    ax[0].plot(param_range,train_mean)
    ax[1].plot(param_range,val_mean)
    # alpha表示透明程度0-1
    ax[0].fill_between(param_range,train_mean-train_std,train_mean+train_std,alpha=0.2)
    ax[1].fill_between(param_range,val_mean-val_std,val_mean+val_std,alpha=0.2)
    plt.show()



