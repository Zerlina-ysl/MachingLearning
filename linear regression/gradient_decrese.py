import numpy as np
import matplotlib.pyplot as plt
# 1. 初始化x,y
iter = 50
# 随机化在0-50均匀分布
X = np.random.rand(iter)*20
# 随机化在0-50呈正态分布
noise = np.random.randn(iter)
y=0.5*X+noise
plt.scatter(X,y)
plt.show()
# 2初始化参数
w=np.random.randn(1)
b=np.zeros(1)
# 学习率α
lr= 0.001
# 3. 根据样本更新参数
for iteration in range(40):
    y_pred = w*X+b
    w_gradient=0
    b_gradient=0
    N=len(X)
    for i in range(N):
            # (h(x)-y)*x
            # θ：=θ+
        w_gradient+=(w*X[i]+b-y[i])*X[i]
      # θ0
        b_gradient+=(w*X[i]+b-y[i])
        # N次求和后除以N
    w-=lr*w_gradient/N
    b-=lr*b_gradient/N
    # 更新后拟合
    y_pred=w*X+b

    plt.scatter(X,y,c='blue')
    plt.plot(X,y_pred,c='red')
    # Show all figures, and block for a time.
    plt.pause(0.2)


