import numpy as np
from sklearn.preprocessing import Normalizer

# L1行正规化-列
zhenggui_lie=Normalizer(norm='l1').fit_transform(np.array([1,1,3,-1,2]).reshape(-1,1))
print(zhenggui_lie)

# L1正规-hang
zhenggui_hang=Normalizer(norm='l1').fit_transform(np.array([[1,1,3,-1,2]]))
print(zhenggui_hang)

# L2正规-hang
zhenggui_hang=Normalizer(norm='l2').fit_transform(np.array([[1,1,3,-1,2]]))
print(zhenggui_hang)



"""生成内容如下 分母部分,L1是绝对值和,L2是平方和再开跟号
[[ 1.]
 [ 1.]
 [ 1.]
 [-1.]
 [ 1.]]
[[ 0.125  0.125  0.375 -0.125  0.25 ]]
[[ 0.25  0.25  0.75 -0.25  0.5 ]]
"""
