# 引入包
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 标签编码
biaoqian1 = LabelEncoder().fit_transform(np.array(['down', 'up', 'up',    'down']).reshape(-1, 1))
biaoqian2 = LabelEncoder().fit_transform(np.array(['low',  'up', 'medium', 'low']).reshape(-1, 1))
print('标签1\n', biaoqian1)
print('标签2\n', biaoqian2)

# 标签编码赋予变量法
lb_encoder = LabelEncoder()
lb_tran_f = lb_encoder.fit_transform(np.array(['red', 'blue', 'yellow', 'green']).reshape(-1, 1))

print('lb_tran_f\n', lb_tran_f)
# 独热编码
oht_encoder = OneHotEncoder().fit(lb_tran_f.reshape(-1, 1))
print('oht_encoder\n', oht_encoder)

oht1 = oht_encoder.transform(lb_encoder.transform(np.array(['red', 'blue', 'yellow', 'green', 'red'])).reshape(-1, 1))
print('oht1\n', oht1)
oht2 = oht_encoder.transform(lb_encoder.transform(np.array(['red', 'blue', 'yellow', 'green', 'red'])).reshape(-1, 1)).toarray()
print('oht2\n', oht2)

"""生成内容如下
标签1
 [0 1 1 0]
标签2
 [0 2 1 0]
lb_tran_f
 [2 0 3 1]
oht_encoder
 OneHotEncoder(categories='auto', drop=None, dtype=<class 'numpy.float64'>,
              handle_unknown='error', sparse=True)
oht1
   (0, 2)	1.0
  (1, 0)	1.0
  (2, 3)	1.0
  (3, 1)	1.0
  (4, 2)	1.0
oht2
 [[0. 0. 1. 0.]
 [1. 0. 0. 0.]
 [0. 0. 0. 1.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]]

Process finished with exit code 0
"""
