import numpy as np
from sklearn import  datasets

iris=datasets.load_iris()
# print(iris)
labels=np.copy(iris.target) #获取label
print('labels个数',len(labels))  #150条数据
print('labels内容',labels)

#随机选一些,赋值为0-1之间的
random_unlabeled_points=np.random.rand(len(iris.target))
#如果数组小于0.3则返回1,大于等于0.3返回0
random_unlabeled_points=random_unlabeled_points<0.3
print('小于0.3的\n',random_unlabeled_points)

# 存个y,是转换之前的
y=labels[random_unlabeled_points]
print('y:',y)
labels[random_unlabeled_points]=-1

# 对比转换结果
print('原始标注',iris.target)
print('labels内容',labels)
print('无标注数据',list(labels).count(-1))

from  sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import  accuracy_score,recall_score,f1_score

lpm=LabelPropagation()
lpm.fit(iris.data,labels)
Y_pred=lpm.predict(iris.data)
Y_pred=Y_pred[random_unlabeled_points]
print('acc',accuracy_score(y,Y_pred))
print('recall score',recall_score(y,Y_pred,average='micro'))
print('f1 score',f1_score(y,Y_pred,average='micro'))
