import numpy as np
import matplotlib.pyplot as plt
# 制造一些点的工具包
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.cluster import KMeans,DBSCAN

n_samples = 1000
# 创建四个聚类点,圆圈\双弧线\数堆\方块
circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)  # factor小圆大圆间距    noise噪声比
moons = make_moons(n_samples=n_samples, noise=0.05)
blobs = make_blobs(n_samples=n_samples, random_state=8,center_box=(-1,1),cluster_std=0.1)  # random_state产生随机值一样
random_data = np.random.rand(n_samples, 2), None  # 2维度调用
print(len(circles),circles[0])
# 制定色系
colors = 'bgrcmyk'
data = [circles, moons, blobs, random_data]

# 创建模型集合
models = [('None', None)]
models.append(('DBSCAN', DBSCAN(min_samples=3,eps=0.5)))
models.append(('DBSCAN', DBSCAN(min_samples=3,eps=0.2)))


# 初始化图像
f = plt.figure()

# 开始计算及绘制图像
for inx, clt in enumerate(models):
    # 名字实体,算法本身
    clt_name, clt_entity = clt
    for i, dataset in enumerate(data):
        # 根据print的数据及可以看到,[0]是数据集,[1]是标注,因此X是数据集
        X, Y = dataset

        if not clt_entity:
            # 如果没有算法实体,则画下数据集合本身
            clt_res = [0 for item in range(len(X))]
        else:
            # 训练模型
            clt_entity.fit(X)
            # 得到标签,转化为整数格式
            clt_res = clt_entity.labels_.astype(np.int)
        # 加入子图
        f.add_subplot(len(models), len(data), inx * len(data) + i + 1)  # 加子图
        # 加图的title
        plt.title(clt_name)
        # 把几个图放成子图模式
        [plt.scatter(X[p, 0], X[p, 1],color=colors[clt_res[p]]) for p in range(len(X))]
plt.savefig('DBscan.png')
plt.show()

