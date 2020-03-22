import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X=np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
Y=np.array([1,1,1,2,2,2])

# 参数n_com...是降到几维的意思
LDA=LinearDiscriminantAnalysis(n_components=1).fit_transform(X,Y)
print('LDA\n',LDA)



clf=LinearDiscriminantAnalysis(n_components=2).fit(X,Y)
print('clf\n',clf)

fisher_classifier=clf.predict([[0.8,1]])
print('fisher_classifier\n',fisher_classifier)


"""
LDA
 [[-1.73205081]
 [-1.73205081]
 [-3.46410162]
 [ 1.73205081]
 [ 1.73205081]
 [ 3.46410162]]
clf
 LinearDiscriminantAnalysis(n_components=2, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001)
fisher_classifier
 [2]

"""
