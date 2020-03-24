def hr_modeling_nn(features, label):
    from sklearn.model_selection import train_test_split
    f_v = features.values
    f_names = features.columns.values
    l_v = label.values
    X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)

    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.optimizers import SGD
    mdl = Sequential()
    mdl.add(Dense(50, input_dim=len(f_v[0])))
    mdl.add(Activation("sigmoid"))
    mdl.add(Dense(2))
    mdl.add(Activation("softmax"))
    sgd = SGD(lr=0.1)
    mdl.compile(loss="mean_squared_error", optimizer="adam")
    mdl.fit(X_train, np.array([[0, 1] if i == 1 else [1, 0] for i in Y_train]), nb_epoch=15000, batch_size=8999)
    xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
    import matplotlib.pyplot as plt
    
    from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
    
    f = plt.figure()
    for i in range(len(xy_lst)):
        X_part = xy_lst[i][0]
        Y_part = xy_lst[i][1]
        Y_pred = mdl.predict(X_part)
        print('Ypred!!!!!',Y_pred)
        Y_pred = np.array(Y_pred[:, 1]).reshape((1, -1))[0]
        print(i)
        f.add_subplot(1, 3, i + 1)
        fpr, tpr, threshold = roc_curve(Y_part, Y_pred)
        plt.plot(fpr, tpr)
        print("NN", "AUC", auc(fpr, tpr))
        print("NN", "AUC_Score", roc_auc_score(Y_part, Y_pred))
    plt.show()
    return
