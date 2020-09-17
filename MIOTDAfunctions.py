from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np # always need it
from numpy import unravel_index
import ot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# functions


def CVsinkhorn(rango_e, xs, ys, xt, yt, clf, metrica="euclidean", Verbose=None):
    """
    This function search for the best reg. parameter based on accuracy
    within the OT-Sinkhorn method
    Parameters
    ----------
    rango_E : array/np.array
        grid of parameter values from the regularization term for entropic
        regularization
    xs : ns x m array, where ns is the number of trials and m the number of
    samples.
        Source data matrix.
    ys : ns array
        source label information.
    xt : nt x m array, where nt is the number of trials and m the number of
    samples.
        Target data matrix.
    yt : nt array
        target label information.
    metrica : string, optional
        distance used within OT. The default is "euclidean".

    Returns
    -------
    regu: float
        best reg. parameter selected based on accuracy
    """
    ACC_ = []
    
    #select subset validation to avoid overfitting
    
        
    for r in range(np.size(rango_e)):
        ot_sinkhorn = ot.da.SinkhornTransport(metric=metrica,reg_e=rango_e[r], 
                                              verbose=Verbose)
        ot_sinkhorn.fit(Xs=xs, Xt=xt)
        # transform
        transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=xs)
        # train new classifier
        clf.fit(transp_Xs_sinkhorn, ys)

        yt_predict = clf.predict(xt)
        # Compute accuracy trad DOAT
        acc_ = accuracy_score(yt, yt_predict)
        ACC_.append(acc_)

    index = np.argmax(ACC_)
    regu = rango_e[index]
    return regu


def CVgrouplasso(rango_e, rango_cl, xs, ys, xt, yt, clf, metrica="euclidean", Verbose=None):
    """
    This function search for the best set of reg. parameters within the OT-L1L2 
    method.

    Parameters
    ----------
    rango_e : array/np.array
        grid of parameter values from the regularization term for entropic
        regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso
        regularization
    xs : ns x m array, where ns is the number of trials and m the number of
    samples.
        Source data matrix.
    ys : ns array
        source label information.
    xt : nt x m array, where nt is the number of trials and m the number of
    samples.
        Target data matrix.
    yt : nt array
        target label information.
    metrica : string, optional
        distance used within OT. The default is "euclidean".

    Returns
    -------
    regu : tuple (2)
        vector with the best reg. parameter for the FOTDA-GL
        regu_trad[0]: entropic regularizer param
        regu_trad[1]; group-lasso regulatizer param
    """
    result = np.empty((np.size(rango_e), np.size(rango_cl)), dtype=float)

    for r in range(np.size(rango_e)):
        for rr in range(np.size(rango_cl)):
            # Sinkhorn Transport with Group lasso regularization
            ot_l1l2 = ot.da.SinkhornL1l2Transport(
                metric=metrica, reg_e=rango_e[r],
                reg_cl=rango_cl[rr], verbose=Verbose)
            ot_l1l2.fit(Xs=xs, ys=ys, Xt=xt)

            # transport source samples onto target samples
            transp_Xs_lpl1 = ot_l1l2.transform(Xs=xs)
            # train on new source
            clf.fit(transp_Xs_lpl1, ys)
            # Compute accuracy
            yt_predict = clf.predict(xt)
            acc_ = accuracy_score(yt, yt_predict)

            result[r, rr] = acc_

    index = unravel_index(result.argmax(), result.shape)
    regu = [rango_e[index[0]], rango_cl[index[1]]]
    return regu


def CVgrouplasso_backward(rango_e, rango_cl, xs, ys, xt, yt, clf, metrica="euclidean", Verbose=False):
    """
    This function search for the best set of reg. parameters within the
    Backward OT-L1L2 method.

    Parameters
    ----------
    rango_e : array/np.array
        grid of parameter values from the regularization term for entropic
        regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso
        regularization
    xs : ns x m array, where ns is the number of trials and m the number of
    samples.
        Source data matrix.
    ys : ns array
        source label information.
    xt : nt x m array, where nt is the number of trials and m the number of
    samples.
        Target data matrix.
    yt : nt array
        target label information.
    metrica : string, optional
        distance used within OT. The default is "euclidean".

    Returns
    -------
    regu : tuple (2)
        vector with the best reg. parameter for the FOTDA-GL
        regu_trad[0]: entropic regularizer param
        regu_trad[1]; group-lasso regulatizer param



    """
    result = np.empty((np.size(rango_e), np.size(rango_cl)), dtype=float)

    for r in range(np.size(rango_e)):

        for rr in range(np.size(rango_cl)):
            # Sinkhorn Transport with Group lasso regularization
            bot_l1l2 = ot.da.SinkhornL1l2Transport(
                metric=metrica, reg_e=rango_e[r], reg_cl=rango_cl[rr],
                verbose=Verbose)
            bot_l1l2.fit(Xs=xt, ys=yt, Xt=xs)
            # transport target samples onto source samples
            transp_Xt_lpl1 = bot_l1l2.transform(Xs=xt)
            # Compute accuracy
            yt_predict = clf.predict(transp_Xt_lpl1)
            acc_ = accuracy_score(yt, yt_predict)
            result[r, rr] = acc_
    index = unravel_index(result.argmax(), result.shape)
    regu = [rango_e[index[0]], rango_cl[index[1]]]

    return regu


def CVsinkhorn_backward(rango_e, xs, ys, xt, yt, clf, metrica="euclidean", Verbose=False):
    """
    This function search for the best set of reg. parameters within the
    OT-Sinkhorn method.

    Parameters
    ----------
    rango_e : array/np.array
        grid of parameter values from the regularization term for entropic
        regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso
        regularization
    xs : ns x m array, where ns is the number of trials and m the number of
    samples.
        Source data matrix.
    ys : ns array
        source label information.
    xt : nt x m array, where nt is the number of trials and m the number of
    samples.
        Target data matrix.
    yt : nt array
        target label information.
    metrica : string, optional
        distance used within OT. The default is "euclidean".

    Returns
    -------
    regu: float
        best reg. parameter selected based on accuracy

    """
    result = []

    for r in range(np.size(rango_e)):
        # Sinkhorn Transport with Group lasso regularization
        bot = ot.da.SinkhornTransport(metric=metrica, reg_e=rango_e[r], 
                                      verbose=Verbose)
        bot.fit(Xs=xt, ys=yt, Xt=xs)
        # transport target samples onto spurce samples
        transp_Xt_lpl1 = bot.transform(Xs=xt)

        # Compute accuracy
        yt_predict = clf.predict(transp_Xt_lpl1)
        acc_ = accuracy_score(yt, yt_predict)

        result.append(acc_)

    result = np.asarray(result)
    index = np.argmax(result)
    regu = rango_e[index]
    return regu


def SelectSubsetTraining_BOTDAl1l2(xs, ys, xv, yv, rango_e, rango_cl, clf,
                                   metrica="euclidean", kfold=20, M=40,
                                   Verbose=False):
    """
    select subset of source data to learn the mapping.

    Parameters
    ----------
    xs : array (ns, m)
        source data matrix.
    ys : array (ns,)
        labels source samples.
    xv : array(nv,m)
        transportation matrix.
    yv : array (nv,)
        labels transportation set samples.
    rango_e : array/np.array
        grid of parameter values from the regularization term for entropic
        regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group
        lasso regularization
    metrica : string, optional
        distance used within OT. The default is "euclidean".
    kfold : number, optional
        times to repeat the resample. The default is 20.
    M : number, optional
        final samples included in the subset. The default is 40.

    Returns
    -------
    subset_xs: array (M, m)
        best selected subset
    subset_ys: (M,)
        corresponding labels of the selected subset_xs
    reg_best: array(2,1)
        regularization parameter for the entropic and and  the group lasso
        terms.

    """

    acc_cv = []
    lista_xs = []
    lista_ys = []

    regu_ = []

    for k in range(kfold):
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(xs, ys, train_size=M, stratify=ys)

        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)
        if np.size(rango_e)==1 and np.size(rango_cl)==1:
            regu = [rango_e, rango_cl]
        else:
            regu = CVgrouplasso_backward(
                rango_e=rango_e, rango_cl=rango_cl,
                xs=xs_daotcv, ys=ys_daotcv,
                xt=xv, yt=yv, clf=clf, metrica=metrica, Verbose=Verbose)

        regu_.append(regu)

        bot_l1l2 = ot.da.SinkhornL1l2Transport(
            metric=metrica, reg_e=regu[0], reg_cl=regu[1])

        bot_l1l2.fit(Xs=xv, ys=yv, Xt=xs_daotcv)
        # transport testing sampless
        transp_Xv_l1l2 = bot_l1l2.transform(Xs=xv)
        acc_cv.append(clf.score(transp_Xv_l1l2, yv))

    index = np.argmax(acc_cv)

    subset_xs = lista_xs[index]
    subset_ys = lista_ys[index]
    reg_best = regu_[index]
    return subset_xs, subset_ys, reg_best


def SelectSubsetTraining_BOTDAs(xs, ys, xv, yv, rango_e, clf,
                                metrica="euclidean", kfold=20, M=40,
                                Verbose=False):
    """
    select subset of source data to learn the mapping.

    Parameters
    ----------
    xs : array (ns, m)
        source data matrix.
    ys : array (ns,)
        labels source samples.
    xv : array(nv,m)
        transportation matrix.
    yv : array (nv,)
        labels transportation set samples.
    rango_e : array/np.array
        grid of parameter values from the regularization term for entropic
        regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso
        regularization
    metrica : string, optional
        distance used within OT. The default is "euclidean".
    kfold : number, optional
        times to repeat the resample. The default is 20.
    M : number, optional
        final samples included in the subset. The default is 40.
    trad : boolean, optional
        if True the outputs are calculard following the re-training OTDA path,
        othverwise, the one-trainig path is followed. The default is True.

    Returns
    -------
    subset_xs: array (M, m)
        best selected subset
    subset_ys: (M,)
        corresponding labels of the selected subset_xs
    reg_best: number
        regularization parameter for the entropic

    """
    acc_cv = []
    lista_xs = []
    lista_ys = []

    regu_ = []

    for k in range(kfold):
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(
            xs, ys, train_size=M, stratify=ys)

        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)
        if np.size(rango_e)==1:
            regu = rango_e
        else:
            regu = CVsinkhorn_backward(rango_e, xs_daotcv, ys_daotcv, xv, yv, clf, metrica, Verbose)
        regu_.append(regu)

        bot = ot.da.SinkhornTransport(metric=metrica, reg_e=regu)

        bot.fit(Xs=xv, ys=yv, Xt=xs_daotcv)
        # transport val sampless
        transp_Xv_l1l2 = bot.transform(Xs=xv)
        acc_cv.append(clf.score(transp_Xv_l1l2, yv))

    index = np.argmax(acc_cv)

    subset_xs = lista_xs[index]
    subset_ys = lista_ys[index]
    reg_best = regu_[index]
    return subset_xs, subset_ys, reg_best


def SelectSubsetTraining_OTDAl1l2(xs, ys, xv, yv, rango_e, rango_cl, clf,
                                  metrica="euclidean", kfold=20, M=40,
                                  Verbose=False):
    """
    select subset of source data to learn the mapping.

    Parameters
    ----------
    xs : array (ns, m)
        source data matrix.
    ys : array (ns,)
        labels source samples.
    xv : array(nv,m)
        transportation matrix.
    yv : array (nv,)
        labels transportation set samples.
    rango_e : array/np.array
        grid of parameter values from the regularization term for entropic
        regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso 
        regularization
    metrica : string, optional
        distance used within OT. The default is "euclidean".
    kfold : number, optional
        times to repeat the resample. The default is 20.
    M : number, optional
        final samples included in the subset. The default is 40.
    trad : boolean, optional
        if True the outputs are calculard following the re-training OTDA path,
        othverwise, the one-trainig path is followed. The default is True.

    Returns
    -------
    subset_xs: array (M, m)
        best selected subset
    subset_ys: (M,)
        corresponding labels of the selected subset_xs
    reg_best: array(2,1)
        regularization parameter for the entropic and and  the group lasso 
        terms.

    """
    acc_cv = []
    lista_xs = []
    lista_ys = []

    regu_ = []

    for k in range(kfold):        
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(
                                            xs, ys, train_size=M, stratify=ys)

        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)
        
        if np.size(rango_e)==1 and np.size(rango_cl)==1:
            regu = [rango_e, rango_cl]
        else:
            regu = CVgrouplasso(
                rango_e=rango_e, rango_cl=rango_cl,
                xs=xs_daotcv, ys=ys_daotcv, xt=xv, yt=yv, clf=clf,
                metrica=metrica, Verbose=Verbose)
        regu_.append(regu)

        ot_l1l2 = ot.da.SinkhornL1l2Transport(
                metric=metrica, reg_e=regu[0], reg_cl=regu[1])

        ot_l1l2.fit(Xs=xs_daotcv, ys=ys_daotcv, Xt=xv)

        # transport source samples
        transp_Xs_l1l2 = ot_l1l2.transform(Xs=xs)

        # lda
        clf.fit(transp_Xs_l1l2, ys)
        acc_cv.append(clf.score(xv, yv))

    index = np.argmax(acc_cv)
    subset_xs = lista_xs[index]
    subset_ys = lista_ys[index]
    reg_best = regu_[index]
    return subset_xs, subset_ys, reg_best


def SelectSubsetTraining_OTDAs(xs, ys, xv, yv, rango_e, clf,
                               metrica="euclidean",
                               kfold=20, M=40, Verbose=False):
    """
    select subset of source data to learn the mapping.

    Parameters
    ----------
    xs : array (ns, m)
        source data matrix.
    ys : array (ns,)
        labels source samples.
    xv : array(nv,m)
        transportation matrix.
    yv : array (nv,)
        labels transportation set samples.
    rango_e : array/np.array
        grid of parameter values from the regularization term for entropic
        regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso
        regularization
    metrica : string, optional
        distance used within OT. The default is "euclidean".
    kfold : number, optional
        times to repeat the resample. The default is 20.
    M : number, optional
        final samples included in the subset. The default is 40.
    trad : boolean, optional
        if True the outputs are calculard following the re-training OTDA path,
        othverwise, the one-trainig path is followed. The default is True.

    Returns
    -------
    subset_xs: array (M, m)
        best selected subset
    subset_ys: (M,)
        corresponding labels of the selected subset_xs
    reg_best: number
        regularization parameter for the entropic

    """
    acc_cv = []
    lista_xs = []
    lista_ys = []

    regu_ = []

    for k in range(0, kfold):
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(
            xs, ys, train_size=M, stratify=ys)

        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)
        
        if np.size(rango_e)==1:
            regu = rango_e
        else:
            regu=CVsinkhorn(rango_e=rango_e, xs=xs_daotcv, ys=ys_daotcv,
                        xt=xv, yt=yv, clf=clf, metrica=metrica,
                        Verbose=Verbose)
        regu_.append(regu)

        ot_sinkhorn = ot.da.SinkhornTransport(metric=metrica, reg_e=regu)
        ot_sinkhorn.fit(Xs=xs_daotcv, ys=ys_daotcv, Xt=xv)
        transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=xs)

        # lda
        clf.fit(transp_Xs_sinkhorn, ys)
        acc_cv.append(clf.score(xv, yv))

    index = np.argmax(acc_cv)
    subset_xs = lista_xs[index]
    subset_ys = lista_ys[index]
    
    reg_best = regu_[index]
    return subset_xs, subset_ys, reg_best
