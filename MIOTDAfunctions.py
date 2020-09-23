import numpy as np
from numpy import unravel_index
import ot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# functions
# Autor: Victoria Peterson <vpeterson@santafe-conicet.gov.ar>


def CVsinkhorn(rango_e, xs, ys, xt, yt, clf, metrica="sqeuclidean", 
               kfold=None, Verbose=False):
    """
    This function search for the best reg. parameter based on accuracy
    within the OT-Sinkhorn method
    Parameters
    ----------
    rango_e : list
        grid of parameter values from the regularization term for entropic
        regularization. list can have only one value
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
    clf : model 
        classifier to be trained by the transported source samples.
    metrica : string, optional
        distance used within OT. The default is "sqeuclidean".
    kfold : dict, optional
        dictionary which contains in "nfold" the number of fold to run the 
        kfold cross-validation and "train_size" a value between 0 and 1
        which indicate the percentage of data keep for training.
        The default is None.
    Verbose : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    regu: float
        best reg. parameter selected based on accuracy

    """
    result = []

    # select subset validation to avoid overfitting

    for r in range(np.size(rango_e)):
        ot_sinkhorn = ot.da.SinkhornTransport(metric=metrica,reg_e=rango_e[r],
                                              verbose=Verbose)
        acc_cv = []
        if kfold is None:
            ot_sinkhorn.fit(Xs=xs, Xt=xt)
            # transform
            transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=xs)
            # train new classifier
            clf.fit(transp_Xs_sinkhorn, ys)

            yt_predict = clf.predict(xt)
            # Compute accuracy trad DOAT
            acc_ = accuracy_score(yt, yt_predict)

            result.append(acc_)
        else:
            for k in range(kfold["nfold"]):
                xs_train, xs_test, ys_train, ys_test = train_test_split(
                    xs, ys, train_size=kfold["train_size"], stratify=ys)
                ot_sinkhorn.fit(Xs=xs_train, Xt=xt)
                # transform
                transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=xs_test)
                # train new classifier
                clf.fit(transp_Xs_sinkhorn, ys_test)

                yt_predict = clf.predict(xt)
                # Compute accuracy trad DOAT
                acc_cv.append(accuracy_score(yt, yt_predict))

            result.append(np.mean(acc_cv))

    index = np.argmax(result)
    regu = rango_e[index]
    return regu


def CVgrouplasso(rango_e, rango_cl, xs, ys, xt, yt, clf, metrica="sqeuclidean",
                 kfold=None, Verbose=False):
    """
    This function search for the best set of reg. parameters within the OT-L1L2
    method.

    Parameters
    ----------
    rango_e : list
        grid of parameter values from the regularization term for entropic
        regularization. list can have only one value
    rango_cl : list
        grid of parameter values from the regularization term  for group lasso
        regularization. list can have only one value
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
    clf : model
        classifier to be trained by the transported source samples.
    metrica : string, optional
        distance used within OT. The default is "sqeuclidean".
    kfold : dict, optional
        dictionary which contains in "nfold" the number of fold to run the
        kfold cross-validation and "train_size" a value between 0 and 1
        which indicate the percentage of data keep for training.
        The default is None.
    Verbose : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    regu : list
        vector with the best reg. parameter for the FOTDA-GL
        regu_trad[0]: entropic regularizer param
        regu_trad[1]; group-lasso regulatizer param
    """
    result = np.empty((np.size(rango_e), np.size(rango_cl)), dtype=float)

    for r in range(np.size(rango_e)):
        for rr in range(np.size(rango_cl)):
            acc_cv = []
            ot_l1l2 = ot.da.SinkhornL1l2Transport(
                    metric=metrica, reg_e=rango_e[r],
                    reg_cl=rango_cl[rr], verbose=Verbose)
            if kfold is None:
                ot_l1l2.fit(Xs=xs, ys=ys, Xt=xt)

                # transport source samples onto target samples
                transp_Xs_lpl1 = ot_l1l2.transform(Xs=xs)
                # train on new source
                clf.fit(transp_Xs_lpl1, ys)
                # Compute accuracy
                yt_predict = clf.predict(xt)
                acc_ = accuracy_score(yt, yt_predict)

                result[r, rr] = acc_
            else:
                for k in range(kfold["nfold"]):
                    xs_train, xs_test, ys_train, ys_test = train_test_split(
                        xs, ys, train_size=kfold["train_size"], stratify=ys)

                    # Sinkhorn Transport with Group lasso regularization
                    ot_l1l2.fit(Xs=xs_train, ys=ys_train, Xt=xt)

                    # transport source samples onto target samples
                    transp_Xs_lpl1 = ot_l1l2.transform(Xs=xs_test)
                    # train on new source
                    clf.fit(transp_Xs_lpl1, ys_test)
                    # Compute accuracy
                    yt_predict = clf.predict(xt)
                    acc_cv.append(accuracy_score(yt, yt_predict))

                result[r, rr] = np.mean(acc_cv)

    index = unravel_index(result.argmax(), result.shape)
    regu = [rango_e[index[0]], rango_cl[index[1]]]
    return regu


def CVgrouplasso_backward(rango_e, rango_cl, xs, ys, xt, yt, clf,
                          metrica="sqeuclidean", kfold=None, Verbose=False):
    """
    This function search for the best set of reg. parameters within the
    Backward OT-L1L2 method.

    Parameters
    ----------
    rango_e : list
        grid of parameter values from the regularization term for entropic
        regularization. list can have only one value
    rango_cl : list
        grid of parameter values from the regularization term  for group lasso
        regularization. list can have only one value
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
    clf : model
        classifier ALREADY trained on SOURCE data. Used to predict
        transported target samples.
    metrica : string, optional
        distance used within OT. The default is "sqeuclidean".
    kfold : dict, optional
        dictionary which contains in "nfold" the number of fold to run the
        kfold cross-validation and "train_size" a value between 0 and 1
        which indicate the percentage of data keep for training.
        The default is None.
    Verbose : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    regu : list
        vector with the best reg. parameter for the FOTDA-GL
        regu_trad[0]: entropic regularizer param
        regu_trad[1]; group-lasso regulatizer param


    """
    result = np.empty((np.size(rango_e), np.size(rango_cl)), dtype=float)

    for r in range(np.size(rango_e)):
        for rr in range(np.size(rango_cl)):
            acc_cv = []
            bot_l1l2 = ot.da.SinkhornL1l2Transport(
                    metric=metrica, reg_e=rango_e[r], reg_cl=rango_cl[rr],
                    verbose=Verbose)
            if kfold is None:
                bot_l1l2.fit(Xs=xt, ys=yt, Xt=xs)
                # transport target samples onto source samples
                transp_Xt_lpl1 = bot_l1l2.transform(Xs=xt)
                # Compute accuracy
                yt_predict = clf.predict(transp_Xt_lpl1)
                acc_ = accuracy_score(yt, yt_predict)

                result[r, rr] = acc_
            else:

                for k in range(kfold["nfold"]):
                    xt_train, xt_test, yt_train, yt_test = train_test_split(
                        xt, yt, train_size=kfold["train_size"], stratify=yt)

                    # Sinkhorn Transport with Group lasso regularizatio
                    bot_l1l2.fit(Xs=xt_train, ys=yt_train, Xt=xs)
                    # transport target samples onto source samples
                    transp_Xt_lpl1 = bot_l1l2.transform(Xs=xt_test)
                    # Compute accuracy
                    yt_predict = clf.predict(transp_Xt_lpl1)
                    acc_cv.append(accuracy_score(yt_test, yt_predict))

                result[r, rr] = np.mean(acc_cv)
    index = unravel_index(result.argmax(), result.shape)
    regu = [rango_e[index[0]], rango_cl[index[1]]]

    return regu


def CVsinkhorn_backward(rango_e, xs, ys, xt, yt, clf, metrica="sqeuclidean",
                        kfold=None, Verbose=False):
    """
    This function search for the best set of reg. parameters within the
    OT-Sinkhorn method.

    Parameters
    ----------
    rango_e : list
        grid of parameter values from the regularization term for entropic
        regularization. list can have only one value
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
    clf : model
        classifier ALREADY trained on SOURCE data. Used to predict
        transported target samples.
    metrica : string, optional
        distance used within OT. The default is "sqeuclidean".
    kfold : dict, optional
        dictionary which contains in "nfold" the number of fold to run the 
        kfold cross-validation and "train_size" a value between 0 and 1
        which indicate the percentage of data keep for training.
        The default is None.
    Verbose : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    regu: float
        best reg. parameter selected based on accuracy

    """
    result = []

    for r in range(np.size(rango_e)):
        acc_cv = []
        bot = ot.da.SinkhornTransport(metric=metrica, reg_e=rango_e[r],
                                          verbose=Verbose)
        if kfold is None:
            # Sinkhorn Transport with Group lasso regularization
            bot.fit(Xs=xt, ys=yt, Xt=xs)

            # transport target samples onto spurce samples
            transp_Xt_lpl1 = bot.transform(Xs=xt)

            # Compute accuracy
            yt_predict = clf.predict(transp_Xt_lpl1)
            acc_ = accuracy_score(yt, yt_predict)

            result.append(acc_)
        else:
            for k in range(kfold["nfold"]):
                xt_train, xt_test, yt_train, yt_test = train_test_split(
                    xt, yt, train_size=kfold["train_size"], stratify=yt)
                # Sinkhorn Transport with Group lasso regularization
                bot.fit(Xs=xt_train, ys=yt_train, Xt=xs)

                # transport target samples onto spurce samples
                transp_Xt_lpl1 = bot.transform(Xs=xt_test)

                # Compute accuracy
                yt_predict = clf.predict(transp_Xt_lpl1)
                acc_cv.append(accuracy_score(yt_test, yt_predict))

            result.append(np.mean(acc_cv))

    result = np.asarray(result)
    index = np.argmax(result)
    regu = rango_e[index]
    return regu


def SelectSubsetTraining_BOTDAl1l2(xs, ys, xv, yv, rango_e, rango_cl, clf,
                                    metrica="sqeuclidean", outerkfold=20,
                                    innerkfold=None, M=40,
                                    Verbose=False):
    """
    select subset of source data to learn the mapping and the best regu
    parameters for that subset.

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
    rango_e : list
        grid of parameter values from the regularization term for entropic
        regularization. list can have only one value.
    rango_cl : list
        grid of parameter values from the regularization term  for group lasso
        regularization. list can have only one value.
    clf : model
        classifier ALREADY trained on Source data. 
        Used to make the prediction on the transported target samples.
    metrica : TYPE, optional
        DESCRIPTION. The default is "sqeuclidean".
    outerkfold : number, optional
        times to repeat the resample. The default is 20.
    innerkfold : dict, optional
        dictionary which contains in "nfold" the number of fold to run the
        kfold cross-validation and "train_size" a value between 0 and 1
        which indicate the percentage of data keep for training.
        The default is None.
    M : number, optional
        final samples included in the subset. The default is 40.
    Verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    subset_xs: array (M, m)
        best selected subset
    subset_ys: (M,)
        corresponding labels of the selected subset_xs
    reg_best: list
        regularization parameter for the entropic and the group lasso
        terms.
    """

    acc_cv = []
    lista_xs = []
    lista_ys = []

    regu_ = []

    for k in range(outerkfold):
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(xs, ys, train_size=M, stratify=ys)

        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)
        if np.size(rango_e) == 1 and np.size(rango_cl) == 1:
            regu = [rango_e, rango_cl]
        else:
            regu = CVgrouplasso_backward(rango_e, rango_cl, xs_daotcv,
                                          ys_daotcv, xv, yv, clf, metrica,
                                          kfold=innerkfold, Verbose=Verbose)

        regu_.append(regu)

        bot_l1l2 = ot.da.SinkhornL1l2Transport(
            metric=metrica, reg_e=regu[0], reg_cl=regu[1])

        bot_l1l2.fit(Xs=xv, ys=yv, Xt=xs_daotcv)
        # transport val sampless
        transp_Xv_l1l2 = bot_l1l2.transform(Xs=xv)
        acc_cv.append(clf.score(transp_Xv_l1l2, yv))

    index = np.argmax(acc_cv)

    subset_xs = lista_xs[index]
    subset_ys = lista_ys[index]
    reg_best = regu_[index]
    return subset_xs, subset_ys, reg_best


def SelectSubsetTraining_BOTDAs(xs, ys, xv, yv, rango_e, clf,
                                metrica="sqeuclidean", outerkfold=20, 
                                innerkfold=None, M=40,
                                Verbose=False):
    """
    select subset of source data to learn the mapping and the best regu
    parameters for that subset.

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
    rango_e : list
        grid of parameter values from the regularization term for entropic
        regularization. list can have only one value.
    clf : model
        classifier ALREADY trained on Source data. 
        Used to make the prediction on the transported target samples.
    metrica : TYPE, optional
        DESCRIPTION. The default is "sqeuclidean".
    outerkfold : number, optional
        times to repeat the resample. The default is 20.
    innerkfold : dict, optional
        dictionary which contains in "nfold" the number of fold to run the
        kfold cross-validation and "train_size" a value between 0 and 1
        which indicate the percentage of data keep for training.
        The default is None.
    M : number, optional
        final samples included in the subset. The default is 40.
    Verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    subset_xs: array (M, m)
        best selected subset
    subset_ys: (M,)
        corresponding labels of the selected subset_xs
    reg_best: float
        regularization parameter for the entropic term
    """
    acc_cv = []
    lista_xs = []
    lista_ys = []

    regu_ = []

    for k in range(outerkfold):
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(
            xs, ys, train_size=M, stratify=ys)

        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)
        if np.size(rango_e) == 1:
            regu = rango_e
        else:
            regu = CVsinkhorn_backward(rango_e, xs_daotcv, ys_daotcv, xv, yv,
                                        clf, metrica, innerkfold, Verbose)
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
                                  metrica="sqeuclidean", outerkfold=20,
                                  innerkfold=None, M=40,
                                  Verbose=False):
    """
    select subset of source data to learn the mapping and the best regu
    parameters for that subset.

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
    rango_e : list
        grid of parameter values from the regularization term for entropic
        regularization. list can have only one value.
    rango_cl : list
        grid of parameter values from the regularization term  for group lasso
        regularization. list can have only one value.
    clf : model
        classifier to be trained by the transported source samples.
    metrica : TYPE, optional
        DESCRIPTION. The default is "sqeuclidean".
    outerkfold : number, optional
        times to repeat the resample. The default is 20.
    innerkfold : dict, optional
        dictionary which contains in "nfold" the number of fold to run the
        kfold cross-validation and "train_size" a value between 0 and 1
        which indicate the percentage of data keep for training.
        The default is None.
    M : number, optional
        final samples included in the subset. The default is 40.
    Verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    subset_xs: array (M, m)
        best selected subset
    subset_ys: (M,)
        corresponding labels of the selected subset_xs
    reg_best: list
        regularization parameter for the entropic and the group lasso
        terms.
    """
    acc_cv = []
    lista_xs = []
    lista_ys = []

    regu_ = []

    for k in range(outerkfold):        
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(
                                            xs, ys, train_size=M, stratify=ys)

        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)

        if np.size(rango_e) == 1 and np.size(rango_cl) == 1:
            regu = [rango_e[0], rango_cl[0]]
        else:
            regu = CVgrouplasso(rango_e, rango_cl, xs_daotcv, ys_daotcv,
                                xv, yv, clf, metrica, innerkfold, Verbose)
        regu_.append(regu)

        ot_l1l2 = ot.da.SinkhornL1l2Transport(
                metric=metrica, reg_e=regu[0], reg_cl=regu[1])

        ot_l1l2.fit(Xs=xs_daotcv, ys=ys_daotcv, Xt=xv)

        # transport source samples
        transp_Xs_l1l2 = ot_l1l2.transform(Xs=xs)

        # clf fit
        clf.fit(transp_Xs_l1l2, ys)
        acc_cv.append(clf.score(xv, yv))

    index = np.argmax(acc_cv)
    subset_xs = lista_xs[index]
    subset_ys = lista_ys[index]
    reg_best = regu_[index]
    return subset_xs, subset_ys, reg_best


def SelectSubsetTraining_OTDAs(xs, ys, xv, yv, rango_e, clf,
                               metrica="euclidean",
                               outerkfold=20, innerkfold=None,
                               M=40, Verbose=False):
    """
    select subset of source data to learn the mapping and the best regu
    parameters for that subset.

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
    rango_e : list
        grid of parameter values from the regularization term for entropic
        regularization
    clf : model
        classifier to be trained by the transported source samples.
    metrica : TYPE, optional
        DESCRIPTION. The default is "sqeuclidean".
    outerkfold : number, optional
        times to repeat the resample. The default is 20.
    innerkfold : dict, optional
        dictionary which contains in "nfold" the number of fold to run the
        kfold cross-validation and "train_size" a value between 0 and 1
        which indicate the percentage of data keep for training.
        The default is None.
    M : number, optional
        final samples included in the subset. The default is 40.
    Verbose : TYPE, optional
        DESCRIPTION. The default is False.

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

    for k in range(outerkfold):
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(
            xs, ys, train_size=M, stratify=ys)

        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)

        if np.size(rango_e) == 1:
            regu = rango_e[0]
        else:
            regu=CVsinkhorn(rango_e, xs_daotcv, ys_daotcv, xv, yv, clf, 
                            metrica, innerkfold, Verbose)
        regu_.append(regu)

        ot_sinkhorn = ot.da.SinkhornTransport(metric=metrica, reg_e=regu)
        ot_sinkhorn.fit(Xs=xs_daotcv, ys=ys_daotcv, Xt=xv)
        transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=xs)

        # clf fit
        clf.fit(transp_Xs_sinkhorn, ys)
        acc_cv.append(clf.score(xv, yv))

    index = np.argmax(acc_cv)
    subset_xs = lista_xs[index]
    subset_ys = lista_ys[index]

    reg_best = regu_[index]
    return subset_xs, subset_ys, reg_best
