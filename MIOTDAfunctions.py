#%%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np # always need it
from numpy import unravel_index
import ot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#%% functions
def CVsinkhorn(rango, xs, ys, xt, yt, metrica="euclidean"):
    """
    This function search for the best reg. parameter within the OT-Sinkhorn 
    method
    Parameters
    ----------
    rango : array/np.array
        grid of parameter values from the regularization term for entropic regularization
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
    regu_trad : number
        best reg. parameter for the traditional OTDA path (retraining).
    regu_inver : number
        best reg. parameter for the inverse OTDA path (one-training).


    """
    #use only the half for validation
    # nb,nf=xt.shape
    # sequence = [i for i in range(nb)]
    # # idx = sample(sequence, 10)  
    # idx = sample(sequence, np.int32(np.round(nb/2)))  
    
    # xt=xt[idx]
    # yt=yt[idx]   
    # np.int32(np.round(nb/2))
    # nb,nf=xs.shape
    # sequence = [i for i in range(nb)]
    # idx = sample(sequence,20)  
    # xs=xs[idx]
    # ys=ys[idx]
    
    ACC_daot_trad_val=[]
    ACC_daot_inverse_val=[]
    for r in range(0,len(rango)):
        ot_sinkhorn= ot.da.SinkhornTransport(metric=metrica,reg_e=rango[r], verbose=None)
        ot_sinkhorn.fit(Xs=xs, Xt=xt)
        #unsupervised
        transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=xs)
        transp_Xt_sinkhorn = ot_sinkhorn.inverse_transform(Xt=xt)
        #
        clf2 = LinearDiscriminantAnalysis()
        clf2.fit(transp_Xs_sinkhorn,ys)
        
        yt_predict2=clf2.predict(xt)
        acc_daot_trad=accuracy_score(yt, yt_predict2)
    
        # Compute accuracy trad DOAT
        ACC_daot_trad_val.append(acc_daot_trad)
            
        # Compute accuracy inverse DAOT
        clf = LinearDiscriminantAnalysis()
        clf.fit(xs,ys)
        yt_predict3=clf.predict(transp_Xt_sinkhorn)
        acc_daot_inver=accuracy_score(yt, yt_predict3)
    
        ACC_daot_inverse_val.append(acc_daot_inver)
    regu_trad=np.argmax(ACC_daot_trad_val)
    regu_inver=np.argmax(ACC_daot_inverse_val)
    return regu_trad, regu_inver
    
def CVgrouplasso(rango_e, rango_cl, xs, ys, xt, yt, metrica="euclidean"):
    """
    This function search for the best set of reg. parameters within the OT-L1L2 
    method.
    
    Parameters
    ----------
    rango_e : array/np.array
        grid of parameter values from the regularization term for entropic regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso regularization
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
    regu_trad : array (2,1)
        vector with the best reg. parameter for the traditional OTDA path (retraining).
        regu_trad[0]: entropic regularization
        regu_trad[1]; group-lasso regulatization
    regu_inver : array(2,1)
        vector with the best reg. parameter for the traditional OTDA path (retraining).
        regu_inver[0]: entropic regularization
        regu_inver[1]; group-lasso regulatization


    """
    result_inverse=[]
    result_trad=[]
    
    # use only the half for validation
    # nb,nf=xt.shape
    # sequence = [i for i in range(nb)]
    # # idx = sample(sequence,10)  
    # idx = sample(sequence, np.int32(np.round(nb/2)))  

    # xt=xt[idx]
    # yt=yt[idx]   
    
    # nb,nf=xs.shape
    # sequence = [i for i in range(nb)]
    # idx = sample(sequence, np.int32(np.round(nb/2)))  
    # xs=xs[idx]
    # ys=ys[idx]
    
    for r in range(0,len(rango_e)):
        ACC_daot_inverse_val=[]
        ACC_daot_trad_val=[]
        for rr in range(0,len(rango_cl)):
            # Sinkhorn Transport with Group lasso regularization
            ot_l1l2 = ot.da.SinkhornL1l2Transport(metric=metrica,reg_e=rango_e[r], reg_cl=rango_cl[rr],verbose=None)
            ot_l1l2.fit(Xs=xs, ys=ys, Xt=xt)
                
            # transport source samples onto target samples
            transp_Xs_lpl1 = ot_l1l2.transform(Xs=xs)
            transp_Xt_inverse=ot_l1l2.inverse_transform(Xt=xt)
            # train on new source
            clf2 = LinearDiscriminantAnalysis()
            clf2.fit(transp_Xs_lpl1,ys)
                
            yt_predict2=clf2.predict(xt)
            acc_daot_trad=accuracy_score(yt, yt_predict2)
            
            # Compute accuracy trad DOAT
            ACC_daot_trad_val.append(acc_daot_trad)
                    
            # Compute accuracy inverse DAOT
            clf = LinearDiscriminantAnalysis()
            clf.fit(xs,ys)
            yt_predict3=clf.predict(transp_Xt_inverse)
            acc_daot_inver=accuracy_score(yt, yt_predict3)
            
            ACC_daot_inverse_val.append(acc_daot_inver)
                
            
        result_inverse.append(ACC_daot_inverse_val)
        result_trad.append(ACC_daot_trad_val)
        
    result_trad=np.asarray(result_trad)
    result_inverse=np.asarray(result_inverse)    
    regu_trad=unravel_index(result_trad.argmax(), result_trad.shape)
    regu_inver=unravel_index(result_inverse.argmax(), result_inverse.shape)

    return regu_trad, regu_inver
            
def CVgrouplasso_backward(rango_e, rango_cl, xs, ys, xt, yt, metrica="euclidean"):
    
    """
    This function search for the best set of reg. parameters within the Backward
    OT-L1L2 method.
    
    Parameters
    ----------
    rango_e : array/np.array
        grid of parameter values from the regularization term for entropic regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso regularization
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
    regu_trad : array (2,1)
        vector with the best reg. parameter for the traditional OTDA path (retraining).
        regu_trad[0]: entropic regularization
        regu_trad[1]; group-lasso regulatization
    regu_inver : array(2,1)
        vector with the best reg. parameter for the traditional OTDA path (retraining).
        regu_inver[0]: entropic regularization
        regu_inver[1]; group-lasso regulatization


    """
    
    result_inverse=[]
    result_trad=[]
    
    
    # nb,nf=xt.shape
    # sequence = [i for i in range(nb)]
    
    # # idx = sample(sequence,10)
    # idx = sample(sequence, np.int32(np.round(nb/2)))  


    # idx = sample(sequence,np.int32(np.round(nb/2)))  
    # xt=xt[idx]
    # yt=yt[idx]   
    # # nb,nf=xs.shape
    # idx=seed.randint(nb, size=20)
    # nb,nf=xs.shape
    # sequence = [i for i in range(nb)]
    # idx = sample(sequence, np.int32(np.round(nb/2)))  
    # xs=xs[idx]
    # ys=ys[idx]        
       
    for r in range(0,len(rango_e)):
        ACC_daot_inverse_val=[]
        ACC_daot_trad_val=[]

        for rr in range(0,len(rango_cl)):
            # Sinkhorn Transport with Group lasso regularization
            ot_l1l2_smll = ot.da.SinkhornL1l2Transport(metric=metrica,reg_e=rango_e[r], reg_cl=rango_cl[rr], verbose=None)
            ot_l1l2_smll.fit(Xs=xt, ys=yt, Xt=xs)
                
            # transport source samples onto target samples
            transp_Xt_lpl1_smll = ot_l1l2_smll.transform(Xs=xt)
            transp_Xs_lpl1_smll = ot_l1l2_smll.inverse_transform(Xt=xs)
                
            # train on new source
            clf2 = LinearDiscriminantAnalysis()
            clf2.fit(transp_Xs_lpl1_smll,ys)
                
            yt_predict2=clf2.predict(xt)
            acc_daot_trad=accuracy_score(yt, yt_predict2)
            
            # Compute accuracy trad DOAT
            ACC_daot_trad_val.append(acc_daot_trad)
                    
            # Compute accuracy inverse DAOT
            clf = LinearDiscriminantAnalysis()
            clf.fit(xs,ys)
            yt_predict3=clf.predict(transp_Xt_lpl1_smll)
            acc_daot_inver=accuracy_score(yt, yt_predict3)
            
            ACC_daot_inverse_val.append(acc_daot_inver)
            
        result_inverse.append(ACC_daot_inverse_val)
        result_trad.append(ACC_daot_trad_val)
        
    result_trad=np.asarray(result_trad)
    result_inverse=np.asarray(result_inverse)    
    regu_trad=unravel_index(result_trad.argmax(), result_trad.shape)
    regu_inver=unravel_index(result_inverse.argmax(), result_inverse.shape)

    return regu_trad, regu_inver

def CVsinkhorn_backward(rango_e, xs, ys, xt, yt, metrica="euclidean"):
    """
    This function search for the best set of reg. parameters within the OT-Sinkhorn 
    method.
    
    Parameters
    ----------
    rango_e : array/np.array
        grid of parameter values from the regularization term for entropic regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso regularization
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
    regu_trad : number
        the best reg. parameter for the traditional OTDA path (retraining).
    regu_inver : number
        the best reg. parameter for the traditional OTDA path (retraining).


    """
    
    result_inverse=[]
    result_trad=[]
    
    # nb,nf=xt.shape
    # sequence = [i for i in range(nb)]
    # # idx = sample(sequence,10) 
    # idx = sample(sequence, np.int32(np.round(nb/2)))  
    
    # xt=xt[idx]
    # yt=yt[idx]   
    # np.int32(np.round(nb/2))      
       
    for r in range(0,len(rango_e)):
        ACC_daot_inverse_val=[]
        ACC_daot_trad_val=[]

           # Sinkhorn Transport with Group lasso regularization
        ot_smll = ot.da.SinkhornTransport(metric=metrica,reg_e=rango_e[r], verbose=None)
        ot_smll.fit(Xs=xt, ys=yt, Xt=xs)
                
        # transport source samples onto target samples
        transp_Xt_lpl1_smll = ot_smll.transform(Xs=xt)
        transp_Xs_lpl1_smll = ot_smll.inverse_transform(Xt=xs)
                
        # train on new source
        clf2 = LinearDiscriminantAnalysis()
        clf2.fit(transp_Xs_lpl1_smll,ys)
                
        yt_predict2=clf2.predict(xt)
        acc_daot_trad=accuracy_score(yt, yt_predict2)
            
        # Compute accuracy trad DOAT
        ACC_daot_trad_val.append(acc_daot_trad)
                    
        # Compute accuracy inverse DAOT
        clf = LinearDiscriminantAnalysis()
        clf.fit(xs,ys)
        yt_predict3=clf.predict(transp_Xt_lpl1_smll)
        acc_daot_inver=accuracy_score(yt, yt_predict3)
        
        ACC_daot_inverse_val.append(acc_daot_inver)
            
        result_inverse.append(ACC_daot_inverse_val)
        result_trad.append(ACC_daot_trad_val)
        
    result_trad=np.asarray(result_trad)
    result_inverse=np.asarray(result_inverse)    
    regu_trad=np.argmax(ACC_daot_trad_val)
    regu_inver=np.argmax(ACC_daot_inverse_val)

    return regu_trad, regu_inver
###############
def SelectSubsetTraining_BOTDAl1l2(xs, ys, xv, yv, rango_e, rango_cl, metrica="euclidean", kfold=20, M=40, trad=True):
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
        grid of parameter values from the regularization term for entropic regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso regularization
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
        regularization parameter for the entropic and and  the group lasso terms.

    """
    
    acc_cv=[]
    lista_xs=[]
    lista_ys=[]

    regu_trad=[]
    regu_inver=[]

    #sequence = [i for i in range(len(ys))]

    for k in range(0, kfold):
        # idx = sample(sequence, M)  
        
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(xs, ys, train_size=M, stratify=ys)


        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)

        # xs_daotcv=xs[idx]
        # ys_daotcv=ys[idx]   
        ldacv = LinearDiscriminantAnalysis()

        ldacv.fit(xs,ys) 
        regu_trad3, regu_inver3=CVgrouplasso_backward(rango_e=rango_e, rango_cl=rango_cl, xs=xs_daotcv, ys=ys_daotcv, xt=xv, yt=yv)
        regu_trad.append(regu_trad3)
        regu_inver.append(regu_inver3)


        ot_l1l2_sup = ot.da.SinkhornL1l2Transport(metric=metrica, reg_e=rango_e[regu_inver3[0]], reg_cl=rango_cl[regu_inver3[1]])

        ot_l1l2_sup.fit(Xs=xv, ys=yv, Xt=xs_daotcv)
        #transport testing sampless
        transp_Xt_l1l2_sup=ot_l1l2_sup.transform(Xs=xv)
        
        
        ot_l1l2_sup = ot.da.SinkhornL1l2Transport(metric=metrica, reg_e=rango_e[regu_trad3[0]], reg_cl=rango_cl[regu_trad3[1]])

        ot_l1l2_sup.fit(Xs=xv, ys=yv, Xt=xs_daotcv)
        #transport testing sampless
        transp_Xs_l1l2_sup=ot_l1l2_sup.inverse_transform(Xt=xs)
        
         # tradicional
        lda3 = LinearDiscriminantAnalysis()
        lda3.fit(transp_Xs_l1l2_sup,ys)
        if trad:
            acc_cv.append(lda3.score(xv, yv))
        else:
            acc_cv.append(ldacv.score(transp_Xt_l1l2_sup, yv))
    index=np.argmax(acc_cv)
    if trad:
        regu=regu_trad
    else:   
        regu=regu_inver    
    
    subset_xs=lista_xs[index]
    subset_ys=lista_ys[index]
    reg_best=regu[index]
    return subset_xs, subset_ys, reg_best


def SelectSubsetTraining_BOTDAs(xs, ys, xv, yv, rango_e, metrica="euclidean", kfold=20, M=40, trad=True):
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
        grid of parameter values from the regularization term for entropic regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso regularization
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
    
    acc_cv=[]
    lista_xs=[]
    lista_ys=[]

    regu_trad=[]
    regu_inver=[]

    #sequence = [i for i in range(len(ys))]

    for k in range(0, kfold):
        # idx = sample(sequence, M)  
        
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(xs, ys, train_size=M, stratify=ys)


        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)

        # xs_daotcv=xs[idx]
        # ys_daotcv=ys[idx]   
        ldacv = LinearDiscriminantAnalysis()

        ldacv.fit(xs,ys) 
        regu_trad3, regu_inver3=CVsinkhorn_backward(rango_e=rango_e, xs=xs_daotcv, ys=ys_daotcv, xt=xv, yt=yv)
        regu_trad.append(regu_trad3)
        regu_inver.append(regu_inver3)


        ot_l1l2_sup = ot.da.SinkhornTransport(metric=metrica, reg_e=rango_e[regu_inver3])

        ot_l1l2_sup.fit(Xs=xv, ys=yv, Xt=xs_daotcv)
        #transport testing sampless
        transp_Xt_l1l2_sup=ot_l1l2_sup.transform(Xs=xv)
        
        
        ot_l1l2_sup = ot.da.SinkhornTransport(metric=metrica, reg_e=rango_e[regu_trad3])

        ot_l1l2_sup.fit(Xs=xv, ys=yv, Xt=xs_daotcv)
        #transport testing sampless
        transp_Xs_l1l2_sup=ot_l1l2_sup.inverse_transform(Xt=xs)
        
         # tradicional
        lda3 = LinearDiscriminantAnalysis()
        lda3.fit(transp_Xs_l1l2_sup,ys)
        if trad:
            acc_cv.append(lda3.score(xv, yv))
        else:
            acc_cv.append(ldacv.score(transp_Xt_l1l2_sup, yv))
    index=np.argmax(acc_cv)
    if trad:
        regu=regu_trad
    else:   
        regu=regu_inver    
    subset_xs=lista_xs[index]
    subset_ys=lista_ys[index]
    reg_best=regu[index]
    return subset_xs, subset_ys, reg_best

def SelectSubsetTraining_OTDAl1l2(xs, ys, xv, yv, rango_e, rango_cl, metrica="euclidean", kfold=20, M=40, trad=True):
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
        grid of parameter values from the regularization term for entropic regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso regularization
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
        regularization parameter for the entropic and and  the group lasso terms.

    """
    
    acc_cv=[]
    lista_xs=[]
    lista_ys=[]

    regu_trad=[]
    regu_inver=[]

    #sequence = [i for i in range(len(ys))]

    for k in range(0, kfold):
        # idx = sample(sequence, M)  
        
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(xs, ys, train_size=M, stratify=ys)


        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)
        # xs_daotcv=xs[idx]
        # ys_daotcv=ys[idx]   
        ldacv = LinearDiscriminantAnalysis()

        ldacv.fit(xs,ys) 
        regu_trad2, regu_inv2=CVgrouplasso(rango_e=rango_e, rango_cl=rango_cl, xs=xs_daotcv, ys=ys_daotcv, xt=xv, yt=yv)
            # ot_l1l2_un = ot.da.SinkhornL1l2Transport(reg_e=1e-1, reg_cl=10)
        regu_trad.append(regu_trad2)
        regu_inver.append(regu_inv2)


        ot_l1l2_un = ot.da.SinkhornL1l2Transport(metric=metrica,reg_e=rango_e[regu_trad2[0]], reg_cl=rango_cl[regu_trad2[1]])

        ot_l1l2_un.fit(Xs=xs_daotcv, ys=ys_daotcv, Xt=xv)
        
        #transport taget samples onto source samples
        transp_Xs_l1l2_un=ot_l1l2_un.transform(Xs=xs)
        
        # tradicional
        lda3 = LinearDiscriminantAnalysis()
        lda3.fit(transp_Xs_l1l2_un,ys)
        
        ot_l1l2_un = ot.da.SinkhornL1l2Transport(metric=metrica,reg_e=rango_e[regu_inv2[0]], reg_cl=rango_cl[regu_inv2[1]])

        ot_l1l2_un.fit(Xs=xs_daotcv, ys=ys_daotcv, Xt=xv)
        
        #transport taget samples onto source samples
        transp_Xt_l1l2_un=ot_l1l2_un.inverse_transform(Xt=xv)
        if trad:
            acc_cv.append(lda3.score(xv, yv))
        else:
            acc_cv.append(ldacv.score(transp_Xt_l1l2_un, yv))

    index=np.argmax(acc_cv)
    if trad:
        regu=regu_trad
    else:   
        regu=regu_inver   
        
    subset_xs=lista_xs[index]
    subset_ys=lista_ys[index]
    reg_best=regu[index]
    return subset_xs, subset_ys, reg_best

def SelectSubsetTraining_OTDAs(xs, ys, xv, yv, rango_e, metrica="euclidean", kfold=20, M=40, trad=True):
    
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
        grid of parameter values from the regularization term for entropic regularization
    rango_cl : array/np.array
        grid of parameter values from the regularization term  for group lasso regularization
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
    acc_cv=[]
    lista_xs=[]
    lista_ys=[]

    regu_trad=[]
    regu_inver=[]

    #sequence = [i for i in range(len(ys))]

    for k in range(0, kfold):
        # idx = sample(sequence, M)  
        
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(xs, ys, train_size=M, stratify=ys)


        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)

        # xs_daotcv=xs[idx]
        # ys_daotcv=ys[idx]   
        ldacv = LinearDiscriminantAnalysis()

        ldacv.fit(xs,ys) 
        regu_trad1, regu_inv1=CVsinkhorn(rango=rango_e, xs=xs_daotcv, ys=ys_daotcv, xt=xv, yt=yv)
        regu_trad.append(regu_trad1)
        regu_inver.append(regu_inv1)

        ot_sinkhorn= ot.da.SinkhornTransport(metric=metrica,reg_e=rango_e[regu_trad1])
        ot_sinkhorn.fit(Xs=xs_daotcv, ys=ys_daotcv, Xt=xv)
        transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=xs)
        
        # tradicional
        lda3 = LinearDiscriminantAnalysis()
        lda3.fit(transp_Xs_sinkhorn,ys)
        
        ot_sinkhorn= ot.da.SinkhornTransport(metric=metrica,reg_e=rango_e[regu_inv1])
        ot_sinkhorn.fit(Xs=xs_daotcv, ys=ys_daotcv, Xt=xv)
        transp_Xt_sinkhorn = ot_sinkhorn.inverse_transform(Xt=xv)
        if trad:
            acc_cv.append(lda3.score(xv, yv))
        else:
            acc_cv.append(ldacv.score(transp_Xt_sinkhorn, yv))

    index=np.argmax(acc_cv)
    if trad:
        regu=regu_trad
    else:   
        regu=regu_inver
        
    subset_xs=lista_xs[index]
    subset_ys=lista_ys[index]
    reg_best=regu[index]
    return subset_xs, subset_ys, reg_best
