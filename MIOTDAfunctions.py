#%%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np # always need it
from numpy import unravel_index
import ot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#%% functions
def CVsinkhorn(rango_e, xs, ys, xt, yt, metrica="euclidean"):
    """
    This function search for the best reg. parameter within the OT-Sinkhorn 
    method
    Parameters
    ----------
    rango_E : array/np.array
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
    regu: float
        best reg. parameter selected based on accuracy
   


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
    
    ACC_=[]
    for r in range(len(rango_e)):
        ot_sinkhorn= ot.da.SinkhornTransport(metric=metrica,reg_e=rango_e[r], verbose=None)
        ot_sinkhorn.fit(Xs=xs, Xt=xt)
        #transform
        transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=xs)
        #train new classifier
        clf2 = LinearDiscriminantAnalysis()
        clf2.fit(transp_Xs_sinkhorn,ys)
        
        yt_predict=clf2.predict(xt)
    
        # Compute accuracy trad DOAT
        acc_=accuracy_score(yt, yt_predict)
        ACC_.append(acc_)
            
       
    index=np.argmax(ACC_)
    regu=rango_e[index]
    return regu
    
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
    regu : tuple (2)
        vector with the best reg. parameter for the FOTDA-GL
        regu_trad[0]: entropic regularizer param
        regu_trad[1]; group-lasso regulatizer param
   


    """
    result=np.empty((len(rango_e), len(rango_cl)), dtype=float)
    
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
    
    for r in range(len(rango_e)):
       
        for rr in range(len(rango_cl)):
            # Sinkhorn Transport with Group lasso regularization
            ot_l1l2 = ot.da.SinkhornL1l2Transport(metric=metrica,reg_e=rango_e[r], reg_cl=rango_cl[rr],verbose=None)
            ot_l1l2.fit(Xs=xs, ys=ys, Xt=xt)
                
            # transport source samples onto target samples
            transp_Xs_lpl1 = ot_l1l2.transform(Xs=xs)
            # train on new source
            clf2 = LinearDiscriminantAnalysis()
            clf2.fit(transp_Xs_lpl1,ys)
            # Compute accuracy     
            yt_predict2=clf2.predict(xt)
            acc_=accuracy_score(yt, yt_predict2)
            
            result[r,rr]=acc_
                    
         
                    
        
    index=unravel_index(result.argmax(), result.shape)
    regu=[rango_e[index[0]], rango_cl[index[1]]]
   
    return regu
          
            
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
    regu : tuple (2)
        vector with the best reg. parameter for the FOTDA-GL
        regu_trad[0]: entropic regularizer param
        regu_trad[1]; group-lasso regulatizer param



    """
    
    result=np.empty((len(rango_e), len(rango_cl)), dtype=float)
       
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
       
    for r in range(len(rango_e)):

        for rr in range(len(rango_cl)):
            # Sinkhorn Transport with Group lasso regularization
            bot_l1l2 = ot.da.SinkhornL1l2Transport(metric=metrica,reg_e=rango_e[r], reg_cl=rango_cl[rr], verbose=None)
            bot_l1l2.fit(Xs=xt, ys=yt, Xt=xs)
                
            # transport target samples onto source samples
            transp_Xt_lpl1 = bot_l1l2.transform(Xs=xt)
            
                    
            # Compute accuracy 
            clf = LinearDiscriminantAnalysis()
            clf.fit(xs,ys)
            yt_predict=clf.predict(transp_Xt_lpl1)
            acc_=accuracy_score(yt, yt_predict)
            
            result[r,rr]=acc_
            
        
    index=unravel_index(result.argmax(), result.shape)
    regu=[rango_e[index[0]], rango_cl[index[1]]]

    return regu

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
    regu: float
        best reg. parameter selected based on accuracy
    


    """
    
    result=[]
    
    # nb,nf=xt.shape
    # sequence = [i for i in range(nb)]
    # # idx = sample(sequence,10) 
    # idx = sample(sequence, np.int32(np.round(nb/2)))  
    
    # xt=xt[idx]
    # yt=yt[idx]   
    # np.int32(np.round(nb/2))      
       
    for r in range(len(rango_e)):
        # Sinkhorn Transport with Group lasso regularization
        bot = ot.da.SinkhornTransport(metric=metrica,reg_e=rango_e[r], verbose=None)
        bot.fit(Xs=xt, ys=yt, Xt=xs)
                
        # transport target samples onto spurce samples
        transp_Xt_lpl1 = bot.transform(Xs=xt)
                
                           
        # Compute accuracy 
        clf = LinearDiscriminantAnalysis()
        clf.fit(xs,ys)
        yt_predict=clf.predict(transp_Xt_lpl1)
        acc_=accuracy_score(yt, yt_predict)
        
        result.append(acc_)
            
        
        
    result=np.asarray(result)
    index=np.argmax(result)
    regu=rango_e[index]
    return regu
###############
def SelectSubsetTraining_BOTDAl1l2(xs, ys, xv, yv, rango_e, rango_cl, metrica="euclidean", kfold=20, M=40):
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

    regu_=[]


    for k in range(kfold):
        # idx = sample(sequence, M)  
        
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(xs, ys, train_size=M, stratify=ys)


        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)

        # xs_daotcv=xs[idx]
        # ys_daotcv=ys[idx]   
        ldacv = LinearDiscriminantAnalysis()

        ldacv.fit(xs,ys) 
        regu=CVgrouplasso_backward(rango_e=rango_e, rango_cl=rango_cl, xs=xs_daotcv, ys=ys_daotcv, xt=xv, yt=yv)
        regu_.append(regu)


        bot_l1l2 = ot.da.SinkhornL1l2Transport(metric=metrica, reg_e=regu[0], reg_cl=regu[1])

        bot_l1l2.fit(Xs=xv, ys=yv, Xt=xs_daotcv)
        #transport testing sampless
        transp_Xt_l1l2=bot_l1l2.transform(Xs=xv)
           
            
        acc_cv.append(ldacv.score(transp_Xt_l1l2, yv))
    
    index=np.argmax(acc_cv)
        
    subset_xs=lista_xs[index]
    subset_ys=lista_ys[index]
    reg_best=regu_[index]
    return subset_xs, subset_ys, reg_best


def SelectSubsetTraining_BOTDAs(xs, ys, xv, yv, rango_e, metrica="euclidean", kfold=20, M=40):
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

    regu_=[]

    #sequence = [i for i in range(len(ys))]

    for k in range(kfold):
        # idx = sample(sequence, M)  
        
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(xs, ys, train_size=M, stratify=ys)


        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)

        # xs_daotcv=xs[idx]
        # ys_daotcv=ys[idx]   
        ldacv = LinearDiscriminantAnalysis()

        ldacv.fit(xs,ys) 
        regu=CVsinkhorn_backward(rango_e=rango_e, xs=xs_daotcv, ys=ys_daotcv, xt=xv, yt=yv)
        regu_.append(regu)


        bot = ot.da.SinkhornTransport(metric=metrica, reg_e=regu)

        bot.fit(Xs=xv, ys=yv, Xt=xs_daotcv)
        #transport val sampless
        transp_Xt_l1l2=bot.transform(Xs=xv)
      
        acc_cv.append(ldacv.score(transp_Xt_l1l2, yv))
    
    index=np.argmax(acc_cv)
    
    subset_xs=lista_xs[index]
    subset_ys=lista_ys[index]
    reg_best=regu_[index]
    return subset_xs, subset_ys, reg_best

def SelectSubsetTraining_OTDAl1l2(xs, ys, xv, yv, rango_e, rango_cl, metrica="euclidean", kfold=20, M=40):
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

    regu_=[]

    #sequence = [i for i in range(len(ys))]

    for k in range(0, kfold):
        # idx = sample(sequence, M)  
        
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(xs, ys, train_size=M, stratify=ys)


        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)
        # xs_daotcv=xs[idx]
        # ys_daotcv=ys[idx]   
        
        regu=CVgrouplasso(rango_e=rango_e, rango_cl=rango_cl, xs=xs_daotcv, ys=ys_daotcv, xt=xv, yt=yv)
        regu_.append(regu)

        ot_l1l2 = ot.da.SinkhornL1l2Transport(metric=metrica,reg_e=regu[0], reg_cl=regu[1])

        ot_l1l2.fit(Xs=xs_daotcv, ys=ys_daotcv, Xt=xv)
        
        #transport source samples o
        transp_Xs_l1l2=ot_l1l2.transform(Xs=xs)
        
        # lda
        lda = LinearDiscriminantAnalysis()
        lda.fit(transp_Xs_l1l2,ys)
        
        
        acc_cv.append(lda.score(xv, yv))
        

    index=np.argmax(acc_cv)
   
    subset_xs=lista_xs[index]
    subset_ys=lista_ys[index]
    reg_best=regu_[index]
    return subset_xs, subset_ys, reg_best

def SelectSubsetTraining_OTDAs(xs, ys, xv, yv, rango_e, metrica="euclidean", kfold=20, M=40):
    
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

    regu_=[]

    #sequence = [i for i in range(len(ys))]

    for k in range(0, kfold):
        # idx = sample(sequence, M)  
        
        xs_daotcv, X_test, ys_daotcv, y_test = train_test_split(xs, ys, train_size=M, stratify=ys)


        lista_xs.append(xs_daotcv)
        lista_ys.append(ys_daotcv)

        # xs_daotcv=xs[idx]
        # ys_daotcv=ys[idx]   
        
        regu=CVsinkhorn(rango_e=rango_e, xs=xs_daotcv, ys=ys_daotcv, xt=xv, yt=yv)
        regu_.append(regu)

        ot_sinkhorn= ot.da.SinkhornTransport(metric=metrica,reg_e=regu)
        ot_sinkhorn.fit(Xs=xs_daotcv, ys=ys_daotcv, Xt=xv)
        transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=xs)
        
        # lda
        lda = LinearDiscriminantAnalysis()
        lda.fit(transp_Xs_sinkhorn,ys)
        
        acc_cv.append(lda.score(xv, yv))
       
    index=np.argmax(acc_cv)
  
        
    subset_xs=lista_xs[index]
    subset_ys=lista_ys[index]
    reg_best=regu_[index]
    return subset_xs, subset_ys, reg_best
