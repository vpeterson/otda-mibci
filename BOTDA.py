import ot
from sklearn.base import BaseEstimator, TransformerMixin


class BOTDA(BaseEstimator, TransformerMixin):
    def __init__(self, reg, clf, metric, make_prediction=True):
        """Initialize instance."""
        self.reg = reg
        self.clf = clf
        self.metric = metric
        self.make_prediction = make_prediction

    def fit(self, Gtr, Gval, Yval):
        # define BOTDA model
        botda = ot.da.SinkhornL1l2Transport(metric=self.metric, reg_e=self.reg[0], reg_cl=self.reg[1])
        # learn params of transport
        botda.fit(Xs=Gval, ys=Yval, Xt=Gtr)
        self.botda = botda
        self.coupling_ = botda.coupling_
        self.cost_ = botda.cost_

    def transform(self, Gte):
        #transport testing samples
        Gte_transported=self.botda.transform(Xs=Gte)
        if self.make_prediction:
            # Compute accuracy without retraining
            yte_predict = self.clf.predict(Gte_transported)
            return Gte_transported, yte_predict
        else:
            return Gte_transported
