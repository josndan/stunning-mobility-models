from hmmlearn import hmm

class HMM:
    def __init__(self,type="MHMM",**kwargs):
        if type == "MHMM":
            self.model = hmm.MultinomialHMM(**kwargs)
        elif type == "GHMM":
            self.model = hmm.GaussianHMM(**kwargs)
    def train(self,X):
        self.model.fit(X[:,None])
    
    def predict(self,X,possible_loc=None):
        if possible_loc is None:
            possible_loc = np.asarray(2956)
        prediction = []
        for seq in X:
            m = float("-inf")
            m_pred = None
            for p in possible_loc:
                t = np.concatenate((seq,np.asarray[p])) 
                s = self.model.score(t)
                if s > m:
                    m = s
                    m_pred = p
            prediction.append(m_pred)
        return np.asarray(prediction) 

class Metrics:
    def accuracy_two_class(self, y_predict, y):
        return y_predict[y_predict == y].shape[0]/y.shape[0]

    def recall_two_class(self, y_predict, y):
        one = np.ones(y.shape)
        total_true_pos = y_predict[one == y].shape[0]
        total_pred_pos = np.sum(y_predict[one == y])

        if total_true_pos==0:
            return float("inf")
        return total_pred_pos / total_true_pos

    def precision_two_class(self, y_predict, y):
        one = np.ones(y_predict.shape)
        total_pred_pos = y[one == y_predict].shape[0]
        total_true_pos = np.sum(y[one == y_predict])
        if total_pred_pos ==0:
            return float("inf")
        return total_true_pos / total_pred_pos

    def make_pos(self, y, val):
        divided = y.copy()
        pos_ind = divided == val
        neg_ind = divided != val
        divided[pos_ind] = 1
        divided[neg_ind] = 0
        return divided

    def multi_class(self, y_predict, y, metric):
        vals = np.unique(y)
        res = []
        if vals.shape[0] == 2:
            return metric(y_predict, y)
        else:
            for val in vals:
                temp = metric(self.make_pos(y_predict, val), self.make_pos(y, val))
                res.append(temp)

        return np.asarray(res).mean()

    def accuracy(self, y_predict, y):
        return self.multi_class(y_predict, y, self.accuracy_two_class)

    def recall(self, y_predict, y):
        return self.multi_class(y_predict, y, self.recall_two_class)

    def precision(self, y_predict, y):
        return self.multi_class(y_predict, y, self.precision_two_class)
    
    def f1_score(self,y_predict,y):
        p =self.precision(y_predict,y)
        a = self.accuracy(y_predict,y)
        if a == float("inf") or p == float("inf"):
            return float("nan")
        return 2 *(p*a)/(p+a)
    
                
                