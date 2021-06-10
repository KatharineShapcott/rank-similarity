import numpy as np
from scipy.stats import rankdata, norm, gaussian_kde
from scipy.sparse import issparse

def rank(data,axis=None):
    # this is only approximate for speed
    return rankdata(data, axis=axis, method='ordinal')

def _unit_norm(data,axis=None):
    # l1-normalization
    if issparse(data):
        sum_data = np.sum(data,axis=axis)
        return data.multiply(1/sum_data)
    
    if axis is not None:
        sum_data = np.sum(data,axis=axis)
        sum_data = np.expand_dims(sum_data, axis=axis)
    else:
        sum_data = np.sum(data)
    return data/sum_data

def confusion(samps1, samps2, res=200):
    xmin = min(samps1.min(), samps2.min())
    xmax = max(samps1.max(), samps2.max())
    xgrid = np.linspace(xmin,xmax,res)
    try:
        kde_1 = gaussian_kde(samps1)
        kde_2 = gaussian_kde(samps2)
        f = kde_1.evaluate(xgrid)
        g = kde_2.evaluate(xgrid)
        integrand = f*g/(f+g)
        c = np.trapz(integrand,xgrid)
    except np.linalg.LinAlgError as error:
        if 'singular matrix' in str(error):
            c = 0.5000001
        else:
            raise
    if np.isnan(c):
        d = 1
    else:
        d = 1-c
    return d - 0.5

def confusion_filters(data, n_each=1000,offset=0.5):
    feat_means = np.mean(data,axis=0) # Feature means
    n_feat = data.shape[1] # Length of features
    order = np.argsort(feat_means) # Features sorted by mean value
    
    if issparse(data):
        order = order.A1
   
    if n_each == 'all': # Take all available samples
        samp_inds = np.arange(np.size(data,0))
    elif type(n_each) == int: # Randomly select a smaller sample of data
        if n_each >= data.shape[0]:
            samp_inds = np.arange(np.size(data,0))
        else:
            samp_inds = np.random.choice(np.size(data,0), n_each, replace=False)
    else:
        raise ValueError('Unrecognised n_each value %s with type %s', n_each, type(n_each))
        
    if issparse(data):
        inc = data[samp_inds,:].toarray()
    else:
        inc = data[samp_inds,:]
   
    dist_map = np.zeros([n_feat,3])
    
    for ward in range(1,n_feat):
        c_top = confusion(inc[:,order[-ward-1]],inc[:,order[-1]])
        c_neighbour = confusion(inc[:,order[-ward-1]],inc[:,order[-ward]])
        c_bottom = confusion(inc[:,order[-ward-1]],inc[:,order[0]])
        dist_map[ward,:] = [c_top,c_neighbour,c_bottom]
        
    p_map=np.zeros(n_feat)
    running_p=offset+dist_map[-1,1]
    p_map[0]=offset
    p_map[1]=running_p
   
    for ward in range(2,n_feat):
        p_neigh=dist_map[-ward,1]
        n_sum=1
        if dist_map[-ward,0]<1 and dist_map[-ward+1,0]<1:
            p_top=(dist_map[-ward,0]-dist_map[-ward+1,0])
            n_sum+=1
        else:
            p_top=0
        if dist_map[-ward,2]<1 and dist_map[-ward+1,2]<1:
            p_bottom=(dist_map[-ward+1,2]-dist_map[ward,2])
            n_sum+=1
        else:
            p_bottom=0
        t_p=(p_top+p_neigh+p_bottom)/n_sum
        running_p+=max(t_p,0)
        p_map[ward] = running_p
        
    return p_map


class FilterFactory():
    """Factory class to build filters for rank similarity estimators.
    
    Parameters
    ----------
    filter_function : {'auto'} or callable, default='auto'
        Function which determines the weights from subsections of the input
        data. 'auto' performs a mean and rank, optionally drawn from a 
        distribution.

    create_distribution : {'confusion'}, callable or None, default=None
        Creates a distribution to draw ranks from. 

        - 'confusion' is a distribution based on the confusibility of
            features in the input data.
        
        Note: the 'confusion' option is extremely slow.

    per_label : bool, default=False
        Creates a separate distribution per class label.

        Only used when ``create_distribution != None``.
    
    """

    def __init__(self, filter_function='auto', create_distribution=None, per_label=False):

        self.filter_function = filter_function
        self.create_distribution = create_distribution
        self.per_label = per_label

        if self.filter_function == 'auto':
            if self.create_distribution == None:
                self.filter_function = self.rank_mean_filter
            elif self.create_distribution == 'confusion':
                self.filter_function = self.distribution_mean_filter
            else:
                self.filter_function = self.distribution_mean_filter
        else:
            #test function
            nsamp = 5
            nfet = 10
            X = np.random.rand(nsamp, nfet)
            distribution = np.random.rand(1,nfet)
            try:
                tmp = self.filter_function(X, distribution, 0)
            except:
                print("Testing filter_function failed\n",
                    "It should accept 3 inputs- X, distribution, index")
                raise

            # checks that output looks as expected
            if not tmp.shape[0] == 1:
                raise ValueError("Output of filter_function is not a single filter, it has shape %i" % tmp.shape[0])
            if not tmp.shape[1] == nfet:
                raise ValueError("N features output of filter_function %i is different than input %i" % (tmp.shape[1], nfet))

        self.dist_index = 0
        self.distribution = None

    def transform(self, X):
        """Transforms data X into a filter.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        filter : ndarray of shape (n_features, )

        """
        return self.filter_function(X, self.distribution, self.dist_index)

    def fit_distribution(self,X,y=None):
        """Fits a distribution to data X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray, shape (n_samples,) or None, default=None
            Target values when distribution is created per label.
        """
        
        if self.create_distribution == 'confusion':
            self.create_distribution = confusion_filters
            
        if self.per_label == True:
            all_labels = np.unique(y)

            self.distribution = []

            for iclass in all_labels:
                self.distribution.append(self.create_distribution(X[y==iclass,:]))
            self.distribution = np.array(self.distribution).astype(np.float32)
        else:
            self.distribution = self.create_distribution(X)
            if type(self.distribution) is not list: 
                self.distribution = [self.distribution]

    @staticmethod
    def mean_filter(X, *unused):
        if len(X.shape) == 1:
            return _unit_norm(X)
        else:
            return _unit_norm(np.mean(X,axis=0))

    @staticmethod
    def rank_mean_filter(X, *unused):
        if len(X.shape) == 1:
            return rankdata(X)
        else:
            return rankdata(np.mean(X,axis=0))

    @staticmethod
    def rank_random_filter(X, *unused):
        if len(X.shape) == 1:
            return rankdata(X)
        else:
            return rankdata(X[np.choice(X.shape[0]),:])

    @staticmethod
    def rank_median_filter(X, *unused):
        if len(X.shape) == 1:
            return rankdata(X)
        else:
            return rankdata(np.median(X,axis=0))

    @staticmethod
    def distribution_mean_filter(X, dist, ind):
        if len(X.shape) == 2:
            X = np.mean(X,axis=0)
        dist_index = rankdata(X).astype(int) - 1
        return dist[ind][dist_index]
    
    @staticmethod
    def match_gaussian(X):
        
        if issparse(X):
            X_mean = np.mean(X, axis=0).A1
        else:
            X_mean = np.mean(X, axis=0)
        center = np.mean(X_mean)
        width = np.std(X_mean)
        
        gaussian = np.sort(norm.rvs(center, width, size = X.shape[1])) # needs more smoothing
        gaussian = gaussian - gaussian[0] + 0.00001
        
        return gaussian
    
    @staticmethod
    def match_mean(X):
        
        sorted_mean = np.mean(np.sort(X,axis=1),axis=0) # should be random sample for speed if X is large
        adjusted_ranks = sorted_mean-sorted_mean.min()+(np.mean(np.diff(sorted_mean))*200)
        
        return adjusted_ranks