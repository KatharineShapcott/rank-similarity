import numpy as np
from numpy.core.numeric import count_nonzero
from scipy.stats import mode
from scipy.sparse import issparse, csr_matrix
import warnings

from ._base import RankSimilarityMixin
from ._filters import _unit_norm

from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution
from sklearn.exceptions import DataDimensionalityWarning


class RankSimilarityTransform(RankSimilarityMixin, TransformerMixin):
    """ Rank Similarity Transform
    
    Transform the data base on responses of rank similarity filters. 
    Output dimensions are equal to n_filters. Values are between 0 and 1.

    Read more in the :ref:`User Guide <transform>`.

    Parameters
    ----------
    n_filters : {'auto'} or int, default='auto'
        Number of filters to use. 'auto' will determine this based on
        max_filters, n_fast_filters and the size of the input data.

    max_filters : int, default=5000
        Maximum number of filters to allocate.

        Only used when ``n_filters='auto'``.

    n_fast_filters: int, default=1000
        Minimum number of filters to allocate, unless the input data has
        fewer samples than this number.

        Only used when ``n_filters='auto'``.

    initialize : {'random','weighted_avg','plusplus'}, default='random'
        Type of filter initialization.

        - 'random', filters are initialized with a random data point.

        - 'weighted_avg', creates filters from similar data, used when 
            there are more filters than input data.
        
        - 'plusplus', filters are initialized with dissimilar data as k-means++

    spreading : {'max', 'weighted_avg'} or None, default='max'
        Determines how data is spread between filters during training

        - 'max', the data point is allocated to the maximum responding
            filter.

        - 'weighted_avg', the weighted average of a fixed number of data 
            points are allocated to the maximum responding filter, used
            when there are more filters than data.
    
    n_iter : int, default=5
        Number of iterations/sweeps over the training dataset to perform
        during training.

    random_state : int, RandomState instance, default=None
        Determines random number generation for filter initialization.
        Pass an int for reproducible results across multiple function calls.
    
    filter_function : {'auto'} or callable, default='auto'
        Function which determines the weights from subsections of the input
        data. 'auto' performs a mean and rank, optionally drawn from a 
        distribution.

    create_distribution : {'confusion'}, callable or None, default=None
        Creates a distribution to draw ranks from. 

        - 'confusion' is a distribution based on the confusibility of
            features in the input data.
        
        Note: the 'confusion' option is extremely slow.

    Attributes
    ----------
    filters_ : ndarray of shape (n_filters\_, n_features)
        Weights of the calculated filters.

    n_filters_ : int
        Number of filters.

    n_iter_ : int
        The number of iterations run by the spreading function.

    n_outputs_ : int
        Number of outputs.

    filterFactory_ : class
        Class used to create the filters.


    Examples
    --------
    >>> from multifilter import RankSimilarityTransform
    >>> from sklearn.datasets import load_digits
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = RankSimilarityTransform(n_filters=10, n_iter=20, 
                                            random_state=0)
    >>> X_transformed = embedding.fit_transform(X)
    >>> X_transformed.shape
    (1797, 10)
    >>> X_transformed[:1, :]
    array([[0.22846891, 0.03269542, 0.        , 0.17862841, 0.23724085,
        0.09489637, 0.        , 1.        , 0.47966508, 0.22846891]])
    """

    @_deprecate_positional_args 
    def __init__(self, *, 
                n_filters = 'auto', max_filters = 5000, n_fast_filters = 1000, 
                initialize = 'random', spreading = 'max', 
                n_iter = 5, random_state = None, 
                filter_function = 'auto', create_distribution = None,
                **kwargs):
        self.n_filters = n_filters
        self.max_filters = max_filters
        self.n_fast_filters = n_fast_filters
        self.initialize = initialize
        self.spreading = spreading
        self.n_iter = n_iter
        self.random_state = random_state
        self.filter_function = filter_function
        self.create_distribution = create_distribution

        super().__init__(**kwargs)

    def fit(self, X, y=None):
        """Fit the rank similarity transform from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        
        X = self._validate_input(X)

        self._random_state = check_random_state(self.random_state)

        n_samples = X.shape[0]
        n_features = X.shape[1]

        self._design_filters(X,y)

        self._check_initialize(n_samples, issparse(X))
        self._check_spreading(n_samples)
        self._check_n_filters(n_samples)

        self._preallocate_filters(n_features)

        self._initialize_filters(X, self.filters_)
        self._spread_filters(X, self.filters_)

        self.filters_[:,:] = _unit_norm(self.filters_, axis=0)#*n_features #does this help?

        return self

    def transform(self, X, n_best=25):
        """ Transforms X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.
            
        Returns
        -------
        X_new : array-like of shape (n_samples, n_filters)
            Transformed version of X where n_filters is the number of filters 
            specified or calculated during fitting.
        """

        check_is_fitted(self)

        X = self._validate_input(X)

        response = self._response_function(X, n_best=n_best)

        response.clip(0, out=response)

        return response


class RankSimilarityClassifier(RankSimilarityMixin, ClassifierMixin):
    """ Rank Similarity Classifier

    Read more in the :ref:`User Guide <classification>`.

    Parameters
    ----------
    n_filters : {'auto'} or int, default='auto'
        Number of filters to use. 'auto' will determine this based on
        max_filters, n_fast_filters and the size of the input data.

    max_filters : int, default=5000
        Maximum number of filters to allocate.

        Only used when ``n_filters='auto'``.

    n_fast_filters: int, default=1000
        Minimum number of filters to allocate, unless the input data has
        fewer samples than this number.

        Only used when ``n_filters='auto'``.

    initialize : {'random','weighted_avg','plusplus'}, default='random'
        Type of filter initialization.

        - 'random', filters are initialized with a random data point.

        - 'weighted_avg', creates filters from similar data, used when 
            there are more filters than input data.
        
        - 'plusplus', filters are initialized with dissimilar data as k-means++

    spreading : {'max', 'weighted_avg'} or None, default='max'
        Determines how data is spread between filters during training

        - 'max', the data point is allocated to the maximum responding
            filter.

        - 'weighted_avg', the weighted average of a fixed number of data 
            points are allocated to the maximum responding filter, used
            when there are more filters than data.

    class_weight : {'balanced','uniform'} or dict, {class_label: weight} \
            default='balanced'
        Used to calculate the number of filters assigned to each class.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    n_iter : int, default=5
        Number of iterations/sweeps over the training dataset to perform
        during training.

    random_state : int, RandomState instance, default=None
        Determines random number generation for filter initialization.
        Pass an int for reproducible results across multiple function calls.

    filter_function : {'auto'} or callable, default='auto'
        Function which determines the weights from subsections of the input
        data. 'auto' performs a mean and rank, optionally drawn from a 
        distribution.

    create_distribution : {'confusion'}, callable or None, default=None
        Creates a distribution to draw ranks from. 

        - 'confusion' is a distribution based on the confusibility of
            features in the input data.
        
        Note: the 'confusion' option is extremely slow.

    Attributes
    ----------
    classes_ : ndarray or list of ndarray of shape (n_classes,)
        Class labels for each output.
        
    filters_ : ndarray of shape (n_filters\_, n_features)
        Weights of the calculated filters.

    filter_labels_ : list of ndarray of shape (n_classes,)
        Label of the datapoints used to make the filter.

    n_class_filters_ : ndarray of shape (n_classes,)
        Number of filters assigned to each class

    n_filters_ : int
        Number of filters.

    n_iter_ : int
        The number of iterations run by the spreading function.

    n_outputs_ : int
        Number of outputs.

    filterFactory_ : class
        Class used to create the filters.


    Examples
    --------
    >>> from multifilter import RankSimilarityClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split 
    >>> X, y = make_classification(n_samples=1000, random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    >>> clf = RankSimilarityClassifier().fit(X_train,y_train)
    >>> clf.predict_proba(X_test[:1, :])
    array([[0.43370805, 0.56629195]])
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    >>> clf.score(X_test, y_test)
    0.888

    """
    
    @_deprecate_positional_args
    def __init__(self, *, 
                n_filters = 'auto', max_filters = 10000, n_fast_filters = 1000,
                initialize = 'random', spreading = 'max', 
                class_weight = 'balanced', n_iter = 5, random_state = None,
                filter_function = 'auto', 
                create_distribution = None, per_label = False,
                **kwargs):
        self.n_filters = n_filters
        self.max_filters = max_filters
        self.n_fast_filters = n_fast_filters
        self.initialize = initialize
        self.spreading = spreading
        self.class_weight = class_weight
        self.n_iter = n_iter
        self.random_state = random_state
        self.filter_function = filter_function
        self.create_distribution = create_distribution
        self.per_label = per_label

        super().__init__(**kwargs)

    def _more_tags(self):
        return {
            'poor_score': True,
            '_xfail_checks': {
                'check_classifiers_classes': 
                'Not enough features to predict 3 classes correctly'
                }
            }

    def fit(self, X, y):
        """Fit rank similarity classifier from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : ndarray, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        
        X, y = self._validate_input(X, y)

        self._random_state = check_random_state(self.random_state)
        
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_samples = y.size
        n_classes = self.classes_.size

        if n_features < n_classes:
            if np.math.factorial(n_features) < n_classes:
                warnings.warn("RankSimilarityClassifier needs at least %i features to separate %i classes."
                             %(np.math.factorial(n_classes), n_classes),
                              DataDimensionalityWarning)

        self._design_filters(X,y)
        self._check_initialize(n_samples, issparse(X))
        self._check_spreading(n_samples)
        self._check_n_filters(n_samples)

        self._n_filters_per_class(y)
        self._assign_filter_labels()

        self._preallocate_filters(n_features)

        st_filt = 0
        for ii, iclass in enumerate(self.classes_):
            if self.filterFactory_.per_label == True:
                self.filterFactory_.index = iclass
            end_filt = st_filt + self.n_class_filters_[ii]

            self._initialize_filters(X[y == iclass,:].reshape(-1,n_features), self.filters_[:,st_filt:end_filt])
            self._spread_filters(X[y == iclass,:].reshape(-1,n_features), self.filters_[:,st_filt:end_filt])

            st_filt += self.n_class_filters_[ii]

        self.filters_[:,:] = _unit_norm(self.filters_, axis=0) # consider multiplying this by some number for large n_features
        
        return self

    def predict(self, X):
        """Predict using the rank similarity classifier

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """

        check_is_fitted(self)
        
        X = self._validate_input(X)
        
        response = (X @ self.filters_)
        max_response = response.max(axis=1)[:,np.newaxis]
        bool_max = np.equal(max_response, response)
        n_max_vals = np.count_nonzero(bool_max,axis=1)
        
        if np.all(n_max_vals==1):
            # simple usual case
            max_filters = np.nonzero(bool_max)[1]
            y_pred = self.filter_labels_[max_filters]
            
        else:
            # deal with single max first
            max_filters = np.zeros(X.shape[0], dtype=int)
            max_filters[n_max_vals==1] = np.nonzero(bool_max[n_max_vals==1, :])[1]
            y_pred = self.filter_labels_[max_filters]
            
            # fix y_pred for multiple max
            ind_multi_max = np.nonzero(n_max_vals > 1)[0]

            for isamp in ind_multi_max:
                pred = self.filter_labels_[bool_max[isamp,:]]
                pred_mode, _ = mode(pred)
                y_pred[isamp] = pred_mode[0]


        return self.classes_[y_pred]
    
    def predict_proba(self, X, n_best=25):
        """Probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """

        check_is_fitted(self)
        X = self._validate_input(X)

        n_samples = X.shape[0]
        n_classes = self.classes_.size
        
        response_match = self._response_function(X, n_best)

        class_response = np.zeros((n_samples, n_classes))
        for ii,iclass in enumerate(self.classes_):
            inc = self.filter_labels_==iclass
            class_response[:,ii] = response_match[:,inc].max(axis=1)
            
        class_response.clip(0, out=class_response)

        # check for multiple max classes and adjust
        max_classes = np.equal(class_response.max(axis=1)[:,np.newaxis],class_response)
        mult_max = np.count_nonzero(max_classes, axis=1)
        for isamp in np.nonzero(mult_max > 1)[0]:
            samp_class_resp = class_response[isamp,:]

            pred = self.filter_labels_[response_match[isamp,:] == samp_class_resp.max()]
            counts = np.bincount(pred, minlength=n_classes)+1
            samp_class_resp[:] = samp_class_resp*counts
            
        return _unit_norm(class_response,axis=1)


class RSPClassifier(RankSimilarityMixin, ClassifierMixin):
    """ Rank Similarity Probabilistic (RSP) Classifier 

    Read more in the :ref:`User Guide <classification>`.

    Parameters
    ----------
    n_filters : {'auto'} or int, default='auto'
        Number of filters to use. 'auto' will determine this based on
        max_filters, n_fast_filters and the size of the input data.

    max_filters : int, default=5000
        Maximum number of filters to allocate.

        Only used when ``n_filters='auto'``.

    n_fast_filters: int, default=1000
        Minimum number of filters to allocate, unless the input data has
        fewer samples than this number.

        Only used when ``n_filters='auto'``.

    initialize : {'random','weighted_avg','plusplus'}, default='random'
        Type of filter initialization.

        - 'random', filters are initialized with a random data point.

        - 'weighted_avg', creates filters from similar data, used when 
            there are more filters than input data.
        
        - 'plusplus', filters are initialized with dissimilar data as k-means++

    spreading : {'max', 'weighted_avg'} or None, default='max'
        Determines how data is spread between filters during training

        - 'max', the data point is allocated to the maximum responding
            filter.

        - 'weighted_avg', the weighted average of a fixed number of data 
            points are allocated to the maximum responding filter, used
            when there are more filters than data.
        
    n_iter : int, default=5
        Number of iterations/sweeps over the training dataset to perform
        during training.

    random_state : int, RandomState instance, default=None
        Determines random number generation for filter initialization.
        Pass an int for reproducible results across multiple function calls.

    filter_function : {'auto'} or callable, default='auto'
        Function which determines the weights from subsections of the input
        data. 'auto' performs a mean and rank, optionally drawn from a 
        distribution.

    create_distribution : {'confusion'}, callable or None, default=None
        Creates a distribution to draw ranks from. 

        - 'confusion' is a distribution based on the confusibility of
            features in the input data.
        
        Note: the 'confusion' option is extremely slow.

    Attributes
    ----------
    classes_ : ndarray or list of ndarray of shape (n_classes,)
        Class labels for each output.
        
    filters_ : ndarray of shape (n_filters\_, n_features)
        Weights of the calculated filters.

    filter_labels_ : list of ndarray of shape (n_classes,)
        Label of the datapoints used to make the filter.

    n_filters_ : int
        Number of filters.

    n_iter_ : int
        The number of iterations run by the spreading function.

    n_outputs_ : int
        Number of outputs.

    filterFactory_ : class
        Class used to create the filters.


    Examples
    --------
    >>> from multifilter import RSPClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=1000, random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    >>> clf = RSPClassifier().fit(X_train,y_train)
    >>> clf.predict_proba(X_test[:1,:])
    array([[0.43370805, 0.56629195]])
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    >>> clf.score(X_test, y_test)
    0.888

    """
    
    @_deprecate_positional_args

    def __init__(self, *, 
                n_filters = 'auto', max_filters = 5000, n_fast_filters = 1000, 
                initialize = 'random', spreading = 'max', 
                n_iter = 5, random_state = None,  
                filter_function = 'auto', create_distribution = None,
                **kwargs):
        self.n_filters = n_filters
        self.max_filters = max_filters
        self.n_fast_filters = n_fast_filters
        self.initialize = initialize
        self.spreading = spreading
        self.n_iter = n_iter
        self.random_state = random_state
        self.filter_function = filter_function
        self.create_distribution = create_distribution

        super().__init__(**kwargs)

    def _more_tags(self):
        return {
            'multioutput': True,
            'poor_score': True,
            '_xfail_checks': {
                'check_classifiers_classes': 
                'Not enough features to predict 3 classes correctly'
                }
            }

    def fit(self, X, y):
        """Fit RSP classifier from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : {array-like, sparse matrix}, shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        
        X, y = self._validate_input(X, y, multi_output=True)
        
        self._random_state = check_random_state(self.random_state)
        
        # remove bad samples from all calculations
        zero_X = np.full((X.shape[0]),False)
        zero_X[:] = (np.sum(X,axis=1) == 0).T
        if np.any(zero_X):
            X = X[np.logical_not(zero_X), :]
            y = y[np.logical_not(zero_X)]
        
        # setup for multilabel
        self.sparse_output_ = False
        
        if issparse(y):
            y = y.toarray()

        if not self.sparse_output_:
            y = np.asarray(y)
            y = np.atleast_1d(y)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
            
        #y = check_array(y)

        self.n_outputs_ = y.shape[1]
        
        self.classes_, self.n_classes_, self.class_prior_ = class_distribution(y) # could include sample_weight here
        
        if len(self.classes_) == 1 and self.n_classes_[0] == 1:
            raise ValueError("RSPClassifier cannot be fit when only one class is present.")

        n_features = X.shape[1]
        n_samples = X.shape[0]

        max_labels = max(self.n_classes_)
        if n_features < max_labels:
            if np.math.factorial(n_features) < max_labels:
                warnings.warn("RSPClassifier needs at least %i features to separate %i labels."
                              %(np.math.factorial(max_labels), max_labels),
                              DataDimensionalityWarning)
        
        # design filters
        self._design_filters(X,y)

        self._check_initialize(n_samples, issparse(X))
        self._check_spreading(n_samples)
        self._check_n_filters(n_samples)

        self._preallocate_filters(n_features)
           
        self._initialize_filters(X, self.filters_)
        self._spread_filters(X, self.filters_)
        self.filters_[:,:] = _unit_norm(self.filters_, axis=0)
        
        self._assign_filter_labels(X, y)

        if self.n_outputs_ == 1:
            self.classes_ = self.classes_[0]
        
        return self

    def _assign_filter_labels(self, X, y):
        classes_ = self.classes_

        response = X @ self.filters_

        winners = response.argmax(axis=1)
        uniq_winners = np.unique(winners) 

        self.filter_labels_ = [np.zeros((self.n_filters_, x.size)) for x in classes_]

        for ifilt in uniq_winners:
            filt_class = y[winners == ifilt, :]

            for ii, classes_i in enumerate(classes_):
                label = self.filter_labels_[ii][ifilt,:]

                inc_labels, counts = np.unique(filt_class[:,ii], return_counts=True)
            
                inc = np.isin(classes_i, inc_labels, assume_unique=True)
                label[inc] = counts
        
        losers = np.setdiff1d(np.arange(self.n_filters_), uniq_winners)
        if losers.size > 0:
            # could take highest responding here
            for ii in range(len(self.filter_labels_)):
                self.filter_labels_[ii][losers,:] = 1
            
        # convert counts to probabilities        
        for ii in range(len(self.filter_labels_)):
            self.filter_labels_[ii] = _unit_norm(self.filter_labels_[ii],axis=1)
            
    def predict(self, X):
        """Predict using the RSP classifier

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        
        prob = self.predict_proba(X)
        
        if self.n_outputs_ == 1:
            predict = self.classes_[prob.argmax(axis=1)]
        else:
            #predict = np.zeros((X.shape[0],len(prob)))
            predict = []
            for iclass in range(len(prob)):
                pred_ind = prob[iclass].argmax(axis=1)
                predict.append(self.classes_[iclass][pred_ind])
            predict = np.asarray(predict).T
        
        return predict
    
    def predict_proba(self, X, n_best=25):
        """Probability estimates for RSP classifier

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """

        check_is_fitted(self)

        X = self._validate_input(X)
        
        n_features = X.shape[1]
        n_samples = X.shape[0]

        if self.n_outputs_ == 1:
            classes_ = [self.classes_]
        else:
            classes_ = self.classes_
        
        response_match = self._response_function(X, n_best)
        response_match.clip(0, out=response_match)
        response_match = csr_matrix(response_match)
        
        probabilities = []
        for ii, classes_i in enumerate(classes_):
            class_resp = np.zeros((n_samples, classes_i.size))
            for ilabel in range(classes_i.size):
                tmp = response_match.multiply(self.filter_labels_[ii][:,ilabel][np.newaxis,:])
                class_resp[:,ilabel] = tmp.max(axis=1).toarray().ravel()
            # turn to probabilites
            zero_row = np.sum(class_resp,axis=1) == 0
            class_resp[zero_row,:] = 1
            class_resp = _unit_norm(class_resp, axis=1)
            
            probabilities.append(class_resp)

        if self.n_outputs_ == 1:
            probabilities = probabilities[0]
        
        return probabilities
