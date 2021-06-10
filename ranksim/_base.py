import numpy as np
import numbers
import warnings

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import ConvergenceWarning

from ._filters import FilterFactory, _unit_norm

class RankSimilarityMixin(BaseEstimator):
    """Mixin class for all rank similarity based estimators."""

    def _validate_input(self, X, y=None, multi_output=False):
        
        if y is not None:
            check_consistent_length(X, y)
            check_classification_targets(y)
            X, y = self._validate_data(X, y, accept_sparse=['csr'], 
                                       ensure_min_features=2, 
                                       multi_output=multi_output)

            out = X, y
        else:
            X = self._validate_data(X, accept_sparse=['csr'], ensure_min_features=2)
            
            if hasattr(self, 'filters_'):
                if X.shape[1] != self.filters_.shape[0]:
                    raise ValueError('Shape of input is different from what was seen '
                                     'in `fit`')

            out = X

        return out

    def _design_filters(self, X, y):

        if not hasattr(self, "per_label"):
            per_label = False
        else:
            per_label = self.per_label

        self.filterFactory_ = FilterFactory(create_distribution = self.create_distribution, 
                                            filter_function = self.filter_function, 
                                            per_label = per_label)
        if not self.filterFactory_.create_distribution == None: 
            self.filterFactory_.fit_distribution(X, y)

    def _check_initialize(self, n_samples, Xissparse=False):

        if self.initialize not in ['random', 'weighted_avg', 'plusplus']:
            raise ValueError("Unrecognized initilization: '%s'" % self.initialize)

        if Xissparse & (self.initialize == 'weighted_avg'):
            raise ValueError("Sparse data must not have initialize='weighted_avg'")
        
        if self.n_filters != 'auto' :
            if self.n_filters >= n_samples:
                if self.initialize == 'plusplus':
                    raise ValueError("When n_filters >= n_samples initialize cannot be 'plusplus'. Setting initialize='weighted_avg' is recommended.")
                if (self.initialize != 'weighted_avg'):
                    print("Setting initialize with 'weighted_avg' is recommended when n_filters > n_samples")


    def _check_spreading(self, n_samples):

        if self.spreading not in ['max','weighted_avg', None]:
            raise ValueError("Unrecognized spreading: '%s'" % self.spreading)

        if self.n_filters != 'auto' :
            if (self.n_filters > n_samples) & (self.spreading != 'weighted_avg'):
                print("Setting spreading with 'weighted_avg' is recommended when n_filters > n_samples")


    def _check_n_filters(self, n_samples):

        if self.n_filters != 'auto':
            if self.n_filters <= 0:
                raise ValueError(
                    "Expected n_filters > 0. Got %d" %
                    self.n_filters)
            elif not isinstance(self.n_filters, numbers.Integral):
                raise TypeError(
                    "n_filters does not take %s value, "
                    "enter integer value" %
                    type(self.n_filters))
            self.n_filters_ = self.n_filters
            return

        # weighted_avg needs more n_filters than other methods
        if self.initialize == 'weighted_avg':
            if n_samples*5 < self.n_fast_filters:
                self.n_filters_ = n_samples*5
            elif n_samples*2 < self.n_fast_filters:
                self.n_filters_ = self.n_fast_filters
            elif n_samples*2 > self.max_filters:
                print('Limiting n_filters to %i for speed/memory considerations. If you need more please specify n_filters manually'%(self.max_filters))
                self.n_filters_ = self.max_filters
            else:
                self.n_filters_ = np.round(n_samples*2).astype(int)
        else:
            # not weighted_avg
            if n_samples <= self.n_fast_filters:
                self.n_filters_ = n_samples
            elif n_samples > self.max_filters*10:
                print('Limiting n_filters to %i for speed/memory considerations. If you need more please specify n_filters manually'%(self.max_filters))
                self.n_filters_ = self.max_filters
            else:
                min_filters = np.round(n_samples/10).astype(int)
                if min_filters <= self.n_fast_filters:
                    self.n_filters_ = self.n_fast_filters
                else:
                    self.n_filters_ = min_filters

    def _n_filters_per_class(self, y):
        n_samples = y.size
        n_classes = np.unique(y).size
        
        if self.n_filters_ < n_classes:
            raise ValueError("n_filters must be greater than or equal to the number of classes; got (n_filters=%i)"
                             % self.n_filters_)
                             
        avg_filts = self.n_filters_/n_classes
        
        if self.class_weight == 'balanced':
            tmp, ind_y = np.unique(y,return_inverse=True)
            bias = (n_classes * np.bincount(ind_y))/n_samples

            self.n_class_filters_ = (bias*avg_filts).astype(int)

        elif self.class_weight == 'uniform':
            self.n_class_filters_ = np.full(n_classes, np.round(avg_filts), dtype=int)

        elif isinstance(self.class_weight, (list, tuple, np.ndarray)):
            self.n_class_filters_ = np.round(_unit_norm(self.class_weight)*self.n_filters_).astype(int)

        elif isinstance(self.class_weight, dict):
            classes = np.unique(y)

            bias = np.ones(classes.shape[0])
            for i_class, weight in self.class_weight.items():
                b_class = classes == i_class
                if not np.any(b_class):
                    raise ValueError("Class: '%s' does not exist in input data" % i_class)
                else:
                    bias[b_class] = weight

            self.n_class_filters_ = np.round(_unit_norm(bias)*self.n_filters_).astype(int)
                
        else:
            raise ValueError("Unrecognized class_weight: '%s'" % self.class_weight)

        # don't want any class unrepresented
        self.n_class_filters_[self.n_class_filters_ == 0] = 1

        # update n_filters to match
        self.n_filters_ = np.sum(self.n_class_filters_).astype(int)
        
    def _assign_filter_labels(self):
        # default behaviour
        self.filter_labels_ = np.repeat(np.arange(len(self.classes_)), self.n_class_filters_)

    def _preallocate_filters(self, n_features):
        dtype = np.float32 # this should have enough precision for the filters
        self.filters_ = np.zeros((n_features, self.n_filters_), dtype=dtype)

    def _initialize_filters(self, X, filters):

        if self.initialize == 'random':
            self._init_random(X, filters)

        elif self.initialize == 'weighted_avg':
            self._init_weighted(X, filters)

        elif self.initialize == 'plusplus':
            n_samples = X.shape[0]
            max_samples = self.n_filters_*100
            if n_samples < max_samples:
                ind_samps = slice(None)
            else:
                # if X is MUCH bigger than n_filters makes sense to first subselect random datapoints
                ind_samps = self._random_state.choice(n_samples, max_samples, replace=False)
            self._init_plusplus(X[ind_samps,:], filters)

    def _init_random(self, X, filters):
        n_filt = filters.shape[1]
        n_samples = X.shape[0]

        if n_filt > n_samples:
            start_samps = self._random_state.choice(n_samples, n_filt)
        elif n_filt == n_samples:
            start_samps = np.arange(n_samples)
            self._random_state.shuffle(start_samps)
        else:
            start_samps = self._random_state.choice(n_samples, n_filt, replace=False)
        
        for ifilt, isamp in enumerate(start_samps):
            filters[:,ifilt] = self.filterFactory_.transform(X[isamp,:])

    def _init_plusplus(self, X, filters, pre_norm=False):
        # k-means++ style initialization with probabilities

        n_features = X.shape[1]
        n_samples = X.shape[0]
        nfilt = filters.shape[1]

        if nfilt >= n_samples:
            print('++ initialization makes no sense with more filters than samples')
            print('Switching to random initialization')
            self.initialize = 'random'
            self._init_random(X, filters)
            return
        
        if not pre_norm:
            norm_X = _unit_norm(X, axis=1)
        else:
            norm_X = X

        start_samp = self._random_state.choice(n_samples, 1)
        filters[:,0] = self.filterFactory_.transform(X[start_samp,:])
        remaining = np.full((n_samples),True)
        remaining[start_samp] = False
        
        pop_response = np.zeros(n_samples)
        pop_response += (norm_X @ filters[:,0])/nfilt
        
        samp_prob = np.zeros(n_samples)
        for ifilt in range(1,nfilt):
            samp_prob[remaining] = self._filter_probabilities(pop_response[remaining], flipped=True)
            next_samp = self._random_state.choice(n_samples,p=samp_prob)

            filters[:,ifilt] = self.filterFactory_.transform(X[next_samp,:])
            pop_response += (norm_X @ filters[:,ifilt])/nfilt
            
            remaining[next_samp] = False
            samp_prob[next_samp] = 0

    def _init_weighted(self, X, filters):
        # This weights while making sure that certain samples are not over represented in the filter space

        n_samples = X.shape[0]
        n_features = X.shape[1]
        nfilt = filters.shape[1]

        if n_samples >= nfilt:
            sz_batch = nfilt
        else:
            sz_batch = int(n_samples*0.8)
            if sz_batch == 0:
                sz_batch = 1

        start_samps = self._random_state.choice(n_samples, sz_batch, replace=False)
        for ifilt, isamp in enumerate(start_samps):
            filters[:,ifilt] = self.filterFactory_.transform(X[isamp,:])
        
        n_batch = np.floor(nfilt/sz_batch).astype(int)

        for ibatch in range(n_batch):
            if (ibatch == 0) & (n_batch > 1):
                continue # don't overwrite single samples if we have more batches

            knn = min(ibatch+2, sz_batch) # necessary only for very small n_samples

            st = sz_batch*ibatch

            if ibatch == 0:
                prev_filters = filters[:,:sz_batch]
            else:
                prev_filters = filters[:,st-sz_batch:st]
            response = X @ prev_filters
            prob, closest_filters = self._knn_probabilities(response, knn, axis=1)

            for ifilt in range(sz_batch):
                inc_filt = closest_filters == ifilt
                if np.any(inc_filt):
                    inc_samples = np.any(inc_filt, axis=1)
                    X_avg = np.average(X[inc_samples,:], axis=0, weights=prob[inc_filt])
                    filters[:,st+ifilt] = self.filterFactory_.transform(X_avg)
                else:
                    filters[:,st+ifilt] = np.zeros(n_features) # to speed up dot product

            filters[:,st:st+sz_batch] = _unit_norm(filters[:,st:st+sz_batch],axis=0)

        st = sz_batch*n_batch

        remaining_filt = nfilt - st
        if remaining_filt == 0:
            return

        prev_filters = filters[:,st-remaining_filt:st] # FIXME: should select non-zero
        response = X @ prev_filters

        knn = min(np.ceil(remaining_filt/10)+1, remaining_filt).astype(int)
        prob, closest_filters = self._knn_probabilities(response, knn, axis=1)

        for ifilt in range(remaining_filt):
            inc_filt = closest_filters == ifilt
            if np.any(inc_filt):
                inc_samples = np.any(inc_filt, axis=1)
                X_avg = np.average(X[inc_samples,:], axis=0, weights=prob[inc_filt])
                filters[:,st+ifilt] = self.filterFactory_.transform(X_avg)
            else:
                filters[:,st+ifilt] = np.zeros(n_features) # to speed up dot product
                
    def _filter_probabilities(self, response, flipped=False, axis=None):
        norm_response = response - np.mean(response, axis=axis)

        if flipped:
            norm_response = norm_response*-1

        norm_response -= norm_response.min() # start from 0

        if np.all(norm_response == 0):
            norm_response = np.ones_like(norm_response)

        if axis is not None:
            prob = _unit_norm(norm_response, axis=axis)
        else:
            sum_response = np.sum(norm_response)
            prob = norm_response/sum_response
        return prob

    def _knn_probabilities(self, response, knn, axis=1):
        if knn > response.shape[axis]:
            raise ValueError("knn %i must be less than response shape %i" %(knn, response.shape[axis]))
        closest = np.argpartition(response, -knn, axis=axis)

        if axis == 1:
            closest = closest[:,-knn:]
        elif axis == 0:
            closest = closest[-knn:,:]
        else:
            raise ValueError("axis must be equal to 0 or 1, not %i" % axis)

        close_response = np.take_along_axis(response, closest, axis=axis)
        prob = _unit_norm(close_response, axis=axis)
        
        return prob, closest

    def _spread_filters(self, X, filters):
        if self.spreading == 'max':
            self._do_max_spreading(X, filters)
        elif self.spreading == 'weighted_avg':
            self._do_weighted_spreading(X, filters)
        elif self.spreading == None:
            pass # do nothing
        else:
            raise ValueError("Spreading type = %s unrecognized" % self.spreading)
        

    def _do_max_spreading(self, X, filters):
        self._winners = None
        n_samples = X.shape[0]
        knn = 5 #int(n_samples/(nfilt*2))+4
        knn = min(n_samples, knn) # only for very small n_samples

        n_samples = X.shape[0]
        nfilt = filters.shape[1]

        spreading = True
        tolerance = int(n_samples/100)+1
        
        i_iter = 0
        while spreading:
            i_iter += 1
            filters[:,:] = _unit_norm(filters,axis=0)
            response = X @ filters

            winners = response.argmax(axis=1)
            uniq_winners = np.unique(winners) #np.where(count_winners>knn)[0]

            for ifilt in uniq_winners:
                filters[:,ifilt] = self.filterFactory_.transform(X[winners == ifilt,:])

            losers = np.setdiff1d(np.arange(nfilt), uniq_winners)
            if losers.size > 0:
                norm_response = response[:,losers]/np.mean(response,axis=1)[:,np.newaxis]
                closest = np.argpartition(norm_response, -knn, axis=0)[-knn:,:]
                for ii, ifilt in enumerate(losers):
                    # move to mean of knn datapoints
                    filters[:,ifilt] = self.filterFactory_.transform(X[closest[:,ii],:])
            
            if self._winners is not None:
                if np.count_nonzero(self._winners - winners) < tolerance:
                    spreading = False
                elif i_iter == self.n_iter:
                    spreading = False
                    warnings.warn('Filter spreading failed to converge after %i iterations, '
                        'increase the number of iterations'%(self.n_iter), ConvergenceWarning)
            elif i_iter == self.n_iter:
                warnings.warn('Filter spreading failed to converge after %i iterations, '
                    'increase the number of iterations'%(self.n_iter), ConvergenceWarning)
                spreading = False
            
            self._winners = winners

        # Store the final counts and iterations
        self.n_iter_ = i_iter
        self._count_winners = np.bincount(winners)

    def _do_weighted_spreading(self, X, filters, pre_norm=False):
        # This weights while making sure that certain samples are not over represented in the filter space
        n_samples = X.shape[0]
        nfilt = filters.shape[1]

        if self.initialize == 'random':
            knn = int(nfilt/10)+1
        else:
            knn = 5

        knn = min(n_samples, knn) # only for very small n_samples
        
        for itry in range(self.n_iter):
            filters[:,:] = _unit_norm(filters,axis=0)
            response = X @ filters
            prob, closest_filters = self._knn_probabilities(response, knn, axis=1)
            
            for ifilt in range(nfilt):
                inc_filt = closest_filters == ifilt
                if np.any(inc_filt):
                    inc_samples = np.any(inc_filt, axis=1)
                    X_avg = np.average(X[inc_samples,:], axis=0, weights=prob[inc_filt])

                    filters[:,ifilt] = self.filterFactory_.transform(X_avg)
                    
    def _response_function(self, X, n_best=25):
        response = X @ self.filters_

        n_best = min(n_best, self.n_filters_-1)
        
        max_response = response.max(axis=1)[:,np.newaxis]
        per_response = np.partition(response, -n_best, axis=1)[:,-n_best][:,np.newaxis]
        
        # if per_response equals max_response then replace by 2nd largest
        zero_max = np.equal(response.max(axis=1)[:,np.newaxis], per_response).ravel()
        if np.any(zero_max):
            zero_max_and_min = np.equal(response[zero_max,:].min(axis=1)[:,np.newaxis], 
                                        per_response[zero_max,:]).ravel()
            if np.any(zero_max_and_min):
                # This is an unhealthy state, all filters are giving identical responses
                bad_ind = np.nonzero(zero_max)[0][zero_max_and_min]
                per_response[bad_ind,:] = 0 # all responses will be 1
                zero_max[bad_ind] = False
                
                no_reponse = response[bad_ind,:].max(axis=1) == 0
                if np.any(no_reponse):
                    # This is the worst case, no filter responds at all
                    worst_ind = bad_ind[no_reponse]
                    max_response[worst_ind] = 1
                
            total_max = np.sum(np.equal(response[zero_max,:], per_response[zero_max,:]),axis=1)
            closest_per = np.partition(response[zero_max,:],-(total_max+1),axis=1)
            closest_per = np.take_along_axis(closest_per, -(total_max+1)[:,np.newaxis], axis=1)
            per_response[zero_max,:] = closest_per
            
        # normalize between per_response and max_response
        response -= per_response
        response /= (max_response-per_response)
        
        return response