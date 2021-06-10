import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_allclose
from sklearn.datasets import load_iris, load_digits
from sklearn.utils.validation import check_is_fitted

from ranksim import RankSimilarityTransform
from ranksim import RankSimilarityClassifier
from ranksim import RSPClassifier

rng = np.random.RandomState(0)
INITILIZATION = ('random', 'weighted_avg', 'plusplus')
SPREADING = ('max', 'weighted_avg', None)


@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    return X[:100,:], y[:100]

ests = [RankSimilarityTransform,
        RankSimilarityClassifier,
        RSPClassifier]
@pytest.mark.parametrize("RankSim", ests)
def test_n_filters_datatype(RankSim):
    # Test to check whether n_filters is integer
    X = [[1, 1], [1, 1], [1, 1]]
    y = [0,1,1]
    expected_msg = "n_filters does not take .*float.* " \
                   "value, enter integer value"
    msg = "Expected n_filters > 0. Got -3"

    clf = RankSim(n_filters=3.)
    with pytest.raises(TypeError, match=expected_msg):
        clf.fit(X,y)

    clf.n_filters =-3
    with pytest.raises(ValueError, match=msg):
        clf.fit(X=X,y=y)

@pytest.mark.parametrize("RankSim", ests)
def test_filters_normalization(RankSim, data):
    # Test to check whether filters sum to 1
    clf = RankSim()
    clf.fit(*data)

    sum_filters = np.sum(clf.filters_, axis=0)
    assert_allclose(sum_filters, np.ones(sum_filters.shape, dtype=np.float32),rtol=1e-05)

@pytest.mark.parametrize("RankSim", ests)
def test_excess_filters(RankSim):

    msg = "When n_filters >= n_samples initialize cannot be 'plusplus'. Setting initialize='weighted_avg' is recommended."

    X_large = rng.random((100,10))
    y_large = np.resize(np.arange(10), (100))

    for init in INITILIZATION:
        for spread in SPREADING:
            for sz in [10,50,100]:
                clf = RankSim(n_filters=100, initialize=init, spreading=spread)
                if init == 'plusplus':
                    with pytest.raises(ValueError, match=msg):
                        clf.fit(X_large[:sz,:], y_large[:sz])
                else:
                    clf.fit(X_large[:sz,:], y_large[:sz])

def test_classifier_fitting(data):
    clf = RankSimilarityClassifier()
    assert clf.n_filters == 'auto'

    X, y = data
    clf.fit(X, y)
    assert check_is_fitted(clf) == None

    y_pred = clf.predict(X)
    assert_array_equal(y_pred, y)


def test_transformer_error(data):
    X, y = data
    trans = RankSimilarityTransform()
    trans.fit(X)
    with pytest.raises(ValueError, match="Shape of input is different"):
        X_diff_size = np.ones((10, X.shape[1] + 1))
        trans.transform(X_diff_size)
