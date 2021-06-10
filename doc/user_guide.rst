.. title:: User guide : contents

.. _user_guide:

==================================================
User guide: Using the Rank Similarity estimators
==================================================

.. currentmodule:: ranksim

.. _transform:

Rank Similarity Transform
-------------------------

The :class:`RankSimilarityTransform` (RST) is a very fast non-linear trasform. It's
created using the responses of rank similarity filters made from the input data.

Use it with the fit and transform methods from scikit-learn:

* at ``fit``, some parameters can be learned from ``X`` and ``y``;
* at ``transform``, ``X`` will be transformed, using the parameters learned
  during ``fit``.

Alternatively you can directly use a combination of ``fit`` and ``transform`` 
called ``fit_transform``:

RST can make non-linear problems solvable by a linear classifier. For example 
using the raw Olivetti faces dataset and a linear support vector machine. 

    >>> from ranksim import RankSimilarityTransform
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.datasets import fetch_olivetti_faces
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.model_selection import train_test_split 
    >>> X, y = fetch_olivetti_faces(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    >>> pipe = make_pipeline(RankSimilarityTransform(n_filters=100,
    ...                                              random_state=10),
    ...                      LinearSVC(random_state=10))
    >>> pipe.fit(X_train, y_train)  # doctest: +ELLIPSIS
    Pipeline(...)
    >>> pipe.score(X_test, y_test)
    0.85

.. _classification:

Classification
--------------

The rank similarity classifiers are very fast non-linear classifiers. They use
the responses of rank similarity filters made from the input data to classify
new samples. Both :class:`RankSimilarityClassifier` and :class:`RSPClassifier` are able to perform binary and
multi-class classification.


Use it with the fit and predict methods from scikit-learn:

* at ``fit``, some parameters can be learned from ``X`` and ``y``;
* at ``predict``, predictions will be computed using ``X`` using the parameters
  learned during ``fit``.
* at ``predict_proba``, will output some probabilities instead.

The predict method can then be used by the score method:

* at ``score``, compute the accuracy score of the predictions.

RankSimilarityClassifier and RSPClassifier are fit using two arrays: an array X
of shape (n_samples, n_features) holding the training samples, and an array y of
class labels (strings or integers), of shape (n_samples):

    >>> from ranksim import RankSimilarityClassifier
    >>> X = [[0, 1], [1, 0]]
    >>> y = [0, 1]
    >>> clf = RankSimilarityClassifier()
    >>> clf.fit(X, y)
    RankSimilarityClassifier()

Rank Similarity Classifier
--------------------------

The :class:`RankSimilarityClassifier` works by fitting each class of data
separately. This makes it suitable for multiclass data. More classes can
actually make it faster because it splits the data into smaller segments. 

.. topic:: Examples:

  * :ref:`sphx_glr_auto_examples_plot_classifier.py`: an example of classification using rank
    similarity classifier.

RSPClassifier
--------------

The rank similarity probabilistic classifier (:class:`RSPClassifier`) fits all
data together and then calculates posthoc the probability that each each filter
belongs to a certain class. This makes it suitable for both multilabel and
multiclass data. 
