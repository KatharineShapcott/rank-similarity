from sklearn.utils.estimator_checks import parametrize_with_checks

from ranksim import RankSimilarityTransform
from ranksim import RankSimilarityClassifier
from ranksim import RSPClassifier

@parametrize_with_checks([RankSimilarityTransform(), 
                          RankSimilarityClassifier(), 
                          RSPClassifier()])
def test_all_estimators(estimator, check):
    check(estimator)                          
