"""
The :mod:`` module implements the rank similarity algorithm.
"""

from ._classes import RankSimilarityTransform
from ._classes import RankSimilarityClassifier
from ._classes import RSPClassifier

from ._version import __version__

__all__ = ['RankSimilarityTransform','RankSimilarityClassifier', 'RSPClassifier',
    '__version__']