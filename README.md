# Rank Similarity

Rank Similarity is a set of non-linear classification and transform tools for large multi-dimensional datasets that use the scikit-learn API. 

## Installation
### Dependencies
rank-similarity requires:

- Scikit-learn (>= 0.23)
- Python (>= 3.7)
- NumPy (>= 1.14.6)
- SciPy (>= 1.1.0)

Optionally for plotting examples:
- matplotlib (>= 2.2.2)

### Install via pip

```
pip install rank-similarity
```

### Development version

To get the latest development version, clone the GitHub repository:

```
git clone https://github.com/KatharineShapcott/rank-similarity
```

## Usage

``` python
from ranksim import RankSimilarityClassifier
X = [[0, 1], [1, 0]]
y = [0, 1]
clf = RankSimilarityClassifier()
clf.fit(X, y)
pred = clf.predict(X)
```

## More Information

### Documentation
More details and background information is available in the
[online documentation](https://katharineshapcott.github.io/rank-similarity/).

### License
The package is new BSD licensed.

### Citation
Please cite the following publication (in preparation) [[1]](#1).

<a id="1">[1]</a> 
Shapcott, Bird, & Singer. Confusion-based rank similarity filters for computationally-efficient machine learning on high dimensional data. In preperation. (2021)
