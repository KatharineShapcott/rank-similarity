#! /usr/bin/env python

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
__version__ = ""
ver_file = os.path.join('ranksim', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'rank-similarity'
DESCRIPTION = 'Rank Similarity is a set of non-linear classification and transform tools for large datasets. '
with codecs.open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Katharine Shapcott'
MAINTAINER_EMAIL = 'katharine.shapcott@esi-frankfurt.de'
URL = 'https://github.com/KatharineShapcott/rank-similarity'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/KatharineShapcott/rank-similarity'
VERSION = __version__
INSTALL_REQUIRES = ['scikit-learn>=0.23', 'numpy>=1.14.6', 'scipy>=1.1.0']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      zip_safe=True,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)