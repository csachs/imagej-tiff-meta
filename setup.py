# -*- coding: utf-8 -*-
"""
documentation
"""

import imagej_tiff_meta
from setuptools import setup, find_packages

setup(
    name='imagej_tiff_meta',
    version=imagej_tiff_meta.__version__,
    description='',
    long_description='',
    author=imagej_tiff_meta.__author__,
    author_email='sachs.christian@gmail.com',
    url='https://github.com/csachs/imagej-tiff-meta',
    packages=find_packages(),
    requires=['numpy'],
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',   # lately no tests
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 2.7',  # tests, not often
        'Programming Language :: Python :: 3',    #
        'Programming Language :: Python :: 3.5',  # main focus
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ]
)
