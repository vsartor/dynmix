'''
Setup file based on setuptools for dynmix

Copyright (c) Victhor S. Sartório. All rights reserved.
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
'''


from os import path
from setuptools import setup


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='dynmix',
    version='0.3.0',
    description='Dynamix Membership Mixture Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vsartor/dynmix',
    author='Victhor S. Sartório',
    author_email='victhor@dme.ufrj.br',
    test_suite='tests',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='mixture model bayesian statistics dynamic',
    packages=['dynmix'],
    install_requires=['numpy', 'scipy'],
)
