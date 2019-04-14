from setuptools import setup

options=dict( name='champ',
    version='2.0.2',
    packages=['champ'],
    url='http://github.com/wweir827/champ',
    license='GPLv3+',
    author='William H. Weir',
    provides=['champ'],
    author_email='wweir@med.unc.edu',
    description='Modularity based networks partition selection tool',
    zip_safe=False,
    classifiers=["Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.6",
                 "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                 "Topic :: Scientific/Engineering :: Information Analysis",
                 ],
    install_requires=['ipython<5.9','matplotlib','future','numpy>1.13','h5py',
                      'scipy','sklearn','louvain','python-igraph','seaborn','tqdm'],
    dependency_links=['https://github.com/wweir827/louvain-igraph/tarball/master#egg=louvain-igraph-0.6.1.champ']
)
#    install_requires=['pyhull','igraph','louvain','matplotlib','numpy',]

setup(**options)
