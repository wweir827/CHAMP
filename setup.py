from setuptools import setup

options=dict( name='champ',
    version='1.0.3',
    packages=['champ'],
    url='http://github.com/wweir827/champ',
    license='GPLv3+',
    author='William H. Weir',
    provides=['champ'],
    author_email='wweir@med.unc.edu',
    description='Modularity based networks partition selection tool',
    zip_safe=False,
        classifiers=["Programming Language :: Python :: 2.7"],
    install_requires=['ipython<5.9','pyhull','matplotlib','future','numpy','sklearn','louvain','python-igraph','h5py']
)
#    install_requires=['pyhull','igraph','louvain','matplotlib','numpy',]

setup(**options)
