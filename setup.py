from setuptools import setup

options=dict( name='champ',
    version='1.0.0',
    packages=['champ'],
    url='http://github.com/wweir827/champ',
    license='',
    author='William Weir',
    author_email='wweir@med.unc.edu',
    description='Modularity based networks partition selection tool',
    zip_safe=False,
    install_requires=['ipython<5.9','pyhull','matplotlib','numpy','sklearn','louvain','igraph']
)
#    install_requires=['pyhull','igraph','louvain','matplotlib','numpy',]

setup(**options)
