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
    install_requires=['pyhull','matplotlib','numpy',]
)
#    install_requires=['pyhull','igraph','louvain','matplotlib','numpy',]

setup(**options)
