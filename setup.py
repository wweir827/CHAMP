from setuptools import setup
import os,re


#read inversion info from single file
PKG = "champ"
VERSIONFILE = os.path.join(PKG, "_version.py")
verstr = "unknown"
try:
    verstrline = open(VERSIONFILE, "rt").read()
    print(verstrline)
except EnvironmentError:
    pass # Okay, there is no version file.
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        print ("unable to find version in %s" % (VERSIONFILE,))
        raise RuntimeError("if %s.py exists, it is required to be well-formed" % (VERSIONFILE,))




options=dict( name='champ',
    version=verstr,
    packages=[PKG],
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
                      'scipy>=0.19','sklearn','louvain','leidenalg','python-igraph','seaborn','tqdm'],
    dependency_links=['https://github.com/wweir827/louvain-igraph/tarball/master#egg=louvain-igraph-0.6.1.champ']
)

setup(**options)
