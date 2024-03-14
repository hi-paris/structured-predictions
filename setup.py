#!/usr/bin/env python

import os
import re
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.extension import Extension

import numpy
from Cython.Build import cythonize
from distutils.command.build_py import build_py as _build_py

sys.path.append(os.path.join("stpredictions", "helpers"))    ########################333
from openmp_helpers import check_openmp_support

# dirty but working
# __version__ = re.search(
#     r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
#     open('stpredictions/OK3/__init__.py').read()).group(1)
# The beautiful part is, I don't even need to check exceptions here.
# If something messes up, let the build process fail noisy, BEFORE my release!

# thanks PyPI for handling markdown now
ROOT = os.path.abspath(os.path.dirname(__file__))
# with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
#     README = f.read()

# #################clean cython output is clean is called
# if 'clean' in sys.argv[1:]:
#     if os.path.isfile('ot/lp/emd_wrap.cpp'):
#         os.remove('ot/lp/emd_wrap.cpp')

# add platform dependant optional compilation argument
openmp_supported, flags = check_openmp_support()
compile_args = ["/O2" if sys.platform == "win32" else "-O3"]
link_args = []

if openmp_supported:
    compile_args += flags + ["/DOMP" if sys.platform == 'win32' else "-DOMP"]
    link_args += flags

if sys.platform.startswith('darwin'):
    compile_args.append("-stdlib=libc++")
    sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path'])
    os.environ['CFLAGS'] = '-isysroot "{}"'.format(sdk_path.rstrip().decode("utf-8"))

# class build_py(_build_py):

#     def find_package_modules(self, package, package_dir):
#         ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
#         modules = super().find_package_modules(package, package_dir)
#         filtered_modules = []
#         for (pkg, mod, filepath) in modules:
#             if os.path.exists(filepath.replace('.py', ext_suffix)):
#                 continue
#             filtered_modules.append((pkg, mod, filepath, ))
#         return filtered_modules


setup(
    name='structured-predictions',
    version='0.1.3',
    description='Structured-Predictions',
    # long_description=README,
    long_description_content_type='text/markdown',
    author=u"Florence d'Alché-Buc (Researcher), Luc Motte (Researcher), Tamim El Ahmad (Researcher) , Awais Sani (Engineer), Danaël  Schlewer-Becker(Engineer), Gaëtan Brison (Engineer)",
    author_email='structured-predictions@gmail.com',
    url='https://github.com/hi-paris/structured-predictions',
    packages=find_packages(exclude=["benchmarks"]),
    include_package_data=True,
    ext_modules=cythonize(Extension(
        name="*",
        sources=["stpredictions/models/OK3/*.pyx"],  # cython/c++ src files
        #language="c++",
        include_dirs=[numpy.get_include(), os.path.join(ROOT, 'stpredictions/models/OK3/test')],
        # include_dirs=[numpy.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args
    )),
    # cmdclass = { 'build_py': build_py},
    include_dirs=[numpy.get_include()],
    platforms=['linux', 'macosx', 'windows'],
    # download_url='https://github.com/PythonOT/POT/archive/{}.tar.gz'.format(version),
    license='MIT',
    scripts=[],
    data_files=[],
    setup_requires=["oldest-supported-numpy", "cython>=0.23"],
    install_requires=["numpy>=1.16", "scipy>=1.0", "scikit-learn", "torch", 
             "liac-arff", "requests", "grakel"],
    # install_requires=["numpy", "scipy", "scikit-learn==0.24.2", "torch",
    #          "liac-arff", "requests"],
    python_requires=">=3.7",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        # 'Programming Language :: C++',
        # 'Programming Language :: C',
        'Programming Language :: Cython',
        'Topic :: Utilities',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ]
)
