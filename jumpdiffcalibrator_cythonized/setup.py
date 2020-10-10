from distutils.core import setup
from Cython.Build import cythonize

setup(name="heston_calibrator",
      ext_modules=cythonize(module_list=["basic_calibrator.pyx", "heston_calibrator.pyx"],
                            compiler_directives={'language_level': "3"}))

