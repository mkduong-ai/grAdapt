from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(name='grAdapt',
      version='0.1.0a6',
      description='grAdapt: Gradient Adaption for Blackbox Optimization.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Manh Khoi Duong',
      author_email='manh.duong@hhu.de',
      url='',
      license='Apache License 2.0',
      # py_modules=['grAdapt'],
      packages=find_packages(),
      # package_dir={'': 'src'},
      classifiers=[
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'License :: OSI Approved :: Apache License 2.0',
          'Operating System :: OS Independent',
      ],
      install_requires=[
          'numpy ~= 1.18',
          'scipy ~= 1.4',
          'scikit-learn ~= 0.22',
          'tqdm ~= 4.44',
          'numba ~=0.49',
          'deprecated ~= 1.2.7'
      ],
      )