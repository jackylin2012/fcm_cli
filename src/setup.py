from setuptools import setup

setup(
      name='fcm',
      version='0.1',
      author_email='maocf1993@gmail.com',
      py_modules=['main'],
      install_requires=['click', 'spacy', 'sklearn', 'pandas', 'torch', 'scipy', 'tqdm', 'pyLDAvis', 'numpy'],
      python_requires='>=3.6',
      entry_points={
        'console_scripts': ['fcm = cli:fcm']
      }
)
