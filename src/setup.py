from setuptools import setup

setup(
      name='fcm',
      version='0.1',
      author_email='maochfe@gmail.com',
      py_modules=['main'],
      install_requires=['click', 'nltk', 'sklearn', 'pandas', 'torch', 'scipy', 'pyLDAvis', 'numpy', 'django', 'gensim'],
      python_requires='>=3.6',
      entry_points={
        'console_scripts': ['fcm = cli:fcm']
      }
)
