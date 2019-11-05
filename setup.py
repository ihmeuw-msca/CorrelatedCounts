from setuptools import setup

setup(
    name='ccount',
    version='0.0.0',
    description='modeling correlated counts',
    url='https://github.com/mbannick/CorrelatedCounts',
    author='Marlena Bannick, Peng Zheng',
    author_email='mnorwood@uw.edu, zhengp@uw.edu',
    license='MIT',
    packages=['ccount'],
    package_dir={'ccount': 'src/ccount'},
    install_requires=['numpy', 'pytest', 'scipy'],
    zip_safe=False
)
