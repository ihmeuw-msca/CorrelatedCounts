from setuptools import setup

setup(
    name='ccount',
    version='0.0.1',
    description='modeling correlated counts, follows model in Rodrigues-Motta et al 2013',
    url='https://github.com/mbannick/CorrelatedCounts',
    author='Marlena Bannick, Peng Zheng',
    author_email='mnorwood@uw.edu, zhengp@uw.edu',
    license='MIT',
    packages=['ccount'],
    package_dir={"": "src"},
    install_requires=['numpy', 'pytest', 'scipy', 'xspline', 'pandas'],
    zip_safe=False
)
