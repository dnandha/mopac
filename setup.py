from distutils.core import setup
from setuptools import find_packages

setup(
    name='mopac',
    packages=find_packages(),
    version='0.1',
    description='Model-based policy optimization',
    long_description=open('./README.md').read(),
    author='',
    author_email='',
    url='',
    entry_points={
        'console_scripts': (
            'mopac=softlearning.scripts.console_scripts:main',
            'viskit=mopac.scripts.console_scripts:main'
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)
