from setuptools import find_packages, setup

setup(
    name='requests_engine',
    packages=find_packages(include=['request_engine']),
    version='0.1.0',
    description='requests_engine',
    author='Edoardo Luciani',
    author_email='edoardo.luciani@gmail.com',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)