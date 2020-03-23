from setuptools import setup
setup(
    name = 'CodeChallenge---TextClassification',
    version = '0.1.0',
    packages = ['textClassifierCLI'],
    entry_points = {
        'console_scripts': [
            'textClassifierCLI = textClassifierCLI.__main__:main'
        ]
    })