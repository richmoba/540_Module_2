from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class CustomInstallCommand(install):
    """Customized install command to run additional setup steps."""
    def run(self):
        install.run(self)
        subprocess.check_call(['python', 'scripts/collect_data.py'])
        subprocess.check_call(['python', 'scripts/preprocess_data.py'])
        subprocess.check_call(['python', 'scripts/classical_model.py'])
        subprocess.check_call(['python', 'scripts/deep_learning_model.py'])

setup(
    name='sentiment_analysis_university_charlotte',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tweepy', 'praw', 'pandas', 'scikit-learn', 'tensorflow', 'streamlit', 'nltk'
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
)
