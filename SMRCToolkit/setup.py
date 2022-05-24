from setuptools import setup
from setuptools import find_packages


setup(name='sogou_mrc',
      version='0.1',
      description='sogou_mrc',
      author='SoGou Search AI Group',
      install_requires=['numpy', 'tensorflow-gpu==2.6.4', 'tensorflow-hub', "tqdm", "spacy", "jieba",
                        "stanfordnlp"],
      packages=find_packages()
      )
