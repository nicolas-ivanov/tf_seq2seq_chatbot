from setuptools import setup
from setuptools import find_packages

install_requires = [
    'tensorflow==0.5.0'
]

setup(
      name='Seq2seq_upgrade',
      version='0.0.1',
      description='Additional Sequence to Sequence Features for TensorFlow',
      author='LeavesBreathe',
      url='https://github.com/LeavesBreathe/Seq2Seq_Upgrade_TensorFlow',
      license='Apache v2',
      install_requires=install_requires,
      packages=find_packages()
)