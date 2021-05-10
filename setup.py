
import os
import setuptools
import urllib.request
import zipfile

from setuptools import setup


setup(
    name='cyolo_score_following',
    version='0.1dev',
    description='Multi-modal Conditional Bounding Box Regression for Music Score Following',
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: MusicInformationRetrieval",
    ],
    author='Florian Henkel',
)


# extract data
DATA_PATH = "./data/msmd.zip"
DATA_URL = "https://zenodo.org/record/4745838/files/msmd.zip?download=1"
if not os.path.exists(DATA_PATH):

    if not os.path.exists(os.path.dirname(DATA_PATH)):
        print('Creating data folder ...')
        os.mkdir(os.path.dirname(DATA_PATH))

    print(f"Downloading data to {DATA_PATH} ...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

    print(f"Extracting data {DATA_PATH} ...")
    zip_ref = zipfile.ZipFile(DATA_PATH, 'r',  zipfile.ZIP_DEFLATED)
    zip_ref.extractall(os.path.dirname(DATA_PATH))
    zip_ref.close()

    # delete zip file
    os.unlink(DATA_PATH)
