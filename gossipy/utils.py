"""This module contains utility functions that are used in multiple modules."""

import tarfile
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import numpy as np
from numpy.random import randint
import torch
from torch.nn import Module
from typing import List, Dict
import matplotlib.pyplot as plt

from . import LOG

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["choice_not_n",
           "torch_models_eq",
           "download_and_unzip",
           "download_and_untar",
           "plot_evaluation"]



# def print_flush(text: str) -> None:
#     """Prints a string and flushes the output buffer."""
#     print(text)
#     sys.stdout.flush()

def choice_not_n(mn: int,
                 mx: int,
                 notn: int) -> int:
    r"""Draws from the uniform distribution an integer between `mn` and `mx`, excluding `notn`.
    
    Parameters
    ----------
    mn : int
        Lowest integer to be drawn from the uniform distribution.
    mx : int
        Highest integer to be drawn from the uniform distribution.
    notn : int
        The value to exclude.

    Returns
    -------
    int
        Random integer :math:`x` s.t. :math:`\textrm{mn} \leq x \leq \textrm{mx}` and :math:`x \neq \textrm{notn}`.
    """

    c: int = randint(mn, mx)
    while c == notn:
        c = randint(mn, mx)
    return c

#def sigmoid(x: float) -> float:
#    return 1 / (1 + np.exp(-x))

def torch_models_eq(m1: Module,
                    m2: Module) -> bool:
    """Checks if two models are equal.

    The equality is defined in terms of the parameters of the models, both architectures and weights.
    
    Parameters
    ----------
    m1 : Module
        First model to compare.
    m2 : Module
        Second model to compare.
    
    Returns
    -------
    bool
        True if the two models are equal, False otherwise.
    """

    for (k1, i1), (k2, i2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        if not k1 == k2 or not torch.equal(i1, i2):
            return False
    return True


def download_and_unzip(url: str, extract_to: str='.') -> str:
    """Downloads a file from `url` and unzips it into `extract_to`.
    
    Parameters
    ----------
    url : str
        URL of the file to download.
    extract_to : str
        Path to extract the file to.
    
    Returns
    -------
    str
        Name of the extracted file.
    """

    LOG.info("Downloading %s into %s" %(url, extract_to))
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)
    return zipfile.namelist()[0]


def download_and_untar(url: str, extract_to: str='.') -> List[str]:
    """Downloads a file from `url` and untar it into `extract_to`.
    
    Parameters
    ----------
    url : str
        URL of the file to download.
    extract_to : str, default="."
        Path to extract the file to.
    
    Returns
    -------
    list of str
        List of names of the extracted files.
    """

    LOG.info("Downloading %s into %s" %(url, extract_to))
    ftpstream = urlopen(url)
    thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
    thetarfile.extractall(path=extract_to)
    return thetarfile.getnames()





def plot_evaluation(evals: List[List[Dict]],
                    title: str="Untitled plot") -> None:
    """Plots the evaluation results.

    Parameters
    ----------
    evals : list of list of dict
        This argument is meant to contain the results of a repeated experiment (outer list).
        For each experiment, the inner list contains the results of the evaluations performed during the 
        simulation. The results are stored in a dictionary where the keys are the names of the metrics and the
        values are the corresponding values.
    title : str, default="Untitled plot"
        Title of the plot.
    """
    if not evals or not evals[0] or not evals[0][0]: return
    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_subplot(111)
    for k in evals[0][0]:
        evs = [[d[k] for d in l] for l in evals]
        mu: float = np.mean(evs, axis=0)
        std: float = np.std(evs, axis=0)
        plt.fill_between(range(1, len(mu)+1), mu-std, mu+std, alpha=0.2)
        plt.title(title)
        plt.xlabel("cycle")
        plt.ylabel("metric value")
        plt.plot(range(1, len(mu)+1), mu, label=k)
        LOG.info(f"{k}: {mu[-1]:.2f}")
    ax.legend(loc="lower right")
    plt.show()