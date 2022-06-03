"""This module contains utility functions."""

import tarfile
from urllib.error import URLError
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import numpy as np
from numpy.random import randint
import torch
from typing import List, Dict
import matplotlib.pyplot as plt
from json import JSONEncoder

from . import LOG

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "Apache License, Version 2.0"
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
    r"""Draws from the uniform distribution an integer between ``mn`` and ``mx``, excluding ``notn``.
    
    Parameters
    ----------
    mn : int
        Lowest integer to be drawn from the uniform distribution.
    mx : int
        Highest integer to be drawn from the uniform distribution.
    notn : int
        The integer value to exclude.

    Returns
    -------
    int
        Random integer :math:`x` s.t. ``mn`` :math:`\leq x \leq` ``mx`` and :math:`x \neq` ``notn``.
    """

    c: int = randint(mn, mx)
    while c == notn:
        c = randint(mn, mx)
    return c


def torch_models_eq(m1: torch.nn.Module,
                    m2: torch.nn.Module) -> bool:
    """Checks if two pytorch models are equal.

    The equality is defined in terms of the parameters of the models, both architecture and weights.
    
    Parameters
    ----------
    m1 : torch.nn.Module
        First model to compare.
    m2 : torch.nn.Module
        Second model to compare.
    
    Returns
    -------
    bool
        True if the two models are equal, False otherwise.
    """

    sd1 = m1.state_dict()
    sd2 = m2.state_dict()

    if len(sd1) != len(sd2):
        return False
    
    for (k1, i1), (k2, i2) in zip(sd1.items(), sd2.items()):
        if  k1 != k2 or not torch.equal(i1, i2):
            return False
    return True


def download_and_unzip(url: str, extract_to: str='.') -> str:
    """Downloads a file from ``url`` and unzips it into ``extract_to``.
    
    Parameters
    ----------
    url : str
        URL of the file to download.
    extract_to : str
        Path to extract the file to.
    
    Returns
    -------
    list of str
        List of names of the extracted files.
    """

    LOG.info("Downloading %s into %s" %(url, extract_to))
    try:
        http_response = urlopen(url)
    except URLError:
        # Handle urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] 
        # certificate verify failed: certificate has expired 
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        http_response = urlopen(url)

    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)
    return zipfile.namelist()


def download_and_untar(url: str, extract_to: str='.') -> List[str]:
    """Downloads a file from ``url`` and untar it into ``extract_to``.
    
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
        This argument is meant to contain the results of a repeated experiment (outer list:
        each element an experiment). For each experiment, the inner list contains the results of
        the evaluations performed during the simulation. The results are stored in a dictionary
        where the keys are the names of the metrics and the values are the
        corresponding performance.
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


class StringEncoder(JSONEncoder):
    # docstr-coverage:excused `internal class to handle logging`
    def default(self, o) -> str:
        return str(o)