import os
import sys
from math import isclose

sys.path.insert(0, os.path.abspath('..'))
from gossipy.utils import print_flush, choice_not_n, sigmoid

def test_print_flush(capsys):
    print_flush("test")
    captured = capsys.readouterr()
    assert captured.out == "test\n"

def test_choice_not_n():
    values = [choice_not_n(0, 3, 1) for _ in range(1000)]
    assert 1 not in values

def test_sigmoid():
    assert sigmoid(0) == .5
    assert isclose(sigmoid(-(100)), 0, rel_tol=1e-8, abs_tol=1e-8)
    assert isclose(sigmoid(100), 1, rel_tol=1e-8, abs_tol=1e-8)
