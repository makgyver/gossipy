from numpy.random import binomial

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["TokenAccount",
           "PurelyProactiveTokenAccount",
           "PurelyReactiveTokenAccount",
           "SimpleTokenAccount",
           "GeneralizedTokenAccount",
           "RandomizedTokenAccount"]


class TokenAccount():
    def __init__(self):
        self.n_tokens = 0
    
    def add(self, n: int=1) -> None:
        self.n_tokens += n

    def sub(self, n: int=1) -> None:
        self.n_tokens = max(0, self.n_tokens - n)
    
    def proactive(self) -> float:
        raise NotImplementedError()

    def reactive(self, utility: int) -> int:
        raise NotImplementedError()


class PurelyProactiveTokenAccount(TokenAccount):
    def proactive(self) -> float:
        return 1

    def reactive(self, utility: int) -> int:
        return 0


class PurelyReactiveTokenAccount(TokenAccount):
    def __init__(self, k: int=1):
        super(PurelyReactiveTokenAccount, self).__init__()
        self.k = k

    def proactive(self) -> float:
        return 0

    def reactive(self, utility: int) -> int:
        return int(utility * self.k)


class SimpleTokenAccount(TokenAccount):
    def __init__(self, C: int=1):
        super(SimpleTokenAccount, self).__init__()
        assert C >= 1, "The capacity C must be strictly positive."
        self.capacity = C
    
    def proactive(self) -> float:
        return int(self.n_tokens >= self.capacity)

    def reactive(self, utility: int) -> int:
        return int(self.n_tokens > 0)


class GeneralizedTokenAccount(SimpleTokenAccount):
    def __init__(self, C: int, A: int): #1
        super(GeneralizedTokenAccount, self).__init__(C)
        assert C >= 1, "The capacity C must be positive."
        assert A >= 1, "The reactivity A must be positive."
        assert A <= C, "The capacity C must be greater or equal than the reactivity A."
        self.reactivity = A

    def reactive(self, utility: int) -> int:
        num = self.reactivity + self.n_tokens - 1
        return int(num / self.reactivity if utility > 0 else num / (2 * self.reactivity))


class RandomizedTokenAccount(GeneralizedTokenAccount):
    def __init__(self, C: int, A: int):
        super(RandomizedTokenAccount, self).__init__(C, A)
    
    def proactive(self) -> float:
        if self.n_tokens < self.reactivity - 1:
            return 0
        elif self.reactivity - 1 <= self.n_tokens <= self.capacity:
            return (self.n_tokens - self.reactivity + 1) / (self.capacity - self.reactivity + 1)
        else:
            return 1

    def reactive(self, utility: int) -> int:
        if utility > 0:
            r = self.n_tokens / self.reactivity
            return int(r) + binomial(1, r - int(r)) #randRound
        return 0
