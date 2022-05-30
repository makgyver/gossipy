from abc import ABC, abstractmethod
from numpy.random import binomial

# AUTHORSHIP
__version__ = "0.0.1"
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


class TokenAccount(ABC):
    def __init__(self):
        """Abstract class representing a generic token account.

        The token account framework has been firstly proposed in :cite:p:`Danner:2018`and it is used
        for gossip learning in :cite:p:`Hegedus:2021`.
        """

        self.n_tokens = 0
    
    def add(self, n: int=1) -> None:
        """Increases the number of tokens by ``n``.

        Parameters
        ----------
        n : int, default=1
            The number of tokens to add.
        """

        self.n_tokens += n

    def sub(self, n: int=1) -> None:
        """Decreases the number of tokens by ``n``.

        Parameters
        ----------
        n : int, default=1
            The number of tokens to remove.
        """

        self.n_tokens = max(0, self.n_tokens - n)
    
    @abstractmethod
    def proactive(self) -> float:
        """Method that along with the ``reactive`` method defines the token account
        strategy.

        Returns
        -------
        float
            The probability of a token being consumed.
        """

        pass

    @abstractmethod
    def reactive(self, utility: int) -> int:
        """Method that along with the ``proactive`` method defines the token account
        strategy.

        It returns the number of messages that the node will send as a reaction to an incoming 
        message, as a function of the account balance and the usefulness (i.e., ``utility``) of the 
        received message.
    
        Parameters
        ----------
        utility : int
            The utility of the current token.
        """

        pass


class PurelyProactiveTokenAccount(TokenAccount):

    def __init__(self):
        """Purely proactive token account.
        
        This is a special case of the token account framework and it essentially implements a standard
        push gossip algorithm.
        """

        pass

    # docstr-coverage:inherited
    def proactive(self) -> float:
        return 1

    # docstr-coverage:inherited
    def reactive(self, utility: int) -> int:
        return 0


class PurelyReactiveTokenAccount(TokenAccount):
    def __init__(self, k: int=1):
        """Purely reactive token account.

        Naive reactive token account variants where every message received triggers message 
        sending immediately.

        Parameters
        ----------
        k : int, default=1
            The number of messages to send in case of reaction.
        """

        super(PurelyReactiveTokenAccount, self).__init__()
        self.k = k

    # docstr-coverage:inherited
    def proactive(self) -> float:
        return 0

    # docstr-coverage:inherited
    def reactive(self, utility: int) -> int:
        return int(utility * self.k)


class SimpleTokenAccount(TokenAccount):
    def __init__(self, C: int=1):
        """Simple token account.
        
        Implements a "standard" token account strategy where the node will proactively send a 
        message if it has at the number of tokens are at least equal to the capacity. It is also 
        reactive iff the node has at least one token.

        Parameters
        ----------
        C : int, default=1
            The capacity of the token account.
        """

        super(SimpleTokenAccount, self).__init__()
        assert C >= 1, "The capacity C must be strictly positive."
        self.capacity = C
    
    # docstr-coverage:inherited
    def proactive(self) -> float:
        return int(self.n_tokens >= self.capacity)

    # docstr-coverage:inherited
    def reactive(self, utility: int) -> int:
        return int(self.n_tokens > 0)


class GeneralizedTokenAccount(SimpleTokenAccount):
    def __init__(self, C: int, A: int): #1
        r"""Generalized token account.

        Implements a generalized simple token account strategy with a reactive function that is able 
        to increase the number of messages sent when the number of tokens is high.
        To this end, the proactive function is the same as in :class:`SimpleTokenAccount` while the
        reactive function is defined as :cite:p:`Danner:2018`:

        :math:`\operatorname{REACTIVE}(a, u)= \begin{cases}\lfloor(A-1+a) / A\rfloor & \text { if } 
        u\\ \lfloor(A-1+a) /(2 A)\rfloor & \text { otherwise }\end{cases}`

        where :math:`u`is the utility of the message, :math:`a` is the account balance, :math:`C` is
        the capacity of the token account, and :math:`A` is the reactivity of the token account.

        Parameters
        ----------
        C : int
            The capacity of the token account.
        A : int
            The reactivity of the token account.
        """

        super(GeneralizedTokenAccount, self).__init__(C)
        assert C >= 1, "The capacity C must be positive."
        assert A >= 1, "The reactivity A must be positive."
        assert A <= C, "The capacity C must be greater or equal than the reactivity A."
        self.reactivity = A

    # docstr-coverage:inherited
    def reactive(self, utility: int) -> int:
        num = self.reactivity + self.n_tokens - 1
        return int(num / self.reactivity if utility > 0 else num / (2 * self.reactivity))


class RandomizedTokenAccount(GeneralizedTokenAccount):
    def __init__(self, C: int, A: int):
        r"""This token account strategy implements a more fine-grained handling of proactive 
        messages. In fact, the proactive function returns 1 when the balance is at least the 
        capacity (as always) but it adds some proactive behavior even when this is not the case: 
        the returned value is linear starting from A − 1 until C. The starting point of this linear
        segment is A − 1 because if the balance is less than A then the reactive function 
        will be able to send less than one messages on average:

        :math:`\operatorname{PROACTIVE}(a)= \begin{cases}0 & \text { if } a<A-1 \\ 
        \frac{a-A+1}{C-A+1} & \text { if } a \in[A-1, C] \\ 1 & \text { otherwise. }\end{cases}`

        Instad, the reactive function is:

        :math:`\operatorname{REACTIVE}(a, u)= \begin{cases} \text{randRound}(a / A) & \text { if } 
        u \\ 0 & \text { otherwise }\end{cases}`

        where the only difference with the :class:`GeneralizedTokenAccount` is that the rounding is 
        performed randomly.

        Parameters
        ----------
        C : int
            The capacity of the token account.
        A : int
            The reactivity of the token account.
        """

        super(RandomizedTokenAccount, self).__init__(C, A)
    
    # docstr-coverage:inherited
    def proactive(self) -> float:
        if self.n_tokens < self.reactivity - 1:
            return 0
        elif self.reactivity - 1 <= self.n_tokens <= self.capacity:
            return (self.n_tokens - self.reactivity + 1) / (self.capacity - self.reactivity + 1)
        else:
            return 1

    # docstr-coverage:inherited
    def reactive(self, utility: int) -> int:
        if utility > 0:
            r = self.n_tokens / self.reactivity
            return int(r) + binomial(1, r - int(r)) #randRound
        return 0
