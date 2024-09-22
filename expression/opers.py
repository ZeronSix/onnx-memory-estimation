from math import floor, ceil
from typing import Callable, Optional, Tuple


def tuple_to_value(a: Tuple[float, ...]) -> float:
    if len(a) != 1:
        raise ValueError('Expected a scalar value')
    return a[0]


class UnaryMinus:
    def __init__(self):
        self.operation = UnaryMinus.unary_minus


    @staticmethod
    def unary_minus(a: Tuple[float, ...]) -> float:
        return - tuple_to_value(a)


class BinaryOperationWithSymbol:
    def __init__(self, symbol: str):
        if BinaryOperationWithSymbol.find(symbol) is None:
            raise ValueError(f'Tried to initialize a binary operation with an unknown symbol')
        
        self.symbol = symbol
        self.operation, self.priority = BinaryOperationWithSymbol.sign_to_oper[symbol]


    @classmethod
    def find(cls, sign: str) -> Optional[Callable[[float, float], float]]:
        return cls.sign_to_oper.get(sign, None)


    @staticmethod
    def __comma(rhs: Tuple[float, ...], lhs: Tuple[float, ...]) -> Tuple[float, ...]:
        return lhs + rhs
    
    @staticmethod
    def __add(rhs: Tuple[float, ...], lhs: Tuple[float, ...]) -> float:
        return tuple_to_value(lhs) + tuple_to_value(rhs)

    @staticmethod
    def __subtract(rhs: Tuple[float, ...], lhs: Tuple[float, ...]) -> float:
        return tuple_to_value(lhs) - tuple_to_value(rhs)
    
    @staticmethod
    def __multiply(rhs: Tuple[float, ...], lhs: Tuple[float, ...]) -> float:
        return tuple_to_value(lhs) * tuple_to_value(rhs)
    
    @staticmethod
    def __divide(rhs: Tuple[float, ...], lhs: Tuple[float, ...]) -> float:
        return tuple_to_value(lhs) / tuple_to_value(rhs)

    sign_to_oper = {
        ',': (__comma, 0),
        '+': (__add, 1),
        '-': (__subtract, 1),
        '*': (__multiply, 2),
        '/': (__divide, 2)
    }

class NamedFunction:
    def __init__(self, s: str):
        if NamedFunction.find(s) is None:
            raise ValueError(f'Tried to initialize a named function with an unknown name')
        
        self.s = s
        self.operation = NamedFunction.sign_to_oper[s]


    @classmethod
    def find(cls, name: str) -> Optional[Callable[[Tuple[float, ...]], float]]:
        return cls.sign_to_oper.get(name, None)


    @staticmethod
    def __id_func(arg: Tuple[float, ...]) -> float:
        return tuple_to_value(arg)

    @staticmethod
    def __ceil_func(arg: Tuple[float, ...]) -> float:
        return ceil(tuple_to_value(arg))

    @staticmethod
    def __floor_func(arg: Tuple[float, ...]) -> float:
        return floor(tuple_to_value(arg))

    @staticmethod
    def __min_func(a: Tuple[float, ...]) -> float:
        return min(a)

    sign_to_oper = {
        'identity': __id_func,
        'ceiling': __ceil_func,
        'floor': __floor_func,
        'Min': __min_func
    }


class NamedPlaceholder:
    def __init__(self, s):
        self.s = s
