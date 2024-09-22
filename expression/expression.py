from typing import Any, Dict, List, Optional, Set, Tuple, Union

from expression.opers import BinaryOperationWithSymbol, NamedFunction, NamedPlaceholder, UnaryMinus

ReprType = Union[NamedPlaceholder, NamedFunction, UnaryMinus, BinaryOperationWithSymbol, float]


class Expression:
    def __init__(self, expression: str, placeholders: Set[str]):
        self.expression = expression
        self.placeholders = placeholders
        self.expression_repr: Optional[List[ReprType]] = None


    def evaluate(self, placeholder_map: Dict[str, Any]) -> float:
        if self.expression_repr is None:
            self.convert_to_representation()

        if self.expression_repr is None:
            raise RuntimeError(f'Expression representation is None after conversion')

        stack: List[Tuple[float, ...]] = list()

        for elem in self.expression_repr:
            if isinstance(elem, float): # tuple of floats is assumed
                stack.append((elem, ))

            elif isinstance(elem, NamedPlaceholder):
                if elem.s not in placeholder_map:
                    raise ValueError(f'Placeholder {elem.s} has no value')
                stack.append((placeholder_map[elem.s], ))

            elif isinstance(elem, NamedFunction):
                if not stack:
                    raise ValueError(f'No arguments in stack for function {elem.s}')
                stack.append((elem.operation(stack.pop()), ))

            elif isinstance(elem, UnaryMinus):
                if not stack:
                    raise ValueError(f'No arguments in stack for unary minus')
                stack.append((elem.operation(stack.pop()), ))

            elif isinstance(elem, BinaryOperationWithSymbol):
                if len(stack) < 2:
                    raise ValueError(f'Not enough arguments in stack for opeation {elem.symbol}')

                # small hack to prevent nested tuples
                res = elem.operation(stack.pop(), stack.pop())
                if elem.symbol != ',':
                    res = (res, )

                stack.append(res)
            
            else:
                raise RuntimeError(f'Cannot evaluate item {elem}')

        if len(stack) != 1 or len(stack[0]) != 1:
            raise RuntimeError(f'Expected the result to be a single scalar value')

        return stack.pop()[0]


    def convert_to_representation(self):
        self.expression_repr = list()

        stack = list()
        start, end = 0, len(self.expression)

        def is_id_char(ch: str):
            if len(ch) != 1:
                raise ValueError('A single character expected')
            return ch == '_' or ch.isalpha()

        can_use_unary_minus_now = True
        while start < end:
            ch = self.expression[start]

            if ch == ' ':
                start += 1
                continue

            can_use_unary_minus_on_next_iteration = False
            if ch == '-' and can_use_unary_minus_now:
                stack.append(UnaryMinus())
                start += 1
                can_use_unary_minus_on_next_iteration = True
            
            elif BinaryOperationWithSymbol.find(ch) is not None:
                op = BinaryOperationWithSymbol(ch)

                while stack:
                    should_pop = False
                    if isinstance(stack[-1], BinaryOperationWithSymbol):
                        if stack[-1].priority >= op.priority:
                            self.expression_repr.append(stack[-1])
                            should_pop = True
                    elif isinstance(stack[-1], UnaryMinus):
                        self.expression_repr.append(stack[-1])
                        should_pop = True
                    else:
                        break

                    if should_pop:
                        stack.pop()
                    else:
                        break
                
                stack.append(op)
                start += 1
                can_use_unary_minus_on_next_iteration = True
            
            elif ch == '(':
                stack.append(NamedFunction('identity'))
                start += 1
                can_use_unary_minus_on_next_iteration = True

            elif ch == ')':
                while stack:
                    if isinstance(stack[-1], (BinaryOperationWithSymbol, UnaryMinus)):
                        self.expression_repr.append(stack[-1])
                        stack.pop()
                    else:
                        break

                if stack and not isinstance(stack[-1], NamedFunction):
                    raise RuntimeError(f'Unexpected closing bracket at position {start}')
                
                self.expression_repr.append(stack[-1])
                stack.pop()
                start += 1
            
            elif is_id_char(ch):
                token = ""

                while start < end and is_id_char(self.expression[start]):
                    token += self.expression[start]
                    start += 1

                if token in self.placeholders:
                    self.expression_repr.append(NamedPlaceholder(token))
                
                else:
                    op = NamedFunction.find(token)
                    if op is None:
                        raise RuntimeError(f'{token} is not a valid function name')
                    if start >= end or self.expression[start] != '(':
                        raise RuntimeError(f'Expected an opening bracket after {token}')
                    start += 1
                    stack.append(NamedFunction(token))
                    can_use_unary_minus_on_next_iteration = True
            
            else:
                next_start = start
                while next_start < end:
                    try:
                        value = float(self.expression[start:next_start+1])
                    except ValueError:
                        break
                    next_start += 1

                if next_start == start:
                    raise RuntimeError(f'Expected a floating-point constant, got {self.expression[start]}')
                
                self.expression_repr.append(value)
                start = next_start
            
            can_use_unary_minus_now = can_use_unary_minus_on_next_iteration
        
        while stack:
            if isinstance(stack[-1], (BinaryOperationWithSymbol, UnaryMinus)):
                self.expression_repr.append(stack[-1])
            else:
                raise RuntimeError(f'No functions are allowed on the operation stack after the parsing ends')
            stack.pop()
