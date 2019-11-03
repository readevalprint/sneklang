"""
Sneklang - (C) 2019 Timothy Watts
-------------------------------------

Minimal subset of Python for safe evaluation

-------------------------------------

Initial idea copied from J.F. Sebastian on Stack Overflow
( http://stackoverflow.com/a/9558001/1973500 ) with
modifications and many improvements.

Since then it has been adapted from simpleeval by danthedeckie.

-------------------------------------
Basic Usage:

>>> from sneklang import snek_eval
>>> snek_eval("20 + 30\\n4+4")
[50, 8]

You can add your own functions easily too:

if file.txt contents is "11"

>>> def get_file():
...     with open("file.txt", 'r') as f:
...         return f.read()

>>> snek_eval('int(get_file()) + 31', scope = {"get_file": get_file})
[42]

For more information, see the full package documentation on pypi, or the github
repo.

-----------


>>> snek_eval("40 + two", scope={"two": 2})
[42]




>>> my_func = snek_eval('''
... a = 1
... def foo(b):
...   c = 3
...   return a, b, c
... foo
... ''')[-1]
>>> my_func(2)
(1, 2, 3)



>>> try:
...   snek_eval('''
... def fib(n):
...   return ((fib(n-1) + fib(n-2)) if n > 1 else n)
... fib(7)
... ''')
... except RecursionError as e:
...  print("Oh no! The current recursion limit is too low for this: %s" % format(sys.getrecursionlimit()))
[None, 13]

>>> try:
...   snek_eval('''
... def bar():
...   bar()
... bar()
... ''')
... except CallTooDeep as e:  # Snek error
...   print(e)
... except RecursionError as e:  # uncaught, this would hit a python error.
...  print("Oh no! The current recursion limit is too low for ths function: %s" % format(sys.getrecursionlimit()))
Sorry, stack is to large. The MAX_CALL_DEPTH is 32.


"""

import inspect
import ast
import operator as op
import sys
import types
import itertools
from collections import Counter, defaultdict
from functools import partial
import forge

########################################
# Module wide 'globals'
MAX_STRING_LENGTH = 100 * 1000
MAX_POWER = 100 * 100  # highest exponent
MAX_SCOPE_SIZE = 2e4
MAX_NODE_CALLS = 10000
MAX_CALL_DEPTH = 32
DISALLOW_PREFIXES = ["_"]
DISALLOW_METHODS = [(str, "format"), (type, "mro"), (str, "format_map")]
WHITLIST_ATTRIBUTES = [
    "dict.get",
    "dict.items",
    "dict.keys",
    "dict.values",
    "list.sort",
    "list.append",
    "list.pop",
]

# Disallow functions:
# This, strictly speaking, is not necessary.  These /should/ never be accessable anyway,
# if DISALLOW_PREFIXES and DISALLOW_METHODS are all right.  This is here to try and help
# people not be stupid.  Allowing these functions opens up all sorts of holes - if any of
# their functionality is required, then please wrap them up in a safe container.  And think
# very hard about it first.  And don't say I didn't warn you.

DISALLOW_FUNCTIONS = {
    type,
    eval,
    getattr,
    setattr,
    help,
    repr,
    compile,
    open,
    exec,
    format,
}

ALLOWED_BUILTINS = ["str.join", "len", "print", "str.count"]

########################################
# Exceptions:


class Return(Exception):
    """ Not actually an exception, just a way to break out of the function """

    def __init__(self, value):
        self.value = value


class Break(Exception):
    """ Not actually an exception, just a way to break out of the loop"""


class InvalidExpression(Exception):
    """ Generic Exception """

    pass


class DangerousValue(Exception):
    """ When you try to pass in something dangerous to snek, it won't catch everything though """

    def __init__(self, msg):
        super().__init__(msg)


class SnekRuntimeError(Exception):
    """Something caused the Snek code to crash"""

    def __init__(self, msg, node=None):
        self.__node = node
        super().__init__(msg)

    @property
    def lineno(self):
        return getattr(self.__node, "lineno", None)

    @property
    def col(self):
        return getattr(self.__node, "col_offset", None)


class SnekArithmeticError(SnekRuntimeError):
    pass


class SnekBufferError(SnekRuntimeError):
    pass


class SnekImportError(SnekRuntimeError):
    pass


class SnekLookupError(SnekRuntimeError):
    pass


class SnekValueError(SnekRuntimeError):
    pass


class SnekAttributeError(SnekRuntimeError):
    pass


class SnekTypeError(SnekRuntimeError):
    pass


class SnekAssertionError(SnekRuntimeError):
    pass


class NameNotDefined(SnekRuntimeError):
    """ a name isn't defined. """

    def __init__(self, node):
        super(NameNotDefined, self).__init__(
            "'{0}' is not defined".format(node.id), node
        )


class FeatureNotAvailable(SnekRuntimeError):
    """ What you're trying to do is not allowed. """


class NumberTooHigh(SnekRuntimeError):
    """ Sorry! That number is too high. I don't want to spend the
        next 10 years evaluating this expression! """

    pass


class CallTooDeep(SnekRuntimeError):
    pass


class IterableTooLong(SnekRuntimeError):
    """ That iterable is **way** too long, baby. """

    pass


class ScopeTooLarge(SnekRuntimeError):
    """ The scope has take too many bytes """

    pass


class ScopeTooComplex(SnekRuntimeError):
    """ The scope has too many nodes """

    pass


class TooManyEvaluations(SnekRuntimeError):
    """ The evaluator has evaluated too may nodes """

    pass


########################################
# Default simple functions to include:


def safe_mod(a, b):
    """ only allow modulo on numbers, not string formating """
    if isinstance(a, str):
        raise TypeError("Sorry, string formating is not supported")
    return a % b


def safe_power(a, b):  # pylint: disable=invalid-name
    """ a limited exponent/to-the-power-of function, for safety reasons """

    if abs(a) > MAX_POWER or abs(b) > MAX_POWER:
        raise ArithmeticError("Sorry! I don't want to evaluate {0} ** {1}".format(a, b))
    return a ** b


def safe_mult(a, b):  # pylint: disable=invalid-name
    """ limit the number of times an iterable can be repeated... """
    if hasattr(a, "__len__") and b * len(str(a)) >= MAX_SCOPE_SIZE:
        raise ArithmeticError("Sorry, I will not evalute something that long.")
    if hasattr(b, "__len__") and a * len(str(b)) >= MAX_SCOPE_SIZE:
        raise ArithmeticError("Sorry, I will not evalute something that long.")

    return a * b


def safe_add(a, b):  # pylint: disable=invalid-name
    """ iterable length limit again """

    if hasattr(a, "__len__") and hasattr(b, "__len__"):
        if len(a) + len(b) > MAX_STRING_LENGTH:
            raise ArithmeticError(
                "Sorry, adding those two together would make something too long."
            )
    return a + b


########################################
# Defaults for the evaluator:


# def get_safe_exec(snek):
#    def safe_exec(expr, scope):
#        snek_eval(expr, scope, snek.call_stack)
#
#    return safe_exec


# class SnekDeferred:
#    def __init__(self, func):
#        self.func = func
#
#    def __call__(self, snek):
#        return self.func(snek)


DEFAULT_SCOPE = {
    "True": True,
    "False": False,
    "None": None,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "len": len,
    # "no one man should have all this power"
    # "exec": SnekDeferred(get_safe_exec),
    "min": min,
    "max": max,
    "any": any,
    "all": all,
    "round": round,
    "isinstance": isinstance,
    "Exception": SnekRuntimeError,
    "enumerate": enumerate,
}


def make_modules(mod_dict):
    return {
        k: (v.__dict__.update(mod_dict[k]) or v)
        for k, v in {k: types.ModuleType(k) for k in mod_dict}.items()
    }


class SnekEval(object):
    nodes_called = 0
    # temp place to track return values
    _last_eval_result = None

    def __init__(self, scope=None, modules=None, call_stack=None):

        if call_stack is None:
            call_stack = []
        self.call_stack = call_stack

        self.operators = {
            ast.Add: safe_add,
            ast.Sub: op.sub,
            ast.Mult: safe_mult,
            ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv,
            ast.Pow: safe_power,
            ast.Mod: safe_mod,
            ast.Eq: op.eq,
            ast.NotEq: op.ne,
            ast.Gt: op.gt,
            ast.Lt: op.lt,
            ast.GtE: op.ge,
            ast.LtE: op.le,
            ast.Not: op.not_,
            ast.USub: op.neg,
            ast.UAdd: op.pos,
            ast.In: lambda x, y: op.contains(y, x),
            ast.NotIn: lambda x, y: not op.contains(y, x),
            ast.Is: op.is_,
            ast.IsNot: op.is_not,
            ast.BitOr: op.or_,
            ast.BitXor: op.xor,
            ast.BitAnd: op.and_,
        }

        if scope is None:
            scope = {}

        tmp_scope = DEFAULT_SCOPE.copy()
        tmp_scope.update(scope)
        scope.update(tmp_scope)

        self.scope = scope

        self.modules = {}
        if modules is not None:
            self.modules = modules

        self.nodes = {
            ast.Num: self._eval_num,
            ast.Bytes: self._eval_bytes,
            ast.Str: self._eval_str,
            ast.Name: self._eval_name,
            ast.UnaryOp: self._eval_unaryop,
            ast.BinOp: self._eval_binop,
            ast.BoolOp: self._eval_boolop,
            ast.Compare: self._eval_compare,
            ast.IfExp: self._eval_ifexp,
            ast.If: self._eval_if,
            ast.Try: self._eval_try,
            ast.ExceptHandler: self._eval_excepthandler,
            ast.Call: self._eval_call,
            ast.keyword: self._eval_keyword,
            ast.Subscript: self._eval_subscript,
            ast.Attribute: self._eval_attribute,
            ast.Index: self._eval_index,
            ast.Slice: self._eval_slice,
            ast.Module: self._eval_module,
            ast.Expr: self._eval_expr,
            ast.Assign: self._eval_assign,
            ast.FunctionDef: self._eval_functiondef,
            ast.arguments: self._eval_arguments,
            ast.Return: self._eval_return,
            ast.JoinedStr: self._eval_joinedstr,  # f-string
            ast.NameConstant: self._eval_nameconstant,
            ast.FormattedValue: self._eval_formattedvalue,
            ast.Dict: self._eval_dict,
            ast.Tuple: self._eval_tuple,
            ast.List: self._eval_list,
            ast.Set: self._eval_set,
            ast.ListComp: self._eval_comprehension,
            ast.SetComp: self._eval_comprehension,
            ast.DictComp: self._eval_comprehension,
            # ast.GeneratorExp: self._eval_comprehension,
            ast.ImportFrom: self._eval_importfrom,
            ast.Import: self._eval_import,
            ast.For: self._eval_for,
            ast.While: self._eval_while,
            ast.Break: self._eval_break,
            ast.Assert: self._eval_assert,
            ast.Delete: self._eval_delete,
            ast.Raise: self._eval_raise,
            # not really none, these are handled differently
            ast.And: None,
            ast.Or: None,
        }

        self.assignments = {
            ast.Name: self._assign_name,
            ast.Subscript: self._assign_subscript,
        }

        self.deletions = {
            ast.Name: self._delete_name,
            ast.Subscript: self._delete_subscript,
        }

        # Check for forbidden functions:
        for name, func in self.scope.items():
            if callable(func):
                try:
                    hash(func)
                except TypeError:
                    raise DangerousValue(
                        "This function '{}' in scope might be a bad idea.".format(name)
                    )
                if func in DISALLOW_FUNCTIONS:
                    raise DangerousValue(
                        "This function '{}' in scope is {} and is in DISALLOW_FUNCTIONS".format(
                            name, func
                        )
                    )

    def eval(self, expr):
        """ evaluate an expresssion, using the operators, functions and
            scope previously set up. """

        # set a copy of the expression aside, so we can give nice errors...

        self.expr = expr

        # and evaluate:
        return self._eval(ast.parse(expr))

    def _eval(self, node):
        """ The internal evaluator used on each node in the parsed tree. """

        self.track(node)
        try:
            handler = self.nodes[type(node)]
        except KeyError:
            raise FeatureNotAvailable(
                "Sorry, {0} is not available in this "
                "evaluator".format(type(node).__name__),
                node,
            )
        # try:
        node.call_stack = self.call_stack
        self._last_eval_result = handler(node)
        self.track(node)
        return self._last_eval_result
        # except (ArithmeticError) as exc:
        #     raise SnekArithmeticError(str(exc), node)
        # except ValueError as exc:
        #     raise SnekValueError(str(exc), node)
        # except TypeError as e:
        #     raise SnekTypeError(str(e), node)
        # exec is special and requires the current stack
        # if scope.get('exec', None) is PlaceHolder:
        #    scope['exec'] = get_safe_exec(self.call_stack)

    def _eval_assert(self, node):
        if not self._eval(node.test):
            raise SnekAssertionError(self._eval(node.msg), node)

    def _eval_while(self, node):
        while self._eval(node.test):
            try:
                for b in node.body:
                    self._eval(b)
            except Break:
                break
        else:
            for b in node.orelse:
                self._eval(b)

    def _eval_for(self, node):
        def recurse_targets(target, value):
            """
                Recursively (enter, (into, (nested, name), unpacking)) = \
                             and, (assign, (values, to), each
            """
            self.track(target)
            if isinstance(target, ast.Name):
                self.scope[target.id] = value
            else:
                for t, v in zip(target.elts, value):
                    recurse_targets(t, v)

        for v in self._eval(node.iter):
            recurse_targets(node.target, v)
            try:
                for b in node.body:
                    self._eval(b)
            except Break:
                break
        else:
            for b in node.orelse:
                self._eval(b)

    def _eval_import(self, node):
        for alias in node.names:
            asname = alias.asname or alias.name
            try:
                self.scope[asname] = self.modules[alias.name]
            except KeyError:
                raise SnekImportError(alias.name, node)

    def _eval_importfrom(self, node):
        for alias in node.names:
            asname = alias.asname or alias.name
            try:
                self.scope[asname] = self.modules[node.module].__dict__[alias.name]
            except KeyError:
                raise SnekImportError(alias.name, node)

    def _eval_expr(self, node):
        return self._eval(node.value)

    def _eval_module(self, node):
        return [self._eval(b) for b in node.body]

    def _eval_arguments(self, node):

        if node.vararg:
            raise FeatureNotAvailable("Sorry, VarArgs are not available", node.vararg)

        if node.kwarg:
            raise FeatureNotAvailable("Sorry, VarKwargs are not available", node.kwarg)
        NONEXISTANT_DEFAULT = object()  # a unique object to contrast with None
        args_and_defaults = []
        for (arg, default) in itertools.zip_longest(
            node.args[::-1], node.defaults[::-1], fillvalue=NONEXISTANT_DEFAULT
        ):
            if default is NONEXISTANT_DEFAULT:
                args_and_defaults.append(forge.arg(arg.arg))
            else:
                args_and_defaults.append(
                    forge.arg(arg.arg, default=self._eval(default))
                )
        args_and_defaults.reverse()

        return {
            "args": args_and_defaults,
            # "vargs": node.vararg and forge.args(node.vararg.arg) or [],
            # "kwargs": {arg.arg: forge.kwarg() for arg in node.kwonlyargs},
            # "varkwargs": node.kwarg and forge.kwargs(node.kwarg.arg) or {}
        }

    def _eval_break(self, node):
        raise Break()

    def _eval_return(self, node):
        ret = None
        if node.value is not None:
            ret = self._eval(node.value)
        raise Return(ret)

    def _eval_functiondef(self, node):

        sig_obj = self._eval(node.args)
        _class = self.__class__

        @forge.sign(*sig_obj["args"])
        def _func(**local_scope):
            s = _class(
                modules=self.modules,
                scope={**self.scope, **local_scope},
                call_stack=self.call_stack,
            )
            s.expr = self.expr
            for b in node.body:
                try:
                    s._eval(b)
                except Return as r:
                    return r.value
                finally:
                    self.track(s)

        _func.__name__ = node.name

        # prevent unwrap from detecting this nested function
        del _func.__wrapped__
        _func.__doc__ = ast.get_docstring(node)
        for decorator_node in node.decorator_list[::-1]:
            decorator = self._eval(decorator_node)
            _func = decorator(_func)

        self.scope[node.name] = _func

    def _assign_name(self, node, value):
        self.scope[node.id] = value
        return value

    def _assign_subscript(self, node, value):
        _slice = self._eval(node.slice)
        self._eval(node.value)[_slice] = value
        return value

    def _delete(self, targets):
        if len(targets) > 1:
            raise FeatureNotAvailable(
                "Sorry, cannot delete {} targets.".format(len(targets)), targets[0]
            )
        target = targets[0]
        try:
            handler = self.deletions[type(target)]
            handler(target)
        except KeyError:
            raise FeatureNotAvailable(
                "Sorry, cannot delete {}".format(type(target).__name__), target
            )

    def _delete_name(self, node):
        del self.scope[node.id]

    def _delete_subscript(self, node):
        _slice = self._eval(node.slice)
        del self._eval(node.value)[_slice]

    def _eval_delete(self, node):
        return self._delete(node.targets)

    def _eval_raise(self, node):
        exc = self._eval(node.exc)
        exc.node = node
        if node.cause is not None:
            cause = self._eval(node.cause)
            raise exc from cause
        raise exc

    def _assign(self, targets, value):
        if len(targets) > 1:
            raise FeatureNotAvailable(
                "Sorry, cannot assign to {} targets.".format(len(targets)), targets[0]
            )
        target = targets[0]
        try:
            handler = self.assignments[type(target)]
            handler(target, value)
        except KeyError:
            raise FeatureNotAvailable(
                "Sorry, cannot assign to {}".format(type(target).__name__), target
            )

    def _eval_assign(self, node):
        value = self._eval(node.value)
        return self._assign(node.targets, value)

    @staticmethod
    def _eval_num(node):
        return node.n

    @staticmethod
    def _eval_bytes(node):
        if len(node.s) > MAX_STRING_LENGTH:
            raise IterableTooLong(
                "Byte Literal in statement is too long!"
                " ({0}, when {1} is max)".format(len(node.s), MAX_STRING_LENGTH),
                node,
            )
        return node.s

    @staticmethod
    def _eval_str(node):
        if len(node.s) > MAX_STRING_LENGTH:
            raise IterableTooLong(
                "String Literal in statement is too long!"
                " ({0}, when {1} is max)".format(len(node.s), MAX_STRING_LENGTH),
                node,
            )
        return node.s

    @staticmethod
    def _eval_nameconstant(node):
        return node.value

    def _eval_unaryop(self, node):
        return self.operators[type(node.op)](self._eval(node.operand))

    def _eval_binop(self, node):
        try:
            return self.operators[type(node.op)](
                self._eval(node.left), self._eval(node.right)
            )
        except ValueError as exc:  # pragma: no cover
            # Is this possible?
            raise SnekValueError(str(exc), node)
        except TypeError as e:
            raise SnekTypeError(str(e), node)
        except ArithmeticError as exc:
            raise SnekArithmeticError(str(exc), node)
        except KeyError:
            raise FeatureNotAvailable(
                "Sorry, {0} is not available in this "
                "evaluator".format(type(node.op).__name__),
                node,
            )

    def _eval_boolop(self, node):
        if isinstance(node.op, ast.And):
            vout = False
            for value in node.values:
                vout = self._eval(value)
                if not vout:
                    return vout
            return vout
        elif isinstance(node.op, ast.Or):
            for value in node.values:
                vout = self._eval(value)
                if vout:
                    return vout
            return vout
        else:  # pragma: no cover
            # This should never happen as there are only two bool operators And and Or
            raise FeatureNotAvailable(
                "Sorry, {0} is not available in this "
                "evaluator".format(type(node).__name__),
                node,
            )

    def _eval_compare(self, node):
        right = self._eval(node.left)
        to_return = True
        for operation, comp in zip(node.ops, node.comparators):
            if not to_return:
                break
            left = right
            right = self._eval(comp)
            to_return = self.operators[type(operation)](left, right)
        return to_return

    def _eval_ifexp(self, node):
        return (
            self._eval(node.body) if self._eval(node.test) else self._eval(node.orelse)
        )

    def _eval_if(self, node):
        if self._eval(node.test):
            [self._eval(b) for b in node.body]
        else:
            [self._eval(b) for b in node.orelse]

    def _eval_try(self, node):
        try:
            for b in node.body:
                self._eval(b)
        except:  # noqa: E722
            caught = False
            for h in node.handlers:
                if self._eval(h):
                    caught = True
                    break
            if not caught:
                raise
        else:
            [self._eval(oe) for oe in node.orelse]
        finally:
            [self._eval(f) for f in node.finalbody]

    def _eval_excepthandler(self, node):
        _type, exc, traceback = sys.exc_info()
        if (node.type is None) or isinstance(exc, self._eval(node.type)):
            if node.name:
                self.scope[node.name] = exc
            [self._eval(b) for b in node.body]
            return True
        return False

    def _eval_call(self, node):
        if len(self.call_stack) >= MAX_CALL_DEPTH:
            # self.call_stack[:] = []  # need stack to pre
            raise CallTooDeep(
                "Sorry, stack is to large. The MAX_CALL_DEPTH is {}.".format(
                    MAX_CALL_DEPTH
                ),
                node,
            )
        func = self._eval(node.func)
        if not callable(func):
            raise SnekRuntimeError(
                "Sorry, {} type is not callable".format(type(func).__name__), node
            )
        func_hash = None
        try:
            func_hash = hash(func)
        except TypeError:
            if func.__qualname__ not in WHITLIST_ATTRIBUTES:
                raise FeatureNotAvailable(
                    "this function is not allowed: {}".format(func.__qualname__), node
                )
        if func_hash and func in DISALLOW_FUNCTIONS:
            raise FeatureNotAvailable("This function is forbidden", node)
        if (
            func_hash
            and isinstance(func, types.BuiltinFunctionType)
            and func.__qualname__ not in ALLOWED_BUILTINS
        ):
            raise FeatureNotAvailable(
                f"This builtin function is not allowed: {func.__qualname__}", node
            )
        kwarg_kwargs = [self._eval(k) for k in node.keywords]

        # some functions need the current context
        # if type(func) is SnekDeferred:
        #    func = func(self)

        f = func
        for a in node.args:
            if a.__class__ == ast.Starred:
                args = self._eval(a.value)
            else:
                args = [self._eval(a)]
            f = partial(f, *args)
        for kwargs in kwarg_kwargs:
            f = partial(f, **kwargs)

        self.call_stack.append([node, self.expr])
        ret = f()
        self.call_stack.pop()
        return ret

    def _eval_keyword(self, node):
        if node.arg is not None:
            return {node.arg: self._eval(node.value)}
        # Not possible until kwargs are enabled
        return self._eval(node.value)

    def _eval_name(self, node):
        try:
            return self.scope[node.id]
        except KeyError:
            raise NameNotDefined(node)

    def _eval_subscript(self, node):
        container = self._eval(node.value)
        key = self._eval(node.slice)
        try:
            return container[key]
        except TypeError as e:
            raise SnekTypeError(str(e), node)
        except (LookupError) as exc:
            raise SnekLookupError(str(exc), node)

    def _eval_attribute(self, node):
        for prefix in DISALLOW_PREFIXES:
            if node.attr.startswith(prefix):
                raise FeatureNotAvailable(
                    "Sorry, access to this attribute "
                    "is not available. "
                    "({0})".format(node.attr),
                    node,
                )
        # eval node
        node_evaluated = self._eval(node.value)
        if (type(node_evaluated), node.attr) in DISALLOW_METHODS:
            raise FeatureNotAvailable(
                "Sorry, this method is not available. "
                "({0}.{1})".format(node_evaluated.__class__.__name__, node.attr),
                node,
            )
        try:
            return getattr(node_evaluated, node.attr)
        except AttributeError as e:
            raise SnekAttributeError(str(e), node)

    def _eval_index(self, node):
        return self._eval(node.value)

    def _eval_slice(self, node):
        lower = upper = step = None
        if node.lower is not None:
            lower = self._eval(node.lower)
        if node.upper is not None:
            upper = self._eval(node.upper)
        if node.step is not None:
            step = self._eval(node.step)
        return slice(lower, upper, step)

    def _eval_joinedstr(self, node):
        length = 0
        evaluated_values = []
        for n in node.values:
            val = str(self._eval(n))
            if len(val) + length > MAX_STRING_LENGTH:
                raise IterableTooLong(
                    "Sorry, I will not evaluate something this long.", node
                )
            length += len(val)
            evaluated_values.append(val)
        return "".join(evaluated_values)

    def _eval_formattedvalue(self, node):
        if node.format_spec:
            # from https://stackoverflow.com/a/44553570/260366
            from collections import namedtuple as nt
            import re

            format_spec = self._eval(node.format_spec)
            r = r"(([\s\S])?([<>=\^]))?([\+\- ])?([#])?([0])?(\d*)([,])?((\.)(\d*))?([sbcdoxXneEfFgGn%])?"
            FormatSpec = nt(
                "FormatSpec",
                "fill align sign alt zero_padding width comma decimal precision type",
            )
            match = re.fullmatch(r, format_spec)

            if match:
                parsed_spec = FormatSpec(
                    *match.group(2, 3, 4, 5, 6, 7, 8, 10, 11, 12)
                )  # skip groups not interested in
                if int(parsed_spec.width or 0) > 100:
                    raise SnekRuntimeError(
                        "Sorry, this format width is too long.", node.format_spec
                    )

                if int(parsed_spec.precision or 0) > 100:
                    raise SnekRuntimeError(
                        "Sorry, this format precision is too long.", node.format_spec
                    )

            fmt = "{:" + format_spec + "}"
            return fmt.format(self._eval(node.value))
        return self._eval(node.value)

    def _eval_dict(self, node):
        if len(node.keys) > MAX_STRING_LENGTH:
            raise IterableTooLong(
                "Dict in statement is too long!"
                " ({0}, when {1} is max)".format(len(node.keys), MAX_STRING_LENGTH),
                node,
            )
        return {self._eval(k): self._eval(v) for (k, v) in zip(node.keys, node.values)}

    def _eval_tuple(self, node):
        if len(node.elts) > MAX_STRING_LENGTH:
            raise IterableTooLong(
                "Tuple in statement is too long!"
                " ({0}, when {1} is max)".format(len(node.elts), MAX_STRING_LENGTH),
                node,
            )
        return tuple(self._eval(x) for x in node.elts)

    def _eval_list(self, node):
        if len(node.elts) > MAX_STRING_LENGTH:
            raise IterableTooLong(
                "List in statement is too long!"
                " ({0}, when {1} is max)".format(len(node.elts), MAX_STRING_LENGTH),
                node,
            )
        return list(self._eval(x) for x in node.elts)

    def _eval_set(self, node):
        return set(self._eval(x) for x in node.elts)

    def track(self, node):
        if hasattr(node, "nodes_called"):
            return

        self.nodes_called += 1
        if self.nodes_called > MAX_NODE_CALLS:
            raise TooManyEvaluations("This program has too many evaluations", node)
        # seen = dict()
        # size = get_size([self.scope, self.return_values], seen)
        size = len(str(self.scope)) + len(str(self._last_eval_result))
        if size > MAX_SCOPE_SIZE:
            raise ScopeTooLarge(
                f"Scope has used too much memory: { size } > {MAX_SCOPE_SIZE}", node
            )

    def _eval_comprehension(self, node):

        if isinstance(node, ast.ListComp) or isinstance(node, ast.GeneratorExp):
            to_return = list()
        elif isinstance(node, ast.DictComp):
            to_return = dict()
        elif isinstance(node, ast.SetComp):
            to_return = set()
        else:  # pragma: no cover
            raise Exception("should never happen")

        extra_scope = {}

        previous_name_evaller = self.nodes[ast.Name]

        def eval_scope_extra(node):
            """
                Here we hide our extra scope for within this comprehension
            """
            if node.id in extra_scope:
                return extra_scope[node.id]
            return previous_name_evaller(node)

        self.nodes.update({ast.Name: eval_scope_extra})

        def recurse_targets(target, value):
            """
                Recursively (enter, (into, (nested, name), unpacking)) = \
                             and, (assign, (values, to), each
            """
            self.track(target)
            if isinstance(target, ast.Name):
                extra_scope[target.id] = value
            else:
                for t, v in zip(target.elts, value):
                    recurse_targets(t, v)

        def do_generator(gi=0):
            g = node.generators[gi]
            if len(g.ifs) > 1:
                raise FeatureNotAvailable(
                    "Sorry, only one `if` allowed in list comprehension, consider booleans or a function",
                    node,
                )

            for i in self._eval(g.iter):
                recurse_targets(g.target, i)
                if all(self._eval(iff) for iff in g.ifs):
                    if len(node.generators) > gi + 1:
                        do_generator(gi + 1)
                    else:
                        if isinstance(node, ast.ListComp) or isinstance(
                            node, ast.GeneratorExp
                        ):
                            to_return.append(self._eval(node.elt))
                        elif isinstance(node, ast.DictComp):
                            to_return[self._eval(node.key)] = self._eval(node.value)
                        elif isinstance(node, ast.SetComp):
                            to_return.add(self._eval(node.elt))
                        else:  # pragma: no cover
                            raise Exception("should never happen")

        do_generator()

        self.nodes.update({ast.Name: previous_name_evaller})

        return to_return


def snek_eval(expr, scope=None, call_stack=None, module_dict=None):
    """ Simply evaluate an expresssion """

    modules = None
    if module_dict:
        modules = make_modules(module_dict)

    s = SnekEval(scope=scope, modules=modules, call_stack=call_stack)
    return s.eval(expr)


class SnekCoverage(SnekEval):

    seen_nodes = defaultdict(int)

    def __init__(self, *args, **kwargs):
        return super(SnekCoverage, self).__init__(*args, **kwargs)

    def eval(self, expr):
        self.seen_nodes = {
            (n.lineno, n.col_offset, n.__class__.__name__): 0
            for n in ast.walk(ast.parse(expr))
            if hasattr(n, "col_offset")
        }
        return super(SnekCoverage, self).eval(expr)

    def _assign(self, targets, value):
        ret = super(SnekCoverage, self)._assign(targets, value)
        # currently only one target is allowed, but still.
        for node in targets:
            self.track(node)
        return ret

    def _eval(self, node):
        ret = super(SnekCoverage, self)._eval(node)
        self.track(node)
        return ret

    def _eval_arguments(self, node):
        ret = super(SnekCoverage, self)._eval_arguments(node)
        for node_arg in node.args:
            self.track(node_arg)
        return ret

    def track(self, node):
        if hasattr(node, "seen_nodes"):
            xx = Counter(node.seen_nodes)
            yy = Counter(self.seen_nodes)
            xx.update(yy)
            self.seen_nodes = dict(xx)
        if hasattr(node, "col_offset"):
            self.seen_nodes[
                (node.lineno, node.col_offset, node.__class__.__name__)
            ] += 1


def snek_test_coverage(expr, scope=None, call_stack=None, module_dict=None):
    """ Simply evaluate an expresssion """

    modules = make_modules(module_dict or {})

    s = SnekCoverage(scope=scope, modules=modules, call_stack=call_stack)
    s.eval(expr)
    test_names = [n for n in s.scope if n.startswith("test_") and callable(s.scope[n])]
    for name in test_names:
        s.scope[name]()
    return sorted(s.seen_nodes.items())


def ascii_format_coverage(coverage, source):
    pct = sum(v > 0 for k, v in coverage) / len(coverage)
    # total = sum(v for k, v in coverage)
    out = ""
    for (r, c, name), v in coverage:
        if v:
            continue
        out += f"Missing {name} on line: {r} col: {c}\n"
        out += (source.splitlines()[r - 1]) + "\n"
        out += (c * "-") + "^\n"
    out += f"{ int(pct * 100) }% coverage\n"
    return out


def get_size(obj, seen):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen[obj_id] = True
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    if hasattr(obj, "__dict__"):
        for cls in obj.__class__.__mro__:  # pragma: no cover
            if "__dict__" in cls.__dict__:
                d = cls.__dict__["__dict__"]
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(
                    d
                ):  # pragma: no cover
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, "__iter__") and not isinstance(
        obj, (str, bytes, bytearray, type)
    ):
        size += sum((get_size(i, seen) for i in obj))

    if hasattr(obj, "__slots__"):  # can have __slots__ with __dict__
        size += sum(
            get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s)
        )

    return size
