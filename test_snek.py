import pytest
import sys

import sneklang
from sneklang import *


def snek_is_still_python(code, snek_scope=None):
    """
    Here we run the code in python exec, then in snek_eval, then compare the `scope['result']`
    """
    py_scope = {}
    snek_scope = snek_scope or {}
    exec(code, py_scope)
    print("py_scope", py_scope["result"])
    print("snek_eval output", snek_eval(code, scope=snek_scope))
    print("snek_scope", snek_scope["result"])
    assert py_scope["result"] == snek_scope["result"], code


def test_snek_comprehension_python():
    CASES = [
        """
non_flat= [ [1,2,3], [4,5,6], [7,8] ]
result = [y for x in non_flat for y in x]
""",
        """
a = 'foo'
l = [a for a in [1,2,3]]
result = a, l
""",
        """
a = {'b':['c', 'd', 'e']}
a['b'][1] =  'D'
result = a
""",
        """
a = [1,2,3,4,5,6]
a[2:5] = ['a', 'b']
result = a
""",
        """
a = [1,2,3,4,5,6,7,8]
a[::2] = [0,0,0,0]
result = a
""",
        """
a = [1,2,3,4,5,6,7,8]
del a[3:6:-1]
result = a
""",
        """
a = [1]
a = [a, a, a, a]
result = a[2]""",
        """
a = [(1,1), (2,2), (3,3)]
result = [(x,y) for x,y in a]
""",
        """
a = [1,2,3,4,5,6,7,8,9,10]
result = [x for x in a if x % 2]
""",
        """
d = {"a": 1, "b": 2, "c": 3}
result = list(d.items()).sort()
""",
    ]
    for code in CASES:
        snek_is_still_python(code)


def test_snek_assertions():
    code = """
assert True, "no"
result = 1"""
    snek_is_still_python(code)

    code = """
result = 1
try:
    assert False, "no"
except:
    pass
result = 2"""
    snek_is_still_python(code)

    code = """
result = 1
try:
    assert False
except:
    pass
result = 2"""
    snek_is_still_python(code)


def test_assignment():
    code = """
a = b,c = 1,2
result = a,b,c """
    snek_is_still_python(code)


def test_augassign():
    for operator in ["+=", "-=", "/=", "//=", "%=", "!=", "*=", "^="]:
        code = f"""
result = 110
result {operator} 21
    """
        snek_is_still_python(code)


def test_snek_delete():
    code = """
a = 1
b = [a, a]
del a
result = [b, b]"""
    snek_is_still_python(code)


def test_snek_kwargs():
    code = """
def foo(a,b,c,d,e,f,g):
    return (a,b,c,d,e,f,g)

result = foo(1, *[2,3], *[4], e=5, **{'f':6}, **{'g':7} )
"""
    snek_is_still_python(code)


def test_args():
    code = """
def standard_arg(arg):
    return arg
result = standard_arg("a") + standard_arg(arg="b")"""
    snek_is_still_python(code)

    code = """
def standard_arg(arg=1):
    return arg
result = standard_arg("a") + standard_arg(arg="b")"""
    snek_is_still_python(code)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
def test_pos_args():
    code = """
def pos_only_arg(arg, /):
    return arg

result = pos_only_arg("a")"""
    snek_is_still_python(code)

    code = """
def pos_only_arg(arg=1, /):
    return arg

result = pos_only_arg("a")"""
    snek_is_still_python(code)


def test_kwd_only_arg():
    code = """
def kwd_only_arg(*, arg):
    return arg

result = kwd_only_arg(arg=42)"""
    snek_is_still_python(code)

    code = """
def kwd_only_arg(*, arg="foo"):
    return arg

result = kwd_only_arg(arg=42)"""
    snek_is_still_python(code)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
def test_func_combined():
    code = """
def combined_example(pos_only, /, standard, *, kwd_only, **kwgs):
    return (pos_only, standard, kwd_only, kwgs)
result = combined_example(1, 2, kwd_only=3, foo=[1,2,3])"""
    snek_is_still_python(code)


def test_snek_lambda():
    code = """
foo = lambda a,b,c,d,e,f,g: (a,b,c,d,e,f,g)
result = foo(1, *[2,3], *[4], e=5, **{'f':6}, **{'g':7} )
"""
    snek_is_still_python(code)


def test_snek_is_python_closure():
    code = """
a = 0
def nest(b):
    a = 4
    def _inner(c):
        a = 5
        def __inner(d):
            return a,b,c,d
        return __inner
    return _inner
result = nest(1)(2)(3)
result
"""
    snek_is_still_python(code)


def test_snek_is_python_ifs():
    snek_is_still_python(
        """
s = 'the answer is '
def foo(i):
    if i < 0:
        res = s + 'too low'
    elif i > 0:
        res = s + 'too high'
    else:
        res = s + 'just right'
    return res

result = [foo(i) for i in [-1,0,1]]
            """
    )


def test_snek_is_python_while():
    snek_is_still_python(
        """
result = []
n = 10
while n:
    n = n -1
    result = result + [n]
"""
    )
    snek_is_still_python(
        """
result = []
n = 0
while n:
    n = n -1
    result = result + [n]
else:
    result = 'nonono'
"""
    )
    snek_is_still_python(
        """
result = []
n = 0
while n < 10:
    n = n + 1
    result = result + [n]
    if n > 5:
        break
"""
    )

    snek_is_still_python(
        """
result = []
n = 0
while n < 10:
    n = n + 1
    result = result + [n]
    if n == 5:
        continue
"""
    )


def test_snek_is_python_for():
    snek_is_still_python(
        """
def evens(s):
    out = ''
    try:
        for i, c in s:
            if i % 2 == 0:
                out = out + c
        else:
            out = -1
    except:
        return False
    return out

result = [evens(s) for s in
[[(0, 'a'),
 (1, 'b'),
 (2, 'c'),
 (3, 'd'),
 (4, 'e'),
 (5, 'f'),
 (6, 'g'),
 (7, 'h')],[], 'asdf'] ]
"""
    )
    snek_is_still_python(
        """
result = []
for n in [1,2,3,4,5,6,7,8,10]:
    result = result + [n]
    if n > 5:
        break
"""
    )

    snek_is_still_python(
        """
result = []
for n in [1,2,3,4,5,6,7,8,10]:
    result = result + [n]
    if n == 5:
        continue
"""
    )


def test_decorators():
    snek_is_still_python(
        """
result = []
def my_decorator(log):
    result.append(f"d1 {log}")
    def _wrapper(func):
        result.append(f"_wrapper {log}")
        def _inner(*args, **kwargs):
            result.append(f"inner {log}")
            func(*args, **kwargs)
        return _inner
    return _wrapper

@my_decorator(1)
@my_decorator(2)
def say(word=None):
    result.append(word or "hi world!")
say()
say('ola mundo')
"""
    )


def test_snek_is_python_try():
    snek_is_still_python(
        """
def foo(a):
    res = 'try|'
    try:
        try:
            res = res + str(1/a) + "|"
        except ArithmeticError as e:
            res= res + 'ArithmeticError|'
    except Exception as e2:
        return 'oops|'
    else:
        res = res + 'else|'
    finally:
        res = res + 'fin|'
    return a, res
result = [foo(i) for i in [-1,0,1, 'a']]
""",
        snek_scope={"Exception": Exception, "ArithmeticError": ArithmeticError},
    )


EXCEPTION_CASES = [
    ("nope", {}, "SnekRuntimeError('NameError(\"\\'nope\\' is not defined\")')"),
    (
        "a=1; a.b",
        {},
        "SnekRuntimeError('AttributeError(\"\\'int\\' object has no attribute \\'b\\'\")')",
    ),
    ("1/0", {}, "SnekRuntimeError(\"ZeroDivisionError('division by zero')\")"),
    (
        "len(str(10000 ** 10001))",
        {},
        "SnekRuntimeError('MemoryError(\"Sorry! I don\\'t want to evaluate 10000 ** 10001\")')",
    ),
    (
        "'aaaa' * 200000",
        {},
        "SnekRuntimeError(\"MemoryError('Sorry, I will not evalute something that long.')\")",
    ),
    (
        "200000 * 'aaaa'",
        {},
        "SnekRuntimeError(\"MemoryError('Sorry, I will not evalute something that long.')\")",
    ),
    (
        "(10000 * 'world!') + (10000 * 'world!')",
        {},
        "SnekRuntimeError(\"MemoryError('Sorry, adding those two together would make something too long.')\")",
    ),
    (
        "4 @ 3",
        {},
        "SnekRuntimeError(\"NotImplementedError('Sorry, MatMult is not available in this evaluator')\")",
    ),
    (
        "",
        {"open": open},
        # dfferences in how python version report the error
        "DangerousValue",
    ),
    (
        "a.clear()",
        {"a": []},
        # dfferences in how python version report the error
        "NotImplementedError",
    ),
    (
        "a @= 3",
        {},
        "SnekRuntimeError(\"NotImplementedError('Sorry, MatMult is not available in this evaluator')\")",
    ),
    ("int.mro()", {}, 'SnekRuntimeError("DangerousValue'),
    (repr("a" * 100001), {}, 'SnekRuntimeError("MemoryError'),
    (
        repr({i: 1 for i in range(1000001)}),
        {},
        "SnekRuntimeError(\"MemoryError('Dict in statement is too long!')\")",
    ),
    (
        repr(tuple(range(1000001))),
        {},
        "SnekRuntimeError(\"MemoryError('Tuple in statement is too long!')\")",
    ),
    (
        repr(set(range(1000001))),
        {},
        "SnekRuntimeError(\"MemoryError('Set in statement is too long!')\")",
    ),
    ("b'" + ("a" * 100001) + "'", {}, 'SnekRuntimeError("MemoryError('),
    (("1" + "0" * sneklang.MAX_STRING_LENGTH), {}, 'SnekRuntimeError("MemoryError('),
    (
        repr(list("a" * 100001)),
        {},
        "SnekRuntimeError(\"MemoryError('List in statement is too long!')\")",
    ),
    ("1()", {}, "SnekRuntimeError(\"TypeError('Sorry, int type is not callable')\")"),
    (
        "forbidden_func()[0]()",
        {"forbidden_func": lambda: [type]},
        "SnekRuntimeError(\"DangerousValue('This function is forbidden: type')\")",
    ),
    (
        "a()([])",
        {"a": lambda: sorted},
        "SnekRuntimeError(\"NotImplementedError('This builtin function is not allowed: sorted')\")",
    ),
    ("a[1]", {"a": []}, "SnekRuntimeError(\"IndexError('list index out of range')\")"),
    (
        "a.__length__",
        {"a": []},
        "SnekRuntimeError(\"NotImplementedError('Sorry, access to this attribute is not available. (__length__)')\")",
    ),
    (
        "'say{}'.format('hi') ",
        {},
        "SnekRuntimeError(\"DangerousValue('Sorry, this method is not available. (str.format)')\")",
    ),
    (
        "class A: 1",
        {},
        "SnekRuntimeError(\"NotImplementedError('Sorry, ClassDef is not available in this evaluator')\")",
    ),
    (
        "a.b",
        {"a": object()},
        "SnekRuntimeError('AttributeError(\"\\'object\\' object has no attribute \\'b\\'\")')",
    ),
    (
        "'a' + 1",
        {},
        "SnekRuntimeError('TypeError(\\'can only concatenate str (not \"int\") to str\\')')",
    ),
    (
        "import non_existant",
        {},
        "SnekRuntimeError(\"ModuleNotFoundError('non_existant')\")",
    ),
    (
        "from nowhere import non_existant",
        {},
        "SnekRuntimeError(\"ModuleNotFoundError('non_existant')\")",
    ),
    ('assert False, "no"', {}, "SnekRuntimeError(\"AssertionError('no')\")"),
    (
        "del a,b,c",
        {},
        "SnekRuntimeError(\"NotImplementedError('Sorry, cannot delete 3 targets.')\")",
    ),
    (
        "del a.c",
        {},
        "SnekRuntimeError(\"NotImplementedError('Sorry, cannot delete Attribute')\")",
    ),
    (
        "[1,2,3][[]]",
        {},
        "SnekRuntimeError(\"TypeError('list indices must be integers or slices, not list')\")",
    ),
    (
        "1<<1",
        {},
        "SnekRuntimeError(\"NotImplementedError('Sorry, LShift is not available in this evaluator')\")",
    ),
    ("assert False", {}, "SnekRuntimeError('AssertionError()')"),
    ("assert False, 'oh no'", {}, "SnekRuntimeError(\"AssertionError('oh no')\")"),
    (
        "(a for a in a)",
        {},
        "SnekRuntimeError(\"NotImplementedError('Sorry, GeneratorExp is not available in this evaluator')\")",
    ),
    (
        "vars(object)",
        {"vars": vars},
        "DangerousValue(\"This function 'vars' in scope is <built-in function vars> and is in DISALLOW_FUNCTIONS\")",
    ),
    (
        "a, b = 1",
        {},
        "SnekRuntimeError(\"TypeError('cannot unpack non-iterable int object')\")",
    ),
    (
        "a, b = 1, 2, 3",
        {},
        "SnekRuntimeError(\"ValueError('too many values to unpack (expected 2)')\")",
    ),
    (
        "a, b, c = 1, 2",
        {},
        "SnekRuntimeError(\"ValueError('not enough values to unpack (expected 3, got 2)')\")",
    ),
    (
        "f'{1:<1000}'",
        {},
        "SnekRuntimeError(\"MemoryError('Sorry, this format width is too long.')\")",
    ),
    (
        "f'{1/3:.1000}'",
        {},
        "SnekRuntimeError(\"MemoryError('Sorry, this format precision is too long.')\")",
    ),
    (
        "f'{1:a}'",
        {},
        "SnekRuntimeError('ValueError(\"Unknown format code \\'a\\' for object of type \\'int\\'\")')",
    ),
]


@pytest.mark.skipif(sys.version_info >= (3, 8), reason="Old way of checking functions")
def test_old_dangerous_values():
    with pytest.raises(sneklang.DangerousValue) as excinfo:
        snek_eval("a", scope={"a": {}.keys})
    assert (
        repr(excinfo.value)
        == "DangerousValue(\"This function 'a' in scope might be a bad idea.\")"
    )


@pytest.mark.filterwarnings("ignore::SyntaxWarning")
def test_exceptions():
    for i, (code, scope, ex_repr) in enumerate(EXCEPTION_CASES):
        try:
            out = snek_eval(code, scope=scope)
        except Exception as e:
            exc = e
            assert ex_repr in repr(
                exc
            ), f"{repr(repr(exc))}\nFailed {code} \nin CASE {i}"
            continue
        pytest.fail("{}\nneeded to raise: {}\nreturned: {}".format(code, ex_repr, out))


def test_smoketests():

    CASES = [
        ("1 + 1", [2]),
        ("1 and []", [[]]),
        ("None or []", [[]]),
        ("3 ** 3", [27]),
        ("len(str(1000 ** 1000))", [3001]),
        ("True != False", [True]),
        ("None is None", [True]),
        ("True is not None", [True]),
        ("'a' in 'abc'", [True]),
        ("'d' not in 'abc'", [True]),
        ("- 1 * 2", [-2]),
        ("True or False", [True]),
        ("1 > 2 > 3", [False]),
        ("'abcd'[1]", ["b"]),
        ("'abcd'[1:3]", ["bc"]),
        ("'abcd'[:3]", ["abc"]),
        ("'abcd'[2:]", ["cd"]),
        ("'abcdefgh'[::3]", ["adg"]),
        ("('abc' 'xyz')", ["abcxyz"]),
        ("(b'abc' b'xyz')", [b"abcxyz"]),
        ("f'{1 + 2}'", ["3"]),
        ("{'a': 1}['a']", [1]),
        (repr([1] * 100), [[1] * 100]),
        (repr(set([1, 2, 3, 3, 3])), [set([1, 2, 3])]),
        ("[a + 1 for a in [1,2,3]]", [[2, 3, 4]]),
        ("[a + 1 for a in [1,2,3]]", [[2, 3, 4]]),
        ("{'a': 1}.get('a')", [1]),
        ("{'a': 1}.items()", [{"a": 1}.items()]),
        ("{'a': 1}.keys()", [{"a": 1}.keys()]),
        ("list({'a': 1}.values())", [[1]]),
        ("[a for a in [1,2,3,4,5,6,7,8,9,10] if a % 2 if a % 5]", [[1, 3, 7, 9]]),
    ]
    for code, out in CASES:
        assert snek_eval(code) == out, f"{code} should equal {out}"
        # Verify evaluates same as python
        assert [eval(code)] == out, code


def test_call_stack():
    scope = {}
    snek_eval("def foo(x): return x, x > 0 and foo(x-1) or 0", scope=scope)
    scope["foo"](3)
    with pytest.raises(SnekRuntimeError) as excinfo:
        scope["foo"](50)
    assert (
        repr(excinfo.value)
        == "SnekRuntimeError(\"RecursionError('Sorry, stack is to large')\")"
    )
    # test fake call stack
    snek_eval(
        "def foo(x): return foo(x - 1) if x > 0 else 0",
        scope=scope,
        call_stack=30 * [1],
    )
    with pytest.raises(SnekRuntimeError) as excinfo:
        scope["foo"](3)
    assert (
        repr(excinfo.value)
        == "SnekRuntimeError(\"RecursionError('Sorry, stack is to large')\")"
    )


def test_settings():
    orig = sneklang.MAX_NODE_CALLS
    with pytest.raises(SnekRuntimeError) as excinfo:
        scope = {}
        sneklang.MAX_NODE_CALLS = 20
        snek_eval("while True: 1", scope=scope)
    assert (
        repr(excinfo.value)
        == "SnekRuntimeError(\"TimeoutError('This program has too many evaluations')\")"
    )
    sneklang.MAX_NODE_CALLS = orig

    orig = sneklang.MAX_SCOPE_SIZE
    with pytest.raises(SnekRuntimeError) as excinfo:
        scope = {}
        sneklang.MAX_SCOPE_SIZE = 500
        snek_eval("a=[]\nwhile True: a=[a, a]", scope=scope)
    assert (
        repr(excinfo.value)
        == "SnekRuntimeError(\"MemoryError('Scope has used too much memory')\")"
    )
    sneklang.MAX_SCOPE_SIZE = orig


def test_importing():
    assert (
        snek_eval("import a as c; c", module_dict={"a": {"b": "123"}})[
            1
        ].__class__.__name__
        == "module"
    )
    assert snek_eval("from a import b as c; c", module_dict={"a": {"b": "123"}}) == [
        None,
        "123",
    ]
    with pytest.raises(SnekRuntimeError) as excinfo:
        snek_eval("from a import d", module_dict={"a": {"b": "123"}})
    assert repr(excinfo.value) == "SnekRuntimeError(\"ImportError('d')\")"


def test_dissallowed_functions():

    snek_eval("", scope={"thing": {}})
    with pytest.raises(DangerousValue):
        snek_eval("", scope={"open": open})


def test_return_nothing():
    assert (
        snek_eval(
            """
def foo():
    return
foo()"""
        )
        == [None, None]
    )


def test_eval_keyword():
    assert (
        snek_eval(
            """
def foo(a,b):
    return a,b
foo(1,b=2)"""
        )
        == [None, (1, 2)]
    )


def test_eval_functiondef_does_nothing():
    # todo: add pass
    assert (
        snek_eval(
            """
def foo(a,b):
    1
foo(1,b=2)"""
        )
        == [None, None]
    )


def test_eval_joinedstr():
    with pytest.raises(SnekRuntimeError):
        sneklang.MAX_SCOPE_SIZE = 10000000
        snek_eval(
            """
a='a' * 50000
f"{a} {a}"
"""
        )
    sneklang.MAX_SCOPE_SIZE = 100000

    assert (
        snek_eval(
            """
width = 10
precision = 4
value = 12.345
f"result: {value:{width}.{precision}}"  # nested fields
"""
        )[-1]
        == "result:      12.35"
    )


def test_coverage():
    src = """
def bar():
    return untested_variable

def foo(a):
    b = 1
    return (b if a else 2)

def test_foo():
    [foo(i) for i in [False, [], 0]]

    """
    coverage = snek_test_coverage(src)

    assert (
        ascii_format_coverage(coverage, src)
        == """Missing Return on line: 3 col: 4
    return untested_variable
----^
Missing Name on line: 3 col: 11
    return untested_variable
-----------^
Missing Name on line: 7 col: 12
    return (b if a else 2)
------------^
87% coverage
"""
    )
