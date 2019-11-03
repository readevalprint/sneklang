import pytest

import sneklang
from sneklang import *


def snek_is_still_python(code, snek_scope=None):
    py_scope = {}
    snek_scope = snek_scope or {}
    print(snek_eval(code, scope=snek_scope))
    exec(code, py_scope)
    print(py_scope["result"])
    print(snek_scope["result"])
    assert py_scope["result"] == snek_scope["result"]


def test_snek_comprehension_python():
    """
    Here we run the code in python exec, then in snek_eval, then compare the `scope['result']`
    """
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


def test_snek_is_python_try():
    snek_is_still_python(
        """
def foo(a):
    res = 'try: '
    try:
        try:
            res = res + str(1/a)
        except (ArithmeticError) as e:
            res= res + 'ArithmeticError'
    except:
        return 'oops'
    else:
        res = res + 'else'
    finally:
        res = res + 'fin'
    return a, res
result = [foo(i) for i in [-1,0,1, 'a']]
""",
        snek_scope={"Exception": Exception, "ArithmeticError": SnekArithmeticError},
    )


EXCEPTION_CASES = [
    ("nope", {}, "NameNotDefined(\"'nope' is not defined\")"),
    (
        "a=1; a.b",
        {},
        "SnekAttributeError(\"'int' object has no attribute 'b'\")",
    ),
    ("1/0", {}, "SnekArithmeticError('division by zero')"),
    (
        "len(str(10000 ** 10001))",
        {},
        'SnekArithmeticError("Sorry! I don\'t want to evaluate 10000 ** 10001")',
    ),
    (
        "'aaaa' * 25000",
        {},
        "SnekArithmeticError('Sorry, I will not evalute something that long.')",
    ),
    (
        "25000 * 'aaaa'",
        {},
        "SnekArithmeticError('Sorry, I will not evalute something that long.')",
    ),
    (
        "(10000 * 'world!') + (10000 * 'world!')",
        {},
        "SnekArithmeticError('Sorry, I will not evalute something that long.')",
    ),
    (
        "4 @ 3",
        {},
        "FeatureNotAvailable('Sorry, MatMult is not available in this evaluator')",
    ),
    (
        "def foo(*args): 1",
        {},
        "FeatureNotAvailable('Sorry, VarArgs are not available')",
    ),
    (
        "def foo(**kwargs): 1",
        {},
        "FeatureNotAvailable('Sorry, VarKwargs are not available')",
    ),
    (
        "a,b = 1, 2",
        {},
        "FeatureNotAvailable('Sorry, cannot assign to Tuple')",
    ),
    (
        "a=b=1",
        {},
        "FeatureNotAvailable('Sorry, cannot assign to 2 targets.')",
    ),
    (
        "int.mro()",
        {},
        "FeatureNotAvailable('Sorry, this method is not available. (type.mro)')",
    ),
    (
        '"a"' * 100_001,
        {},
        "IterableTooLong('String Literal in statement is too long! (100001, when 100000 is max)')",
    ),
    (
        "b'" + ("a" * 100_001) + "'",
        {},
        "IterableTooLong('Byte Literal in statement is too long! (100001, when 100000 is max)')",
    ),
    (
        repr(list("a" * 100001)),
        {},
        "IterableTooLong('List in statement is too long! (100001, when 100000 is max)')",
    ),
    (
        "1()",
        {},
        "SnekRuntimeError('Sorry, int type is not callable')",
    ),
    (
        "forbidden_func()[0]()",
        {"forbidden_func": lambda: [type]},
        "FeatureNotAvailable('This function is forbidden')",
    ),
    (
        "a()([])",
        {"a": lambda: sorted},
        "FeatureNotAvailable('This builtin function is not allowed: sorted')",
    ),
    (
        "a()",
        {"a": [].sort},
        "DangerousValue(\"This function 'a' in scope might be a bad idea.\")",
    ),
    (
        "a[1]",
        {"a": []},
        "SnekLookupError('list index out of range')",
    ),
    (
        "a.__length__",
        {"a": []},
        "FeatureNotAvailable('Sorry, access to this attribute is not available. (__length__)')",
    ),
    (
        "'say{}'.format('hi') ",
        {},
        "FeatureNotAvailable('Sorry, this method is not available. (str.format)')",
    ),
    (
        "[a for a in [] if True if True]",
        {},
        (
            "FeatureNotAvailable('Sorry, only one `if` allowed in list comprehension, "
            "consider booleans or a function')"
        ),
    ),
    (
        "class A: 1",
        {},
        "FeatureNotAvailable('Sorry, ClassDef is not available in this evaluator')",
    ),
    (
        "a.b",
        {"a": object()},
        "SnekAttributeError(\"'object' object has no attribute 'b'\")",
    ),
    (
        "'a' + 1",
        {},
        "SnekTypeError('can only concatenate str (not \"int\") to str')",
    ),
    (
        "import non_existant",
        {},
        "SnekImportError('non_existant')",
    ),
    ('assert False, "no"', {}, "SnekAssertionError('no')"),
    (
        "del a,b,c",
        {},
        "FeatureNotAvailable('Sorry, cannot delete 3 targets.')",
    ),
    (
        "del a.c",
        {},
        "FeatureNotAvailable('Sorry, cannot delete Attribute')",
    ),
    (
        "[1,2,3][[]]",
        {},
        "SnekTypeError('list indices must be integers or slices, not list')",
    ),
]


def test_exceptions():
    for code, scope, ex_repr in EXCEPTION_CASES:
        try:
            out = snek_eval(code, scope=scope)
        except Exception as exc:
            assert repr(exc) == ex_repr, f"Failed {code}"
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
    ]
    for code, out in CASES:
        assert snek_eval(code) == out, f"{code} should equal {out}"
        # Verify evaluates same as python
        assert [eval(code)] == out, code


def test_call_stack():
    scope = {}
    snek_eval("def foo(x): return x, x > 0 and foo(x-1) or 0", scope=scope)
    scope["foo"](3)
    with pytest.raises(CallTooDeep):
        scope["foo"](50)

    snek_eval(
        "def foo(x): return foo(x - 1) if x > 0 else 0",
        scope=scope,
        call_stack=30 * [1],
    )
    with pytest.raises(CallTooDeep):
        scope["foo"](3)


def test_settings():
    orig = sneklang.MAX_NODE_CALLS
    with pytest.raises(sneklang.TooManyEvaluations):
        scope = {}
        sneklang.MAX_NODE_CALLS = 20
        snek_eval("while True: 1", scope=scope)
    sneklang.MAX_NODE_CALLS = orig

    orig = sneklang.MAX_SCOPE_SIZE
    with pytest.raises(sneklang.ScopeTooLarge):
        scope = {}
        sneklang.MAX_SCOPE_SIZE = 500
        snek_eval("a=[]\nwhile True: a=[a, a]", scope=scope)
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


def test_dissallowed_functions():

    snek_eval("", scope={"thing": {}})
    with pytest.raises(DangerousValue):
        snek_eval("", scope={"open": open})


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
    with pytest.raises(IterableTooLong):
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
    return None

def foo(a):
    b = 1
    return (1 if a else 2)

def test_foo():
    [foo(i) for i in [False, [], 0]]

    """
    coverage = snek_test_coverage(src)

    assert (
        ascii_format_coverage(coverage, src)
        == """Missing Return on line: 3 col: 4
    return None
----^
Missing NameConstant on line: 3 col: 11
    return None
-----------^
Missing Num on line: 7 col: 12
    return (1 if a else 2)
------------^
87% coverage
"""
    )
