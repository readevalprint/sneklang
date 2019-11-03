
.. image:: logo.jpeg
   :target: logo.jpeg

Sneklang 
========================

.. image:: https://travis-ci.org/readevalprint/sneklang.svg?branch=master
   :target: https://travis-ci.org/readevalprint/sneklang
   :alt: Build Status

.. image:: https://coveralls.io/repos/github/readevalprint/sneklang/badge.svg?branch=master&1
   :target: https://coveralls.io/r/readevalprint/sneklang?branch=master
   :alt: Coverage Status

.. image:: https://badge.fury.io/py/sneklang.svg
   :target: https://badge.fury.io/py/sneklang
   :alt: PyPI Version

Try online
----------

https://sneklang.functup.com

Basic Usage
-----------

``snek_eval`` returns a list of all the expressions in the provided code.
Generally you care about the last one. 


To get very simple evaluating:

.. code-block:: python

    from sneklang import snek_eval

    snek_eval("'Hi!' + ' world!'")

returns ``[Hi! World!]``.

Expressions can be as complex and convoluted as you want:

.. code-block:: python

    snek_eval("21 + 19 / 7 + (8 % 3) ** 9")

returns ``[535.714285714]``.

You can add your own functions in as well.

.. code-block:: python

    snek_eval("square(11)", scope={"square": lambda x: x*x})

returns ``[121]``.


Try some dictionary or set comprehension.

.. code-block:: python

    >>> from sneklang import snek_eval
    >>> snek_eval("{a:b for a,b in [('a', 1), ('b',2)]}")
    [{'a': 1, 'b': 2}]

    >>> snek_eval("{a*a for a in [1,2,3]}")
    [{1, 4, 9}]


You can even define functions within the sand box at evaluation time.

.. code-block:: python

    >>> from sneklang import snek_eval
    >>> snek_eval('''
    ... def my_function(x): 
    ...     return x + 3
    ... 
    ... my_function(5)
    ... 
    ... ''')
    [None, 8]


Advanced Usage
--------------




Some times you will want to run a dynamically defined sandboxed funtion in your app.

.. code-block:: python

    >>> user_scope = {}
    >>> out = snek_eval('''
    ... def my_function(x=2): 
    ...    return x ** 3
    ... ''', scope=user_scope)
    >>> user_func = user_scope['my_function']
    >>> user_func()
    8


Or maybe create a decorator

.. code-block:: python

    >>> user_scope = {}
    >>> out = snek_eval('''
    ... def foo_decorator(func): 
    ...     def inner(s):
    ...        return "this is foo", func(s)
    ...     return inner
    ...
    ... @foo_decorator 
    ... def bar(s):
    ...     return "this is bar", s
    ... 
    ... output = bar("BAZ")
    ... ''', scope=user_scope)
    >>> user_scope['output']
    ('this is foo', ('this is bar', 'BAZ'))



You can also delete variables and catch exception

.. code-block:: python

    >>> user_scope = {}
    >>> out = snek_eval('''
    ... a = [1, 2, 3, 4, 5, 6, 7]
    ... del a[3:5]
    ... try:
    ...     a[10]
    ... except Exception as e:
    ...     b = "We got an error: " + str(e)
    ... ''', scope=user_scope)
    >>> user_scope['a']
    [1, 2, 3, 6, 7]
    >>> user_scope['b']
    'We got an error: list index out of range'


.. code-block:: python

    >>> user_scope = {}
    >>> out = snek_eval('''
    ... try:
    ...    raise Exception("this is my last resort")
    ... except Exception as e:
    ...     exc = e
    ... ''', scope=user_scope)
    >>> user_scope['exc']
    SnekRuntimeError('this is my last resort')

.. code-block:: python

    >>> user_scope = {}
    >>> out = snek_eval('''
    ... try:
    ...     try:
    ...         1/0
    ...     except Exception as e:
    ...         raise Exception("oh no") from e
    ... except Exception as e:
    ...     exc = e
    ... ''', scope=user_scope)
    >>> user_scope['exc']
    SnekRuntimeError('oh no')


And sometimes, users write crappy code... `MAX_CALL_DEPTH` is configurable, of course.
Here you can see some extreamly ineffecient code to multiply a number by 2

.. code-block:: python

    >>> from sneklang import InvalidExpression, CallTooDeep
    >>> user_scope = {}
    >>> out = snek_eval('''
    ... def multiply_by_2(x): 
    ...    return (2 + multiply_by_2(x-1)) if x > 0 else 0
    ... ''', scope=user_scope)

    >>> multiply_by_2 = user_scope['multiply_by_2']
    >>> multiply_by_2(5)
    10
    >>> try:
    ...     multiply_by_2(50)
    ... except CallTooDeep as e:
    ...     print(f'oh no! "{e}" On line:{e.lineno} col:{e.col}')
    oh no! "Sorry, stack is to large. The MAX_CALL_DEPTH is 32." On line:3 col:15



    >>> try:
    ...     snek_eval("int('foo is not a number')")
    ... except ValueError as e:
    ...     print('oh no! {}'.format(e))
    oh no! invalid literal for int() with base 10: 'foo is not a number'



Limited Power
~~~~~~~~~~~~~

Also note, the ``**`` operator has been locked down by default to have a
maximum input value of ``4000000``, which makes it somewhat harder to make
expressions which go on for ever.  You can change this limit by changing the
``sneklang.POWER_MAX`` module level value to whatever is an appropriate value
for you (and the hardware that you're running on) or if you want to completely
remove all limitations, you can set the ``s.operators[ast.Pow] = operator.pow``
or make your own function.

On my computer, ``9**9**5`` evaluates almost instantly, but ``9**9**6`` takes
over 30 seconds.  Since ``9**7`` is ``4782969``, and so over the ``POWER_MAX``
limit, it throws a ``NumberTooHigh`` exception for you. (Otherwise it would go
on for hours, or until the computer runs out of memory)

Strings (and other Iterables) Safety
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are also limits on string length (100000 characters,
``MAX_STRING_LENGTH``).  This can be changed if you wish.

Related to this, if you try to create a silly long string/bytes/list, by doing
``'i want to break free'.split() * 9999999999`` for instance, it will block you.

If Expressions
--------------

You can use python style ``if x then y else z`` type expressions:

.. code-block:: python

    >>> snek_eval("'equal' if x == y else 'not equal'", scope={"x": 1, "y": 2})
    ['not equal']

which, of course, can be nested:

.. code-block:: python

    >>> snek_eval("'a' if 1 == 2 else 'b' if 2 == 3 else 'c'")
    ['c']


Functions
---------

You can define functions which you'd like the expresssions to have access to:

.. code-block:: python

    >>> snek_eval("double(21)", scope={"double": lambda x:x*2})
    [42]

You can define "real" functions to pass in rather than lambdas, of course too,
and even re-name them so that expressions can be shorter

.. code-block:: python

    >>> def square(x):
    ...     return x ** 2
    >>> snek_eval("s(10) + square(2)", scope={"s": square, "square":square})
    [104]

If you don't provide your own ``scope`` dict, then the the following defaults
are provided in the ``DEFAULT_SCOPE`` dict:

+----------------+--------------------------------------------------+
| ``int(x)``     | Convert ``x`` to an ``int``.                     |
+----------------+--------------------------------------------------+
| ``float(x)``   | Convert ``x`` to a ``float``.                    |
+----------------+--------------------------------------------------+
| ``str(x)``     | Convert ``x`` to a ``str`` (``unicode`` in py2)  |
+----------------+--------------------------------------------------+

.. code-block:: python

    >>> snek_eval("a + b", scope={"a": 11, "b": 100})
    [111]

    >>> snek_eval("a + b", scope={"a": "Hi ", "b": "world!"})
    ['Hi world!']

You can also hand the scope of variable enames over to a function, if you prefer:


.. code-block:: python

    >>> class case_insensitive_scope(dict):
    ...    def __getitem__(self, key):
    ...        return super().__getitem__(key.lower())
    ...    def __setitem__(self, key, value):
    ...        return super().__setitem__(key.lower(), value)

    >>> snek_eval('''
    ... FOOBAR
    ... foobar
    ... FooBar''', scope=case_insensitive_scope({'foobar': 42}))
    [42, 42, 42]

.. code-block:: python

    >>> import sneklang
    >>> import random
    >>> my_scope = {}
    >>> my_scope.update(
    ...        square=(lambda x:x*x),
    ...        randint=(lambda top: int(random.random() * top))
    ...    )
    >>> snek_eval('square(randint(int("1")))', scope=my_scope)
    [0]



Other...
--------


Object attributes that start with ``_`` or ``func_`` are disallowed by default.
If you really need that (BE CAREFUL!), then modify the module global
``sneklang.DISALLOW_PREFIXES``.

A few builtin functions are listed in ``sneklang.DISALLOW_FUNCTIONS``.  ``type``, ``open``, etc.
If you need to give access to this kind of functionality to your expressions, then be very
careful.  You'd be better wrapping the functions in your own safe wrappers.

The initial idea came from J.F. Sebastian on Stack Overflow
( http://stackoverflow.com/a/9558001/1973500 ) with modifications and many improvements,
see the head of the main file for contributors list.

Then danthedeckie on Github with simpleeval(https://github.com/danthedeckie/simpleeval)

I've filled it out a bit more to allow safe funtion definitions, and better scope management.

Please read the ``test_snek.py`` file for other potential gotchas or
details.  I'm very happy to accept pull requests, suggestions, or other issues.
Enjoy!

Developing
----------

Run tests::

    $ make test

Or to set the tests running on every file change:

    $ make autotest

(requires ``entr``) 

