from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace, field, fields
from dataclasses import is_dataclass

import apsw
import textwrap

from typing import Any, Type, TypeVar, ParamSpec, Generic, cast, Callable, ClassVar
from typing import Literal

import typing as t
import typing_extensions as tx
import contextlib
import re

# from . import serializer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing_extensions import Concatenate
    from typing_extensions import Self

from datetime import datetime, timedelta
import functools
import importlib
import inspect
import json

from pprint import pp

class PP:
    def __call__(self, thing: A) -> A:
        pp(thing)
        return thing

    def __or__(self, thing: A) -> A:
        self(thing)
        return thing

    def __ror__(self, thing: A) -> A:
        self(thing)
        return thing

p = PP()

def asdict_shallow(x: Any) -> dict[str, Any]:
    assert is_dataclass(x)
    return {
        f.name: getattr(x, f.name)
        for f in fields(x)
    }

@dataclass(frozen=True)
class Serializer:

    reg: dict[str, Type[Any]] = field(default_factory=dict, hash=False)

    def register(self, cls: Type[Any]):
        self.reg[cls.__qualname__] = cls

    @functools.cache
    def lookup(self, qual_name: str) -> Any:
        if cls := self.reg.get(qual_name):
            return cls
        module_name, _sep, class_name = qual_name.rpartition('.')
        module = importlib.import_module(module_name)
        return module.__dict__[class_name]

    def from_json(self, x: Any) -> Any:
        if isinstance(x, dict):
            x = cast(dict[str, Any], x)
            type = x.get('type')
            if type == 'datetime':
                return datetime.fromisoformat(x['value'])
            elif type == 'timedelta':
                return timedelta(seconds=x['total_seconds'])
            elif type == 'tuple':
                return tuple(self.from_json(x['items']))
            elif type == 'dict':
                return dict(self.from_json(x['items']))
            elif type:
                cls = self.lookup(type)
                return cls(**{k: self.from_json(v) for k, v in x.items() if k != 'type'})
            else:
                return {k: self.from_json(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self.from_json(v) for v in cast(list[Any], x)]
        elif isinstance(x, None | float | int | bool | str):
            return x
        else:
            raise ValueError()

    def to_json(self, x: Any) -> dict[str, Any] | list[Any] | None | float | int | bool | str:
        if isinstance(x, Syntax):
            return x._replace(_args=[self.to_json(a) for a in x._args])
            # type: ignore
        elif isinstance(x, datetime):
            return {
                'type': 'datetime',
                'value': x.isoformat(sep=' '),
            }
        elif isinstance(x, timedelta):
            return {
                'type': 'timedelta',
                'total_seconds': x.total_seconds(),
            }
        elif is_dataclass(x):
            d = asdict_shallow(x)
            cls = x.__class__
            type = cls.__qualname__
            assert self.lookup(type) == cls
            assert 'type' not in d
            return {'type': type} | cast(dict[str, Any], self.to_json(d))
        elif isinstance(x, dict):
            d: dict[Any, Any] = x
            if 'type' not in d and all(isinstance(k, str) for k in d.keys()):
                return {k: self.to_json(v) for k, v in d.items()}
            else:
                return {
                    'type': 'dict',
                    'items': self.to_json([[k, v] for k, v in d.items()]),
                }
        elif isinstance(x, list):
            return [self.to_json(v) for v in cast(list[Any], x)]
        elif isinstance(x, tuple):
            t: tuple[Any, ...] = cast(Any, x)
            return {
                'type': 'tuple',
                'items': [self.to_json(v) for v in t]
            }
        elif isinstance(x, None | float | int | bool | str):
            return x
        else:
            raise ValueError(x)

    def dumps(self, x: Any) -> str:
        return json.dumps(self.to_json(x))

    def loads(self, s: str) -> Any:
        return self.from_json(json.loads(s))

serializer = Serializer()
# serializer.lookup('imager.utils.mixins.Serializer') | p

def collect_fields(cls: Any, args: tuple[Any], kws: dict[str, Any]) -> dict[str, Any]:
    for field, arg in zip(fields(cls), args):
        kws[field.name] = arg
    return kws

class ReplaceMixin:
    @property
    def replace(self) -> Type[Self]:
        def replacer(*args: Any, **kws: Any) -> Self:
            return replace(self, **collect_fields(self, args, kws))
        return replacer # type: ignore

class PrivateReplaceMixin:
    @property
    def _replace(self) -> Type[Self]:
        def replacer(*args: Any, **kws: Any) -> Self:
            return replace(self, **collect_fields(self, args, kws))
        return replacer # type: ignore



P = ParamSpec('P')
R = TypeVar('R')

@dataclass(frozen=True)
class SelectOptions(ReplaceMixin):
    order: str = 'id'
    limit: int | None = None
    offset: int | None = None
    where_str: list[str] = field(default_factory=list)
    where_ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    verbose: bool = False

    def add_where_op(self, op: str, *args: Any, **kws: Any) -> SelectOptions:
        return self.replace(where_ops=[*self.where_ops, (op, args, kws)])

    def add_where_str(self, *clauses: str) -> SelectOptions:
        return self.replace(where_str=[*self.where_str, *clauses])


# class API:
#     @classmethod
#     def

def _wrap(name: str):
    def inner(self, *args: Any):
        if eman := revbinops.get(name):
            assert len(args) == 1
            return Syntax(eman, [self, *args][::-1])
        else:
            return Syntax(name, [self, *args])
    return inner

unops = dict(
    __invert__        = '~',
    __neg__           = '-',
    __pos__           = '+',
) | {
    ' not ': 'not',
}
functions = dict(
    __abs__           = 'abs',
    __ceil__          = 'ceil',
    __floor__         = 'floor',
    __trunc__         = 'trunc',
    __round__         = 'round',
)
binops = dict(
    __add__           = '+',
    __and__           = '&',
    __eq__            = '==',
    __floordiv__      = '//',
    __ge__            = '>=',
    __gt__            = '>',
    __le__            = '<=',
    __lshift__        = '<<',
    __lt__            = '<',
    __matmul__        = '@',
    __mod__           = '%',
    __mul__           = '*',
    __ne__            = '!=',
    __or__            = '|',
    __pow__           = '**',
    __rshift__        = '>>',
    __sub__           = '-',
    __truediv__       = '/',
    __xor__           = '^',
) | {
    ' or ':           'or',
    ' and ':          'and',
}
revbinops = dict(
    __radd__          = '__add__',
    __rand__          = '__and__',
    __rfloordiv__     = '__floordiv__',
    __rlshift__       = '__lshift__',
    __rmatmul__       = '__matmul__',
    __rmod__          = '__mod__',
    __rmul__          = '__mul__',
    __ror__           = '__or__',
    __rpow__          = '__pow__',
    __rrshift__       = '__rshift__',
    __rsub__          = '__sub__',
    __rtruediv__      = '__truediv__',
    __rxor__          = '__xor__',
)

class SQL_API:
    def neg(self, x: bool) -> bool:
        return Syntax(' not ', [x])
    def len(self, x: str) -> int:
        return Syntax('__call__', [Syntax('length'), x])
    def iif(self, b: bool, t: A, f: A) -> A:
        return Syntax('__call__', [Syntax('iif'), b, t, f])

sql = SQL_API()

def call_str(head: str, *args: str):
    return head + '(' + ','.join(args) + ')'

def is_root(thing: Any) -> bool:
    return isinstance(thing, Syntax) and thing._is_root

def is_str_type(thing: Any) -> bool:
    match thing:
        case str():
            return True
        case Syntax(str(s), [lhs, rhs]) if binops.get(s) == '+':
            return is_str_type(lhs) or is_str_type(rhs)
        case Syntax('__getattr__', [head, str(field_name)]) if is_root(head):
            return bool(re.search(r'\bstr\b', inspect.get_annotations(head._root_type)[field_name]))
        case _:
            return False

def json_quote(thing: str) -> str:
    return call_str('json_quote', thing)

def show_ex(thing: Any) -> str:
    match thing:
        case int() | float() | bool():
            return str(thing)
        case str(s):
            return sqlquote(thing)
        case Syntax('__getattr__', [head, attr]):
            return f'({show(head)} ->> {show(attr)})'
        case Syntax(str(s), [lhs, rhs]) if '+' == binops.get(s) and is_str_type(thing):
            return f'{show_ex(lhs)} || {show_ex(rhs)}'
        case _:
            return call_str('json_extract', show(thing), "'$'")


def show(thing: Syntax | Prims | Any) -> str:
    # root_path = sel(thing)
    # if root_path is not None:
    #     root, path = root_path
    #     return f'({root} ->> {sqlquote(path)})'
    match thing:
        case Syntax(name, [], _is_root=True):
            # return f"({name}.value ->> '$')" # , '$'
            return f"json({name}.value)"
            # return f"{name}.value"
        case list(xs_untyped): # type: ignore
            xs: list[Any] = xs_untyped
            return call_str(
                'json_array',
                *[
                    show(item)
                    for item in xs
                ]
            )
        case dict(d_untyped): # type: ignore
            d: dict[str, Any] = d_untyped
            return call_str(
                'json_object',
                *[
                    show(item)
                    for k, v in d.items()
                    for item in [k, v]
                ]
            )

        case Syntax(str(s), [lhs, rhs]) if '+' == binops.get(s) and is_str_type(thing):
            return json_quote(show_ex(thing))

        case Syntax(str(s), [lhs, rhs]) if (
            (op := binops.get(s)) in '< <= > >='.split()
        ):
            if is_root(lhs) or is_root(rhs):
                raise ValueError('Refusing to order over serialized json')
            else:
                return f'({show_ex(lhs)} {op} {show_ex(rhs)})'

        case Syntax(str(s), [lhs, rhs]) if op := binops.get(s):
            return f'({show(lhs)} {op} {show(rhs)})'
        case Syntax(str(s), [lhs, rhs]) if op := revbinops.get(s):
            return f'({show(rhs)} {op} {show(lhs)})'
        case Syntax(str(s), [arg]) if op := unops.get(s):
            return f'({op}{show(arg)})'
        case Syntax(str(name), args) if x := functions.get(name):
            return show(Syntax('__call__', [x, *args]))

        case Syntax('__call__', [Syntax('__getattr__', [head, str(method_name)]), *args]):
            return show(Syntax(method_name, [head, *args]))

        case Syntax('startswith', [s, prefix]):
            return call_str('glob', show(prefix + "*"), show(s))
        case Syntax('endswith', [s, suffix]):
            return call_str('glob', show("*" + suffix), show(s))

        case Syntax('lower', [s]):
            return json_quote(call_str('lower', show_ex(s)))
        case Syntax('upper', [s]):
            return json_quote(call_str('upper', show_ex(s)))

        case Syntax('strip', [s, *args]):
            return json_quote(call_str('trim', show_ex(s), *map(show_ex, args)))
        case Syntax('lstrip', [s, *args]):
            return json_quote(call_str('ltrim', show_ex(s), *map(show_ex, args)))
        case Syntax('rstrip', [s, *args]):
            return json_quote(call_str('rtrim', show_ex(s), *map(show_ex, args)))

        case Syntax('replace', args):
            return json_quote(call_str('replace', *map(show_ex, args)))

        case Syntax('__getitem__', [head, start, stop]) if is_str_type(head):
            if isinstance(stop, int) and stop < 0:
                raise ValueError(f'Negative stop ({stop}) not implemented (but is possible, see https://sqlite.org/lang_corefunc.html#substr)')
            if start is None:
                start = 0
            return call_str('substr', show(head), show(start), show(stop - start + 1))
        case Syntax('__getitem__', [head, pos]) if is_str_type(head):
            return call_str('substr', show(head), show(pos), show(1))
        case Syntax('__getitem__', [head, int(pos)]):
            if pos >= 0:
                return f"json({show(head)} ->> '$[{pos}]')"
            else:
                return f"json({show(head)} ->> '$[#{pos}]')"

        case Syntax('__call__', [self, *args]):
            return call_str(show(self), *map(show, args))
        case Syntax('__getattr__', [head, attr]):
            if is_str_type(thing):
                return f'({show(head)} -> {show(attr)})'
            else:
                return f'json({show(head)} ->> {show(attr)})'

        case Syntax(str(name), []):
            return name
        case str(s):
            return sqlquote(s)
        case None:
            return "NULL"
        case int() | float() | bool():
            return repr(thing)
        case _:
            raise ValueError(f'Cannot convert {thing} of type {type(thing)} to sql')

def roots(*ss: Syntax | Any) -> list[Syntax]:
    out: list[Syntax] = []
    for s in ss:
        xs: list[Syntax] = []
        if isinstance(s, list):
            xs = roots(*s)
        elif isinstance(s, dict):
            xs = roots(*s.values())
        elif isinstance(s, Syntax):
            xs = roots(*s._args)
            if s._is_root:
                xs += [s]
        for x in xs:
            if not any(x._op == y._op for y in out):
                out += [x]
    return out

@dataclass(frozen=True)
class Syntax(PrivateReplaceMixin):
    _op: Prims = None
    _args: list[Any] = field(default_factory=list)
    _is_root: bool = False
    _root_type: Any = None
    _table: str = ''

    def __repr__(self):
        return f'{self._op}({",".join(map(repr, self._args))})'

    def __getattr__(self, attr: self):
        if attr.startswith('__'):
            # required for hasattr(..., '__iter__') to return false
            raise AttributeError
        else:
            return _wrap('__getattr__')(self, attr)

    def __getitem__(self, x: int | slice):
        if isinstance(x, slice):
            assert x.step is None or x.step == 1
            return _wrap('__getitem__')(self, x.start, x.stop)
        else:
            return _wrap('__getitem__')(self, x)

    __abs__           = _wrap('__abs__')
    __add__           = _wrap('__add__')
    __and__           = _wrap('__and__')
    __call__          = _wrap('__call__')
    __ceil__          = _wrap('__ceil__')
    __complex__       = _wrap('__complex__')
    __eq__            = _wrap('__eq__')
    __floor__         = _wrap('__floor__')
    __floordiv__      = _wrap('__floordiv__')
    __ge__            = _wrap('__ge__')
    __gt__            = _wrap('__gt__')
    __int__           = _wrap('__int__')
    __invert__        = _wrap('__invert__')
    __le__            = _wrap('__le__')
    __lshift__        = _wrap('__lshift__')
    __lt__            = _wrap('__lt__')
    __matmul__        = _wrap('__matmul__')
    __mod__           = _wrap('__mod__')
    __mul__           = _wrap('__mul__')
    __ne__            = _wrap('__ne__')
    __neg__           = _wrap('__neg__')
    __or__            = _wrap('__or__')
    __pos__           = _wrap('__pos__')
    __pow__           = _wrap('__pow__')
    __radd__          = _wrap('__radd__')
    __rand__          = _wrap('__rand__')
    __rfloordiv__     = _wrap('__rfloordiv__')
    __rlshift__       = _wrap('__rlshift__')
    __rmatmul__       = _wrap('__rmatmul__')
    __rmod__          = _wrap('__rmod__')
    __rmul__          = _wrap('__rmul__')
    __ror__           = _wrap('__ror__')
    __round__         = _wrap('__round__')
    __rpow__          = _wrap('__rpow__')
    __rrshift__       = _wrap('__rrshift__')
    __rshift__        = _wrap('__rshift__')
    __rsub__          = _wrap('__rsub__')
    __rtruediv__      = _wrap('__rtruediv__')
    __rxor__          = _wrap('__rxor__')
    __sub__           = _wrap('__sub__')
    __truediv__       = _wrap('__truediv__')
    __trunc__         = _wrap('__trunc__')
    __xor__           = _wrap('__xor__')

    # __complex__       = _wrap('__complex__'),
    # __int__           = _wrap('__int__'),
    # __float__         = _wrap('__float__'),

    # __aenter__        = _wrap('__aenter__')
    # __aexit__         = _wrap('__aexit__')
    # __aiter__         = _wrap('__aiter__')
    # __anext__         = _wrap('__anext__')
    # __await__         = _wrap('__await__')
    # __bool__          = _wrap('__bool__')
    # __bytes__         = _wrap('__bytes__')
    # __call__          = _wrap('__call__')
    # __class__         = _wrap('__class__')
    # __class_getitem__ = _wrap('__class_getitem__')
    # __contains__      = _wrap('__contains__')
    # __copy__          = _wrap('__copy__')
    # __deepcopy__      = _wrap('__deepcopy__')
    # __del__           = _wrap('__del__')
    # __delattr__       = _wrap('__delattr__')
    # __delete__        = _wrap('__delete__')
    # __delitem__       = _wrap('__delitem__')
    # __dict__          = _wrap('__dict__')
    # __dir__           = _wrap('__dir__')
    # __divmod__        = _wrap('__divmod__')
    # __doc__           = _wrap('__doc__')
    # __enter__         = _wrap('__enter__')
    # __exit__          = _wrap('__exit__')
    # __format__        = _wrap('__format__')
    # __get__           = _wrap('__get__')
    # __getattribute__  = _wrap('__getattribute__')
    # __getnewargs__    = _wrap('__getnewargs__')
    # __getnewargs_ex__ = _wrap('__getnewargs_ex__')
    # __getstate__      = _wrap('__getstate__')
    # __hash__          = _wrap('__hash__')
    # __iadd__          = _wrap('__iadd__')
    # __iand__          = _wrap('__iand__')
    # __ifloordiv__     = _wrap('__ifloordiv__')
    # __ilshift__       = _wrap('__ilshift__')
    # __imatmul__       = _wrap('__imatmul__')
    # __imod__          = _wrap('__imod__')
    # __imul__          = _wrap('__imul__')
    # __index__         = _wrap('__index__')
    # __init_subclass__ = _wrap('__init_subclass__')
    # __instancecheck__ = _wrap('__instancecheck__')
    # __ior__           = _wrap('__ior__')
    # __ipow__          = _wrap('__ipow__')
    # __irshift__       = _wrap('__irshift__')
    # __isub__          = _wrap('__isub__')
    # __iter__          = _wrap('__iter__')
    # __itruediv__      = _wrap('__itruediv__')
    # __ixor__          = _wrap('__ixor__')
    # __len__           = lambda self: 12345 # _wrap('__len__')
    # __length_hint__   = _wrap('__length_hint__')
    # __missing__       = _wrap('__missing__')
    # __name__          = _wrap('__name__')
    # __next__          = _wrap('__next__')
    # __objclass__      = _wrap('__objclass__')
    # __prepare__       = _wrap('__prepare__')
    # __reduce__        = _wrap('__reduce__')
    # __reduce_ex__     = _wrap('__reduce_ex__')
    # __repr__          = _wrap('__repr__')
    # __reversed__      = _wrap('__reversed__')
    # __rdivmod__       = _wrap('__rdivmod__')
    # __set__           = _wrap('__set__')
    # __setattr__       = _wrap('__setattr__')
    # __setitem__       = _wrap('__setitem__')
    # __set_name__      = _wrap('__set_name__')
    # __setstate__      = _wrap('__setstate__')
    # __slots__         = _wrap('__slots__')
    # __str__           = _wrap('__str__')
    # __subclasscheck__ = _wrap('__subclasscheck__')
    # __weakref__       = _wrap('__weakref__')

Prims: t.TypeAlias = int | bool | float | None | bytes | str
Prim = TypeVar('Prim', Prims, tuple[Prims, ...])
Key = TypeVar('Key')
Item = TypeVar('Item')
Items = TypeVar('Items')

@dataclass(frozen=True)
class Group(Generic[Key, R], PrivateReplaceMixin):
    def dict(self) -> dict[Key, list[R]]:
        return dict(self.list())

    def list(self) -> list[tuple[Key, list[R]]]:
        raise

    def select(self, v: A) -> Group[Key, A]:
        raise

    def having(self, *cond: bool | t.Iterable[bool]) -> Group[Key, R]:
        raise

    '''
    todo: order by within the group needs a nested select
    select
        value, count(*), json_group_array(value)
      from
        (select value from generate_series(0, 10) order by value desc)
      group by
        value % 2;
    value  count(*)  json_group_array(value)
    -----  --------  -----------------------
    10     6         [10,8,6,4,2,0]
    9      5         [9,7,5,3,1]
    '''

S = t.TypeVar('S')

@dataclass(frozen=True)
class Select(Generic[R], PrivateReplaceMixin):

    _focus: Syntax = Syntax('None')
    _where: list[Syntax] = field(default_factory=list)
    _db: DB = cast(Any, None)

    def get(self, t: Type[A]) -> Select[A]:
        return self._replace(_focus=t)

    def select(self, v: A) -> Select[A]:
        return self._replace(_focus=v)

    def sql(self) -> str:
        def join(none: str, sep: str, values: list[Any]):
            if not values:
                return none
            else:
                return sep.join(values)
        focus = [serializer.to_json(self._focus)]
        where = [serializer.to_json(w) for w in self._where]
        rs = roots(focus, *where)
        select = show(focus) # , to_json_text=True)
        from_ = ', '.join(
            f'{r._table} {r._op}'
            for r in rs
        )
        stmt = {'select': select}
        if from_:
            stmt['from'] = from_
        if where:
            stmt['where'] = '\n  and '.join(map(show, where))
        return '\n'.join(k + '\n  ' + v for k, v in stmt.items()) + ';'

    def one(self) -> R:
        return self.list()[0]

    def list(self) -> list[R]:
        stmt = self.sql()
        def throw(e: Exception):
            raise e
        return [
            serializer.loads(v)[0] if isinstance(v, str) else throw(ValueError(f'{v} is not string'))
            for v, in self._db.con.execute(stmt).fetchall()
        ]

    def __iter__(self):
        yield from self.list()

    def where(self, *cond: bool) -> Select[R]:
        return self._replace(_where=[*self._where, *cond]) # type: ignore

    def where_some(self, *cond: bool) -> Select[R]:
        if not cond:
            syntax = Syntax(0, [])
        else:
            syntax = cond[0]
            for c in cond[1:]:
                syntax = Syntax(' or ', [syntax, c])
        return self._replace(_where=[*self._where, syntax]) # type: ignore

    def group(self, by: Prim) -> Group[Prim, R]:
                                                          #  key   items
        ...

    def limit(self, bound: int | None = None, offset: int | None = None) -> Select[R]:
        return self._replace(self._opts.replace(limit=bound, offset=offset))

    def order(self, by: Prim) -> Select[R]:
        return self._replace(self._opts.replace(order=by))

    def show(self) -> Select[R]:
        print(self.sql())
        return self

def sqlquote(s: str) -> str:
    c = "'"
    return c + s.replace(c, c+c) + c

A = TypeVar('A')

@dataclass
class DB:
    con: apsw.Connection
    transaction_depth: int = 0

    @property
    @contextlib.contextmanager
    def transaction(self):
        '''
        Exclusive transaction (begin exclusive .. commit), context manager version.
        '''
        self.transaction_depth += 1
        if self.transaction_depth == 1:
            self.con.execute('begin exclusive')
            yield
            self.con.execute('commit')
        else:
            yield
        self.transaction_depth -= 1

    def with_transaction(self, do: Callable[[], A]) -> A:
        '''
        Exclusive transaction (begin exclusive .. commit), expression version.
        '''
        with self.transaction:
            return do()

    def has_table(self, name: str, type: Literal['table', 'view', 'index', 'trigger']="table") -> bool:
        return any(
            self.con.execute(f'''
                select 1 from sqlite_master where name = ? and type = ?
            ''', [name, type])
        )

    def table_name(self, t: Callable[P, Any]):
        Table = t.__name__
        TableView = f'{Table}View'
        if not self.has_table(Table):
            self.con.execute(textwrap.dedent(f'''
                create table if not exists {Table} (
                    id integer as (value ->> 'id') unique,
                    value text,
                    check (typeof(id) = 'integer'),
                    check (id >= 0),
                    check (json_valid(value))
                );
                create index if not exists {Table}_id on {Table} (id);
            '''))
        if is_dataclass(t) and not self.has_table(TableView, 'view'):
            meta = getattr(t, '__meta__', None)
            views: dict[str, str] = {
                f.name: f'value ->> {sqlquote(f.name)}'
                for f in sorted(
                    fields(t),
                    key=lambda f: f.name != 'id',
                )
            }
            if isinstance(meta, Meta):
                views.update(meta.views)
            xs = [
                f'({expr}) as {sqlquote(name)}'
                for name, expr in views.items()
            ]
            self.con.execute(textwrap.dedent(f'''
                create view {TableView} as select
                    {""",
                    """.join(xs)}
                    from {Table} order by id
            '''))
        return Table

    def get(self, t: Type[R]) -> Select[R]:
        return Select(t._proxy, _db=self) # type: ignore

    def where(self, *cond: bool) -> Select[None]:
        return Select[None](_db=self).where(*cond)

    def select(self, v: A) -> Select[A]:
        return Select[A](v, _db=self)

    def __post_init__(self):
        self.con.execute('pragma journal_mode=WAL')

    @contextmanager
    @staticmethod
    def open(path: str):
        db = DB.connect(path)
        yield db
        db.con.close()

    @staticmethod
    def connect(path: str):
        con = apsw.Connection(path)
        con.setbusytimeout(2000)
        return DB(con)

class DBMixin(ReplaceMixin):
    id: int

    @classmethod
    def get(cls, db: DB) -> Select[Self]:
        return db.get(cls)

    @classmethod
    def proxy(cls) -> Self:
        p = Syntax(cls.__name__.lower(), [], _is_root=True, _root_type=cls, _table=cls.__name__)
        for f in fields(cls):
            setattr(cls, f.name, getattr(p, f.name))
        return p # type: ignore

    @classmethod
    def autoproxy(cls):
        cls._proxy = cls.proxy()

    def save(self, db: DB) -> Self:
        with db.transaction:
            Table = db.table_name(self.__class__)
            meta = getattr(self.__class__, '__meta__', None)

            if isinstance(meta, Meta) and meta.log:
                LogTable = meta.log_table or f'{Table}Log'
                exists = any(
                    db.con.execute(f'''
                        select 1 from sqlite_master where type = "table" and name = ?
                    ''', [LogTable])
                )
                if not exists:
                    db.con.execute(textwrap.dedent(f'''
                        create table {LogTable} (
                            t timestamp default (strftime('%Y-%m-%d %H:%M:%f', 'now', 'localtime')),
                            action text,
                            old json,
                            new json
                        );
                        create trigger {Table}_insert after insert on {Table} begin
                            insert into {LogTable}(action, old, new) values ("insert", NULL, NEW.value);
                        end;
                        create trigger {Table}_update after update on {Table} begin
                            insert into {LogTable}(action, old, new) values ("update", OLD.value, NEW.value);
                        end;
                        create trigger {Table}_delete after delete on {Table} begin
                            insert into {LogTable}(action, old, new) values ("delete", OLD.value, NULL);
                        end;
                    '''))
            if self.id == -1:
                exists = False
            else:
                exists = any(
                    db.con.execute(f'''
                        select 1 from {Table} where id = ?
                    ''', [self.id])
                )
            if exists:
                db.con.execute(f'''
                    update {Table} set value = ? -> '$' where id = ?
                ''', [serializer.dumps(self), self.id])
                # db.con.commit()
                return self
            else:
                if self.id == -1:
                    reply = db.con.execute(f'''
                        select ifnull(max(id) + 1, 0) from {Table};
                    ''').fetchone()
                    assert reply is not None
                    id, = reply
                    res = self.replace(id=id) # type: ignore
                else:
                    id = self.id
                    res = self
                db.con.execute(f'''
                    insert into {Table} values (? -> '$')
                ''', [serializer.dumps(res)])
                # db.con.commit()
                return res

    def delete(self, db: DB):
        Table = self.__class__.__name__
        db.con.execute(f'''
            delete from {Table} where id = ?
        ''', [self.id])

    def reload(self, db: DB) -> Self:
        return db.get(self.__class__).where(id=self.id)[0]

@dataclass(frozen=True)
class Meta:
    log: bool = False
    log_table: None | str = None
    views: dict[str, str] = field(default_factory=dict)

import sys

@functools.cache
def get_annotations(cls: Type[Any]):
    return inspect.get_annotations(
        cls,
        globals=sys.modules[cls.__module__].__dict__,
        locals=dict(vars(cls)),
        eval_str=True,
    )

def test():
    '''
    python -m imager.utils.mixins
    '''
    from datetime import datetime
    from pprint import pp

    @dataclass
    class Todo(DBMixin):
        msg: str = ''
        done: bool = False
        created: datetime = field(default_factory=lambda: datetime.now())
        deleted: None | datetime = None
        id: int = -1
        __meta__: ClassVar = Meta(
            log=True,
            views={
                'created': 'value ->> "created.value"',
                'deleted': 'value ->> "deleted.value"',
            },
        )

    @dataclass
    class Todos(DBMixin):
        head: str
        todos: list[Todo]
        id: int = -1

    Todo.__qualname__ = 'Todo'
    serializer.register(Todo)

    Todos.__qualname__ = 'Todos'
    serializer.register(Todos)

    Todo.autoproxy()
    Todos.autoproxy()

    with DB.open(':memory:') as db:

        t0 = Todo('hello world').save(db)
        t1 = Todo('goodbye world').save(db)
        t2 = Todo('hello again').save(db)
        t3 = Todo('\\\\llo there').save(db)
        ts = Todos('hehe', [t0, t2]).save(db)

        # db.con.execute('select * from Todo').fetchall() | p

        todo = Todo
        t2: Any = Syntax('t2', _is_root=True, _root_type=Todo)

        Todo('{}').save(db)

        # db.get(Todo).show()
        stmt = (
            db.select((
                Todo.msg + " " + Todo.msg,
                Todo.msg.upper(),
                Todo.msg.replace('hello', 'goodbye'),
                Todo.msg.strip('he'),
                Todo.msg[-1],
                Todo.msg[0:2],
                Todo.id + 1,
                Todo(Todo.msg, Todo.done, Todo.created, Todo.deleted, Todo.id),
                {"foo": Todo.msg},
                Todo.created.value,
            ))
                # .where(Todo._proxy == Todo('a'))
                # .where(todo.id < 2)
                # .where_some(todo.done, sql.neg(todo.done))
                # .where(todo.msg.startswith('hello'))
                # .where(sql.len(todo.msg) > 5)
                # .where(sql.iif(sql.len(todo.msg) > 5, True, False))
                # .where_some(todo.msg == 'world', todo.msg == 'plopp')
                # .show()
        )

        ea = Syntax('ea', [], True, Todo, call_str('json_each', show(Todos._proxy.todos)))

        stmt = (
            db.select(
                [
                    Todo.id * 3 + 1,
                    (' ... ' + Todo.msg + " ! ").strip(),
                    Todo.msg.strip() + Todo.msg.strip(), # incorrect type
                    Todo.msg,
                    Todo._proxy,
                    ea,
                    Todos.head,
                    # Todos.todos[0].msg # incorrect type
                ]
            ).where(
                Todo.msg < ea.msg,
                ea.id < 2,
                Todo.id < 2,
                Todo.msg < "z"
                # (Todos.head, ea)
                # Todos.head
                # Todos._proxy
                # Todos.todos[0]
                # Todo._proxy
                # Todo.msg
                # Todo._proxy
                # "{}"
            ).show()
        )
        pp(stmt.list())
        quit()
        for z in [
            'a',
            '"a"',
            "'a'",
            '{"type":"dict","items":[]}',
            {"type":"dict","items":[]},
            {},
        ]:
            stmt = db.select(z).show()
            pp(
                stmt.list()
            )

        quit()

        e = (
            Todo.msg.startswith("boop") &
            Todo.msg.endswith(Todo.msg) &
            todo |
            todo.id ^ todo.id |
            1 // todo.id |
            1 / todo.id |
            todo.id << 1 |
            todo.id >> 1 |
            ~-+todo.id |
            abs(todo.id) |
            round(todo.id, 2)
        )

        print(
            repr(e),
            show(e),
            sep='\n',
        )

        quit()

        Todos1 = db.get(Todo) # .verbose
        Todos2 = Todo.get(db)

        x = db.get(Todo)

        pp(
            db.get(Todo.msg).where(Todo.id > 2).list()
        )

        t1 = Todo.proxy()
        t2 = Todo.proxy()

        u = db.select((t1, t2))
        u.where(t1.msg < t2.msg)
        for v1, v2 in u:
            ...

        j = db.where(Todo.msg.startswith('hello')).select(Todo)
        # j = Todo.select(db).where(Todo.msg.startswith('hello'))
        j = db.select(Todo).where(Todo.msg.startswith('hello'))
        db.select(Todo())

        z.list(db)

        # print(fields(Todo))
        # print(Todo.created)

        # Todo.msg='aaaaa'
        print(Todo())
        print(Todo.msg + "3", Todo.id + Todo.created.minute)
        quit()


        print(todo.id > 1)
        # print(lambda todo: todo.id > 1 and todo.id > 5)
        print(todo.msg.startswith("banana"))
        print(todo.msg.startswith(todo.msg))
        print(3 >> ~todo.id << todo.id ^ todo.id ** 8)
        print(1 // todo.id)
        print(1 / todo.id)
        print(1 % -+todo.id)
        # print(1 // todo.m + len(todo.msg))

        db.select(Todo).where(Todo.msg == 'apa')

        x = Todos.join(Todo)
        x.where(lambda t1, t2: t1.id > t2.id)
        g2 = x.group(lambda t1, t2: t1.msg + t2.msg)
        d = g2.dict()

        g = Todos.group(by=lambda todo: todo.msg)

        Todos.where(lambda todo: todo.id > 1)
        Todos.where(lambda todo: todo.id > 1)
        r = Todos.where(lambda todo: todo.id > 1).select(lambda todo: todo.msg)
        r = Todos.where(lambda todo: todo.id > 1).select(lambda todo: (todo.msg, "bosse"))

        Todos[todo.id > 1]
        Todos.where(todo.id > 1)

        # len,


        Todos.where(lambda todo: todo.id > 1)
        quit()

        print(*Todos, sep='\n')
        print(*Todos.where_glob(msg='*world'), sep='\n')
        print(*Todos.where_like(msg='hello%'), sep='\n')
        print(*Todos.where_ge(id=1).where_le(id=2), sep='\n')
        now = datetime.now()
        t3.replace(deleted=now).save(db)
        print(*Todos.where_not(deleted=None), sep='\n')
        print(*Todos.where(deleted=now), sep='\n')
        quit()
        print()
        t2.replace(done=True).save(db)
        print(*Todos.where(done=False), sep='\n')
        print()
        print(*Todos.where(done=True), sep='\n')
        t1.delete(db)
        t3.replace(deleted=datetime.now()).save(db)
        import tempfile
        from subprocess import check_output
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            db.con.execute('vacuum into ?', [tmp.name])
            out = check_output([
                'sqlite3',
                tmp.name,
                '.mode box',
                'select t, action, ifnull(new, old) from TodoLog',
                'select * from TodoView where done',
                'select * from TodoView where not done',
                'select * from TodoView where msg glob "*world"',
            ], encoding='utf8')
            print(out)

if __name__ == '__main__':
    test()
