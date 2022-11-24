import copy
import dataclasses
import os
import math
import pytest
import typing as tp

from pytest import approx

from compgraph import operations as ops
from compgraph import algorithms as alg

from compgraph.graph import Graph
from examples import run_pmi,  run_word_count
from click.testing import CliRunner
Runner = CliRunner()

KiB = 1024
MiB = 1024 * KiB


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


@dataclasses.dataclass
class MapCase:
    mapper: ops.Mapper
    data: list[ops.TRow]
    etalon: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    mapper_item: int = 0
    mapper_etalon_items: tuple[int, ...] = (0,)


MAP_CASES = [
    MapCase(
        mapper=ops.DummyMapper(),
        data=[
            {'test_id': 1, 'text': 'one two three'},
            {'test_id': 2, 'text': 'testing out stuff'}
        ],
        etalon=[
            {'test_id': 1, 'text': 'one two three'},
            {'test_id': 2, 'text': 'testing out stuff'}
        ],
        cmp_keys=('test_id', 'text')
    ),
    MapCase(
        mapper=ops.LowerCase(column='text'),
        data=[
            {'test_id': 1, 'text': 'camelCaseTest'},
            {'test_id': 2, 'text': 'UPPER_CASE_TEST'},
            {'test_id': 3, 'text': 'wEiRdTeSt'}
        ],
        etalon=[
            {'test_id': 1, 'text': 'camelcasetest'},
            {'test_id': 2, 'text': 'upper_case_test'},
            {'test_id': 3, 'text': 'weirdtest'}
        ],
        cmp_keys=('test_id', 'text')
    ),
    MapCase(
        mapper=ops.FilterPunctuation(column='text'),
        data=[
            {'test_id': 1, 'text': 'Hello, world!'},
            {'test_id': 2, 'text': 'Test. with. a. lot. of. dots.'},
            {'test_id': 3, 'text': r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'}
        ],
        etalon=[
            {'test_id': 1, 'text': 'Hello world'},
            {'test_id': 2, 'text': 'Test with a lot of dots'},
            {'test_id': 3, 'text': ''}
        ],
        cmp_keys=('test_id', 'text')
    ),
    MapCase(
        mapper=ops.Split(column='text'),
        data=[
            {'test_id': 1, 'text': 'one two three'},
            {'test_id': 2, 'text': 'tab\tsplitting\ttest'},
            {'test_id': 3, 'text': 'more\nlines\ntest'},
            {'test_id': 4, 'text': 'tricky\u00A0test'}
        ],
        etalon=[
            {'test_id': 1, 'text': 'one'},
            {'test_id': 1, 'text': 'three'},
            {'test_id': 1, 'text': 'two'},

            {'test_id': 2, 'text': 'splitting'},
            {'test_id': 2, 'text': 'tab'},
            {'test_id': 2, 'text': 'test'},

            {'test_id': 3, 'text': 'lines'},
            {'test_id': 3, 'text': 'more'},
            {'test_id': 3, 'text': 'test'},

            {'test_id': 4, 'text': 'test'},
            {'test_id': 4, 'text': 'tricky'}
        ],
        cmp_keys=('test_id', 'text'),
        mapper_etalon_items=(0, 1, 2)
    ),
    MapCase(
        mapper=ops.Product(columns=['speed', 'time'], result_column='distance'),
        data=[
            {'test_id': 1, 'speed': 5, 'time': 10},
            {'test_id': 2, 'speed': 60, 'time': 2},
            {'test_id': 3, 'speed': 3, 'time': 15},
            {'test_id': 4, 'speed': 100, 'time': 0.5},
            {'test_id': 5, 'speed': 48, 'time': 15},
        ],
        etalon=[
            {'test_id': 1, 'speed': 5, 'time': 10, 'distance': 50},
            {'test_id': 2, 'speed': 60, 'time': 2, 'distance': 120},
            {'test_id': 3, 'speed': 3, 'time': 15, 'distance': 45},
            {'test_id': 4, 'speed': 100, 'time': 0.5, 'distance': 50},
            {'test_id': 5, 'speed': 48, 'time': 15, 'distance': 720},
        ],
        cmp_keys=('test_id', 'speed', 'time', 'distance')
    ),
    MapCase(
        mapper=ops.Filter(condition=lambda row: row['f'] ^ row['g']),
        data=[
            {'test_id': 1, 'f': 0, 'g': 0},
            {'test_id': 2, 'f': 0, 'g': 1},
            {'test_id': 3, 'f': 1, 'g': 0},
            {'test_id': 4, 'f': 1, 'g': 1}
        ],
        etalon=[
            {'test_id': 2, 'f': 0, 'g': 1},
            {'test_id': 3, 'f': 1, 'g': 0}
        ],
        cmp_keys=('test_id', 'f', 'g'),
        mapper_etalon_items=tuple()
    ),
    MapCase(
        mapper=ops.Project(columns=['value']),
        data=[
            {'test_id': 1, 'junk': 'x', 'value': 42},
            {'test_id': 2, 'junk': 'y', 'value': 1},
            {'test_id': 3, 'junk': 'z', 'value': 144}
        ],
        etalon=[
            {'value': 42},
            {'value': 1},
            {'value': 144}
        ],
        cmp_keys=('value',)
    ),
    MapCase(
        mapper=ops.Inverse(column='value'),
        data=[
            {'test_id': 1, 'word': 'zxc', 'value': 10},
            {'test_id': 2, 'word': 'asd', 'value': 1},
            {'test_id': 3, 'word': 'qwe', 'value': 0.1}
        ],
        etalon=[
            {'test_id': 1, 'word': 'zxc', 'value': 0.1},
            {'test_id': 2, 'word': 'asd', 'value': 1},
            {'test_id': 3, 'word': 'qwe', 'value': 10}
        ],
        cmp_keys=('test_id', 'word', 'value')
    ),
    MapCase(
        mapper=ops.Log(column='value'),
        data=[
            {'test_id': 1, 'word': 'zxc', 'value': math.e},
            {'test_id': 2, 'word': 'asd', 'value': 1},
            {'test_id': 3, 'word': 'qwe', 'value': 1 / math.e}
        ],
        etalon=[
            {'test_id': 1, 'word': 'zxc', 'value': 1},
            {'test_id': 2, 'word': 'asd', 'value': 0},
            {'test_id': 3, 'word': 'qwe', 'value': -1}
        ],
        cmp_keys=('test_id', 'word', 'value')
    ),
    MapCase(
        mapper=ops.Distance(),
        data=[
            {'start': [37.84870228730142, 55.73853974696249], 'end': [37.8490418381989, 55.73832445777953]},
            {'start': [37.524768467992544, 55.88785375468433], 'end': [37.52415172755718, 55.88807155843824]},
            {'start': [37.84870228730142, 55.73853974696249], 'end': [37.84870228730142, 55.73853974696249]}
        ],
        etalon=[
            {'distance': 0.03202394407224201, 'end': [37.8490418381989, 55.73832445777953],
             'start': [37.84870228730142, 55.73853974696249]},
            {'distance': 0.045464188432109455,
             'end': [37.52415172755718, 55.88807155843824],
             'start': [37.524768467992544, 55.88785375468433]},
            {'distance': 0.0,
             'end': [37.84870228730142, 55.73853974696249],
             'start': [37.84870228730142, 55.73853974696249]}
        ],
        cmp_keys=('distance', 'start', 'end')
    ),
    MapCase(
        mapper=ops.TimeDelta(),
        data=[
            {'leave_time': '20211121T202530.427000', 'enter_time': '20211121T192530.427000'},
        ],
        etalon=[
            {'leave_time': '20211121T202530.427000', 'enter_time': '20211121T192530.427000', 'time_delta': 3600}

        ],
        cmp_keys=("leave_time", "enter_time", "time_delta"),
    ),
    MapCase(
        mapper=ops.AverageSpeed(),
        data=[
            {'distance': 1000, 'time_delta': 36000},
            {'distance': 1, 'time_delta': 360000},
        ],
        etalon=[
            {'distance': 1000, 'time_delta': 36000, 'avg_speed': 100},
            {'distance': 1, 'time_delta': 360000, 'avg_speed': 0.01},
        ],
        cmp_keys=("distance", 'time_delta', 'avg_speed')
    ),
    MapCase(
        mapper=ops.DayHour(),
        data=[
            {'datetime': '20211122T203330.427000'},

        ],
        etalon=[
            {'weekday': 'Mon', 'hour': 20, 'datetime': '20211122T203330.427000'},
        ],
        cmp_keys=("datetime", "weekday", "hour")
    ),
]


@pytest.mark.parametrize("case", MAP_CASES)
def test_mapper(case: MapCase) -> None:
    mapper_data_row = copy.deepcopy(case.data[case.mapper_item])
    mapper_etalon_rows = [copy.deepcopy(case.etalon[i]) for i in case.mapper_etalon_items]

    key_func = _Key(*case.cmp_keys)

    mapper_result = case.mapper(mapper_data_row)
    assert isinstance(mapper_result, tp.Iterator)
    assert sorted(mapper_etalon_rows, key=key_func) == sorted(mapper_result, key=key_func)

    result = ops.Map(case.mapper)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(case.etalon, key=key_func) == sorted(result, key=key_func)


@dataclasses.dataclass
class ReduceCase:
    reducer: ops.Reducer
    reducer_keys: tp.Tuple[str, ...]
    data: tp.List[ops.TRow]
    etalon: tp.List[ops.TRow]
    cmp_keys: tp.Tuple[str, ...]
    reduce_data_items: tp.Tuple[int, ...] = (0,)
    reduce_etalon_items: tp.Tuple[int, ...] = (0,)


REDUCE_CASES = [
    ReduceCase(
        reducer=ops.FirstReducer(),
        reducer_keys=('test_id',),
        data=[
            {'test_id': 1, 'text': 'hello, world'},
            {'test_id': 2, 'text': 'bye!'}
        ],
        etalon=[
            {'test_id': 1, 'text': 'hello, world'},
            {'test_id': 2, 'text': 'bye!'}
        ],
        cmp_keys=("test_id", "text")
    ),
    ReduceCase(
        reducer=ops.TopN(column='rank', n=3),
        reducer_keys=('match_id',),
        data=[
            {'match_id': 1, 'player_id': 1, 'rank': 42},
            {'match_id': 1, 'player_id': 2, 'rank': 7},
            {'match_id': 1, 'player_id': 3, 'rank': 0},
            {'match_id': 1, 'player_id': 4, 'rank': 39},

            {'match_id': 2, 'player_id': 5, 'rank': 15},
            {'match_id': 2, 'player_id': 6, 'rank': 39},
            {'match_id': 2, 'player_id': 7, 'rank': 27},
            {'match_id': 2, 'player_id': 8, 'rank': 7}
        ],
        etalon=[
            {'match_id': 1, 'player_id': 1, 'rank': 42},
            {'match_id': 1, 'player_id': 2, 'rank': 7},
            {'match_id': 1, 'player_id': 4, 'rank': 39},

            {'match_id': 2, 'player_id': 5, 'rank': 15},
            {'match_id': 2, 'player_id': 6, 'rank': 39},
            {'match_id': 2, 'player_id': 7, 'rank': 27}
        ],
        cmp_keys=("match_id", "player_id", "rank"),
        reduce_data_items=(0, 1, 2, 3),
        reduce_etalon_items=(0, 1, 2)
    ),
    ReduceCase(
        reducer=ops.TermFrequency(words_column='text'),
        reducer_keys=('doc_id',),
        data=[
            {'doc_id': 1, 'text': 'hello', 'count': 1},
            {'doc_id': 1, 'text': 'little', 'count': 1},
            {'doc_id': 1, 'text': 'world', 'count': 1},

            {'doc_id': 2, 'text': 'little', 'count': 1},

            {'doc_id': 3, 'text': 'little', 'count': 3},
            {'doc_id': 3, 'text': 'little', 'count': 3},
            {'doc_id': 3, 'text': 'little', 'count': 3},

            {'doc_id': 4, 'text': 'little', 'count': 2},
            {'doc_id': 4, 'text': 'hello', 'count': 1},
            {'doc_id': 4, 'text': 'little', 'count': 2},
            {'doc_id': 4, 'text': 'world', 'count': 1},

            {'doc_id': 5, 'text': 'hello', 'count': 2},
            {'doc_id': 5, 'text': 'hello', 'count': 2},
            {'doc_id': 5, 'text': 'world', 'count': 1},

            {'doc_id': 6, 'text': 'world', 'count': 4},
            {'doc_id': 6, 'text': 'world', 'count': 4},
            {'doc_id': 6, 'text': 'world', 'count': 4},
            {'doc_id': 6, 'text': 'world', 'count': 4},
            {'doc_id': 6, 'text': 'hello', 'count': 1}
        ],
        etalon=[
            {'doc_id': 1, 'text': 'hello', 'tf': approx(0.3333, abs=0.001)},
            {'doc_id': 1, 'text': 'little', 'tf': approx(0.3333, abs=0.001)},
            {'doc_id': 1, 'text': 'world', 'tf': approx(0.3333, abs=0.001)},

            {'doc_id': 2, 'text': 'little', 'tf': approx(1.0)},

            {'doc_id': 3, 'text': 'little', 'tf': approx(1.0)},

            {'doc_id': 4, 'text': 'hello', 'tf': approx(0.25)},
            {'doc_id': 4, 'text': 'little', 'tf': approx(0.5)},
            {'doc_id': 4, 'text': 'world', 'tf': approx(0.25)},

            {'doc_id': 5, 'text': 'hello', 'tf': approx(0.666, abs=0.001)},
            {'doc_id': 5, 'text': 'world', 'tf': approx(0.333, abs=0.001)},

            {'doc_id': 6, 'text': 'hello', 'tf': approx(0.2)},
            {'doc_id': 6, 'text': 'world', 'tf': approx(0.8)}
        ],
        cmp_keys=("doc_id", "text", "tf"),
        reduce_data_items=(0, 1, 2),
        reduce_etalon_items=(0, 1, 2)
    ),
    ReduceCase(
        reducer=ops.Count(column='count'),
        reducer_keys=("word",),
        data=[
            {'sentence_id': 2, 'word': 'hell'},
            {'sentence_id': 1, 'word': 'hello'},
            {'sentence_id': 2, 'word': 'hello'},
            {'sentence_id': 1, 'word': 'little'},
            {'sentence_id': 2, 'word': 'little'},
            {'sentence_id': 2, 'word': 'little'},
            {'sentence_id': 1, 'word': 'my'},
            {'sentence_id': 2, 'word': 'my'},
            {'sentence_id': 1, 'word': 'world'},
        ],
        etalon=[
            {'count': 1, 'word': 'hell'},
            {'count': 1, 'word': 'world'},
            {'count': 2, 'word': 'hello'},
            {'count': 2, 'word': 'my'},
            {'count': 3, 'word': 'little'}
        ],
        cmp_keys=("count", "word"),
        reduce_data_items=(1, 2),
        reduce_etalon_items=(2,)
    ),
    ReduceCase(
        reducer=ops.Sum(column='score'),
        reducer_keys=("match_id",),
        data=[
            {'match_id': 1, 'player_id': 1, 'score': 42},
            {'match_id': 1, 'player_id': 2, 'score': 7},
            {'match_id': 1, 'player_id': 3, 'score': 0},
            {'match_id': 1, 'player_id': 4, 'score': 39},

            {'match_id': 2, 'player_id': 5, 'score': 15},
            {'match_id': 2, 'player_id': 6, 'score': 39},
            {'match_id': 2, 'player_id': 7, 'score': 27},
            {'match_id': 2, 'player_id': 8, 'score': 7}
        ],
        etalon=[
            {'match_id': 1, 'score': 88},
            {'match_id': 2, 'score': 88}
        ],
        cmp_keys=("test_id", "text"),
        reduce_data_items=(0, 1, 2, 3),
        reduce_etalon_items=(0,)
    ),
    ReduceCase(
        reducer=ops.Mean(column='count'),
        reducer_keys=("word",),
        data=[
            {'count': 0, 'word': 'zxc'},
            {'count': 1, 'word': 'zxc'},
            {'count': 3, 'word': 'qwe'},
            {'count': -3, 'word': 'qwe'},
            {'count': 100, 'word': '123'},
            {'count': 100, 'word': '123'},
            {'count': 1000, 'word': '123'},

        ],
        etalon=[
            {'count': 0.5, 'word': 'zxc'},
            {'count': 0., 'word': 'qwe'},
            {'count': 400, 'word': '123'},
        ],
        cmp_keys=('word', 'count'),
        reduce_data_items=(0, 1),
        reduce_etalon_items=(0,)
    ),
    ReduceCase(
        reducer=ops.TermFrequencyAfterCount(words_column='text'),
        reducer_keys=('doc_id',),
        data=[
            {'doc_id': 1, 'text': 'hello', 'count': 1},
            {'doc_id': 1, 'text': 'little', 'count': 1},
            {'doc_id': 1, 'text': 'world', 'count': 1},

            {'doc_id': 2, 'text': 'little', 'count': 1},

            {'doc_id': 3, 'text': 'little', 'count': 3},
            {'doc_id': 3, 'text': 'little', 'count': 3},
            {'doc_id': 3, 'text': 'little', 'count': 3},

            {'doc_id': 4, 'text': 'little', 'count': 2},
            {'doc_id': 4, 'text': 'hello', 'count': 1},
            {'doc_id': 4, 'text': 'little', 'count': 2},
            {'doc_id': 4, 'text': 'world', 'count': 1},

            {'doc_id': 5, 'text': 'hello', 'count': 2},
            {'doc_id': 5, 'text': 'hello', 'count': 2},
            {'doc_id': 5, 'text': 'world', 'count': 1},

            {'doc_id': 6, 'text': 'world', 'count': 4},
            {'doc_id': 6, 'text': 'world', 'count': 4},
            {'doc_id': 6, 'text': 'world', 'count': 4},
            {'doc_id': 6, 'text': 'world', 'count': 4},
            {'doc_id': 6, 'text': 'hello', 'count': 1}
        ],
        etalon=[
            {'doc_id': 1, 'text': 'hello', 'tf': approx(0.3333, abs=0.001)},
            {'doc_id': 1, 'text': 'little', 'tf': approx(0.3333, abs=0.001)},
            {'doc_id': 1, 'text': 'world', 'tf': approx(0.3333, abs=0.001)},

            {'doc_id': 2, 'text': 'little', 'tf': approx(1.0)},

            {'doc_id': 3, 'text': 'little', 'tf': approx(1.0)},

            {'doc_id': 4, 'text': 'hello', 'tf': approx(0.1666, abs=0.001)},
            {'doc_id': 4, 'text': 'little', 'tf': approx(0.6666, abs=0.001)},
            {'doc_id': 4, 'text': 'world', 'tf': approx(0.1666, abs=0.001)},

            {'doc_id': 5, 'text': 'hello', 'tf': approx(0.8)},
            {'doc_id': 5, 'text': 'world', 'tf': approx(0.2)},

            {'doc_id': 6, 'text': 'hello', 'tf': approx(0.0588, abs=0.001)},
            {'doc_id': 6, 'text': 'world', 'tf': approx(0.9411, abs=0.001)}
        ],
        cmp_keys=("doc_id", "text", "tf"),
        reduce_data_items=(0, 1, 2),
        reduce_etalon_items=(0, 1, 2)
    ),

]


@pytest.mark.parametrize("case", REDUCE_CASES)
def test_reducer(case: ReduceCase) -> None:
    reducer_data_rows = [copy.deepcopy(case.data[i]) for i in case.reduce_data_items]
    reducer_etalon_rows = [copy.deepcopy(case.etalon[i]) for i in case.reduce_etalon_items]

    key_func = _Key(*case.cmp_keys)

    reducer_result = case.reducer(case.reducer_keys, iter(reducer_data_rows))
    assert isinstance(reducer_result, tp.Iterator)
    assert sorted(reducer_etalon_rows, key=key_func) == sorted(reducer_result, key=key_func)

    result = ops.Reduce(case.reducer, case.reducer_keys)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(case.etalon, key=key_func) == sorted(result, key=key_func)


@dataclasses.dataclass
class GraphMapCase:
    mappers: tp.List[tp.Any]
    etalon: int


GRAPH_MAP_CASES = [
    GraphMapCase(
        mappers=[ops.LowerCase, ops.FilterPunctuation, ops.Split],
        etalon=4
    ),
    GraphMapCase(
        mappers=[],
        etalon=1
    ),
]


@pytest.mark.parametrize("case", GRAPH_MAP_CASES)
def test_graph_map_case(case: GraphMapCase) -> None:
    g = Graph.graph_from_iter('zxc')
    for mapper in case.mappers:
        g = g.map(mapper)

    assert len(g._operations) == case.etalon


@dataclasses.dataclass
class GraphReduceCase:
    reducers: tp.List[tp.Any]
    keys: tp.List[tp.List[str]]
    etalon: int


GRAPH_REDUCE_CASES = [
    GraphReduceCase(
        reducers=[ops.TermFrequencyAfterCount, ops.TermFrequency],
        keys=[["1"], ["2"]],
        etalon=3
    ),
    GraphReduceCase(
        reducers=[],
        keys=[],
        etalon=1
    ),
]


@pytest.mark.parametrize("case", GRAPH_REDUCE_CASES)
def test_graph_reduce_case(case: GraphReduceCase) -> None:
    g = Graph.graph_from_iter('zxc')
    for reducer, key in zip(case.reducers, case.keys):
        g = g.reduce(reducer, key)

    assert len(g._operations) == case.etalon


@dataclasses.dataclass
class GraphSortCase:
    keys: tp.List[tp.List[str]]
    etalon: int


GRAPH_SORT_CASES = [
    GraphSortCase(
        keys=[["1"], ["1"]],
        etalon=3
    ),
    GraphSortCase(
        keys=[],
        etalon=1
    ),
]


@pytest.mark.parametrize("case", GRAPH_SORT_CASES)
def test_graph_sort_case(case: GraphSortCase) -> None:
    g = Graph.graph_from_iter('zxc')
    for key in case.keys:
        g = g.sort(key)

    assert len(g._operations) == case.etalon


@dataclasses.dataclass
class GraphJoinCase:
    joiners: tp.List[tp.Any]
    graphs: tp.List['Graph']
    keys: tp.List[tp.List[str]]
    etalon: int


GRAPH_JOIN_CASES = [
    GraphJoinCase(
        joiners=[ops.OuterJoiner, ops.InnerJoiner, ops.OuterJoiner],
        graphs=[Graph.graph_from_iter('qwe'), Graph.graph_from_iter('asd'), Graph.graph_from_iter('zxc')],
        keys=[["1"], ["2"], ["3"]],
        etalon=3
    ),
    GraphJoinCase(
        joiners=[],
        graphs=[],
        keys=[],
        etalon=0
    ),
]


@pytest.mark.parametrize("case", GRAPH_JOIN_CASES)
def test_graph_join_case(case: GraphJoinCase) -> None:
    g = Graph.graph_from_iter('123')
    for joiner, graph, key in zip(case.joiners, case.graphs, case.keys):
        g = g.join(joiner, graph, key)

    assert len(g._join_list) == case.etalon


def test_graph_run_case_1() -> None:
    graph = alg.word_count_graph('docs', text_column='text', count_column='count')

    docs = [
        {'doc_id': 1, 'text': 'hello, my little WORLD'},
        {'doc_id': 2, 'text': 'Hello, my little little hell'}
    ]

    expected = [
        {'count': 1, 'text': 'hell'},
        {'count': 1, 'text': 'world'},
        {'count': 2, 'text': 'hello'},
        {'count': 2, 'text': 'my'},
        {'count': 3, 'text': 'little'}
    ]

    result = graph.run(docs=lambda: iter(docs))

    assert expected == list(result)


def test_sequent_calls_case_0() -> None:
    docs = [
        {'doc_id': 1, 'text': 'hello, my little WORLD'},
    ]

    expected = [
        {'count': 1, 'text': 'hello'},

        {'count': 1, 'text': 'little'},
        {'count': 1, 'text': 'my'},
        {'count': 1, 'text': 'world'},
    ]

    g = Graph.graph_from_iter('docs')
    text_column = 'text'
    count_column = 'count'
    g = g \
        .map(ops.FilterPunctuation(text_column)) \
        .map(ops.LowerCase(text_column)) \
        .map(ops.Split(text_column)) \
        .sort([text_column]) \
        .reduce(ops.Count(count_column), [text_column]) \
        .sort([count_column, text_column])

    res = g.run(docs=lambda: iter(docs))
    assert list(res) == expected


def test_sequent_calls_case_1() -> None:
    docs = [
        {'doc_id': 1, 'text': 'hello, my little WORLD'},
    ]

    expected = [
        {'count': 1, 'text': 'hello'},

        {'count': 1, 'text': 'my'},
        {'count': 1, 'text': 'little'},

        {'count': 1, 'text': 'world'},
    ]

    g = Graph.graph_from_iter('docs')
    text_column = 'text'
    count_column = 'count'
    g = g \
        .map(ops.FilterPunctuation(text_column)) \
        .map(ops.LowerCase(text_column)) \
        .map(ops.Split(text_column)) \
        .reduce(ops.Count(count_column), [text_column]) \

    res = g.run(docs=lambda: iter(docs))
    assert list(res) == expected


def test_sequent_calls_case_2() -> None:
    docs = [
        {'doc_id': 1, 'text': 'hello, my little WORLD'},
    ]

    expected = [
        {'doc_id': 1, 'text': 'hello'},
        {'doc_id': 1, 'text': 'my'},
        {'doc_id': 1, 'text': 'little'},
        {'doc_id': 1, 'text': 'world'},
    ]

    g = Graph.graph_from_iter('docs')
    text_column = 'text'
    g = g \
        .map(ops.FilterPunctuation(text_column)) \
        .map(ops.LowerCase(text_column)) \
        .map(ops.Split(text_column))

    res = g.run(docs=lambda: iter(docs))
    assert list(res) == expected


def test_word_count() -> None:
    filename_in = 'file_in.txt'
    with open(filename_in, 'w') as fp:
        fp.write("{'doc_id': 1, 'text': 'hello, my little WORLD'}\n"
                 "{'doc_id': 2, 'text': 'Hello, my little little hell'}\n")

    filename_out = 'file_out.txt'
    t = open('file_out.txt', 'w+')
    t.close()

    expected = [
        {'count': 1, 'text': 'hell'},
        {'count': 1, 'text': 'world'},
        {'count': 2, 'text': 'hello'},
        {'count': 2, 'text': 'my'},
        {'count': 3, 'text': 'little'}
    ]

    Runner.invoke(run_word_count.main, filename_in + ' ' + filename_out)

    with open(filename_out) as f:
        for i, line in enumerate(f):
            assert expected[i] == eval(line)

    os.remove(filename_in)
    os.remove(filename_out)


test_word_count()


def test_pmi() -> None:
    filename_in = 'tmp_file_for_test.txt'
    with open(filename_in, 'w') as fp:
        fp.write(
            "{'doc_id': 1, 'text': 'hello, little world'}\n"
            "{'doc_id': 2, 'text': 'little'}\n"
            "{'doc_id': 3, 'text': 'little little little'}\n"
            "{'doc_id': 4, 'text': 'little? hello little world'}\n"
            "{'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'}\n"
            "{'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!! HELLO!!!!!!!'}\n")

    filename_out = 'tmp_file_for_test_out.txt'
    t = open('tmp_file_for_test_out.txt', 'w+')
    t.close()

    expected = [
        {'doc_id': 3, 'text': 'little', 'pmi': approx(0.9555, 0.001)},
        {'doc_id': 4, 'text': 'little', 'pmi': approx(0.9555, 0.001)},
        {'doc_id': 5, 'text': 'hello', 'pmi': approx(1.1786, 0.001)},
        {'doc_id': 6, 'text': 'world', 'pmi': approx(0.7731, 0.001)},
        {'doc_id': 6, 'text': 'hello', 'pmi': approx(0.0800, 0.001)},
    ]

    Runner.invoke(run_pmi.main, filename_in + ' ' + filename_out)

    with open(filename_out) as f:
        for i, line in enumerate(f):
            assert expected[i] == eval(line)

    os.remove(filename_in)
    os.remove(filename_out)


test_pmi()
