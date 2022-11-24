import heapq
import itertools
import math
import re
import typing as tp
from abc import abstractmethod, ABC
from datetime import datetime


TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""
    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            yield from self.mapper.__call__(row)


class Reducer(ABC):
    """Base class for reducers"""
    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for k, group in itertools.groupby(rows, key=lambda x: [x[key] for key in self.keys]):
            yield from self.reducer.__call__(tuple(self.keys), group)


class Joiner(ABC):
    """Base class for joiners"""
    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        left_data = itertools.groupby(rows, key=lambda x: [x[key] for key in self.keys])
        right_data = itertools.groupby(args[0], key=lambda x: [x[key] for key in self.keys])
        key1, group1 = next(left_data)
        key2, group2 = next(right_data)

        flag1, flag2 = False, False
        empty: TRow = {}
        while True:
            if flag1 and flag2:
                break
            if flag1:
                yield from self.joiner.__call__(self.keys, [empty], group2)
                for k2, g2 in right_data:
                    yield from self.joiner.__call__(self.keys, [empty], g2)
                break
            elif flag2:
                yield from self.joiner.__call__(self.keys, group1, [empty])
                for k1, g1 in left_data:
                    yield from self.joiner.__call__(self.keys, g1, [empty])
                break

            if key1 == key2:
                yield from self.joiner.__call__(self.keys, group1, group2)
                try:
                    key1, group1 = next(left_data)
                except StopIteration:
                    flag1 = True
                    pass
                try:
                    key2, group2 = next(right_data)
                except StopIteration:
                    flag2 = True
                    pass

            elif key2 < key1:
                while key2 < key1:
                    yield from self.joiner.__call__(self.keys, [empty], group2)
                    try:
                        key2, group2 = next(right_data)
                    except StopIteration:
                        flag2 = True
                        pass

            else:
                while key1 < key2:
                    yield from self.joiner.__call__(self.keys, group1, [empty])
                    try:
                        key1, group1 = next(left_data)
                    except StopIteration:
                        flag1 = True
                        pass


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""
    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = re.sub(r'[^a-zA-Z0-9\s]', '', row[self.column])
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self._lower_case(row[self.column])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: tp.Optional[str] = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    @staticmethod
    def _split_iter(separator: str) -> tp.Iterator[str]:
        return (x.group(0) for x in re.finditer(r"[A-Za-z']+", separator))

    def __call__(self, row: TRow) -> TRowsGenerator:
        for word in self._split_iter(row[self.column]):
            ans: TRow = {}
            for key in row.keys():
                if key != self.column:
                    ans[key] = row[key]
                else:
                    ans[self.column] = word
            yield ans


class Product(Mapper):
    """Calculates product of multiple columns"""
    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        ans = 1
        for cur_col in self.columns:
            ans *= row[cur_col]
        row[self.result_column] = ans
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""
    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        is_ok = self.condition(row)
        if is_ok:
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""
    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        ans: TRow = {}
        for cur_col in self.columns:
            ans[cur_col] = row[cur_col]
        yield ans


class Inverse(Mapper):
    """x -> 1/x for numbers in given column"""
    def __init__(self, column: str) -> None:
        """
        :param column: name of the column to apply
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = 1 / row[self.column]
        yield row


class Log(Mapper):
    """x -> log(x) for numbers in given column"""
    def __init__(self, column: str = 'idf') -> None:
        """
        :param column: name of the column to apply
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = math.log(row[self.column])
        yield row


class Distance(Mapper):
    """Calculating the distance in kilometers from the given coordinates in lon/lat format"""
    def __init__(self, start: str = 'start', end: str = 'end', distance: str = 'distance') -> None:
        """
        :param start: start coordinates in lon/lat format
        :param end: end coordinate in lon/lat format
        :param distance: name for result column
        """
        self.start = start
        self.end = end
        self.result_column = distance
        self.planet_rad = 6373.0

    def __call__(self, row: TRow) -> TRowsGenerator:
        lon_start, lat_start = math.radians(row[self.start][0]), math.radians(row[self.start][1])
        lon_end, lat_end = math.radians(row[self.end][0]), math.radians(row[self.end][1])

        length = (math.sin(lat_start) * math.sin(lat_end)) + \
                 (math.cos(lat_start) * math.cos(lat_end) * math.cos(lon_start-lon_end))

        row[self.result_column] = self.planet_rad * math.acos(length)
        yield row


class DayHour(Mapper):
    """Calculating weekday and hour of datetime in ISO-8601 format"""
    def __init__(self, time: str = 'datetime', weekday: str = 'weekday', hour: str = 'hour') -> None:
        """
        :param time: datetime in ISO-8601 format
        :param weekday: name of result column for weekday
        :param hour: name of result column for hour
        """
        self.time = time
        self.weekday = weekday
        self.hour = hour

    def __call__(self, row: TRow) -> TRowsGenerator:
        time = datetime.strptime(row[self.time], "%Y%m%dT%H%M%S.%f")
        row[self.hour] = time.hour

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        row[self.weekday] = days[time.weekday()]
        yield row


class TimeDelta(Mapper):
    """Calculating timedelta in seconds by given start and end time in ISO-8601 format"""
    def __init__(self, start: str = 'enter_time', end: str = 'leave_time', time_delta: str = 'time_delta') -> None:
        """
        :param start: start time in ISO-8601 format
        :param end: end time in ISO-8601 format
        :param time_delta: name for result column
        """
        self.start = start
        self.end = end
        self.result_column = time_delta

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result_column] = (datetime.strptime(row[self.end], "%Y%m%dT%H%M%S.%f") -
                                   datetime.strptime(row[self.start], "%Y%m%dT%H%M%S.%f")).total_seconds()
        yield row


class AverageSpeed(Mapper):
    """Calculating average speed (km/h) by given distance (km) and time (s)"""
    def __init__(self, distance: str = 'distance', time_delta: str = 'time_delta', avg_speed: str = 'avg_speed'):
        """
        :param distance: total distance
        :param time_delta: total time
        :param avg_speed: name of result column for avg speed
        """
        self.distance = distance
        self.time = time_delta
        self.avg_speed = avg_speed
        self.hour = 3600

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.avg_speed] = row[self.distance] * self.hour / row[self.time]
        yield row


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""
    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        h: list[tuple[int, tuple[tp.Any, ...]]] = []
        i = 0
        for row in rows:
            if len(h) < self.n:
                heapq.heappush(h, (row[self.column_max], tuple(row.items())))
            else:
                heapq.heappushpop(h, (row[self.column_max], tuple(row.items())))
            i += 1
        for item in h:
            yield dict(item[1])


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""
    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        words_counter: dict[str, int] = {}
        ans: TRow = {}
        n = 0
        for row in rows:
            if not ans:
                for key in group_key:
                    ans[key] = row[key]
            n += 1
            if row[self.words_column] in words_counter.keys():
                words_counter[row[self.words_column]] += 1
            else:
                words_counter[row[self.words_column]] = 1

        for key, value in words_counter.items():
            ans_row: TRow = {}
            for k in ans.keys():
                ans_row[k] = ans[k]
            ans_row[self.words_column] = key
            ans_row[self.result_column] = value / n
            yield ans_row


class TermFrequencyAfterCount(Reducer):
    """Calculate frequency of values in column with given count"""
    def __init__(self, words_column: str, count_column: str = 'count', result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param count_column: name for column with counts
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.count_column = count_column
        self.result_column = result_column

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        words_counter: dict[str, int] = {}
        ans: TRow = {}
        n = 0
        for row in rows:
            if not ans:
                for key in group_key:
                    ans[key] = row[key]
            n += row[self.count_column]
            if row[self.words_column] in words_counter.keys():
                words_counter[row[self.words_column]] += row[self.count_column]
            else:
                words_counter[row[self.words_column]] = row[self.count_column]
        for key, value in words_counter.items():
            ans_row: TRow = {}
            for k in ans.keys():
                ans_row[k] = ans[k]
            ans_row[self.words_column] = key
            ans_row[self.result_column] = value / n
            yield ans_row


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        ans: TRow = {}
        cnt = 0
        ok = False
        for row in rows:
            if not ok:
                for key in group_key:
                    ans[key] = row[key]
                ok = True
            cnt += 1
        ans[self.column] = cnt
        yield ans


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        column_sum = 0
        ans: TRow = {}
        ok = False
        for row in rows:
            if not ok:
                for key in group_key:
                    ans[key] = row[key]
                ok = True
            column_sum += row[self.column]
        ans[self.column] = column_sum
        yield ans


class Mean(Reducer):
    """
    Mean value aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 4, 'c': 5}
        =>
        {'a': 1, 'b': 3}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        ans: TRow = {}
        column_sum = 0
        n = 0
        for row in rows:
            if not ans:
                for key in group_key:
                    ans[key] = row[key]
            n += 1
            column_sum += row[self.column]
        ans[self.column] = column_sum / n
        yield ans

# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        empty: TRow = {}
        rows = list(rows_b)
        if rows != [empty]:
            for row_a in rows_a:
                if not row_a:
                    continue
                for row_b in rows:
                    ans: TRow = {}
                    for col in row_a.keys():
                        if col in row_b.keys() and row_a[col] != row_b[col]:
                            ans[col + self._a_suffix] = row_a[col]
                            ans[col + self._b_suffix] = row_b[col]
                        else:
                            ans[col] = row_a[col]
                    for col in row_b.keys():
                        if col not in row_a.keys():
                            ans[col] = row_b[col]
                    yield ans


class OuterJoiner(Joiner):
    """Join with outer strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows = list(rows_b)
        if not rows:
            empty: TRow = {}
            rows.append(empty)
        for row_a in rows_a:
            for row_b in rows:
                ans: TRow = {}
                for col in row_a.keys():
                    if col in row_b.keys() and row_a[col] != row_b[col]:
                        ans[col + self._a_suffix] = row_a[col]
                        ans[col + self._b_suffix] = row_b[col]
                    else:
                        ans[col] = row_a[col]
                for col in row_b.keys():
                    if col not in row_a.keys():
                        ans[col] = row_b[col]
                yield ans


class LeftJoiner(Joiner):
    """Join with left strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows = list(rows_b)
        for row_a in rows_a:
            if row_a == {}:
                continue
            for row_b in rows:
                ans: TRow = {}
                for col in row_a.keys():
                    if col in row_b.keys() and row_a[col] != row_b[col]:
                        ans[col + self._a_suffix] = row_a[col]
                        ans[col + self._b_suffix] = row_b[col]
                    else:
                        ans[col] = row_a[col]
                for col in row_b.keys():
                    if col not in row_a.keys():
                        ans[col] = row_b[col]
                yield ans


class RightJoiner(Joiner):
    """Join with right strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows = list(rows_a)
        for row_b in rows_b:
            if row_b == {}:
                continue
            for row_a in rows:
                ans: TRow = {}
                for col in row_a.keys():
                    if col in row_b.keys() and row_a[col] != row_b[col]:
                        ans[col + self._a_suffix] = row_a[col]
                        ans[col + self._b_suffix] = row_b[col]
                    else:
                        ans[col] = row_a[col]
                for col in row_b.keys():
                    if col not in row_a.keys():
                        ans[col] = row_b[col]
                yield ans
