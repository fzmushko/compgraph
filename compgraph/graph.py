import typing as tp
from . import operations as ops
from .external_sort import ExternalSort as Esort


class Graph:
    """Computational graph implementation"""
    def __init__(self, operations: list[ops.Operation]) -> None:
        self._operations = operations
        self._join_list: list['Graph'] = []

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        return Graph(operations=[ops.ReadIterFactory(name=name)])

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        return Graph(operations=[ops.Read(filename=filename, parser=parser)])

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        self._operations.append(ops.Map(mapper=mapper))
        return self

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        self._operations.append(ops.Reduce(keys=keys, reducer=reducer))
        return self

    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        self._operations.append(Esort(keys=keys))
        return self

    def join(self, joiner: ops.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        self._operations.append(ops.Join(joiner=joiner, keys=keys))
        self._join_list.append(join_graph)
        return self

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        assert len(self._operations) > 0 and (isinstance(self._operations[0], ops.ReadIterFactory) or
                                              isinstance(self._operations[0], ops.Read)), 'Wrong first operation'
        table = self._operations[0](**kwargs)
        cur_joiner = 0

        for operation in self._operations[1:]:
            if isinstance(operation, ops.Join):
                right_table = self._join_list[cur_joiner].run(**kwargs)
                cur_joiner += 1

                table = operation(table, right_table)
            else:
                table = operation(table)

        yield from table
