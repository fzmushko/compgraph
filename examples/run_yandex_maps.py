import click

from compgraph.algorithms import yandex_maps_graph


@click.command()
@click.argument('input_stream_name_time', type=click.Path(exists=True))
@click.argument('input_stream_name_length', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=True))
@click.argument('enter_time_column', type=str, default='enter_time')
@click.argument('leave_time_column', type=str, default='leave_time')
@click.argument('edge_id_column', type=str, default='edge_id')
@click.argument('start_coord_column', type=str, default='start')
@click.argument('end_coord_column', type=str, default='end')
@click.argument('weekday_result_column', type=str, default='weekday')
@click.argument('weekday_result_column', type=str, default='weekday')
@click.argument('hour_result_column', type=str, default='hour')
@click.argument('speed_result_column', type=str, default='speed')
def main(input_stream_name_time: str, input_stream_name_length: str, output: str,
         enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
         edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
         weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
         speed_result_column: str = 'speed') -> None:

    graph = yandex_maps_graph(input_stream_name_time, input_stream_name_length,
                              enter_time_column, leave_time_column,
                              edge_id_column, start_coord_column, end_coord_column,
                              weekday_result_column, hour_result_column,
                              speed_result_column, parser=eval)

    result = graph.run()
    with open(output, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == '__main__':
    main()
