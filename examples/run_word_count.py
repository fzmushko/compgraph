import click

from compgraph.algorithms import word_count_graph


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=True))
@click.argument('text_column', type=str, default='text')
@click.argument('count_column', type=str, default='count')
def main(input_file: str, output: str, text_column: str = 'text', count_column: str = 'count') -> None:
    graph = word_count_graph(input_file, text_column, count_column, parser=eval)

    result = graph.run()
    with open(output, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    main()
