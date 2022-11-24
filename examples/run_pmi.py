import click

from compgraph.algorithms import pmi_graph


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=True))
@click.argument('doc_column', type=str, default='doc_id')
@click.argument('text_column', type=str, default='text')
@click.argument('result_column', type=str, default='pmi')
def main(input_file: str, output: str, doc_column: str = 'doc_id',
         text_column: str = 'text', result_column: str = 'pmi') -> None:
    graph = pmi_graph(input_file, doc_column, text_column, result_column, eval)

    result = graph.run()
    with open(output, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == '__main__':
    main()
