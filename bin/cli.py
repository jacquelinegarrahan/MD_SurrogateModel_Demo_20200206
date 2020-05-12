import click
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from bin.commands.serve import serve


@click.group()
def cli():
    pass


cli.add_command(serve)

if __name__ == "__main__":
    cli()
