import click
import os


@click.group()
def serve():
    pass


@serve.command()
@click.argument("protocol")
@click.argument("data_file")
@click.option("--from-xarray/--from-classes", default=False)
def start_server(protocol: str, data_file: str, from_xarray: bool):
    """
    Start server using given PROTOCOL.

    PROTOCOL options are 'ca' and 'pva'
    """
    os.environ["PROTOCOL"] = "pva"

    from online_model import server
    from online_model.model.MySurrogateModel import MySurrogateModel
    from online_model import MODEL_KWARGS, PREFIX

    pv_server = server.get_server(
        PREFIX, MySurrogateModel, MODEL_KWARGS, protocol, data_file, from_xarray
    )
    pv_server.start_server()
