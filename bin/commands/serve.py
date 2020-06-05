import click
import os
import numpy as np


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

    from online_model import server
    from online_model.model.MySurrogateModel import MySurrogateModel

    PREFIX = "smvm"

    MODEL_FILE = "online_model/files/CNN_051620_SurrogateModel.h5"
    STOCK_LASER_IMAGE = "online_model/files/example_input_image.npy"
    ARRAY_PVS = ["x:y", "image"]

    MODEL_KWARGS = {
        "model_file": MODEL_FILE,
        "stock_image_input": np.load(STOCK_LASER_IMAGE),
    }

    pv_server = server.get_server(
        PREFIX,
        MySurrogateModel,
        MODEL_KWARGS,
        protocol,
        data_file,
        from_xarray,
        ARRAY_PVS,
    )
    pv_server.start_server()
