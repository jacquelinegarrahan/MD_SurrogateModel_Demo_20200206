import click
import os


@click.group()
def serve():
    pass


@serve.command()
@click.argument("protocol")
def start_server(protocol):
    """
    Start server using given PROTOCOL.

    PROTOCOL options are 'ca' and 'pva'
    """

    # the protocol must be set to properly assemble the CMD_PVDB, SIM_PVDB etc.
    os.environ["PROTOCOL"] = protocol

    from online_model.model.MySurrogateModel import MySurrogateModel
    from online_model import CMD_PVDB, SIM_PVDB, MODEL_KWARGS, PREFIX

    if protocol == "ca":
        from online_model.server.ca import CAServer

        server = CAServer(MySurrogateModel, MODEL_KWARGS, CMD_PVDB, SIM_PVDB, PREFIX)
        server.start_server()

    elif protocol == "pva":
        from online_model.server.pva import PVAServer

        server = PVAServer(MySurrogateModel, MODEL_KWARGS, CMD_PVDB, SIM_PVDB, PREFIX)
        server.start_server()

    else:
        print("Given protocol %s is not supported.", protocol)
