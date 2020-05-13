import click
import json


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

    vmname = "smvm"

    from online_model.util import fix_units
    from online_model.model.surrogate_model import OnlineSurrogateModel
    from online_model import CMD_PVDB, SIM_PVDB

    if protocol == "ca":
        from online_model.server.ca import SyncedSimPVServer

        model = OnlineSurrogateModel()

        noise_params = {}

        server = SyncedSimPVServer(
            vmname, CMD_PVDB, SIM_PVDB, model, noise_params=noise_params
        )
        server.start_server()

    elif protocol == "pva":
        from online_model.server.pva import PVAServer

        server = PVAServer(CMD_PVDB, SIM_PVDB)
        server.start_server()

    else:
        print("Given protocol %s is not supported.", protocol)
