import click


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
    from online_model import CMD_PVDB, SIM_PVDB

    if protocol == "ca":
        from online_model.server.ca import CAServer

        server = CAServer(CMD_PVDB, SIM_PVDB)
        server.start_server()

    elif protocol == "pva":
        from online_model.server.pva import PVAServer

        server = PVAServer(CMD_PVDB, SIM_PVDB)
        server.start_server()

    else:
        print("Given protocol %s is not supported.", protocol)
