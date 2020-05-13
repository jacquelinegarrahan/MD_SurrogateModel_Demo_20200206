import click
import json


@click.group()
def serve():
    pass


@serve.command()
def start_ca_server():
    from online_model.model.surrogate_model import OnlineSurrogateModel
    from online_model.server.ca import SyncedSimPVServer
    from online_model.util import fix_units

    vmname = "smvm"

    sm = OnlineSurrogateModel()

    # Start with the nice example from Lipi
    default_inputs = {
        "maxb(2)": 0.06125866317542922,
        "phi(1)": 8.351877669807294,
        "q_total": 0.020414630732101164,
        "sig_x": 0.4065596830730608,
    }
    default_output = sm.run(default_inputs)

    cmd_pvdb = {}
    for ii, input_name in enumerate(sm.scalar_model.input_names):
        cmd_pvdb[input_name] = {
            "type": "float",
            "prec": 8,
            "value": default_inputs[input_name],
            "units": fix_units(sm.scalar_model.input_units[ii]),
            "range": list(sm.scalar_model.input_ranges[ii]),
        }

    sim_pvdb = {}
    for ii, output_name in enumerate(sm.scalar_model.output_names):
        sim_pvdb[output_name] = {
            "type": "float",
            "prec": 8,
            "value": default_output[output_name],
            "units": fix_units(sm.scalar_model.output_units[ii]),
        }

    # sim_pvdb['z:pz']={'type': 'float', 'prec': 8, 'count':len(default_output['z:pz']),'units':'mm:delta','value':list(default_output['z:pz'])}
    sim_pvdb["x:y"] = {
        "type": "float",
        "prec": 8,
        "count": len(default_output["x:y"]),
        "units": "mm:mm",
        "value": list(default_output["x:y"]),
    }

    pv_def = {"prefix": vmname + ":", "input": cmd_pvdb, "output": sim_pvdb}
    with open("pvdef.json", "w") as fp:
        json.dump(pv_def, fp, sort_keys=True, indent=4)

    # Add in noise for fun
    # sim_pvdb['x_95coremit']['scan']=0.2
    # noise_params = {'x_95coremit':{'sigma':0.5e-7,'dist':'uniform'}}
    noise_params = {}

    server = SyncedSimPVServer(vmname, cmd_pvdb, sim_pvdb, noise_params, sm)
    server.start_server()


@serve.command()
@click.argument("protocol")
def start_server(protocol):
    """
    Start server using given PROTOCOL.

    PROTOCOL options are 'ca' and 'pva'
    """

    vmname = "smvm"

    from online_model.model.surrogate_model import OnlineSurrogateModel
    from online_model.util import fix_units

    sm = OnlineSurrogateModel()

    # Start with the nice example from Lipi
    default_inputs = {
        "maxb(2)": 0.06125866317542922,
        "phi(1)": 8.351877669807294,
        "q_total": 0.020414630732101164,
        "sig_x": 0.4065596830730608,
    }
    default_output = sm.run(default_inputs)

    cmd_pvdb = {}
    for ii, input_name in enumerate(sm.scalar_model.input_names):
        cmd_pvdb[input_name] = {
            "type": "float",
            "prec": 8,
            "value": default_inputs[input_name],
            "units": fix_units(sm.scalar_model.input_units[ii]),
            "range": list(sm.scalar_model.input_ranges[ii]),
        }

    sim_pvdb = {}
    for ii, output_name in enumerate(sm.scalar_model.output_names):
        sim_pvdb[output_name] = {
            "type": "float",
            "prec": 8,
            "value": default_output[output_name],
            "units": fix_units(sm.scalar_model.output_units[ii]),
        }

    # sim_pvdb['z:pz']={'type': 'float', 'prec': 8, 'count':len(default_output['z:pz']),'units':'mm:delta','value':list(default_output['z:pz'])}
    sim_pvdb["x:y"] = {
        "type": "float",
        "prec": 8,
        "count": len(default_output["x:y"]),
        "units": "mm:mm",
        "value": list(default_output["x:y"]),
    }

    pv_def = {"prefix": vmname + ":", "input": cmd_pvdb, "output": sim_pvdb}

    with open("pvdef.json", "w") as fp:
        json.dump(pv_def, fp, sort_keys=True, indent=4)

    if protocol == "ca":
        from online_model.server.ca import SyncedSimPVServer

        noise_params = {}

        server = SyncedSimPVServer(vmname, cmd_pvdb, sim_pvdb, noise_params, sm)
        server.start_server()

    elif protocol == "pva":
        from online_model.server.pva import PVAServer

        server = PVAServer(cmd_pvdb, sim_pvdb)
        server.start_server()

    else:
        print("Given protocol %s is not supported.", protocol)
