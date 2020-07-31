from functools import partial
from typing import Union, List

from bokeh.models import Slider

from online_model.app.controllers import Controller
from online_model import PREFIX, ARRAY_PVS, EXCLUDE_SLIDERS


def set_pv_from_slider(
    attr: str,
    old: float,
    new: float,
    pvname: str,
    scale: Union[float, int],
    controller: Controller,
) -> None:
    """
    Callback function for slider change.

    Parameters
    ----------
    attr:str
        Attribute to update

    old:float
        Old value

    new:float
        New value

    pvname: str
        Process variable name

    scale:float/int
        Scale of the slider


    controller: online_model.app.widgets.controllers.Controller
        Controller object for getting pv values

    """
    controller.put(pvname, new * scale)


def build_slider(
    title: str, pvname, scale, start, end, step, controller, server="bokeh"
) -> Slider:
    """
    Utility function for building a slider.

    Parameters
    ----------
    title:str
        Slider title

    pvname:str
        Process variable name

    scale:float/int
        Scale of the slider

    start:float
        Lower range of the slider

    end:float
        Upper range of the slider

    step:np.float64
        The step between consecutive values

    controller: online_model.app.widgets.controllers.Controller
        Controller object for getting pv values

    Returns
    -------
    bokeh.models.widgets.sliders.Slider

    """

    # initialize value
    try:
        start_val = controller.get(pvname)

    except TimeoutError:
        print(f"No process variable found for {pvname}")
        start_val = 0

    slider = Slider(
        title=title, value=scale * start_val, start=start, end=end, step=step
    )

    # set up callback
    slider.on_change(
        "value",
        partial(set_pv_from_slider, pvname=pvname, scale=scale, controller=controller),
    )

    return slider


def build_sliders(cmd_pvdb: dict, controller: Controller) -> List[Slider]:
    """
    Build sliders from the cmd_pvdb.

    Parameters
    ----------
    cmd_pvdb: dict

    Return
    ------
    list
        List of slider objects


    controller: online_model.app.widgets.controllers.Controller
        Controller object for getting pv values

    """
    sliders = []

    for ii, pv in enumerate(cmd_pvdb):
        if pv not in EXCLUDE_SLIDERS:  # temporarily exclude the extent sliders
            title = pv + " (" + cmd_pvdb[pv]["units"] + ")"
            pvname = PREFIX + ":" + pv
            step = (cmd_pvdb[pv]["range"][1] - cmd_pvdb[pv]["range"][0]) / 100.0
            scale = 1

            slider = build_slider(
                title,
                pvname,
                scale,
                cmd_pvdb[pv]["range"][0],
                cmd_pvdb[pv]["range"][1],
                step,
                controller,
            )
            sliders.append(slider)

    return sliders
