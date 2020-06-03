from typing import List

from bokeh.models import ColumnDataSource, DataTable, TableColumn, StringFormatter

from online_model.app.controllers import Controller
from online_model.app.monitors import PVScalar


class ValueTable:
    def __init__(
        self, sim_pvdb, controller: Controller, array_pvs: List[str], prefix: str
    ) -> None:
        """
        View for value table item. Maps process variable name to its value.

        Parameters
        ----------
        sim_pvdb: dict
            Dictionary of process variable values

        controller: online_model.app.widgets.controllers.Controller
            Controller object for getting pv values

        array_pvs: list
            List of pvs to be excluded due to image formatting etc.

        Notes
        -----
        The array_pvs is kind of a hacky fix that should be fixed and accounted for
        when stronger parameter type definitions are implemented.

        """
        # only creating pvs for non-image pvs
        self.pv_monitors = {}
        self.output_values = []
        self.names = []

        # be sure to surface units in the table
        self.unit_map = {}

        for pv in sim_pvdb:
            if pv not in array_pvs:
                self.pv_monitors[pv] = PVScalar(
                    f"{prefix}:{pv}", sim_pvdb[pv]["units"], controller
                )
                v = self.pv_monitors[pv].poll()

                self.output_values.append(v)
                self.names.append(pv)
                self.unit_map[pv] = sim_pvdb[pv]["units"]

        self.create_table()

    def create_table(self) -> None:
        """
        Create the table and populate prelim data.
        """
        self.table_data = dict(x=self.names, y=self.output_values)
        self.source = ColumnDataSource(self.table_data)
        columns = [
            TableColumn(
                field="x", title="Outputs", formatter=StringFormatter(font_style="bold")
            ),
            TableColumn(field="y", title="Current Value"),
        ]

        self.table = DataTable(
            source=self.source, columns=columns, width=400, height=280
        )

    def update(self):
        """
        Update data source.
        """
        output_values = []
        for pv in self.pv_monitors:
            v = self.pv_monitors[pv].poll()
            output_values.append(v)

        self.source.data = dict(x=self.names, y=output_values)
