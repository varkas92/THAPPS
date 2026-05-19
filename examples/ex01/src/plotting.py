# plotting.py

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import get_cti_option_text

def get_rgb(
        temp: float,
        ) -> str:
    """
    Map temperature to an RGB colour string for Plotly.

    Parameters
    ----------
    temp : float
        Temperature value [°C].

    Returns
    -------
    str
        RGB colour string in Plotly format.
    """
    t_min = 0.0
    t_max = 100.0

    ratio = (temp - t_min) / (t_max - t_min)
    ratio = max(0.0, min(1.0, ratio))

    r = int(255 * ratio)
    g = 0
    b = int(255 * (1.0 - ratio))

    return f"rgb({r},{g},{b})"


def seconds_to_clock(
        secs: int,
        include_day: bool,
        ) -> str:
    """
    Convert a time value in seconds to a clock-style label.

    Parameters
    ----------
    secs : int
        Time in seconds.
    include_day : bool
        Whether to append the day number to the label.

    Returns
    -------
    str
        Time formatted as HH:MM:SS, optionally including the day number.

    Notes
    -----
    When `include_day` is True, the output is formatted as:
    'HH:MM:SS (Day N)'.
    """
    if include_day:
        days = int(np.floor(secs / (24 * 3600)))
    else:
        days = 0

    if days != 0:
        secs = secs - days * 24 * 3600

    hours = int(np.floor(secs / 3600))
    if hours != 0:
        secs = secs - hours * 3600

    minutes = int(np.floor(secs / 60))
    if minutes != 0:
        secs = secs - minutes * 60

    if hours < 10:
        hours = "0" + str(hours)
    else:
        hours = str(hours)

    if minutes < 10:
        minutes = "0" + str(minutes)
    else:
        minutes = str(minutes)

    if secs < 10:
        secs = "0" + str(secs)
    else:
        secs = str(secs)

    if include_day:
        return f"{hours}:{minutes}:{secs} (Day {days + 1})"

    return f"{hours}:{minutes}:{secs}"


def get_plot_time_marks(
        slider_range: int,
        ) -> dict:
    """
    Build slider mark labels for the Dash time slider.

    Parameters
    ----------
    slider_range : int
        Maximum slider value / simulation duration [s].

    Returns
    -------
    dict
        Dictionary mapping slider positions to Dash label dictionaries.

    Notes
    -----
    The mark interval is currently fixed at 2 hours. A more adaptive
    interval-selection rule can be reintroduced later if needed.
    """
    # Future improvement:
    # choose mark_interval automatically based on slider_range
    mark_interval = 3600 * 2

    mark_dic = {}
    total_marks = int(np.floor(slider_range / mark_interval)) + 1

    for i in range(total_marks):
        mark_value = i * mark_interval
        mark_label = f"{int(mark_value / 3600):02d}:00"
        mark_dic[mark_value] = {"label": mark_label}

    return mark_dic


def _prepare_plot_options(
        opt: dict,
        low_interval_temp: float,
        high_interval_temp: float,
        ) -> dict:
    """
    Normalise plotting options and derive the CTI label.

    Parameters
    ----------
    opt : dict
        Raw plotting options dictionary.
    low_interval_temp : float
        Lower bound of the CTI interval [°C].
    high_interval_temp : float
        Upper bound of the CTI interval [°C].

    Returns
    -------
    dict
        Normalised plotting options.
    """
    cti_option_text = get_cti_option_text(
        low_interval_temp, high_interval_temp
    ).lower()

    return {
        "elem_to_plot_2": opt["elem_to_plot_2"],
        "var_to_plot_1": opt["var_to_plot_1"].lower(),
        "var_to_plot_2": opt["var_to_plot_2"].lower(),
        "sim_to_plot_1": opt["sim_to_plot_1"],
        "sim_to_plot_2": opt["sim_to_plot_2"],
        "obj_to_plot_2": opt["obj_to_plot_2"],
        "cti_option_text": cti_option_text,
    }


def _prepare_result_timeslice(
        result: dict,
        t: int,
        time_step: int,
        ) -> dict:
    """
    Extract time-slice data for one simulation result.

    Parameters
    ----------
    result : dict
        One entry from the results list.
    t : int
        Current simulation time [s].
    time_step : int
        Simulation time step [s].

    Returns
    -------
    dict
        Time-slice data and full series needed for plotting.
    """
    res = result["results"]

    demands = res.node["demand"]
    node_ages = res.node["quality"]
    flows = res.link["flowrate"]
    link_ages = res.link["quality"]

    node_temps = result["temperatures"]["nodes"]
    link_temps = result["temperatures"]["links"]
    link_cti = result["cti"]["links"]

    return {
        "demands": demands,
        "node_ages": node_ages,
        "flows": flows,
        "link_ages": link_ages,
        "node_temps": node_temps,
        "link_temps": link_temps,
        "link_cti": link_cti,
        "node_demands_at_time": round(demands.loc[t, :] * 1000 * 60, 2),
        "node_ages_at_time": round(node_ages.loc[t, :] / 3600, 4),
        "flow_at_time": flows.loc[t, :],
        "link_ages_at_time": round(link_ages.loc[t, :] / 3600, 4),
        "node_temps_at_time": node_temps[int(t / time_step)],
        "link_temps_at_time": link_temps[int(t / time_step)],
    }


def _build_node_plot_data(
        node_dics: list,
        timeslice: dict,
        demands_at_time,
        age_units: str,
        demand_units: str,
        ) -> dict:
    """
    Build node-level plotting arrays and annotations.

    Parameters
    ----------
    node_dics : list
        Node metadata dictionaries.
    timeslice : dict
        Prepared time-slice result data.
    demands_at_time : pandas.Series
        Demand at the selected time [L/min].
    age_units : str
        Display units for age.
    demand_units : str
        Display units for demand.

    Returns
    -------
    dict
        Node plotting arrays and metadata.
    """
    node_ages_at_time = timeslice["node_ages_at_time"]
    node_temps_at_time = timeslice["node_temps_at_time"]

    n_labels = []
    n_colors = []
    n_symbols = []
    n_sizes = []
    n_annotations = []

    n_plot = []
    n_plot_colors = []
    n_plot_symbols = []
    n_plot_sizes = []
    n_plot_annotations = []

    n_tanks = []

    Xn = []
    Yn = []
    Zn = []

    for node in node_dics:
        node_name = node["name"]
        node_age = node_ages_at_time[node_name]
        node_temp = round(node_temps_at_time[node_name], 2)

        name_label = "Node " + node_name
        age_label = f"Age = {node_age} {age_units}"
        temp_label = f"T = {node_temp} °C"

        node_x = node["coordinates"][0]
        node_y = node["coordinates"][1]
        n_type = node["node_type"]

        if n_type == "Junction":
            node_z = node["elevation"]
            b_demand = node["demand_timeseries_list"][0]["base_val"]
            node_demand = demands_at_time[node_name]
            node_symbol = "circle"

            if abs(b_demand) > 0:
                n_plot.append(node_name)
                node_demand_label = f"q = {round(node_demand, 2)} {demand_units}"
                node_label = (
                    name_label + "<br>" + age_label + "<br>" + temp_label + "<br>" + node_demand_label
                )
                n_plot_annotations.append(node_label)

                node_size = 5 + 2 / np.pi * np.arctan(node_age / 10) * 25

                if abs(round(node_demand, 2)) > 0:
                    node_color = "blue" if node_demand > 0 else "magenta"
                else:
                    node_label = name_label + "<br>" + age_label + "<br>" + temp_label
                    node_color = "cyan"

                n_plot_colors.append(node_color)
                n_plot_symbols.append(node_symbol)
                n_plot_sizes.append(node_size)
            else:
                node_color = "lightgray"
                node_size = 1
                node_name = ""
                node_label = ""

        elif n_type == "Tank":
            n_plot.append(node_name)
            n_tanks.append(node_name)
            node_z = node["elevation"]
            volume = np.pi * node["diameter"] ** 2 / 4 * node["max_level"]
            volume_label = f"V = {round(volume, 2)} m³"
            node_label = name_label + "<br>" + volume_label + "<br>" + age_label + "<br>" + temp_label
            n_plot_annotations.append(node_label)
            node_symbol = "square"
            node_color = get_rgb(node_temp)
            node_size = 2 / np.pi * np.arctan(volume * 264.172 / 10) * 30
            n_plot_colors.append(node_color)
            n_plot_symbols.append(node_symbol)
            n_plot_sizes.append(5 + 2 / np.pi * np.arctan(node_age / 10) * 25)

        else:
            node_z = node["base_head"]
            elevation_label = "Elevation = " + str(round(node["base_head"], 1)) + " m"
            node_label = name_label + "<br>" + elevation_label + "<br>" + age_label + "<br>" + temp_label
            node_symbol = "square"
            node_color = get_rgb(node_temp)
            node_size = 10

        node_annotation = dict(
            showarrow=False,
            x=node_x,
            y=node_y,
            z=node_z,
            text=node_name,
            xanchor="left",
            xshift=10,
            opacity=0.7,
        )

        n_labels.append(node_label)
        n_symbols.append(node_symbol)
        n_colors.append(node_color)
        n_sizes.append(node_size)
        n_annotations.append(node_annotation)

        Xn.append(node_x)
        Yn.append(node_y)
        Zn.append(node_z)

    return {
        "n_labels": n_labels,
        "n_colors": n_colors,
        "n_symbols": n_symbols,
        "n_sizes": n_sizes,
        "n_annotations": n_annotations,
        "n_plot": n_plot,
        "n_plot_colors": n_plot_colors,
        "n_plot_symbols": n_plot_symbols,
        "n_plot_sizes": n_plot_sizes,
        "n_plot_annotations": n_plot_annotations,
        "n_tanks": n_tanks,
        "Xn": Xn,
        "Yn": Yn,
        "Zn": Zn,
    }


def _build_link_plot_data(
    link_dics: list,
    node_dics: list,
    timeslice: dict,
) -> dict:
    """
    Build link-level plotting arrays and annotations.

    Parameters
    ----------
    link_dics : list
        Link metadata dictionaries.
    node_dics : list
        Node metadata dictionaries.
    timeslice : dict
        Prepared time-slice result data.

    Returns
    -------
    dict
        Link plotting arrays and coordinates.
    """
    link_ages_at_time = timeslice["link_ages_at_time"]
    link_temps_at_time = timeslice["link_temps_at_time"]
    #flow_at_time = timeslice["flow_at_time"]

    node_lookup = {node["name"]: node for node in node_dics}

    l_names = []
    l_labels = []
    l_colors = []
    l_sizes = []

    Xe = []
    Ye = []
    Ze = []

    for link in link_dics:
        link_name = link["name"]
        link_age = link_ages_at_time[link_name]
        link_temp = round(link_temps_at_time[link_name], 2)

        name_label = "Link " + link_name
        age_label = "Age = " + str(link_age) + " hrs"
        temp_label = "T = " + str(link_temp) + " °C"

        initial = link["start_node_name"]
        final = link["end_node_name"]
        #flow = round(flow_at_time[link_name] * 1000 * 60, 2)

        link_color = get_rgb(link_temp)

        start_node = node_lookup[initial]
        end_node = node_lookup[final]

        link["x_start"] = start_node["coordinates"][0]
        link["y_start"] = start_node["coordinates"][1]
        link["z_start"] = start_node["base_head"] if start_node["node_type"] == "Reservoir" else start_node["elevation"]

        link["x_end"] = end_node["coordinates"][0]
        link["y_end"] = end_node["coordinates"][1]
        link["z_end"] = end_node["base_head"] if end_node["node_type"] == "Reservoir" else end_node["elevation"]

        Xe.append([link["x_start"], link["x_end"], None])
        Ye.append([link["y_start"], link["y_end"], None])
        Ze.append([link["z_start"], link["z_end"], None])

        try:
            l_diameter = link["diameter"]
            l_diameter_in = round(l_diameter * 39.3701, 1)
        except Exception:
            l_diameter_in = "NA"

        link_label = (
            name_label
            + "<br>"
            + 'd = '
            + str(l_diameter_in)
            + '"<br>'
            + age_label
            + "<br>"
            + temp_label
        )

        l_names.append(link_name)
        l_labels.append(link_label)
        l_colors.append(link_color)
        l_sizes.append(5)

    return {
        "l_names": l_names,
        "l_labels": l_labels,
        "l_colors": l_colors,
        "l_sizes": l_sizes,
        "Xe": np.array(Xe),
        "Ye": np.array(Ye),
        "Ze": np.array(Ze),
    }


def _build_line_plot_data(
    plot_opts: dict,
    timeslice: dict,
    node_plot_data: dict,
    obj_to_plot_2: str,
    low_interval_temp: float,
    high_interval_temp: float,
) -> tuple:
    """
    Build the time-series data and axis labels for the secondary plot.

    Parameters
    ----------
    plot_opts : dict
        Normalised plotting options.
    timeslice : dict
        Prepared time-slice result data for the selected simulation.
    node_plot_data : dict
        Node plotting data returned by `_build_node_plot_data`.
    obj_to_plot_2 : str
        Selected object name for the detailed time-series plot.
    low_interval_temp : float
        Lower CTI threshold [°C].
    high_interval_temp : float
        Upper CTI threshold [°C].

    Returns
    -------
    tuple
        Tuple containing:
        - line_plot : sequence
            Time-series values for the selected object.
        - y_axis : str
            Y-axis title.
        - title1 : str
            Variable description used in the figure title.
        - title2 : str
            Object description used in the figure title.
    """
    elem_to_plot_2 = plot_opts["elem_to_plot_2"]
    var_to_plot_2 = plot_opts["var_to_plot_2"]
    cti_option_text = plot_opts["cti_option_text"]

    demands = timeslice["demands"]
    node_ages = timeslice["node_ages"]
    link_ages = timeslice["link_ages"]
    link_temps = timeslice["link_temps"]
    link_cti = timeslice["link_cti"]

    n_plot = node_plot_data["n_plot"]
    n_tanks = node_plot_data["n_tanks"]

    sum_ages = round(node_ages.loc[:, n_plot].sum(axis=1) / 3600, 4)
    common = set(n_plot).intersection(set(n_tanks))
    n_plot_no_tanks = [k for k in n_plot if k not in common]
    total_demand = round(demands.loc[:, n_plot_no_tanks].sum(axis=1) * 1000 * 60, 2)

    if elem_to_plot_2 == "Node":
        if var_to_plot_2 == "age":
            y_axis = "AGE (HOURS)"
            title1 = "age"
            if obj_to_plot_2 == "SUM OF ALL":
                line_plot = sum_ages
                title2 = "the sum of all appliance nodes and tanks"
            else:
                line_plot = round(node_ages.loc[:, obj_to_plot_2] / 3600, 4)
                title2 = "node " + obj_to_plot_2

        elif var_to_plot_2 == "demand":
            y_axis = "DEMAND (LPM)"
            title1 = "demand"
            if obj_to_plot_2 == "SUM OF ALL":
                line_plot = total_demand
                title2 = "the sum of all appliance nodes"
            else:
                line_plot = round(demands.loc[:, obj_to_plot_2] * 1000 * 60, 2)
                title2 = "node " + obj_to_plot_2

    elif elem_to_plot_2 == "Link":
        if var_to_plot_2 == "age":
            y_axis = "AGE (HOURS)"
            title1 = "age"
            line_plot = round(link_ages.loc[:, obj_to_plot_2] / 3600, 4)

        elif var_to_plot_2 == "temperature":
            y_axis = "TEMP. (°C)"
            title1 = "temperature"
            line_plot = [round(d[obj_to_plot_2], 2) for d in link_temps]

        elif var_to_plot_2 == cti_option_text:
            y_axis = (
                "CTI "
                + str(low_interval_temp)
                + "-"
                + str(high_interval_temp)
                + "°C (hours)"
            )
            title1 = (
                "time between "
                + str(low_interval_temp)
                + "°C and "
                + str(high_interval_temp)
                + "°C"
            )
            line_plot = [round(d[obj_to_plot_2], 4) for d in link_cti]

        title2 = "link " + obj_to_plot_2

    return line_plot, y_axis, title1, title2


def _build_time_labels(n_steps: int, time_step: int, include_days: bool = True) -> list[str]:
    """
    Build formatted simulation-time labels for plotting.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    time_step : int
        Simulation time step [s].
    include_days : bool, optional
        Whether to append the day number to the label.

    Returns
    -------
    list[str]
        Formatted time labels.
    """
    return [seconds_to_clock(i * time_step, include_days) for i in range(n_steps)]


def _build_max_node_summary(
    plot_opts: dict,
    timeslice: dict,
    node_plot_data: dict,
    age_units: str,
    demand_units: str,
) -> dict:
    """
    Compute maximum node metrics and corresponding hover labels.
    """
    node_ages = timeslice["node_ages"]
    demands = timeslice["demands"]
    n_plot = node_plot_data["n_plot"]

    max_ages = []
    max_demands = []
    max_sizes = []
    max_labels = []

    include_days = True

    for node in n_plot:
        all_ages = round(node_ages.loc[:, node] / 3600, 1)
        max_age = round(max(all_ages), 4)
        max_ages.append(max_age)

        all_demands = round(demands.loc[:, node] * 1000 * 60, 2)
        max_demand = round(max(all_demands), 2)
        max_demands.append(max_demand)

        max_sizes.append(5 + 2 / np.pi * np.arctan(max_age / 10) * 25)

        if plot_opts["var_to_plot_1"] == "age":
            max_time = all_ages.idxmax()
            max_time = seconds_to_clock(max_time, include_days)
            max_labels.append(
                f"Max. age = {max_age} {age_units}<br> At time = {max_time}"
            )
        elif plot_opts["var_to_plot_1"] == "demand":
            max_time = all_demands.idxmax()
            max_time = seconds_to_clock(max_time, include_days)
            max_labels.append(
                f"Max. demand = {max_demand} {demand_units}<br> At time = {max_time}"
            )

    return {
        "max_ages": max_ages,
        "max_demands": max_demands,
        "max_sizes": max_sizes,
        "max_labels": max_labels,
    }


def _initialise_network_figure(node_annotations: list):
    """
    Initialise the combined 3D-network and summary-scatter figure.

    Parameters
    ----------
    node_annotations : list
        Plotly annotation dictionaries for node labels.

    Returns
    -------
    plotly.graph_objects.Figure
        Initialised subplot figure.
    """
    axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title="",
    )

    layout = go.Layout(
        uirevision="true",
        height=600,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
            annotations=node_annotations,
        ),
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.4, 0.6],
        row_heights=[1],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
    )

    fig.update_layout(layout)
    return fig


def _add_network_traces(fig1, link_plot_data: dict, node_plot_data: dict) -> None:
    """
    Add 3D link and node traces to the network figure.

    Parameters
    ----------
    fig1 : plotly.graph_objects.Figure
        Target figure.
    link_plot_data : dict
        Link plotting data.
    node_plot_data : dict
        Node plotting data.
    """
    Xe = link_plot_data["Xe"]
    Ye = link_plot_data["Ye"]
    Ze = link_plot_data["Ze"]
    l_names = link_plot_data["l_names"]
    l_labels = link_plot_data["l_labels"]
    l_colors = link_plot_data["l_colors"]
    l_sizes = link_plot_data["l_sizes"]

    Xn = node_plot_data["Xn"]
    Yn = node_plot_data["Yn"]
    Zn = node_plot_data["Zn"]
    n_symbols = node_plot_data["n_symbols"]
    n_sizes = node_plot_data["n_sizes"]
    n_colors = node_plot_data["n_colors"]
    n_labels = node_plot_data["n_labels"]

    for i in range(len(l_names)):
        fig1.add_trace(
            go.Scatter3d(
                x=[Xe[i, 0], Xe[i, 1]],
                y=[Ye[i, 0], Ye[i, 1]],
                z=[Ze[i, 0], Ze[i, 1]],
                name=l_names[i],
                mode="lines",
                line=dict(color=l_colors[i], width=l_sizes[i]),
                hoverinfo="text",
                text=l_labels[i],
            ),
            row=1,
            col=1,
        )

    fig1.add_trace(
        go.Scatter3d(
            x=Xn,
            y=Yn,
            z=Zn,
            mode="markers",
            name="nodes",
            marker=dict(
                symbol=n_symbols,
                size=n_sizes,
                color=n_colors,
                colorscale="Viridis",
                line=dict(color="rgb(50,50,50)", width=0.5),
            ),
            text=n_labels,
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )


def _add_summary_scatter(
    fig1,
    plot_opts: dict,
    node_plot_data: dict,
    max_data: dict,
    timeslice: dict,
) -> None:
    """
    Add the right-hand node summary scatter plot to the network figure.

    Parameters
    ----------
    fig1 : plotly.graph_objects.Figure
        Target figure.
    plot_opts : dict
        Normalised plotting options.
    node_plot_data : dict
        Node plotting data.
    max_data : dict
        Maximum-value summary data.
    timeslice : dict
        Prepared result data for the current simulation.
    """
    n_plot = node_plot_data["n_plot"]
    n_plot_symbols = node_plot_data["n_plot_symbols"]
    n_plot_sizes = node_plot_data["n_plot_sizes"]
    n_plot_colors = node_plot_data["n_plot_colors"]
    n_plot_annotations = node_plot_data["n_plot_annotations"]

    node_ages_at_time = timeslice["node_ages_at_time"]
    node_demands_at_time = timeslice["node_demands_at_time"]

    if plot_opts["var_to_plot_1"] == "age":
        var_plot = node_ages_at_time[n_plot]
        max_var = max_data["max_ages"]
    elif plot_opts["var_to_plot_1"] == "demand":
        var_plot = node_demands_at_time[n_plot]
        max_var = max_data["max_demands"]
    else:
        var_plot = node_ages_at_time[n_plot]
        max_var = max_data["max_ages"]

    fig1.add_trace(
        go.Scatter(
            x=n_plot,
            y=var_plot,
            mode="markers",
            marker=dict(
                symbol=n_plot_symbols,
                size=n_plot_sizes,
                color=n_plot_colors,
                colorscale="Viridis",
                line=dict(color="rgb(50,50,50)", width=0.5),
            ),
            text=n_plot_annotations,
            hoverinfo="text",
        ),
        row=1,
        col=2,
    )

    fig1.add_trace(
        go.Scatter(
            x=n_plot,
            y=max_var,
            mode="markers",
            marker=dict(
                symbol=n_plot_symbols,
                size=max_data["max_sizes"],
                color="lightgray",
                colorscale="Viridis",
                line=dict(color="rgb(50,50,50)", width=0.5),
            ),
            text=max_data["max_labels"],
            hoverinfo="text",
        ),
        row=1,
        col=2,
    )

    age_range = [0, int(np.ceil(max(max_var))) + 5]
    fig1.update_layout(yaxis_range=age_range)
    

def _initialise_line_figure(title1: str, title2: str, y_axis: str):
    """
    Initialise the secondary time-series figure.

    Parameters
    ----------
    title1 : str
        Variable name shown in the title.
    title2 : str
        Object description shown in the title.
    y_axis : str
        Y-axis title.

    Returns
    -------
    plotly.graph_objects.Figure
        Initialised line figure.
    """
    fig = go.Figure()
    fig.update_layout(
        title="Showing " + title1 + " for " + title2,
        xaxis_title="SIMULATION TIME",
        yaxis_title=y_axis,
        legend_title="Simulation with:",
        font=dict(
            family="Courier New, monospace",
            size=10,
            color="RebeccaPurple",
        ),
    )
    return fig


def _add_temperature_threshold_lines(
    fig2,
    all_times: list,
    low_interval_temp: float,
    high_interval_temp: float,
) -> None:
    """
    Add horizontal threshold lines to the temperature time-series figure.

    Parameters
    ----------
    fig2 : plotly.graph_objects.Figure
        Target figure.
    all_times : list
        Formatted time labels.
    low_interval_temp : float
        Lower threshold temperature [°C].
    high_interval_temp : float
        Upper threshold temperature [°C].
    """
    fig2.add_trace(
        go.Scatter(
            x=all_times,
            y=[high_interval_temp] * len(all_times),
            name=str(high_interval_temp) + "°C",
            mode="lines",
            line=dict(dash="dash"),
            uirevision="true",
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=all_times,
            y=[low_interval_temp] * len(all_times),
            name=str(low_interval_temp) + "°C",
            mode="lines",
            line=dict(dash="dash"),
            uirevision="true",
        )
    )


def draw_network_fig(
    data: dict,
    res_dic: list,
    t: int,
    opt: dict,
    time_step: int,
    low_interval_temp: float,
    high_interval_temp: float,
    demand_units: str,
    age_units: str,
):
    """
    Build the network visualisation and companion time-series figure.

    This function coordinates the plotting workflow by:
    1. normalising the plotting options,
    2. extracting the required time-slice data for each simulation result,
    3. preparing node and link plotting data, and
    4. assembling the two Plotly figures.

    Parameters
    ----------
    data : dict
        Network metadata used for plotting. Expected to contain at least:
        - data["nodes"] : list of node dictionaries
        - data["links"] : list of link dictionaries
    res_dic : list[dict]
        List of simulation result dictionaries.
    t : int
        Current simulation time [s].
    opt : dict
        Plotting options dictionary.
    time_step : int
        Simulation time step [s].
    low_interval_temp : float
        Lower bound of the CTI interval [°C].
    high_interval_temp : float
        Upper bound of the CTI interval [°C].
    demand_units : str
        Display units for demand.
    age_units : str
        Display units for age.

    Returns
    -------
    list
        Two-item list containing:
        - network_fig : plotly.graph_objects.Figure
        - graph_fig : plotly.graph_objects.Figure
    """
    plot_opts = _prepare_plot_options(
        opt=opt,
        low_interval_temp=low_interval_temp,
        high_interval_temp=high_interval_temp,
    )

    fig1 = None
    fig2 = None
    all_times = None

    for result in res_dic:
        result["show2"] = result["legend"] in plot_opts["sim_to_plot_2"]

    for result in res_dic:
        timeslice = _prepare_result_timeslice(
            result=result,
            t=t,
            time_step=time_step,
        )

        node_dics = data["nodes"]
        link_dics = data["links"]

        node_plot_data = _build_node_plot_data(
            node_dics=node_dics,
            timeslice=timeslice,
            demands_at_time=timeslice["node_demands_at_time"],
            age_units=age_units,
            demand_units=demand_units,
        )

        link_plot_data = _build_link_plot_data(
            link_dics=link_dics,
            node_dics=node_dics,
            timeslice=timeslice,
        )

        all_times = _build_time_labels(
            n_steps=len(timeslice["node_ages"]),
            time_step=time_step,
            include_days=True,
        )

        max_data = _build_max_node_summary(
            plot_opts=plot_opts,
            timeslice=timeslice,
            node_plot_data=node_plot_data,
            age_units=age_units,
            demand_units=demand_units,
        )

        if fig1 is None:
            fig1 = _initialise_network_figure(
                node_annotations=node_plot_data["n_annotations"]
            )

        if result["legend"] == plot_opts["sim_to_plot_1"]:
            _add_network_traces(
                fig1=fig1,
                link_plot_data=link_plot_data,
                node_plot_data=node_plot_data,
            )

            _add_summary_scatter(
                fig1=fig1,
                plot_opts=plot_opts,
                node_plot_data=node_plot_data,
                max_data=max_data,
                timeslice=timeslice,
            )

        line_plot, y_axis, title1, title2 = _build_line_plot_data(
            plot_opts=plot_opts,
            timeslice=timeslice,
            node_plot_data=node_plot_data,
            obj_to_plot_2=plot_opts["obj_to_plot_2"],
            low_interval_temp=low_interval_temp,
            high_interval_temp=high_interval_temp,
        )

        if fig2 is None:
            fig2 = _initialise_line_figure(
                title1=title1,
                title2=title2,
                y_axis=y_axis,
            )

        if result["show2"]:
            fig2.add_trace(
                go.Scatter(
                    x=all_times,
                    y=line_plot,
                    name=result["legend"],
                    mode="lines",
                    uirevision="true",
                )
            )

    if plot_opts["var_to_plot_2"] == "temperature" and fig2 is not None and all_times is not None:
        _add_temperature_threshold_lines(
            fig2=fig2,
            all_times=all_times,
            low_interval_temp=low_interval_temp,
            high_interval_temp=high_interval_temp,
        )

    return [fig1, fig2]