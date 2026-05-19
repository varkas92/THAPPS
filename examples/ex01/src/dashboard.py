# dashboard.py

import webbrowser

from dash import Dash, html, dcc, ctx, Output, Input

from config import DEMAND_UNITS, AGE_UNITS

from .plotting import draw_network_fig, get_plot_time_marks


def nodes_for_dash(
        my_appliances: list[dict],
        ) -> list[str]:
    """
    Build the list of demand-node names to show in the dashboard.

    Parameters
    ----------
    my_appliances : list
        List of appliance dictionaries from the model configuration.
        Each dictionary is expected to contain at least the key "name".

    Returns
    -------
    list[str]
        List of appliance or demand-node names used for dashboard selection.
    """
    plot_nodes = []
    for appliance in my_appliances:
        appliance_name = appliance['name']
        plot_nodes.append(appliance_name)
    return plot_nodes


def _get_slider_intervals(
        duration: int,
        ) -> tuple[int, int]:
    """
    Return the slider step and button step for the dashboard time controls.

    Parameters
    ----------
    duration : int
        Total simulation duration [s].

    Returns
    -------
    tuple[int, int]
        (slider_step, button_step) in seconds.
    """
    slider_step = 300
    button_step = 300
    return slider_step, button_step


def _clic_prev_next(
        t0: int,
        btn_previous: int,
        btn_next: int,
        button_step: int,
        duration: int,
        ) -> int:
    """
    Update slider time from previous/next button clicks.

    Parameters
    ----------
    t0 : int
        Current slider value [s].
    btn_previous : int
        Number of clicks on the previous button.
    btn_next : int
        Number of clicks on the next button.
    button_step : int
        Increment/decrement step [s].
    duration : int
        Maximum allowed time [s].

    Returns
    -------
    int
        Updated slider value [s].
    """
    trigger = ctx.triggered_id

    if trigger == "btn-previous":
        return max(0, t0 - button_step)
    if trigger == "btn-next":
        return min(duration, t0 + button_step)

    return t0


def _on_select_element(
        elem_to_plot_2: str,
        data: dict,
) -> tuple[list[str], str | None]:
    """
    Return available object options depending on whether the user selected
    links or nodes.

    Parameters
    ----------
    elem_to_plot_2 : str
        Selected element type, e.g. 'Link' or 'Node'.
    data : dict
        Network metadata dictionary.

    Returns
    -------
    tuple[list, str]
        (options, default_value)
    """
    if elem_to_plot_2 == "Link":
        options = [link["name"] for link in data["links"]]
    else:
        options = [node["name"] for node in data["nodes"]]

    default_value = options[0] if options else None
    return options, default_value


def create_dashboard_app(
    *,
    data: dict,
    results_dic: list,
    duration: int,
    time_step: int,
    number_of_simulations: int,
    number_of_people: int,
    variable_options_nodes,
    variable_options_links,
    variable_default,
    simulation_options,
    simulation_default,
    time_options,
    time_default,
    element_options,
    element_default,
    link_options,
    link_default,
    low_interval_temp: float,
    high_interval_temp: float,
    ) -> Dash:
    """
    Create and return the Dash application.

    Parameters
    ----------
    data : dict
        Network metadata dictionary.
    results_dic : list
        Result dictionaries used by plotting.
    duration : int
        Total simulation duration [s].
    time_step : int
        Simulation time step [s].
    number_of_simulations : int
        Number of simulations represented in the current results.
    number_of_people : int
        Number of people represented in the demand model.
    variable_options_nodes : list
        Variable options for the left-side node summary controls.
    variable_options_links : list
        Variable options for the right-side detailed plot controls.
    variable_default : str
        Default selected variable.
    simulation_options : list
        Available simulation labels for selection.
    simulation_default : str
        Default simulation label.
    time_options : list
        Available time-mode options.
    time_default : str
        Default time-mode option.
    element_options : list
        Available element types, e.g. Link/Node.
    element_default : str
        Default element type.
    link_options : list
        Default selectable objects for the detailed plot.
    link_default : str
        Default selected object.
    low_interval_temp : float
        Lower CTI threshold [°C].
    high_interval_temp : float
        Upper CTI threshold [°C].

    Returns
    -------
    dash.Dash
        Configured Dash application.
    """
    app = Dash(__name__)

    slider_step, button_step = _get_slider_intervals(duration)
    mark_dic = get_plot_time_marks(duration)

    app.layout = html.Div(
        children=[
            html.Div("Show results of:", style={"color": "darkblue", "fontSize": 14}),
            dcc.RadioItems(
                variable_options_nodes,
                variable_default,
                inline=True,
                id="radio_variable_1",
            ),

            html.Div(
                (
                    "For the simulation (out of "
                    + str(number_of_simulations)
                    + " random simulations with "
                    + str(number_of_people)
                    + " people) with:"
                ),
                style={"color": "darkblue", "fontSize": 14},
            ),
            dcc.RadioItems(
                simulation_options,
                simulation_default,
                inline=True,
                id="radio_simulation_1",
            ),

            dcc.Graph(figure={}, id="my-network", config={"displayModeBar": True}),

            dcc.RadioItems(time_options, time_default, inline=True, id="radio_time_1"),
            html.Button("Previous", id="btn-previous", n_clicks=0),
            html.Button("Next", id="btn-next", n_clicks=0),

            html.Div(
                id="slider-value-text",
                style={"font-size": 30, "color": "lime", "text-align": "center"},
            ),

            dcc.Slider(
                0,
                duration,
                slider_step,
                value=0,
                marks=mark_dic,
                id="my-slider",
            ),

            dcc.Graph(figure={}, id="my-graph", config={"displayModeBar": False}),

            html.Div("Show results for:", style={"color": "darkblue", "fontSize": 14}),
            dcc.RadioItems(
                element_options,
                element_default,
                inline=True,
                id="radio_element_2",
            ),

            html.Div("Variable:", style={"color": "darkblue", "fontSize": 14}),
            dcc.RadioItems(
                variable_options_links,
                variable_default,
                inline=True,
                id="radio_variable_2",
            ),

            html.Div(
                (
                    "For the simulation(s) (out of "
                    + str(number_of_simulations)
                    + " random simulations with "
                    + str(number_of_people)
                    + " people) with:"
                ),
                style={"color": "darkblue", "fontSize": 14},
            ),
            dcc.Checklist(
                simulation_options,
                simulation_options,
                inline=True,
                id="check_simulation_2",
            ),

            html.Div("Select Object:", style={"color": "darkblue", "fontSize": 14}),
            dcc.RadioItems(
                link_options,
                link_default,
                inline=False,
                id="radio_selection_2",
            ),
        ]
    )

    @app.callback(
        Output("radio_selection_2", "options"),
        Output("radio_selection_2", "value"),
        Input("radio_element_2", "value"),
    )
    def update_selectable_objects(elem_to_plot_2):
        return _on_select_element(elem_to_plot_2, data)

    @app.callback(
        Output("my-slider", "value"),
        Input("btn-previous", "n_clicks"),
        Input("btn-next", "n_clicks"),
        Input("my-slider", "value"),
    )
    def update_slider_from_buttons(btn_previous, btn_next, t0):
        return _clic_prev_next(t0, btn_previous, btn_next, button_step, duration)

    @app.callback(
        Output("slider-value-text", "children"),
        Output("my-network", "figure"),
        Output("my-graph", "figure"),
        Input("my-slider", "value"),
        Input("radio_variable_1", "value"),
        Input("radio_simulation_1", "value"),
        Input("radio_time_1", "value"),
        Input("radio_element_2", "value"),
        Input("radio_variable_2", "value"),
        Input("check_simulation_2", "value"),
        Input("radio_selection_2", "value"),
    )
    def update_app(
        t,
        var_to_plot_1,
        sim_to_plot_1,
        time_to_plot_1,
        elem_to_plot_2,
        var_to_plot_2,
        sim_to_plot_2,
        obj_to_plot_2,
        ):
        slider_label = f"t = {t} s"

        my_options = {
            "elem_to_plot_2": elem_to_plot_2,
            "var_to_plot_1": var_to_plot_1,
            "var_to_plot_2": var_to_plot_2,
            "sim_to_plot_1": sim_to_plot_1,
            "sim_to_plot_2": sim_to_plot_2,
            "obj_to_plot_2": obj_to_plot_2,
            "time_to_plot_1": time_to_plot_1,
        }

        network_fig, graph_fig = draw_network_fig(
            data=data,
            res_dic=results_dic,
            t=t,
            opt=my_options,
            time_step=time_step,
            low_interval_temp=low_interval_temp,
            high_interval_temp=high_interval_temp,
            demand_units=DEMAND_UNITS,
            age_units=AGE_UNITS,
        )

        return slider_label, network_fig, graph_fig

    return app


def run_dashboard(app, open_browser: bool = True, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    """
   Launch the Dash application.

   Parameters
   ----------
   app : dash.Dash
       Dash application instance.
   open_browser : bool, optional
       Whether to open the default browser automatically.
   host : str, optional
       Host address used by the Dash server.
   port : int, optional
       Port number used by the Dash server.
   debug : bool, optional
       Whether to run Dash in debug mode.

   Returns
   -------
   None
   """
    if open_browser:
        webbrowser.open(f"http://{host}:{port}/")

    app.run(host=host, port=port, debug=debug)