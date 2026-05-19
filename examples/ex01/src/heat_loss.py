# temperature.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import G
from .properties import get_property, get_pipe_properties, get_R

EXCLUDE_LINKS = {"extra", "HWH"}

def _estimate_surface_temperatures(
        t_bulk: float,
        t_room: float,
        ) -> tuple[float, float, float]:
    """
    Estimate inner/outer pipe surface temperatures needed by get_R().
    """
    t_arit_mean = (t_bulk + t_room) / 2

    if t_bulk * t_room >= 0:
        t_geom_mean = (t_bulk * t_room) ** 0.5
    else:
        t_geom_mean = t_arit_mean

    if t_bulk >= t_room:
        t_s_in = t_arit_mean
        t_s_ex = t_geom_mean
    else:
        t_s_in = t_geom_mean
        t_s_ex = t_arit_mean

    return t_s_in, t_s_ex, t_s_ex


def _prepare_pipe_for_heat_loss(
        pipe: Any,
        pipes_df: pd.DataFrame,
        ) -> None:
    """
    Attach the thermal attributes required by get_R() to a WNTR pipe object.
    """
    (
        t_room,
        t_initial,
        mat_pipe,
        e_pipe,
        k_insulator,
        e_insulator,
        k_wall,
        z_wall,
        is_insulated,
        has_wall,
    ) = get_pipe_properties(pipe.name, pipes_df)

    pipe.temperature_infinity = t_room
    pipe.temperature = t_initial
    pipe.mat_pipe = mat_pipe
    pipe.e_link = e_pipe
    pipe.k_insulator = k_insulator
    pipe.e_insulator = e_insulator
    pipe.k_wall = k_wall
    pipe.z_wall = z_wall
    pipe.insulator = is_insulated
    pipe.wall = has_wall

    # Defaults used by get_R(). Adjust if you already calculate orientation elsewhere.
    pipe.theta = getattr(pipe, "theta", 0.0)


def _calculate_total_heat_capacity(
        pipe: Any,
        t_bulk: float,
        interpolators: dict[str, Any],
        ) -> float:
    """
    Calculate water + pipe-wall heat capacity for one full pipe segment [J/°C].
    """
    d = pipe.diameter
    length = pipe.length
    e_pipe = pipe.e_link
    mat_pipe = pipe.mat_pipe

    a_cross = np.pi * d ** 2 / 4
    a_pipe = np.pi / 4 * ((d + 2 * e_pipe) ** 2 - d ** 2)

    rho_water = get_property(t_bulk, "w_rho", interpolators)
    cp_water = get_property(t_bulk, "w_cp", interpolators)

    rho_pipe = get_property(t_bulk, f"{mat_pipe}_rho", interpolators)
    cp_pipe = get_property(t_bulk, f"{mat_pipe}_cp", interpolators)

    c_water = rho_water * cp_water * a_cross * length
    c_pipe = rho_pipe * cp_pipe * a_pipe * length

    return c_water + c_pipe


def build_heat_loss_tables_from_timeseries(
        temperature_df: pd.DataFrame,
        flow_df: pd.DataFrame,
        wn: Any,
        pipes_df: pd.DataFrame,
        interpolators: dict[str, Any],
        time_step: int,
        g: float = G,
        ) -> dict[str, pd.DataFrame]:
    """
    Build heat-loss result tables from existing link temperature and flow results.

    Parameters
    ----------
    temperature_df : pandas.DataFrame
        Link temperature results. Index must be simulation time [s].
        Columns must be link names.

    flow_df : pandas.DataFrame
        Link flowrate results. Index must be simulation time [s].
        Columns must be link names.

    wn : object
        WNTR network model loaded from the same INP file.

    pipes_df : pandas.DataFrame
        Pipe-property table loaded from network_properties.xlsx.

    interpolators : dict
        Temperature-dependent property interpolators.

    time_step : int
        Time step [s].

    g : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary of heat-loss result tables ready for Excel export.
    """
    # Keep only common time index.
    common_index = temperature_df.index.intersection(flow_df.index)
    temperature_df = temperature_df.loc[common_index]
    flow_df = flow_df.loc[common_index]
    flow_df = flow_df / 1000 / 60

    link_names = [
        name
        for name in temperature_df.columns
        if name in flow_df.columns
        and name not in EXCLUDE_LINKS
        and not str(name).startswith("HWH")
    ]

    radial_heat_flowrate = pd.DataFrame(index=common_index, columns=link_names, dtype=float)
    radial_heat_transferred = pd.DataFrame(index=common_index, columns=link_names, dtype=float)
    net_heat_change = pd.DataFrame(index=common_index, columns=link_names, dtype=float)

    for link_name in tqdm(link_names, desc="Processing links", unit="link"):
        pipe = wn.get_link(link_name)

        if getattr(pipe, "link_type", None) != "Pipe":
            continue

        _prepare_pipe_for_heat_loss(pipe, pipes_df)

        for i, time in enumerate(common_index):
            current_temperature = float(temperature_df.loc[time, link_name])
            current_flow = abs(float(flow_df.loc[time, link_name]))

            t_room = pipe.temperature_infinity
            
            t_s_in, t_s_ex, t_s_ex_air = _estimate_surface_temperatures(
                t_bulk=current_temperature,
                t_room=t_room,
            )

            package = {
                "volume": pipe.diameter ** 2 * np.pi / 4 * pipe.length,
                "temperature": current_temperature,
                "temperature_s_in": t_s_in,
                "temperature_s_ex": t_s_ex,
                "temperature_s_ex_air": t_s_ex_air,
                "old_pipe": link_name,
            }

            r_overall, *_ = get_R(
                pipe=pipe,
                package=package,
                flow=current_flow,
                interpolators=interpolators,
                g=g,
            )

            # Positive = heat transferred from pipe/water to room.
            qdot_radial = (current_temperature - t_room) / r_overall
            radial_heat_flowrate.loc[time, link_name] = qdot_radial
            
            if i == 0:
                net_heat_change.loc[time, link_name] = 0.0
                continue

            previous_time = common_index[i - 1]
            previous_temperature = float(temperature_df.loc[previous_time, link_name])

            radial_heat_transferred.loc[time, link_name] = qdot_radial * time_step

            t_capacity = (current_temperature + previous_temperature) / 2

            c_total = _calculate_total_heat_capacity(
                pipe=pipe,
                t_bulk=t_capacity,
                interpolators=interpolators,
            )
    
            net_heat_change.loc[time, link_name] = c_total * (
                current_temperature - previous_temperature
            )
    
        radial_heat_transferred[link_name] = (
            radial_heat_flowrate[link_name].shift(1) * time_step
        )
        radial_heat_transferred.loc[common_index[0], link_name] = 0.0
    
    aggregated = pd.DataFrame(index=common_index)

    aggregated["Total Net Heat Change (J)"] = net_heat_change.sum(axis=1)

    aggregated["Total Heat Transferred - Pipes to Room (J)"] = (
        radial_heat_transferred.where(radial_heat_transferred > 0, 0.0).sum(axis=1)
    )

    room_to_pipes_raw = (
        radial_heat_transferred.where(radial_heat_transferred < 0, 0.0).sum(axis=1)
    )

    # Report as positive magnitude for readability.
    aggregated["Total Heat Transferred - Room to Pipes (J)"] = -room_to_pipes_raw

    aggregated["Heat Transferred - HWH to Pipes (J)"] = (
        aggregated["Total Net Heat Change (J)"]
        + aggregated["Total Heat Transferred - Pipes to Room (J)"]
        - aggregated["Total Heat Transferred - Room to Pipes (J)"]
    )

    aggregated["Net Radial Heat Loss from Pipes (J)"] = (
        aggregated["Total Heat Transferred - Pipes to Room (J)"]
        - aggregated["Total Heat Transferred - Room to Pipes (J)"]
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Total Net Heat Change (kJ)",
                "Total Heat Transferred - Pipes to Room (kJ)",
                "Total Heat Transferred - Room to Pipes (kJ)",
                "Heat Transferred - HWH to Pipes (kJ)",
                "Net Radial Heat Loss from Pipes (kJ)",
            ],
            "Value": [
                aggregated["Total Net Heat Change (J)"].sum() / 1000,
                aggregated["Total Heat Transferred - Pipes to Room (J)"].sum() / 1000,
                aggregated["Total Heat Transferred - Room to Pipes (J)"].sum() / 1000,
                aggregated["Heat Transferred - HWH to Pipes (J)"].sum() / 1000,
                aggregated["Net Radial Heat Loss from Pipes (J)"].sum() / 1000,
            ],
        }
    )

    return {
        "Radial Heat Flowrate (W)": radial_heat_flowrate,
        "Radial Heat Transferred (J)": radial_heat_transferred,
        "Net Heat Change in Pipes (J)": net_heat_change,
        "Aggregated Heat Transferred (J)": aggregated,
        "Heat Loss (kJ)": summary,
    }
    

def _normalise_heat_records(
        q_dics: dict[str, dict[str, Any]],
        duration: int,
        time_step: int,
        ) -> tuple[pd.Index, list[str]]:
    times = pd.Index(range(0, duration + time_step, time_step), name="Time (s)")
    link_names = [
        name for name in q_dics
        if name not in EXCLUDE_LINKS and not name.startswith("HWH")
    ]
    return times, link_names


def _series_to_dataframe(
        q_dics: dict[str, dict[str, Any]],
        field_name: str,
        duration: int,
        time_step: int,
        fill_first_value: float | None = None,
        ) -> pd.DataFrame:
    times, link_names = _normalise_heat_records(q_dics, duration, time_step)

    data = {}
    for link_name in link_names:
        values = list(q_dics[link_name].get(field_name, []))

        if fill_first_value is not None:
            values = [fill_first_value] + values

        values = values[:len(times)]
        if len(values) < len(times):
            values += [None] * (len(times) - len(values))

        data[link_name] = values

    return pd.DataFrame(data, index=times)


def build_heat_loss_tables(
        q_dics: dict[str, dict[str, Any]],
        duration: int,
        time_step: int,
        ) -> dict[str, pd.DataFrame]:
    radial_heat_flowrate = _series_to_dataframe(
        q_dics=q_dics,
        field_name="heat_flow",
        duration=duration,
        time_step=time_step,
    )

    radial_heat_transferred = _series_to_dataframe(
        q_dics=q_dics,
        field_name="radial_heat_transferred",
        duration=duration,
        time_step=time_step,
        fill_first_value=0.0,
    )

    net_heat_change = _series_to_dataframe(
        q_dics=q_dics,
        field_name="net_heat_change",
        duration=duration,
        time_step=time_step,
        fill_first_value=0.0,
    )
    
    netQ="Net Heat Change in Pipes (J)"
    pipes_to_room="Heat Transferred - Pipes to Room (J)"
    room_to_pipes="Heat Transferred - Room to Pipes (J)"
    HWH_to_pipes="Heat Transferred - HWH to Pipes (J)"
    loss_pipes="Net Heat Loss from Pipes (J)"

    aggregated = pd.DataFrame(index=radial_heat_transferred.index)

    aggregated[netQ] = net_heat_change.sum(axis=1)

    aggregated[pipes_to_room] = (
        radial_heat_transferred.where(radial_heat_transferred > 0, 0.0).sum(axis=1)
    )

    aggregated[room_to_pipes] = (
        radial_heat_transferred.where(radial_heat_transferred < 0, 0.0).sum(axis=1)
    )
    
    aggregated[room_to_pipes] = -aggregated[room_to_pipes]

    aggregated[HWH_to_pipes] = (
        aggregated[netQ]
        + aggregated[pipes_to_room]
        - aggregated[room_to_pipes]
    )

    aggregated[loss_pipes] = (
        aggregated[pipes_to_room]
        - aggregated[room_to_pipes]
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Total Net Heat Change in Pipes (kJ)",
                "Total Heat Transferred - Pipes to Room (kJ)",
                "Total Heat Transferred - Room to Pipes (kJ)",
                "Total Heat Transferred - HWH to Pipes (kJ)",
                "Total Net Heat Loss from Pipes (kJ)",
            ],
            "Value": [
                aggregated[netQ].sum() / 1000,
                aggregated[pipes_to_room].sum() / 1000,
                aggregated[room_to_pipes].sum() / 1000,
                aggregated[HWH_to_pipes].sum() / 1000,
                aggregated[loss_pipes].sum() / 1000,
            ],
        }
    )

    return {
        "Radial Heat Flowrate (W)": radial_heat_flowrate,
        "Radial Heat Transferred (J)": radial_heat_transferred,
        "Net Heat Change in Pipes (J)": net_heat_change,
        "Aggregated Heat Transferred (J)": aggregated,
        "Heat Loss Summary (kJ)": summary,
    }


def export_heat_loss_tables(
        tables: dict[str, pd.DataFrame],
        network_name: str,
        output_dir: str | Path = ".",
        ) -> Path:
    """
    Export prebuilt heat-loss result tables to an Excel workbook.
    """
    output_path = Path(output_dir) / f"heat_loss_{network_name}.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_name)

    return output_path


def export_heat_loss_results(
        q_dics,
        network_name: str,
        duration: int,
        time_step: int,
        output_dir: str | Path = ".",
) -> Path:
    tables = build_heat_loss_tables(
        q_dics=q_dics,
        duration=duration,
        time_step=time_step,
    )

    return export_heat_loss_tables(
        tables=tables,
        network_name=network_name,
        output_dir=output_dir,
    )