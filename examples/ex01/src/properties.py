# properties.py

import numpy as np
import pandas as pd

from config import READ_INSULATION

def get_property(
        temperature: float,
        sheet_name: str,
        interpolators: dict
        ) -> float:
    """
    Interpolate a temperature-dependent property from the physical-properties workbook.

    Parameters
    ----------
    temperature : float
        Temperature at which the property is evaluated [°C].
    sheet_name : str
        Name of the sheet/property to read from the interpolator dictionary.
        Examples include:
        - 'w_mu', 'w_rho', 'w_beta', 'w_cp', 'w_k'
        - 'a_mu', 'a_rho', 'a_cp', 'a_k'
        - 'cu_rho', 'cu_cp', 'cu_k'
        - 'pex_rho', 'pex_cp', 'pex_k'
    interpolators : dict[str, callable]
        Dictionary of prebuilt interpolation functions keyed by sheet name.

    Returns
    -------
    float
        Interpolated property value.

    Notes
    -----
    The workbook structure already distinguishes between water, air,
    and solid-material properties through the sheet-name prefixes.
    """
    interpolator = interpolators[sheet_name]
    return float(interpolator(temperature))


def get_pipe_properties(
        pipe_id: str,
        pipes_df: pd.DataFrame,
        ) -> tuple:
    """
    Retrieve additional thermal properties for a pipe from the 'Pipes' sheet.

    Parameters
    ----------
    pipe_id : str
        Pipe identifier.
    pipes_df : pandas.DataFrame
        DataFrame loaded from the 'Pipes' sheet of the network-properties workbook.

    Returns
    -------
    tuple
        Tuple containing ambient temperature, initial temperature, material
        identifier, wall thickness, insulation properties, wall properties,
        and boolean flags indicating whether insulation and wall layers exist.

    Raises
    ------
    ValueError
        If the pipe ID is not found in the spreadsheet.
    """
    pipe_id = str(pipe_id)

    pipe_row = pipes_df.loc[pipes_df.iloc[:, 0] == pipe_id]

    if pipe_row.empty:
        raise ValueError(f"Pipe ID '{pipe_id}' not found in the Pipes sheet.")

    # Special handling for hot-water-heater placeholder elements,
    if pipe_id.startswith("HWH"):
        t_room = float(pipe_row.iloc[0, 1])
        t_initial = float(pipe_row.iloc[0, 2])
        mat_pipe = "cu"
        e_pipe = 0.0
        k_insulator = 0.0
        e_insulator = 0.0
        k_wall = 0.0
        z_wall = 0.0
        is_insulated = False
        has_wall = False

        return (
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
        )

    t_room = float(pipe_row.iloc[0, 1])
    t_initial = float(pipe_row.iloc[0, 2])

    mat_pipe = pipe_row.iloc[0, 3]
    if mat_pipe == 0:
        mat_pipe = "cu"
    elif mat_pipe == 1:
        mat_pipe = "pex"

    e_pipe = float(pipe_row.iloc[0, 4]) / 1000.0

    if READ_INSULATION:
        k_insulator = pipe_row.iloc[0, 5] if pd.notna(pipe_row.iloc[0, 5]) else 0.0
        e_insulator = float(pipe_row.iloc[0, 6]) / 1000.0 if pd.notna(pipe_row.iloc[0, 6]) else 0.0
    else:
        k_insulator = 0.0
        e_insulator = 0.0

    k_wall = pipe_row.iloc[0, 7] if pd.notna(pipe_row.iloc[0, 7]) else 0.0
    z_wall = float(pipe_row.iloc[0, 8]) / 1000.0 if pd.notna(pipe_row.iloc[0, 8]) else 0.0

    is_insulated = e_insulator > 0
    has_wall = z_wall > 0

    return (
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
    )


def get_Nusselt_natural(
        Gr: float,
        Pr: float,
        ) -> float:
    """
    Compute the Nusselt number for natural convection (external).

    Parameters
    ----------
    Gr : float
        Grashof number [-].
    Pr : float
        Prandtl number [-].

    Returns
    -------
    float
        Nusselt number [-].

    Notes
    -----
    Uses piecewise empirical correlation based on the Rayleigh number.
    """
    Ra = Gr * Pr

    if Ra >= 4.545e9:
        C = 0.021
        m = 0.4
    elif Ra >= 1e4:
        C = 0.59
        m = 0.25
    else:
        Ra = 1e4
        C = 0.59
        m = 0.25

    return C * Ra ** m


def get_Nusselt_forced(
        Re: float,
        Pr: float,
        d: float,
        L: float,
        cooling: bool,
        ) -> float:
    """
    Compute the Nusselt number for forced convection (internal).

    Parameters
    ----------
    Re : float
        Reynolds number [-].
    Pr : float
        Prandtl number [-].
    d : float
        Inner diameter [m].
    L : float
        Characteristic length [m].
    cooling : bool
        True if the fluid is cooling, False if it is heating.

    Returns
    -------
    float
        Nusselt number [-].

    Notes
    -----
    - For lower Reynolds numbers, uses the laminar correlation.
    - For higher Reynolds numbers, uses the Dittus-Boelter form with a
      different exponent depending on cooling or heating.
    """
    if Re < 10000:
        return 3.66 + 0.065 * Re * Pr * d / L / (1 + 0.04 * (Re * Pr * d / L) ** (2 / 3))

    C = 0.023
    m = 4 / 5
    n = 1 / 3 if cooling else 2 / 5
    return C * Re ** m * Pr ** n


def get_R(
        pipe,
        package: dict,
        flow: float, interpolators: dict,
        g: float,
        ) -> tuple:
    """
    Compute radial thermal resistances for a fluid package inside a pipe.

    Parameters
    ----------
    pipe : object
        Link/pipe object containing geometry, orientation, and material attributes.
        Expected attributes include diameter, theta, e_link, mat_pipe,
        insulator, e_insulator, k_insulator, wall, z_wall, k_wall,
        and temperature_infinity.
    package : dict
        Package state dictionary containing at least:
        - 'temperature'
        - 'temperature_s_in'
        - 'temperature_s_ex_air'
        - 'volume'
    flow : float
        Absolute volumetric flow rate through the link [m³/s].
    interpolators : dict[str, callable]
        Dictionary of interpolation functions keyed by sheet name.
    g : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    tuple
        Tuple containing:
        - R_overall : float
            Total radial thermal resistance [K/W].
        - R_water : float
            Internal convection resistance [K/W].
        - R_pipe : float
            Pipe-wall conduction resistance [K/W].
        - R_insulator : float
            Insulation conduction resistance [K/W].
        - R_wall : float
            Wall conduction resistance [K/W].
        - R_air : float
            External air-side convection resistance [K/W].

    Notes
    -----
    - water properties are read from 'w_*'
    - air properties are read from 'a_*'
    - pipe-wall material properties are read from '{mat_pipe}_*'
    - resistance values are updated iteratively until the estimated heat flow
      converges within the tolerance
    """
    t_room = pipe.temperature_infinity
    t_bulk = package["temperature"]
    t_s_in = package["temperature_s_in"]
    t_s_ex_air = package["temperature_s_ex_air"]

    d1 = pipe.diameter
    A_cross = np.pi * d1 ** 2 / 4
    vel = flow / A_cross if A_cross > 0 else 0.0

    P1 = np.pi * d1
    L_package = package["volume"] / A_cross
    A1 = L_package * P1

    theta_pipe = pipe.theta

    epsilon_pipe = pipe.e_link
    d2 = d1 + 2 * epsilon_pipe
    mat_pipe = pipe.mat_pipe
    k_pipe = get_property(t_bulk, f"{mat_pipe}_k", interpolators)

    has_insulator = pipe.insulator
    epsilon_insulator = pipe.e_insulator
    k_insulator = pipe.k_insulator

    z_wall = pipe.z_wall
    k_wall = pipe.k_wall

    heat_flow = 0.0
    err = np.inf
    iteration = 0
    finished = False

    while err > 0.01 and not finished:
        t_film_water = (t_bulk * t_s_in) ** 0.5
        t_film_air = (t_room + t_s_ex_air) / 2

        is_cooling = t_bulk > t_room

        mu_water = get_property(t_film_water, "w_mu", interpolators)
        rho_water = get_property(t_film_water, "w_rho", interpolators)
        beta_water = get_property(t_film_water, "w_beta", interpolators)
        cp_water = get_property(t_film_water, "w_cp", interpolators)
        k_water = get_property(t_film_water, "w_k", interpolators)
        nu_water = mu_water / rho_water
        Pr_water = mu_water * cp_water / k_water

        mu_air = get_property(t_film_air, "a_mu", interpolators)
        rho_air = get_property(t_film_air, "a_rho", interpolators)
        beta_air = 1 / (t_film_air + 273.15)
        cp_air = get_property(t_film_air, "a_cp", interpolators)
        k_air = get_property(t_film_air, "a_k", interpolators)
        nu_air = mu_air / rho_air
        Pr_air = mu_air * cp_air / k_air

        Re = vel * d1 / nu_water

        if theta_pipe < np.pi / 6:
            Lc = d1
            Gr_water = g * beta_water * abs(t_bulk - t_s_in) * Lc ** 3 / nu_water ** 2
        else:
            Lc = L_package
            Gr_water = g * np.sin(theta_pipe) * beta_water * abs(t_bulk - t_s_in) * Lc ** 3 / nu_water ** 2

        h_water_forced = 0.0
        h_water_natural = 0.0

        if Re > 0:
            Ri = Gr_water / Re ** 2
            if Ri > 0.1:
                Nu_natural = get_Nusselt_natural(Gr_water, Pr_water)
                h_water_natural = Nu_natural * k_water / Lc
            if Ri < 10:
                Nu_forced = get_Nusselt_forced(Re, Pr_water, d1, L_package, is_cooling)
                h_water_forced = Nu_forced * k_water / Lc
        else:
            Nu_natural = get_Nusselt_natural(Gr_water, Pr_water)
            h_water_natural = Nu_natural * k_water / Lc

        h_water_convection = h_water_forced + h_water_natural
        if h_water_convection == 0:
            h_water_convection = 1e-10

        R_water = 1 / (h_water_convection * A1)
        R_pipe = np.log(d2 / d1) / (2 * np.pi * L_package * k_pipe)

        if has_insulator:
            d3 = d2 + 2 * epsilon_insulator
            R_insulator = np.log(d3 / d2) / (2 * np.pi * L_package * k_insulator)
        else:
            d3 = d2
            R_insulator = 0.0

        P3 = np.pi * d3
        A3 = L_package * P3

        if pipe.wall:
            R_air = 0.0
            S_wall = 2 * np.pi * L_package / np.log(8 * z_wall / (np.pi * d3))
            R_wall = 1 / (k_wall * S_wall)
        else:
            R_wall = 0.0
            if theta_pipe < np.pi / 6:
                Lc_air = d3
                Gr_air = g * beta_air * abs(t_s_ex_air - t_room) * Lc_air ** 3 / nu_air ** 2
                Ra_air = Gr_air * Pr_air
                Nu_air = (
                    0.6
                    + 0.387 * Ra_air ** (1 / 6) / (1 + (0.559 / Pr_air) ** (9 / 16)) ** (8 / 27)
                ) ** 2
            else:
                Lc_air = L_package
                Gr_air = g * np.sin(theta_pipe) * beta_air * abs(t_s_ex_air - t_room) * Lc_air ** 3 / nu_air ** 2
                Ra_air = Gr_air * Pr_air
                if Ra_air < 1e9:
                    Nu_lp = 0.68 + 0.67 * Ra_air ** (1 / 4) / (1 + (0.492 / Pr_air) ** (9 / 16)) ** (4 / 9)
                else:
                    Nu_lp = (
                        0.825
                        + 0.387 * Ra_air ** (1 / 6) / (1 + (0.492 / Pr_air) ** (9 / 16)) ** (8 / 27)
                    ) ** 2
                Nu_air = Nu_lp

            h_air = Nu_air * k_air / Lc_air
            R_air = 1 / (h_air * A3)

        R_overall = R_water + R_pipe + R_insulator + R_wall + R_air
        new_heat_flow = (t_bulk - t_room) / R_overall

        err = abs(new_heat_flow - heat_flow)
        heat_flow = new_heat_flow

        t_s_in = t_bulk - heat_flow * R_water
        t_s_ex_air = t_room + heat_flow * R_air

        iteration += 1
        if iteration > 100:
            finished = True

    return R_overall, R_water, R_pipe, R_insulator, R_wall, R_air