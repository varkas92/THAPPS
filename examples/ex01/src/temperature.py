# temperature.py

from time import sleep

import numpy as np
from tqdm import tqdm

from config import G

from .properties import get_property, get_pipe_properties, get_R

from dataclasses import dataclass

@dataclass
class SimulationContext:
    """
    Shared simulation dependencies and settings used across the thermal model.

    Parameters
    ----------
    wn : object
        WNTR network model.
    pipes_df : object
        Pipe-property table.
    duration : int
        Total simulation duration [s].
    time_step : int
        Simulation time step [s].
    interpolators : dict
        Interpolation functions for temperature-dependent properties.
    factor : float
        Model factor used in thermal calculations.
    q_dics : dict
        Heat-tracking dictionary.
    cold_water : float
        Reference cold-water temperature [°C].
    low_interval_temp : float
        Lower CTI threshold [°C].
    high_interval_temp : float
        Upper CTI threshold [°C].
    """
    wn: object
    pipes_df: object
    duration: int
    time_step: int
    interpolators: dict
    factor: float
    q_dics: dict
    cold_water: float
    low_interval_temp: float
    high_interval_temp: float


def get_network_data(wn):
    """
    Extract node and link metadata from a WNTR network.

    Parameters
    ----------
    wn : wntr.network.WaterNetworkModel
        WNTR network model containing nodes and links.

    Returns
    -------
    dict
        Dictionary containing serialised network metadata.

    Notes
    -----
    This function currently serialises the network to JSON and reads it back
    into a plain Python dictionary.
    """
    nodes = []
    for name, node in wn.nodes():
        node_data = {
            "name": name,
            "coordinates": node.coordinates,
            "node_type": node.node_type,
        }

        if node.node_type == "Junction":
            node_data["elevation"] = node.elevation
            node_data["demand_timeseries_list"] = []
            for d in node.demand_timeseries_list:
                node_data["demand_timeseries_list"].append({
                    "base_val": d.base_value
                })

        elif node.node_type == "Tank":
            node_data["elevation"] = node.elevation
            node_data["diameter"] = node.diameter
            node_data["max_level"] = node.max_level

        elif node.node_type == "Reservoir":
            node_data["base_head"] = node.base_head

        nodes.append(node_data)

    links = []
    for name, link in wn.links():
        link_data = {
            "name": name,
            "start_node_name": link.start_node_name,
            "end_node_name": link.end_node_name,
        }

        if hasattr(link, "diameter"):
            link_data["diameter"] = link.diameter

        if hasattr(link, "length"):
            link_data["length"] = link.length

        links.append(link_data)

    return {"nodes": nodes, "links": links}


def init_heat_tracking(links_iterable):
    """
    Initialise data structures for tracking heat transfer in the network.

    Parameters
    ----------
    links_iterable : iterable
        Iterable of link names to include in the heat-tracking structure.

    Returns
    -------
    dict
        Dictionary storing heat-transfer tracking variables for each included link.

    Notes
    -----
    Links listed in the local exclusion set are skipped.
    """
    EXCLUDE_LINKS = {"extra", "HWH"}

    Q = {}
    for name in links_iterable:
        if name in EXCLUDE_LINKS:
            continue

        Q[name] = {
            "heat_flow": [],
            "radial_heat_transferred": [],
            "net_heat_change": [],
            "sum_net": 0.0,
            "sum_pos": 0.0,
            "sum_neg": 0.0,
        }

    return Q


def update_heat(heat_dic,
                link_name: str, 
                heat_flow_W_radial: float,
                net_heat_change_J: float,
                dt_s: float,
) -> None:
    """
    Update heat tracking for a link.

    Parameters
    ----------
    heat_dic : dict
        Dictionary storing accumulated heat values.
    link_name : str
        Name of the link being updated.
    heat_flow_W_radial : float
        Radial heat flow for the current time step [W].
    net_heat_change_J : float
        Net heat change for the current time step [J].
    dt_s : float
        Time-step duration [s].

    Returns
    -------
    None

    Notes
    -----
    This function updates the heat tracking dictionary in place.
    If the link name is not present in the tracking dictionary, the update is skipped.
    """
    if link_name not in heat_dic:
        return

    heat_J = heat_flow_W_radial * dt_s
    rec = heat_dic[link_name]

    rec["heat_flow"].append(heat_flow_W_radial)
    rec["radial_heat_transferred"].append(heat_J)
    rec["net_heat_change"].append(net_heat_change_J)
    
    rec["sum_net"] += net_heat_change_J

    if heat_J >= 0:
        rec["sum_pos"] += heat_J
    else:
        rec["sum_neg"] += heat_J


def move_packages(start_node_name,
                  car,
                  flows,
                  t,
                  ctx,
):
    """
    Move water packages through the network from a starting node.

    Parameters
    ----------
    start_node_name : str
        Name of the node from which package movement starts.
    car : list
        List of water packages currently to be propagated through the network.
        Each package contains state variables such as temperature,
        age, volume, and position-related information.
    flows : pandas.DataFrame
        Link flow results from the hydraulic simulation.
    t : int
        Current simulation time [s].
    ctx : SimulationContext
        Shared simulation context.

    Returns
    -------
    tuple
        Updated package-routing information after movement during the current
        time step. The exact tuple contents depend on the existing model logic.

    Notes
    -----
    This function propagates discrete water packages across links and nodes
    according to the hydraulic results at time `t`.

    The function uses the network topology stored in `wn` and the current
    flow directions/magnitudes stored in `flows` to determine where packages
    move during the current time step. This is meant to be use recursively.
    """
    
    wn = ctx.wn
    time_step = ctx.time_step
    
    node = wn.get_node(start_node_name)
    node.temperature = car[0]['temperature']
    node.volume = 0
    if len(node.links_out) > 0:
        for link_name in node.links_out:
            link = wn.get_link(link_name)
            link_flow = abs(round(flows.loc[t * time_step, link_name], 6))
            weight = link_flow / node.outflow
            sub_car = []
            for package in car:
                p_vol = package['volume'] * weight
                p_temp = package['temperature']
                p_temp_s_in = package['temperature_s_in']
                p_temp_s_ex = package['temperature_s_ex']
                p_temp_s_ex_air = package['temperature_s_ex_air']
                p_old_pipe = package['old_pipe']
                sub_car.append({'volume': p_vol, 'temperature': p_temp, 'temperature_s_in': p_temp_s_in, 'temperature_s_ex': p_temp_s_ex, 'temperature_s_ex_air': p_temp_s_ex_air, 'old_pipe': p_old_pipe})
            if link.link_type == 'Pipe':
                for p in link.packages:
                    sub_car.append(p)
                link.packages = []
                av_volume = link.volume
                temp_vol = 0
                temp_vol_s_in = 0
                temp_vol_s_ex = 0
                temp_vol_s_ex_air = 0
                while av_volume > 0:
                    package = sub_car[0]
                    p_vol = package['volume']
                    p_temp = package['temperature']
                    p_temp_s_in = package['temperature_s_in']
                    p_temp_s_ex = package['temperature_s_ex']
                    p_temp_s_ex_air = package['temperature_s_ex_air']
                    p_old_pipe = package['old_pipe']
                    if p_old_pipe != link.name:
                        t_bulk = p_temp
                        t_room = link.temperature_infinity
                        t_arit_mean = (t_bulk + t_room) / 2
                        t_geom_mean = (t_bulk * t_room) ** 0.5                          
                        if t_bulk >= t_room: #if cooling
                            t_s_in = t_arit_mean
                            t_s_ex = t_geom_mean
                        else: #heating
                            t_s_in = t_geom_mean
                            t_s_ex = t_arit_mean
                        package['temperature_s_in'] = t_s_in
                        package['temperature_s_ex'] = t_s_ex
                        package['temperature_s_ex_air'] = t_s_ex
                        package['old_pipe'] = link.name
                    if p_vol < av_volume:
                        link.packages.append(package)
                        sub_car.remove(package)
                        temp_vol += p_temp * p_vol
                        temp_vol_s_in += p_temp_s_in * p_vol
                        temp_vol_s_ex += p_temp_s_ex * p_vol
                        temp_vol_s_ex_air += p_temp_s_ex_air * p_vol
                        av_volume = av_volume - p_vol
                    else:
                        link.packages.append({'volume': av_volume, 'temperature': p_temp, 'temperature_s_in': p_temp_s_in, 'temperature_s_ex': p_temp_s_ex, 'temperature_s_ex_air': p_temp_s_ex_air, 'old_pipe': p_old_pipe})
                        sub_car[0]['volume'] = p_vol - av_volume
                        temp_vol += p_temp * av_volume
                        temp_vol_s_in += p_temp_s_in * av_volume
                        temp_vol_s_ex += p_temp_s_ex * av_volume
                        temp_vol_s_ex_air += p_temp_s_ex_air * av_volume
                        link.temperature = temp_vol / link.volume
                        link.temperature_s_in = temp_vol_s_in / link.volume
                        link.temperature_s_ex = temp_vol_s_ex / link.volume
                        link.temperature_e_ex_air = temp_vol_s_ex_air / link.volume
                        av_volume = 0
            
            if link.end_node_name == start_node_name:
                end_node = wn.get_node(link.start_node_name)
            else:
                end_node = wn.get_node(link.end_node_name)
                
            if end_node.node_type == 'Junction':
                end_node.links_check = end_node.links_check - 1
                if end_node.links_in != 1:
                    for package in sub_car:
                        p_vol = package['volume']
                        p_temp = package['temperature']
                        p_temp_s_in = package['temperature_s_in']
                        p_temp_s_ex = package['temperature_s_ex']
                        p_temp_s_ex_air = package['temperature_s_ex_air']
                        p_old_pipe = package['old_pipe']
                        end_node.temperature = (end_node.temperature * end_node.volume + p_temp * p_vol) / (end_node.volume + p_vol)
                        end_node.volume += p_vol
                    sub_car = [{'volume': end_node.volume, 'temperature': end_node.temperature, 'temperature_s_in': p_temp_s_in, 'temperature_s_ex': p_temp_s_ex, 'temperature_s_ex_air': p_temp_s_ex_air, 'old_pipe': p_old_pipe}]
                if end_node.links_check == 0:
                    move_packages(end_node.name, sub_car, flows, t, ctx)
                    

def update_pipe_temperature(link,
                            flows,
                            t,
                            ctx,
):
    """
    Update the temperature state of water packages inside a pipe.

    Parameters
    ----------
    link : object
        Network link or pipe object whose package temperatures are updated.
    flows : pandas.DataFrame
        Link flow results from the hydraulic simulation.
    t : int
        Current simulation time [s].
    ctx : SimulationContext
        Shared simulation context.

    Returns
    -------
    None

    Notes
    -----
    This function updates package temperatures in place for the specified link.
    The calculation includes radial heat exchange with the surroundings and
    axial exchange between neighbouring packages and connected links, while
    also updating link-level heat-tracking variables.
    """
    
    wn = ctx.wn
    time_step = ctx.time_step
    interpolators = ctx.interpolators
    g = G
    factor = ctx.factor
    q_dics = ctx.q_dics
    
    name = link.name
    start_node_name = link.start_node_name
    start_node = wn.get_node(start_node_name)
    start_links = start_node.links
    end_node_name = link.end_node_name
    end_node = wn.get_node(end_node_name)
    end_links = end_node.links
    flow = abs(round(flows.loc[t * time_step, name], 6))
    t_room = link.temperature_infinity
    if name.startswith('HWH'):
        link.packages = [{'volume': link.volume, 'temperature': t_room, 'temperature_s_in': t_room, 'temperature_s_ex': t_room, 'temperature_s_ex_air': t_room, 'old_pipe': name}]
    else:
        d = link.diameter
        A_cross = np.pi * d ** 2 / 4
        v_cross = flow / A_cross
        mat_pipe = link.mat_pipe
        e_pipe = link.e_link
        A_pipe = np.pi / 4 * ((d + 2 * e_pipe) ** 2 - d ** 2)
        theta = link.theta
        high_node = link.high_node
            
        link_packages = link.packages
        
        for i, package in enumerate(link_packages):
            package_volume = package['volume']
            package_length = package_volume / A_cross
            #-----Radial
            R_overall, R_water, R_pipe, R_insulator, R_wall, R_air = get_R(
                link, package, flow, interpolators, g
            )
            
            t_bulk = package['temperature']
            
            rho_water = get_property(t_bulk, 'w_rho', interpolators)
            cp_water = get_property(t_bulk, 'w_cp', interpolators)
            C_water = rho_water * cp_water * A_cross * package_length
            rho_pipe = get_property(t_bulk, mat_pipe + '_rho', interpolators)
            cp_pipe = get_property(t_bulk, mat_pipe + '_cp', interpolators)
            C_pipe = rho_pipe * cp_pipe * A_pipe * package_length
            C_total = C_water + C_pipe
            b = 1 / (C_total * R_overall)
            
            dt_r = t_room + (t_bulk - t_room) * np.e ** (-1 * b * time_step) - t_bulk
            
            axial = True #TEST #TODO
            dt_a_conduction = 0
            dt_a_convection = 0
            if axial == True:
                #-----Axial
                dt_a_conduction = 0
                dt_a_convection = 0
                
                mu_water = get_property(t_bulk, 'w_mu', interpolators)
                nu_water = mu_water / rho_water
                beta_water = get_property(t_bulk, 'w_beta', interpolators)
                Re_water = v_cross * d / nu_water
                k_water = get_property(t_bulk, 'w_k', interpolators)
                k_pipe = get_property(t_bulk, mat_pipe + '_k', interpolators)
                k_average = (k_water * A_cross + k_pipe * A_pipe) / (A_cross + A_pipe) #this is equivalent to adding the heat flowrate because wall and water are in parallel with respect to the x-axis
                
                R_0 = package_length / 2 / (k_average * (A_cross + A_pipe))
                if theta < np.pi / 6:
                    g_water = g
                    Lc = d
                    #Lc = d / 4 #TEST
                else:
                    g_water = g * np.sin(theta)
                    Lc = package_length #TODO check if it is always d or rather d/4 in the axial case
                    #Lc = d
                    #Lc = d / 4 #TEST
                    
                #if the package is the one next to the start node
                if (i == 0 and start_node_name == link.start_node_packages) or (i == len(link_packages) - 1 and end_node_name == link.start_node_packages):
                    #For start node: other pipe is upstream and current pipe is downstream
                    for other_name in start_links:
                        if other_name != name:
                            other_link = wn.get_link(other_name)
                            if start_node_name == other_link.start_node_packages:
                                other_package = other_link.packages[0]
                            else:
                                other_package = other_link.packages[-1]
                            #---Conduction
                            other_d = other_link.diameter
                            other_A_cross = np.pi * other_d ** 2 / 4
                            other_mat = other_link.mat_pipe
                            other_e_pipe = other_link.e_link
                            other_A_pipe = np.pi / 4 * ((other_d + 2 * other_e_pipe) ** 2 - other_d ** 2)
                            other_volume = other_package['volume']
                            other_length = other_volume / other_A_cross
                            other_t = other_package['temperature']
                            other_k = get_property(other_t, 'w_k', interpolators)
                            other_k_pipe = get_property(other_t, other_mat + '_k', interpolators)
                            other_k_average = (other_k * other_A_cross + other_k_pipe * other_A_pipe) / (other_A_cross + other_A_pipe)
                            other_R = other_length / 2 / (other_k_average * (other_A_cross + other_A_pipe))
                            R_0n = R_0 + other_R
                            b_0n = 1 / (C_total * R_0n)
                            dt_a_conduction += other_t + (t_bulk - other_t) * np.e ** (-1 * b_0n * time_step) - t_bulk
                            #---Convection
                            other_mu = get_property(other_t, 'w_mu', interpolators)
                            other_rho = get_property(other_t, 'w_rho', interpolators)
                            other_nu = other_mu / other_rho
                            other_flow = abs(round(flows.loc[t * time_step, other_name], 6))
                            other_v_cross = other_flow / other_A_cross
                            other_Re = other_v_cross * other_d / other_nu
                            other_beta = get_property(other_t, 'w_beta', interpolators)
                            other_theta = other_link.theta
                            if other_theta < np.pi / 6:
                                other_g = g
                                other_Lc = other_d 
                                #other_Lc = other_d / 4 #TEST
                            else:
                                other_g = g * np.sin(other_theta)
                                other_Lc = other_length
                                #other_Lc = other_d / 4 #TEST
                                
                            A_cross_conv = np.mean([A_cross, other_A_cross])
                            if flow == 0 and other_flow == 0:
                                convection = True
                            else:
                                Gr_water = g_water * beta_water * abs(t_bulk - other_t) * Lc ** 3 / nu_water ** 2
                                other_Gr = other_g * other_beta * abs(t_bulk - other_t) * other_Lc ** 3 / other_nu ** 2
                                Gr_conv = np.mean([Gr_water, other_Gr])
                                Re_conv = np.mean([Re_water, other_Re])
                                Ri_conv = Gr_conv / Re_conv ** 2
                                #Ri_conv = 0 #TEST
                                if Ri_conv > 0.1:
                                    convection = True
                                else:
                                    convection = False
                                    
                            if convection: #verify positioning
                                other_high_node = other_link.high_node
                                if t_bulk > other_t: #if downstream pipe is warmer
                                    if (high_node[0] is None) or (start_node_name in high_node): #if downstream hot pipe is horizontal or going down
                                        other_high_node = other_link.high_node
                                        if start_node_name not in other_high_node: #if upstream cold pipe is going down or is horizontal
                                            convection = True
                                        else: #if upstream cold pipe is going up
                                            convection = False
                                    else: #if downstream hot pipe is going up
                                        convection = False
                                elif t_bulk < other_t: #if downstream pipe is colder
                                    if start_node_name not in high_node: #if downstream cold pipe is horizontal or going up
                                        other_high_node = other_link.high_node
                                        if (other_high_node[0] is None) or (start_node_name in other_high_node): #if upstream hot pipe is going up or is horizontal
                                            convection = True
                                        else: #if upstream hot pipe is going down
                                            convection = False
                                    else: #if downstream cold pipe is going down
                                        convection = False
                                else: #if the temperature is the same
                                    convection = False
                                    
                            #convection = False #TEST
                            if convection: #if it passed all tests, calculate convection
                                v_water = factor * g_water * beta_water * Lc ** 2 * abs(t_bulk - other_t) / nu_water
                                other_v = factor * other_g * other_beta* other_Lc ** 2 * abs(t_bulk - other_t) / other_nu
                                v_conv = np.min([v_water, other_v])
                                #v_conv = np.mean([v_water, other_v])
                                volume_conv = v_conv * A_cross_conv * time_step #total volume exchanged between packages
                                if volume_conv > package_volume * 2:
                                    volume_conv = package_volume * 2
                                if volume_conv > other_volume * 2:
                                    volume_conv = other_volume * 2
                                dt_a_convection += (volume_conv / 2 * other_t + (package_volume - volume_conv / 2) * t_bulk) / package_volume - t_bulk #weighted average temperature of the volume enetering the package and the volume that is left, minus the previous temperature
                                #print(f"-start-{name} with {other_name}-start-")
                                    
                #if the package is the one next to the end node
                if (i == 0 and end_node_name == link.start_node_packages) or (i == len(link_packages) - 1 and start_node_name == link.start_node_packages):
                    #For end node: other pipe is downstream and current pipe is upstream
                    for other_name in end_links:
                        if other_name != name:
                            other_link = wn.get_link(other_name)
                            if end_node_name == other_link.start_node_packages:
                                other_package = other_link.packages[0]
                            else:
                                other_package = other_link.packages[-1]
                            #---Conduction
                            other_d = other_link.diameter
                            other_A_cross = np.pi * other_d ** 2 / 4
                            other_mat = other_link.mat_pipe
                            other_e_pipe = other_link.e_link
                            other_A_pipe = np.pi / 4 * ((other_d + 2 * other_e_pipe) ** 2 - other_d ** 2)
                            other_volume = other_package['volume'] 
                            other_length = other_volume / other_A_cross
                            other_t = other_package['temperature']
                            other_k = get_property(other_t, 'w_k', interpolators)
                            other_k_pipe = get_property(other_t, other_mat + '_k', interpolators)
                            other_k_average = (other_k * other_A_cross + other_k_pipe * other_A_pipe) / (other_A_cross + other_A_pipe)
                            other_R = other_length / 2 / (other_k_average * (other_A_cross + other_A_pipe))
                            R_0n = R_0 + other_R
                            b_0n = 1 / (C_total * R_0n)
                            dt_a_conduction += other_t + (t_bulk - other_t) * np.e ** (-1 * b_0n * time_step) - t_bulk
                            #---Convection
                            other_mu = get_property(other_t, 'w_mu', interpolators)
                            other_rho = get_property(other_t, 'w_rho', interpolators)
                            other_nu = other_mu / other_rho
                            other_flow = abs(round(flows.loc[t * time_step, other_name], 6))
                            other_v_cross = other_flow / other_A_cross
                            other_Re = other_v_cross * other_d / other_nu
                            other_beta = get_property(other_t, 'w_beta', interpolators)
                            other_theta = other_link.theta
                            if other_theta < np.pi / 6:
                                other_g = g
                                other_Lc = other_d
                                #other_Lc = other_d / 4 #TEST
                            else:
                                other_g = g * np.sin(other_theta)
                                other_Lc = other_length
                                #other_Lc = other_d / 4 #TEST
                                
                            A_cross_conv = np.mean([A_cross, other_A_cross])
                            if flow == 0 and other_flow == 0:
                                convection = True
                            else:
                                Gr_water = g_water * beta_water * abs(t_bulk - other_t) * Lc ** 3 / nu_water ** 2
                                other_Gr = other_g * other_beta * abs(t_bulk - other_t) * other_Lc ** 3 / other_nu ** 2
                                Gr_conv = np.mean([Gr_water, other_Gr])
                                Re_conv = np.mean([Re_water, other_Re])
                                Ri_conv = Gr_conv / Re_conv ** 2
                                #Ri_conv = 0 #TEST
                                if Ri_conv > 0.1:
                                    convection = True
                                else:
                                    convection = False
                            
                            if convection: #verify positioning
                                other_high_node = other_link.high_node
                                if t_bulk > other_t: #if upstream pipe is warmer
                                    if (high_node[0] is None) or (end_node_name in high_node): #if upstream hot pipe is horizontal or going up
                                        other_high_node = other_link.high_node
                                        if end_node_name not in other_high_node: #if downstream cold pipe is going up or is horizontal
                                            convection = True
                                        else: #if downstream cold pipe is going down
                                            convection = False
                                    else: #if upstream hot pipe is going down
                                        convection = False
                                elif t_bulk < other_t: #if upstream pipe is colder
                                    if end_node_name not in high_node: #if upstream cold pipe is horizontal or going down
                                        other_high_node = other_link.high_node
                                        if (other_high_node[0] is None) or (end_node_name in other_high_node): #if downstream hot pipe is going down or is horizontal
                                            convection = True
                                        else: #if downstream hot pipe is going up
                                            convection = False
                                    else: #if upstream cold pipe is going up
                                        convection = False
                                else: #if the temperature is the same
                                    convection = False
                                            
                            #convection = False #TEST
                            if convection: #if it passed all tests, calculate convection
                                v_water = factor * g_water * beta_water * Lc ** 2 * abs(t_bulk - other_t) / nu_water
                                other_v = factor * other_g * other_beta * other_Lc ** 2 * abs(t_bulk - other_t) / other_nu
                                v_conv = np.min([v_water, other_v])
                                #v_conv = np.mean([v_water, other_v])
                                volume_conv = v_conv * A_cross_conv * time_step #total volume exchanged between packages
                                if volume_conv > package_volume * 2:
                                    volume_conv = package_volume * 2
                                    #print(f"a-end-{name} with {other_name}-end-")
                                if volume_conv > other_volume * 2:
                                    volume_conv = other_volume * 2
                                    #print(f"b-end-{name} with {other_name}-end-")
                                dt_a_convection += (volume_conv / 2 * other_t + (package_volume - volume_conv / 2) * t_bulk) / package_volume - t_bulk #weighted average temperature of the volume enetering the package and the volume that is left, minus the previous temperature                  
                                
                #if the package is not the first one, take the previous one
                if i > 0:
                    other_package = link.packages[i - 1]
                    #---Conduction
                    other_volume = other_package['volume'] 
                    other_length = other_volume / A_cross
                    other_t = other_package['temperature']
                    other_k = get_property(other_t, 'w_k', interpolators)
                    other_k_pipe = get_property(other_t, mat_pipe + '_k', interpolators)
                    other_k_average = (other_k * A_cross + other_k_pipe * A_pipe) / (A_cross + A_pipe)
                    other_R = other_length / 2 / (other_k_average * (A_cross + A_pipe))
                    R_0n = R_0 + other_R
                    b_0n = 1 / (C_total * R_0n)
                    dt_a_conduction += other_t + (t_bulk - other_t) * np.e ** (-1 * b_0n * time_step) - t_bulk
                    #---Convection
                    other_mu = get_property(other_t, 'w_mu', interpolators)
                    other_rho = get_property(other_t, 'w_rho', interpolators)
                    other_nu = other_mu / other_rho
                    other_Re = v_cross * d / other_nu
                    other_beta = get_property(other_t, 'w_beta', interpolators)
                    other_Lc = other_length
                    #other_Lc = other_d / 4 #TEST
                    other_g = g_water
                    A_cross_conv = A_cross
                    Re_conv = Re_water
                    if flow == 0:
                        convection = True
                    else:
                        Gr_water = g_water * beta_water * abs(t_bulk - other_t) * Lc ** 3 / nu_water ** 2
                        other_Gr = other_g * other_beta * abs(t_bulk - other_t) * other_Lc ** 3 / other_nu ** 2
                        Gr_conv = np.mean([Gr_water, other_Gr])
                        #Gr_conv = g_conv * beta_conv * abs(t_bulk - other_t) * Lc_conv ** 3 / nu_conv ** 2
                        Ri_conv = Gr_conv / Re_conv ** 2
                        #Ri_conv = 0 #TEST
                        if Ri_conv > 0.1:
                            convection = True
                        else:
                            convection = False
                            
                    if convection: #verify positioning
                        if t_bulk > other_t: #if downstream package is warmer
                            if (high_node[0] is None) or (link.start_node_packages in high_node): #if packages arranged counting from more to less elevated or horizontal
                                convection = True
                            else:
                                convection = False
                        elif t_bulk < other_t: #if downstream package is colder
                            if high_node == link.start_node_packages in high_node: #if packages arranged counting from more to less elevated
                                convection = False
                            else:
                                convection = True
                    
                    #convection = False #TEST
                    if convection: #if it passed all tests, calculate convection
                        v_water = factor * g_water * beta_water * Lc ** 2 * abs(t_bulk - other_t) / nu_water
                        other_v = factor * other_g * other_beta* other_Lc ** 2 * abs(t_bulk - other_t) / other_nu
                        v_conv = np.min([v_water, other_v])
                        #v_conv = np.mean([v_water, other_v])
                        volume_conv = v_conv * A_cross_conv * time_step #total volume exchanged between packages
                        if volume_conv > package_volume * 2:
                            volume_conv = package_volume * 2
                        if volume_conv > other_volume * 2:
                            volume_conv = other_volume * 2
                        dt_a_convection += (volume_conv / 2 * other_t + (package_volume - volume_conv / 2) * t_bulk) / package_volume - t_bulk #weighted average temperature of the volume enetering the package and the volume that is left, minus the previous temperature
                #if the package is not the last one, take the next one
                if i < len(link_packages) - 1:
                    other_package = link.packages[i + 1]
                    #---Conduction
                    other_volume = other_package['volume'] 
                    other_length = other_volume / A_cross
                    other_t = other_package['temperature']
                    other_k = get_property(other_t, 'w_k', interpolators)
                    other_k_pipe = get_property(other_t, mat_pipe + '_k', interpolators)
                    other_k_average = (other_k * A_cross + other_k_pipe * A_pipe) / (A_cross + A_pipe)
                    other_R = other_length / 2 / (other_k_average * (A_cross + A_pipe))
                    R_0n = R_0 + other_R
                    b_0n = 1 / (C_total * R_0n)
                    dt_a_conduction += other_t + (t_bulk - other_t) * np.e ** (-1 * b_0n * time_step) - t_bulk
                    #---Convection
                    other_mu = get_property(other_t, 'w_mu', interpolators)
                    other_rho = get_property(other_t, 'w_rho', interpolators)
                    other_nu = other_mu / other_rho
                    other_Re = v_cross * d / other_nu
                    other_beta = get_property(other_t, 'w_beta', interpolators)
                    other_Lc = other_length
                    #other_Lc = other_d / 4 #TEST
                    other_g = g_water
                    A_cross_conv = A_cross
                    Re_conv = Re_water
                    if flow == 0:
                        convection = True
                    else:
                        Gr_water = g_water * beta_water * abs(t_bulk - other_t) * Lc ** 3 / nu_water ** 2
                        other_Gr = other_g * other_beta * abs(t_bulk - other_t) * other_Lc ** 3 / other_nu ** 2
                        Gr_conv = np.mean([Gr_water, other_Gr])
                        Ri_conv = Gr_conv / Re_conv ** 2
                        #Ri_conv = 0 #TEST
                        if Ri_conv > 0.1:
                            convection = True
                        else:
                            convection = False
                            
                    if convection: #verify positioning
                        if t_bulk > other_t: #if upstream package is warmer
                            if link.start_node_packages in high_node: #if packages arranged counting from more to less elevated
                                convection = False
                            else:
                                convection = True
                        elif t_bulk < other_t: #if upstream package is colder
                            if (high_node[0] is None) or (link.start_node_packages in high_node): #if packages arranged counting from more to less elevated or horizontal
                                convection = True
                            else:
                                convection = False
                      
                    #convection = False #TEST
                    if convection: #if it passed all tests, calculate convection
                        v_water = factor * g_water * beta_water * Lc ** 2 * abs(t_bulk - other_t) / nu_water
                        other_v = factor * other_g * other_beta* other_Lc ** 2 * abs(t_bulk - other_t) / other_nu
                        v_conv = np.min([v_water, other_v])
                        #v_conv = np.mean([v_water, other_v])
                        volume_conv = v_conv * A_cross_conv * time_step #total volume exchanged between packages
                        if volume_conv > package_volume * 2:
                            volume_conv = package_volume * 2
                        if volume_conv > other_volume * 2:
                            volume_conv = other_volume * 2
                        dt_a_convection += (volume_conv / 2 * other_t + (package_volume - volume_conv / 2) * t_bulk) / package_volume - t_bulk #weighted average temperature of the volume enetering the package and the volume that is left, minus the previous temperature
            old_t = package['temperature']
            new_t = old_t + dt_r + dt_a_conduction + dt_a_convection
            heat_flow_radial = (new_t - t_room) / R_overall #possitive if cooling
            heat_flow_radial_old = (old_t - t_room) / R_overall #possitive if cooling
            t_s_in = new_t - heat_flow_radial * R_water
            t_s_ex = t_s_in - heat_flow_radial * R_pipe
            t_s_ex_air = t_room + heat_flow_radial * R_air
            
            package['temperature'] = new_t
            package['temperature_s_in'] = t_s_in
            package['temperature_s_ex'] = t_s_ex
            package['temperature_s_ex_air'] = t_s_ex_air
            
            net_energy_change = C_total * (new_t - old_t)
            update_heat(q_dics, name, heat_flow_radial_old, net_energy_change, time_step)
    

def get_temp_dics(results,
                  ctx,
):
    """
    Run the temperature model and return thermal outputs for one hydraulic run.

    Parameters
    ----------
    results : object
        Hydraulic simulation results returned by the hydraulic solver.
    ctx : SimulationContext
        Shared simulation context containing the network model, thermal
        property tables, timestep settings, and heat-tracking structures.

    Returns
    -------
    dict
        Dictionary containing:
        - "temperatures" : dict
            Time-dependent node and link temperatures.
        - "cti" : dict
            Time-dependent CTI values for nodes and links.

    Notes
    -----
    This function is the main orchestration routine of the temperature model.
    It initialises thermal state variables, advances water packages through
    the network, updates package temperatures, and stores the resulting
    temperatures and CTI values for later plotting and analysis.
    """
    
    wn = ctx.wn
    pipes_df = ctx.pipes_df
    duration = ctx.duration
    time_step = ctx.time_step
    cold_water = ctx.cold_water
    low_interval_temp = ctx.low_interval_temp
    high_interval_temp = ctx.high_interval_temp
    
    res = results
    demands = res.node['demand']
    flows = res.link['flowrate']
        
    for name, node in wn.nodes():
        node.links = []
        node.temperature = 0
        node.cti = 0
        
    for name, link in wn.pipes():
        t_room, t_initial, mat_pipe, e_pipe, k_insulator, e_insulator, k_wall, z_wall, is_insulated, has_wall = get_pipe_properties(name, pipes_df)
        link.temperature_infinity = t_room
        link.temperature = t_initial
        link.temperature_s_in = link.temperature_infinity
        link.temperature_s_ex = link.temperature_infinity
        link.temperature_s_ex_air = link.temperature_infinity
        link.cti = 0
        link.flow_direction = 0
        link.start_node_packages = link.start_node_name
        link.volume = link.length * np.pi * link.diameter ** 2 / 4
        link.insulator = is_insulated #Not insulated if False
        link.wall = has_wall #No wall if False
        link.k_insulator = k_insulator
        link.k_wall = k_wall
        link.mat_pipe = mat_pipe
        link.z_wall = z_wall
        link.e_link = e_pipe
        link.e_insulator = e_insulator
        start_node = wn.get_node(link.start_node_name)
        x_start, y_start = start_node.coordinates
        start_node.links.append(name)
        end_node = wn.get_node(link.end_node_name)
        x_end, y_end = end_node.coordinates
        end_node.links.append(name)
        if start_node.node_type == 'Reservoir':
            z_start = start_node.base_head
        else:
            z_start = start_node.elevation
        if end_node.node_type == 'Reservoir':
            z_end = end_node.base_head
        else:
            z_end = end_node.elevation
        delta_xy = ((x_end - x_start) ** 2 + (y_end - y_start) ** 2) ** (1 / 2)
        delta_z = abs(z_end - z_start)
        if delta_xy == 0:
            theta = np.pi / 2
        else:
            theta = np.arctan(delta_z / delta_xy)
        link.theta = theta
        theta_critical = np.arctan(link.diameter / link.length)
        if theta <= theta_critical:
            link.high_node = [None]
        elif z_end > z_start:
            link.high_node = [link.end_node_name]
        else:
            link.high_node = [link.start_node_name]
        link.packages = [{'volume': link.volume, 'temperature': t_initial, 'temperature_s_in': link.temperature_infinity, 'temperature_s_ex': link.temperature_infinity, 'temperature_s_ex_air': link.temperature_infinity, 'old_pipe': link.name}]
    
    for i in range(2):
        exclude_names = {n for n, _ in wn.pipes()}
        for name, link in wn.links():
            if name not in exclude_names:
                t_room, t_initial, mat_pipe, e_pipe, k_insulator, e_insulator, k_wall, z_wall, is_insulated, has_wall = get_pipe_properties(
                    name, pipes_df
                )
                if link.link_type == 'Pump':
                    link.diameter = "NA"
                link.temperature_infinity = t_room #maybe not necessary
                link.temperature = t_initial
                link.temperature_s_in = link.temperature_infinity #maybe not necessary
                link.temperature_s_ex = link.temperature_infinity #maybe not necessary
                link.cti = 0 #maybe not necessary
                link.flow_direction = 0 #maybe not necessary #TODO
                start_node = wn.get_node(link.start_node_name)
                links_of_start = start_node.links
                end_node = wn.get_node(link.end_node_name)
                links_of_end = end_node.links
                start_name = link.start_node_name
                end_name = link.end_node_name
                for pipe_name in links_of_start: #for all pipes that connect to the valve from the start
                    pipe = wn.get_link(pipe_name)
                    high_nodes = pipe.high_node
                    if (start_name in high_nodes) and (end_name not in high_nodes): #if the valve start node is the high node of that pipe
                        pipe.high_node += [end_name] #then the end node of the valve must also be its high node
                    if pipe_name not in links_of_end:
                        end_node.links.append(pipe_name) #all the start node links are also links of the end node
                for pipe_name in links_of_end: #for all pipes that connect to the valve from the end
                    pipe = wn.get_link(pipe_name)
                    high_nodes = pipe.high_node
                    if end_name in high_nodes: #if the valve end node is the high node of that pipe
                        pipe.high_node += [start_name] #then the start node of the valve must also be its high node
                    if pipe_name not in links_of_start:
                        start_node.links.append(pipe_name) #all the end node links are also links of the start node
            
    temp_dics = {'links': [],'links_s_ex': [], 'nodes': []}
    cti_dics = {'links': [], 'nodes': []}
    
    for t in tqdm(range(int(duration / time_step + 1))):
        sleep(0.0001)
        
        #Verify if there are no demands for the current time to speed up #TODO
        #res_outflow = 'test'
        # for name, reservoir in wn.reservoirs():
        #     downstream_links = wn.get_links_for_node(name, 'OUTLET')
        #     for link_name in downstream_links:
        #         link_outflow = abs(round(flows.loc[t * time_step, link_name], 6))
        #         res_outflow += link_outflow

        #Add demand to the total outflow count of each demand node
        for name, node in wn.nodes():
            node.volume = 0
            node.sum_count = 0
            node.links_out = []
            node.links_in = 0
            node.links_check = 0
            if node.node_type == 'Junction':
                node.outflow = round(demands.loc[t * time_step, name], 6)
            else:
                node.outflow = 0
                
        for name, link in wn.links():
            start_node = wn.get_node(link.start_node_name)
            end_node = wn.get_node(link.end_node_name)
            link_flow = round(flows.loc[t * time_step, name], 6)
            if link_flow < 0:
                new_direction = -1
                start_node.links_in += 1
                start_node.links_check += 1
                end_node.links_out.append(name)
                end_node.outflow += abs(link_flow)
                link.start_node_packages = end_node.name
            elif link_flow > 0:
                new_direction = 1
                end_node.links_in += 1
                end_node.links_check += 1
                start_node.links_out.append(name)
                start_node.outflow += link_flow
                link.start_node_packages = start_node.name
            else:
                new_direction = 0
            #Update temperatures of all packages
            if link.link_type == 'Pipe':
                #Merge packages if no flow or if first and last temperature are very similar
                first_package = link.packages[0]
                last_package = link.packages[-1]
                first_temp = first_package['temperature']
                last_temp = last_package['temperature']
                diff_temp = abs(first_temp - last_temp)
                if new_direction == 0 or diff_temp <= 0.1:
                    temp_vol = 0
                    temp_vol_s_in = 0
                    temp_vol_s_ex = 0
                    temp_vol_s_ex_air = 0
                    for package in link.packages:
                        temp_vol += package['temperature'] * package['volume']
                        temp_vol_s_in += package['temperature_s_in'] * package['volume']
                        temp_vol_s_ex += package['temperature_s_ex'] * package['volume']
                        temp_vol_s_ex_air += package['temperature_s_ex_air'] * package['volume']
                    link.temperature = temp_vol / link.volume
                    link.temperature_s_in = temp_vol_s_in / link.volume
                    link.temperature_s_ex = temp_vol_s_ex / link.volume
                    link.temperature_s_ex_air = temp_vol_s_ex_air / link.volume
                    t_bulk = link.temperature
                    t_s_in = link.temperature_s_in
                    t_s_ex = link.temperature_s_ex
                    t_s_ex_air = link.temperature_s_ex_air

                    link.packages = [{'temperature': t_bulk, 'volume': link.volume, 'temperature_s_in': t_s_in, 'temperature_s_ex': t_s_ex, 'temperature_s_ex_air': t_s_ex_air, 'old_pipe': link.name}]
                    start_node.temperature = (start_node.temperature * start_node.sum_count + t_bulk) / (start_node.sum_count + 1)
                    start_node.sum_count += 1
                    end_node.temperature = (end_node.temperature * end_node.sum_count + t_bulk) / (end_node.sum_count + 1)
                    end_node.sum_count += 1
                #reverse package order if flow changes to opposite direction
                if new_direction != 0 and new_direction != link.flow_direction: #This is ok because we merge all packages at zero flow (flow_direction = 0)
                    link.packages.reverse()
                #Function that updates the temperature of a link
                update_pipe_temperature(link, flows, t, ctx)
                    
            link.flow_direction = new_direction
                    
        for name, reservoir in wn.reservoirs():
            reservoir.temperature = cold_water
            
            if reservoir.outflow != 0:
                vol_new = reservoir.outflow * time_step
                temp_new = cold_water
                car = [{'volume': vol_new, 'temperature': temp_new, 'temperature_s_in': temp_new, 'temperature_s_ex': temp_new, 'temperature_s_ex_air': temp_new, 'old_pipe': 'COLD_SOURCE'}] #TODO name from file
                move_packages(name, car, flows, t, ctx)

        for name, valve in wn.valves():
            start_node = wn.get_node(valve.start_node_name)
            valve.temperature = start_node.temperature
                
        for name, pump in wn.pumps():
            start_node = wn.get_node(pump.start_node_name)
            pump.temperature = start_node.temperature
            
        node_temps = {}
        link_temps = {}
        link_temps_s_ex = {}
        node_cti = {}
        link_cti = {}
        for name, link in wn.links(): #TODO: Check if valves not needed
            link_temps[name] = link.temperature
            link_temps_s_ex[name] = link.temperature_s_ex
            if t > 0:
                if link.temperature >= low_interval_temp and link.temperature <= high_interval_temp:
                    link.cti = link.cti + time_step / 3600
                else:
                    link.cti = 0
            link_cti[name] = link.cti
            start = link.start_node_name
            end = link.end_node_name
            if not(start in node_temps.keys()):
                start_node = wn.get_node(start)
                node_temps[start] = start_node.temperature
                if t > 0:
                    if start_node.temperature >= low_interval_temp and start_node.temperature <= high_interval_temp:
                        start_node.cti = start_node.cti + time_step / 3600
                    else:
                        start_node.cti = 0
                node_cti[name] = start_node.cti
            if not(end in node_temps.keys()):
                end_node = wn.get_node(end)
                node_temps[end] = end_node.temperature
                if t > 0:
                    if end_node.temperature >= low_interval_temp and end_node.temperature <= high_interval_temp:
                        end_node.cti = end_node.cti + time_step / 3600
                    else:
                        end_node.cti = 0
                node_cti[name] = end_node.cti
                
        temp_dics['links'].append(link_temps)
        temp_dics['links_s_ex'].append(link_temps_s_ex)
        temp_dics['nodes'].append(node_temps)
        cti_dics['links'].append(link_cti)
        cti_dics['nodes'].append(node_cti)
        
    return {"temperatures": temp_dics, "cti": cti_dics}