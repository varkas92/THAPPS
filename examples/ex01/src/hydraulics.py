# hydraulics.py

import numpy as np
from copy import copy

import wntr
import wntr.network.controls as controls
try:
    import pysimdeum
except ImportError:
    pysimdeum = None
    
from config import(
    NUMBER_OF_PEOPLE,
)

from .plotting import (
    seconds_to_clock,
)
    
# ============================================================
# Optional helper for pySIMDEUM house creation
# ============================================================

def build_house_if_needed(random_patterns):
    """
    Build a pySIMDEUM house only when random demand patterns are enabled.
    
    Parameters
    ----------
   random_patterns : bool
        Indicates if using pySIMDEUM or fixed demands

    Returns
    -------
    house : object | None
        pySIMDEUM house object, or None if random patterns are disabled.
    """
    if not random_patterns:
        return None

    if pysimdeum is None:
        raise ImportError(
            "pysimdeum is required when RANDOM_PATTERNS = True, but it is not installed."
        )

    house = pysimdeum.built_house(house_type="one_person")
    base_user_male = house.users[0]
    base_user_male.gender = "male"

    my_users = [base_user_male]
    for i in range(NUMBER_OF_PEOPLE - 1):
        new_user = copy(base_user_male)
        new_user.id = f"user_{i + 2}"
        my_users.append(new_user)

    house = pysimdeum.built_house(house_type="family")
    house.users = my_users
    return house


# def get_appliances(my_appliances):
#     """
#     Fetch pySIMDEUM appliance objects matching appliance type of the input.

#     Parameters
#     ----------
#     my_appliances : list
#         List of appliance dictionaries. Each dictionary is expected to contain
#         a key "type" matching an appliance type available in pySIMDEUM.

#     Returns
#     -------
#     list
#         List of pySIMDEUM appliance objects corresponding to the requested
#         appliance types.

#     Notes
#     -----
#     The function repeatedly builds a temporary pySIMDEUM house and searches
#     its appliance library until all requested appliance types have been found.
#     """
#     obj_appliances = []
#     my_appliances_del = copy(my_appliances)
#     while len(my_appliances_del) > 0:
#         #Build a house (family household)
#         house = pysimdeum.built_house(house_type='family')
#         house_appliances = house.appliances
#         for appliance in house_appliances:
#             for my_app in my_appliances_del:
#                 if appliance.name == my_app['type']:
#                     obj_appliances.append(appliance)
#                     my_appliances_del.remove(my_app)
#     return obj_appliances


def generate_weekly_patterns(appliances,
                             house,
                             wn,
                             random_patterns,
                             duration,
                             time_step
):
    """
    Generate weekly demand patterns for appliances.

    Parameters
    ----------
    appliances : list
        List of appliance definitions used to generate demand patterns.
    house : object or dict
        House configuration used for pattern generation.
    random_patterns : bool
        If True, stochastic demand patterns are generated.
        If False, deterministic or predefined patterns are used.
    wn : wntr.network.WaterNetworkModel
        Network model used to read deterministic demand patterns.
    duration : int
        Total simulation duration [s].
    time_step : int
        Simulation time step [s].

    Returns
    -------
    dict
        Dictionary mapping appliance or node names to generated demand patterns.
    """
    pattern_list = {}

    if random_patterns:
        for j in range(7):
            if j >= 5:
                for user in house.users:
                    user.age = 'home_ad'
                    user.job = False
            else:
                for user in house.users:
                    user.age = 'work_ad'
                    user.job = True

            consumption = house.simulate(num_patterns=1, duration='1 days')
            tot_cons_house = consumption.max(['user'])

            for i in range(len(appliances)):
                appliance = appliances[i]['name']
                appliance_type = appliances[i]['type2']
                pattern = tot_cons_house.sel(enduse=appliance_type)

                if j == 0:
                    pattern_list[appliance] = pattern
                else:
                    pattern_list[appliance] = np.hstack((pattern_list[appliance], pattern))

    else:
        for appliance in appliances:
            name = appliance['name']
            junction = wn.get_node(name)

            for demand in junction.demand_timeseries_list:
                if demand.pattern is not None:
                    base_demand = demand.base_value
                    pattern_multipliers = demand.pattern.multipliers
                    adjusted_multipliers = []

                    time = 0
                    while time < duration:
                        for multiplier in pattern_multipliers:
                            adjusted_multipliers.append(multiplier * base_demand)
                        time = len(adjusted_multipliers) * time_step

                    array = np.zeros((1, len(adjusted_multipliers)))
                    array[0, :] = adjusted_multipliers
                    pattern_list[name] = array

    return pattern_list


def assign_patterns(wn,
                    duration,
                    time_step,
                    random_patterns,
                    pattern_list,
                    my_appliances,
                    cold_water,
                    hot_water
):
    """
    Assign demand patterns and mixing-valve control settings to the network.

    Parameters
    ----------
    wn : wntr.network.WaterNetworkModel
        Network model to which the patterns and controls are applied.
    duration : int
        Total simulation duration [s].
    time_step : int
        Simulation time step [s].
    random_patterns : bool
        Whether the supplied patterns were generated stochastically.
    pattern_list : dict
        Dictionary of demand patterns keyed by appliance or demand-node name.
    my_appliances : list
        Appliance configuration list. Entries may include target mixed-water
        temperatures for nodes connected to both hot and cold branches.
    cold_water : float
        Cold-water source temperature [°C].
    hot_water : float
        Hot-water source temperature [°C].

    Returns
    -------
    wntr.network.WaterNetworkModel
        Network model with assigned patterns and any required controls.

    Notes
    -----
    For appliances connected to both hot and cold branches, this function
    computes the cold- and hot-side demand fractions required to match the
    requested target temperature and applies the corresponding valve settings.
    """
    for appliance in my_appliances:
        appliance_name = appliance['name']
        appliance_pattern = pattern_list[appliance_name]
        pattern = []
        time = 0
        valve_cold_name = 'VC_' + appliance_name
        valve_hot_name = 'VH_' + appliance_name
        try:
            valve_cold = wn.get_link(valve_cold_name)
            has_cold = True
        except:
            valve_cold = None
            has_cold = False

        try:
            valve_hot = wn.get_link(valve_hot_name)
            has_hot = True
        except:
            valve_hot = None
            has_hot = False

        if has_hot and has_cold:
            desired_temp = appliance['target_temp']
            setting_multiplier_cold = (hot_water - desired_temp) / (hot_water - cold_water)
            setting_multiplier_hot = 1 - setting_multiplier_cold
            
            downstream_hot_node_name = valve_hot.end_node_name
            downstream_hot_links = wn.get_links_for_node(downstream_hot_node_name, 'OUTLET')
            for link_name in downstream_hot_links:
                link = wn.get_link(link_name)
                if isinstance(link, wntr.network.Pipe):
                    if link.check_valve != True:
                        link.check_valve = True
                        #print(f"Set check valve on downstream pipe: {link_name}")
                        
            downstream_cold_node_name = valve_cold.end_node_name
            downstream_cold_links = wn.get_links_for_node(downstream_cold_node_name, 'OUTLET')
            for link_name in downstream_cold_links:
                link = wn.get_link(link_name)
                if isinstance(link, wntr.network.Pipe):
                    if link.check_valve != True:
                        link.check_valve = True
                        #print(f"Set check valve on downstream pipe: {link_name}")
            
        past_multiplier = 0
        while time < duration:
            day = int(np.floor(time / 86400))
            second = time - day * 86400
            if random_patterns:
                multiplier = appliance_pattern[second, day] / 1000
            else:
                multiplier = appliance_pattern[0, int(time / time_step - 1)]
            pattern.append(multiplier)
            if multiplier != past_multiplier and (has_hot and has_cold):
                setting_cold = setting_multiplier_cold * multiplier
                setting_hot = setting_multiplier_hot * multiplier
                act_cold = controls.ControlAction(valve_cold, 'setting', setting_cold)
                act_hot = controls.ControlAction(valve_hot, 'setting', setting_hot)
                clock_time = seconds_to_clock(time, False)
                cond = controls.SimTimeCondition(wn, '=', clock_time)
                ctrl_name_cold = valve_cold_name + clock_time
                ctrl_name_hot = valve_hot_name + clock_time
                ctrl_cold = controls.Control(cond, act_cold, name=ctrl_name_cold)
                ctrl_hot = controls.Control(cond, act_hot, name=ctrl_name_hot)
                try:
                    wn.add_control(ctrl_name_cold, ctrl_cold)
                    wn.add_control(ctrl_name_hot, ctrl_hot)
                except:
                    wn.remove_control(ctrl_name_cold)
                    wn.remove_control(ctrl_name_hot)
                    wn.add_control(ctrl_name_cold, ctrl_cold)
                    wn.add_control(ctrl_name_hot, ctrl_hot)
                    
            past_multiplier = multiplier
            time = time + time_step
            
        if random_patterns:
            wn.add_pattern(appliance_name, pattern)
            demand_node = wn.get_node(appliance_name)
            base_1 = 1
            demand_node.demand_timeseries_list.clear()
            my_category = 'pysimdeum'
            demand_node.add_demand(base = base_1, pattern_name = appliance_name, category = my_category)
    
    return wn


def setup_network(inp_file,
                  time_step,
                  duration,
                  random_patterns,
                  my_appliances,
                  cold_water,
                  hot_water
):
    """
    Build and prepare the hydraulic network for simulation.

    Parameters
    ----------
    inp_file : str
        Path to the EPANET input file.
    time_step : int
        Simulation time step [s].
    duration : int
        Total simulation duration [s].
    random_patterns : bool
        Whether demand patterns are generated stochastically.
    my_appliances : list
        Appliance configuration list used for demand assignment.
    cold_water : float
        Cold-water source temperature [°C].
    hot_water : float
        Hot-water source temperature [°C].

    Returns
    -------
    wntr.network.WaterNetworkModel
        Prepared network model ready for hydraulic simulation.

    Notes
    -----
    This function is intended to centralise hydraulic-network setup, including
    demand-pattern assignment and any required control logic.
    """
    wn = wntr.network.WaterNetworkModel(inp_file) #Use existing network from .inp file
    wn.options.quality.parameter = 'AGE'

    wn.options.time.hydraulic_timestep = time_step
    wn.options.time.quality_timestep = time_step
    wn.options.time.pattern_timestep =  time_step
    wn.options.time.report_timestep = time_step
    wn.options.time.pattern_start = 0

    wn.options.time.duration = duration
    
    for name, control in wn.controls():
        wn.remove_control(name)
    
    house = build_house_if_needed(random_patterns)
    
    pattern_list = generate_weekly_patterns(
        appliances=my_appliances,
        house=house,
        wn=wn,
        random_patterns=random_patterns,
        duration=duration,
        time_step=time_step,
    )
    
    wn = assign_patterns(
        wn=wn,
        duration=duration,
        time_step=time_step,
        random_patterns=random_patterns,
        pattern_list=pattern_list,
        my_appliances=my_appliances,
        cold_water=cold_water,
        hot_water=hot_water,
    )
    
    return wn

def run_hydraulic_simulation(wn):
    """
    Run EPANET hydraulic simulation.

    Parameters
    ----------
    wn : WaterNetworkModel

    Returns
    -------
    results : SimulationResults
    """
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim(file_prefix=os.path.join(tmpdir, "temp"))

    return results