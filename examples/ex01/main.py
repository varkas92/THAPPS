# main.py

from copy import copy
from time import sleep
from tqdm import tqdm
import pandas as pd

from copy import deepcopy

from config import (
    INP_FILE,
    PIPE_SPREADSHEET,
    PHYSICAL_PROPERTY_SPREADSHEET,
    OUTPUT_DIR,
    MY_APPLIANCES,
    NUMBER_OF_PEOPLE,
    RANDOM_PATTERNS,
    NUMBER_OF_SIMULATIONS,
    DURATION,
    TIME_STEP,
    FACTOR,
    COLD_WATER,
    HOT_WATER,
    LOW_INTERVAL_TEMP,
    HIGH_INTERVAL_TEMP,
    RESULTS_TEMPLATE,
    ELEMENT_OPTIONS,
    ELEMENT_DEFAULT,
    VARIABLE_OPTIONS_NODES,
    VARIABLE_DEFAULT,
    TIME_OPTIONS,
    TIME_DEFAULT,
    SIMULATION_OPTIONS,
    SIMULATION_DEFAULT,
    load_physical_property_tables,
    build_interpolators,
    load_pipes_df,
    get_variable_options_links,
)

from src.plotting import (
    seconds_to_clock,
)

from src.hydraulics import (
    setup_network,
    run_hydraulic_simulation,   
)

from src.temperature import (
    SimulationContext,
    get_network_data,
    get_temp_dics,
    init_heat_tracking,
)

from src.heat_loss import (
    export_heat_loss_results,
)

from src.dashboard import (
    nodes_for_dash,
    create_dashboard_app,
    run_dashboard,
)


"""
START
"""

physical_property_tables = load_physical_property_tables(PHYSICAL_PROPERTY_SPREADSHEET)

# Precompute interpolators for all sheets
interpolators = build_interpolators(physical_property_tables)

# Read the Pipes sheet
pipes_df = load_pipes_df(PIPE_SPREADSHEET)
results_dic = deepcopy(RESULTS_TEMPLATE)

variable_options_links = get_variable_options_links(
    LOW_INTERVAL_TEMP, HIGH_INTERVAL_TEMP
)


for i in tqdm(range(NUMBER_OF_SIMULATIONS)):
    sleep(0.0001)
    
    wn = setup_network(INP_FILE,
                       TIME_STEP,
                       DURATION,
                       RANDOM_PATTERNS,
                       MY_APPLIANCES,
                       COLD_WATER,
                       HOT_WATER,
    )
    
    plot_links = list(wn.links.keys())
    
    plot_nodes = nodes_for_dash(MY_APPLIANCES)
    
    results = run_hydraulic_simulation(wn)
    
    data = get_network_data(wn)
    demands = results.node['demand']
    #node_ages = results.node['quality']
    link_ages = results.link['quality']
    
    max_age = link_ages.max().max()
    
    total_demand = demands.loc[:,plot_nodes].sum(axis = 1).sum(axis = 0)
    
    for result in results_dic:
        name = result['name']
        value = result['value']
        
        if (max_age > value) and (name == 'max_age'):
            max_age_time = link_ages.max(axis=1).idxmax()
            max_age_pipe = link_ages.max(axis=0).idxmax()
            result['value'] = max_age
            result['coordinates']['pipe'] = max_age_pipe
            result['coordinates']['time'] = seconds_to_clock(max_age_time, True)
            result['results'] = copy(results)
            result['wn'] = deepcopy(wn)

        if (max_age < value) and (name == 'min_age'):
            max_age_time = link_ages.max(axis=1).idxmax()
            max_age_pipe = link_ages.max(axis=0).idxmax()
            result['value'] = max_age
            result['coordinates']['pipe'] = max_age_pipe
            result['coordinates']['time'] = seconds_to_clock(max_age_time, True)
            result['results'] = copy(results)
            result['wn'] = deepcopy(wn)
        
        # if (total_demand > value) and (name == 'max_demand'):
        #     result['value'] = total_demand
        #     result['results'] = copy(results)
        #     result['wn'] = deepcopy(wn)
            
        # if (total_demand < value) and (name == 'min_demand'):
        #     result['value'] = total_demand
        #     result['results'] = copy(results)
        #     result['wn'] = deepcopy(wn)

for result in results_dic:
    if result["results"] is not None and result.get("wn") is not None:
        selected_wn = result["wn"]
        selected_data = get_network_data(selected_wn)
        selected_q_dics = init_heat_tracking([name for name, _ in selected_wn.links()])

        selected_sim_ctx = SimulationContext(
            wn=selected_wn,
            pipes_df=pipes_df,
            duration=DURATION,
            time_step=TIME_STEP,
            interpolators=interpolators,
            factor=FACTOR,
            q_dics=selected_q_dics,
            cold_water=COLD_WATER,
            low_interval_temp=LOW_INTERVAL_TEMP,
            high_interval_temp=HIGH_INTERVAL_TEMP,
        )

        thermal_outputs = get_temp_dics(result["results"], selected_sim_ctx)
        result["temperatures"] = thermal_outputs["temperatures"]
        result["cti"] = thermal_outputs["cti"]

try:
    plot_tanks = wn.tank_name_list
except:
    plot_tanks = []

plot_nodes_plus_tanks = plot_nodes + plot_tanks

#plot_ages = node_ages.loc[:, plot_nodes_plus_tanks].sum(axis = 1)
plot_ages = link_ages.sum(axis = 1)
#max_age = max(plot_ages)

link_options = plot_links
link_default = plot_links[0]
appliance_options = ['SUM OF ALL'] + plot_nodes_plus_tanks #+ plot_links
appliance_default = 'SUM OF ALL'

# Initialize the app
app = create_dashboard_app(
    data=data,
    results_dic=results_dic,
    duration=DURATION,
    time_step=TIME_STEP,
    number_of_simulations=NUMBER_OF_SIMULATIONS,
    number_of_people=NUMBER_OF_PEOPLE,
    variable_options_nodes=VARIABLE_OPTIONS_NODES,
    variable_options_links=variable_options_links,
    variable_default=VARIABLE_DEFAULT,
    simulation_options=SIMULATION_OPTIONS,
    simulation_default=SIMULATION_DEFAULT,
    time_options=TIME_OPTIONS,
    time_default=TIME_DEFAULT,
    element_options=ELEMENT_OPTIONS,
    element_default=ELEMENT_DEFAULT,
    link_options=link_options,
    link_default=link_default,
    low_interval_temp=LOW_INTERVAL_TEMP,
    high_interval_temp=HIGH_INTERVAL_TEMP,
)

#run_dashboard(app, open_browser=True, host="127.0.0.1", port=8050, debug=False) #TODO enable when finished

# -------------------------------
# Process the simulation results
# -------------------------------

# --- Node Pressure ---
# Get node pressure results from the simulation. The DataFrame’s index is the simulation time.
node_pressure = results.node['pressure']
# Exclude reservoirs from the results
node_pressure = node_pressure[[node for node in node_pressure.columns 
                               if node not in wn.reservoir_name_list]]
# Reset the index so that the time becomes the first column
node_pressure = node_pressure.reset_index().rename(columns={'index': 'Time'})
# Order columns: "Time" first, then the rest sorted alphabetically/numerically
node_cols = ['Time'] + sorted([col for col in node_pressure.columns if col != 'Time'])
node_pressure = node_pressure[node_cols]

# --- Link Flow ---
# Get link flow results; the DataFrame’s index is the simulation time.
link_flow = results.link['flowrate'] * 1000 * 60
link_flow = link_flow.reset_index().rename(columns={'index': 'Time'})
# Order columns: "Time" first, then the rest sorted
flow_cols = ['Time'] + sorted([col for col in link_flow.columns if col != 'Time'])
link_flow = link_flow[flow_cols]

# --- Link Age ---
# Get link quality (age) results from the simulation.
link_quality = results.link['quality']
# Exclude valves from the link age results. (Assumes that each link in wn.links has an attribute "link_type".)
links_to_include = [link for link in link_quality.columns 
                    if wn.get_link(link).link_type != 'Valve']
link_age = link_quality[links_to_include]
link_age = link_age.reset_index().rename(columns={'index': 'Time'})
# Order columns: "Time" first, then the rest sorted
age_cols = ['Time'] + sorted([col for col in link_age.columns if col != 'Time'])
link_age = link_age[age_cols]

# --- Link Temperatures ---
# Get link temperature results from the simulation.
selected_result = results_dic[0]
varz1 = selected_result["temperatures"]["links"]

df1 = pd.DataFrame(varz1)

# Convert iteration index to simulation time [s]
df1.index = df1.index * TIME_STEP
df1.index.name = "Time"

df1 = df1.reset_index()

# Order columns: "Time" first, then the rest sorted
temperature_cols = ["Time"] + sorted([col for col in df1.columns if col != "Time"])
df1 = df1[temperature_cols]
# -------------------------------
# Export the results to an Excel file
# -------------------------------

network_name = INP_FILE.stem
output_file = OUTPUT_DIR / f"temperatures_{network_name}.xlsx"
heat_loss_output_file = export_heat_loss_results(
    q_dics=selected_q_dics,
    network_name=network_name,
    duration=DURATION,
    time_step=TIME_STEP,
    output_dir=OUTPUT_DIR,
)

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Write the first dataframe to a sheet
    df1.to_excel(writer, sheet_name='Link Temperatures (°C)', index=False)
    link_flow.to_excel(writer, sheet_name="Link Flow (LPM)", index=False)
    link_age.to_excel(writer, sheet_name="Link Age (s)", index=False)
    node_pressure.to_excel(writer, sheet_name="Node Pressure (m)", index=False)
        
print(f"\nSimulation complete for {network_name}. Results saved in:")
print(f"  - {output_file.stem}.xlsx")
print(f"  - {heat_loss_output_file.stem}.xlsx\n")

