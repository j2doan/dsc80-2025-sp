# project.py


import pandas as pd
import numpy as np
from pathlib import Path

###
from collections import deque
from shapely.geometry import Point
###

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'

import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def create_detailed_schedule(schedule, stops, trips, bus_lines):

    schedule_stops = schedule.merge(stops, on='stop_id', how='left')
    detailed_schedule = schedule_stops.merge(trips, on='trip_id', how='left')

    detailed_schedule['route_id'] = pd.Categorical(detailed_schedule['route_id'], categories=bus_lines, ordered=True)

    detailed_schedule = detailed_schedule.sort_values(by=['route_id', 'trip_id', 'stop_sequence']) # account for multiple trip_id in a route_d, and then order by stop_sequence 12345...
    
    # IDEA: instead of for loop, make a new column for trip_lengths (based on stops), and then assign those total trip_lengths to each trip_id, then can sort faster! 
    trip_lengths = detailed_schedule.groupby('trip_id')['stop_id'].count() # make new col
    detailed_schedule['trip_length'] = detailed_schedule['trip_id'].map(trip_lengths) # map them to the correct trip_ids
    
    detailed_schedule = detailed_schedule.sort_values(by=['route_id', 'trip_length', 'trip_id', 'stop_sequence'])  # fewer stops come FIRST
    
    detailed_schedule = detailed_schedule.drop(columns=['trip_length']) # drop the temporary trip_len col
    
    detailed_schedule = detailed_schedule.set_index('trip_id') # set index

    detailed_schedule = detailed_schedule[detailed_schedule['route_id'].isna() == False]
    
    return detailed_schedule


def visualize_bus_network(bus_df):
    # Load the shapefile for San Diego city boundary
    san_diego_boundary_path = 'data/data_city/data_city.shp'
    san_diego_city_bounds = gpd.read_file(san_diego_boundary_path)
    
    # Ensure the coordinate reference system is correct
    san_diego_city_bounds = san_diego_city_bounds.to_crs("EPSG:4326")
    
    san_diego_city_bounds['lon'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.x)
    san_diego_city_bounds['lat'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.y)
    
    fig = go.Figure()
    
    # Add city boundary
    fig.add_trace(go.Choroplethmapbox(
        geojson=san_diego_city_bounds.__geo_interface__,
        locations=san_diego_city_bounds.index,
        z=[1] * len(san_diego_city_bounds),
        colorscale="Greys",
        showscale=False,
        marker_opacity=0.5,
        marker_line_width=1,
    ))

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": san_diego_city_bounds['lat'].mean(), "lon": san_diego_city_bounds['lon'].mean()},
            zoom=10,
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    color_palette = px.colors.qualitative.Plotly # get the colors
    route_ids = bus_df['route_id'].unique() # get all the unique route_ids
    color_map = {route_id: color_palette[i % len(color_palette)] for i, route_id in enumerate(route_ids)} # create a dictionary for each route_id:color
    # Example: if route 201 is first, it might get blue, route 202 might get orange, etc.

    # Group by trip_id to plot each route
    for trip_id, trip_data in bus_df.groupby(level=0): # not aggreggating data, need index (trip_id), and the rest of the data in that group
        #trip_data_sorted = trip_data#.sort_values(by='stop_sequence')

        fig.add_trace(go.Scattermapbox(
            lat=trip_data['stop_lat'], # a series of stop_lat
            lon=trip_data['stop_lon'], # series of stop_lon
            mode='markers+lines',
            marker=go.scattermapbox.Marker(size=8),
            line=go.scattermapbox.Line(width=2, color=color_map[trip_data['route_id'].iloc[0]]), # make the color of the line match with the route in the dictionary
            text=trip_data['stop_name'],
            hoverinfo='text',
            name=f"Route {trip_data['route_id'].iloc[0]} ({trip_id})" # EX: Route 201 (17287595)
        ))

    return fig




# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def find_neighbors(station_name, detailed_schedule):
    station_data = detailed_schedule[detailed_schedule['stop_name'] == station_name]

    if len(station_data) == 0:
        return np.array([])


    station_data['stop_sequence'] = station_data['stop_sequence'] + 1

    merged_data = pd.merge(
        detailed_schedule.reset_index(),  # Reset index so we can merge on columns
        station_data.reset_index(),  # Reset index so we can merge on trip_id
        how='inner',  # Only keep rows that match in both DataFrames
        left_on=['trip_id', 'stop_sequence'],  # Merge on trip_id and stop_sequence
        right_on=['trip_id', 'stop_sequence']  # Merge on trip_id and stop_sequence (since 'index' is no longer present)
    )

    merged_data

    output = merged_data['stop_name_x'].unique().tolist()
    return np.array(output)


def bfs(start_station, end_station, detailed_schedule):
    
    if start_station not in detailed_schedule['stop_name'].values:
        return f"Start station {start_station} not found."
    if end_station not in detailed_schedule['stop_name'].values:
        return f"End station '{end_station}' not found."

    visited = set()
    queue = deque([[start_station]])

    while queue:
        path = queue.popleft()
        current_stop = path[-1]

        if current_stop == end_station:
            result_rows = []
            for i, stop_name in enumerate(path):
                row = detailed_schedule[detailed_schedule['stop_name'] == stop_name].iloc[0]
                result_rows.append({
                    'stop_name': stop_name,
                    'stop_lat': row['stop_lat'],
                    'stop_lon': row['stop_lon'],
                    'stop_num': i + 1
                })
            return pd.DataFrame(result_rows)

        if current_stop not in visited:
            visited.add(current_stop)
            neighbors = find_neighbors(current_stop, detailed_schedule)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(path + [neighbor])

    return "No path found."


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def simulate_bus_arrivals(tau, seed=12):
    
    np.random.seed(seed) # Random seed -- do not change
    
    start_time = 360
    end_time = 1440

    num_buses = int((end_time - start_time) / tau) # AVG NUM BUSSES

    arrival_minutes = np.random.uniform(start_time, end_time, num_buses)
    # here, instead taking n number of uniform distributions and adding the time,
    # just take 1 uniform distribution for all busses, it will evenly distribute out
    arrival_minutes.sort()

    bus_times = {
        'Arrival Time': [],
        'Interval': []
    }

    for i in range(len(arrival_minutes)):

        if i == 0:
            interval = arrival_minutes[i] - start_time
        else:
            interval = arrival_minutes[i] - arrival_minutes[i - 1]

        total_sec = int(arrival_minutes[i] * 60)
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        seconds = total_sec % 60

        arrival = f"{str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}"
        bus_times['Arrival Time'].append(arrival)
        bus_times['Interval'].append(round(interval, 2))

    return pd.DataFrame(bus_times)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

# HELPER FUNCTIONS
def time_str_to_minutes(t):
    """
    >>> time_str_to_minutes('06:00:00')
    360
    """
    h, m, s = map(int, t.split(":"))
    return h * 60 + m + s / 60

def minutes_to_time_str(mins):
    """
    >>> minutes_to_time_str(360)
    '06:00:00'
    """
    total_seconds = int(mins * 60)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{str(h).zfill(2)}:{str(m).zfill(2)}:{str(s).zfill(2)}"


def simulate_wait_times(arrival_times_df, n_passengers):
    
    # Convert EVERYTHING BACK TO RAW MINUTES for data use

    bus_minutes = arrival_times_df["Arrival Time"].apply(time_str_to_minutes).values

    start_time = 360
    end_time = max(bus_minutes)

    # NOW WE BEGIN. JUST LIKE PART 3
    passenger_arrival_minutes = np.random.uniform(start_time, end_time, n_passengers) # same idea as bus uniform dist
    passenger_arrival_minutes.sort()

    passenger_data = {
        "Passenger Arrival Time": [],
        "Bus Arrival Time": [],
        "Bus Index": [],
        "Wait Time": []
    }

    bus_idx = 0

    for when_passenger_comes in passenger_arrival_minutes:
        # Only increase / go to the next bus once the current passenger is over the departure time of the current bus.
        while bus_idx < len(bus_minutes) and bus_minutes[bus_idx] < when_passenger_comes:
            bus_idx += 1
        
        # End of buses
        if bus_idx >= len(bus_minutes):
            break

        wait_time = bus_minutes[bus_idx] - when_passenger_comes

        passenger_data["Passenger Arrival Time"].append(minutes_to_time_str(when_passenger_comes))
        passenger_data["Bus Arrival Time"].append(minutes_to_time_str(bus_minutes[bus_idx]))
        passenger_data["Bus Index"].append(bus_idx)
        passenger_data["Wait Time"].append(round(wait_time, 2))

    return pd.DataFrame(passenger_data)

def visualize_wait_times(wait_times_df, timestamp):
    h = timestamp.hour
    m = timestamp.minute
    s = timestamp.second

    begin = h * 60 + m + s / 60
    end = begin + 60

    edited_wait_times_df = wait_times_df.copy()
    edited_wait_times_df['Passenger Arrival Time Min'] = edited_wait_times_df['Passenger Arrival Time'].apply(time_str_to_minutes)
    edited_wait_times_df['Bus Arrival Time Min'] = edited_wait_times_df['Bus Arrival Time'].apply(time_str_to_minutes)
    edited_wait_times_df = edited_wait_times_df[(edited_wait_times_df['Passenger Arrival Time Min'] >= begin) & (edited_wait_times_df['Bus Arrival Time Min'] >= begin)]
    edited_wait_times_df = edited_wait_times_df[(edited_wait_times_df['Passenger Arrival Time Min'] <= end) & (edited_wait_times_df['Bus Arrival Time Min'] <= end)]
    edited_wait_times_df

    bus_times = edited_wait_times_df['Bus Arrival Time Min']
    passenger_times_x = edited_wait_times_df['Passenger Arrival Time Min']
    waits_y = edited_wait_times_df['Wait Time']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bus_times,
        y=[0] * len(bus_times),
        mode='markers',
        marker=dict(color='blue', size=8),
        name='Bus Arrivals'
    ))

    for x, y in zip(passenger_times_x, waits_y):
        fig.add_trace(go.Scatter(
            x=[x, x],
            y=[0, y],
            mode='lines',
            line=dict(color='red', width=2, dash='dot'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers',
            marker=dict(color='red', size=5),
            showlegend=False
        ))

    fig.update_layout(
        title='Passenger Wait Times in a 60-Minute Block',
        xaxis_title='Time (minutes in a block)',
        yaxis_title='Wait Time (minutes)',
        height=500
    )  

    return fig
