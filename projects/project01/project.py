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
    ...


def bfs(start_station, end_station, detailed_schedule):
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def simulate_bus_arrivals(tau, seed=12):
    
    np.random.seed(seed) # Random seed -- do not change
    
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def simulate_wait_times(arrival_times_df, n_passengers):

    ...

def visualize_wait_times(wait_times_df, timestamp):
    ...
