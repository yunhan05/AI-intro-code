import pandas as pd
import plotly.express as px

# Load the data
data = pd.read_csv('us_pop_by_state.csv')

# Custom color scale intervals based on the data quartiles
color_scale = [
    (0.000, 'rgb(255,255,255)'),  # white
    (0.0017, 'rgb(255,230,230)'), # very light red
    (0.0054, 'rgb(255,204,204)'), # lighter red
    (0.0135, 'rgb(255,179,179)'), # light red
    (0.0224, 'rgb(255,153,153)'), # less light red
    (0.1191, 'rgb(255,102,102)'), # intermediate red
    (1.000, 'rgb(255,0,0)')       # red
]

# Create the choropleth map using adjusted density intervals
fig = px.choropleth(
    data_frame=data,
    locations='state_code',  # column in the data frame containing state codes
    locationmode='USA-states',  # set of locations match entries in locations
    color='2020_census',  # column giving the color intensity of the regions
    color_continuous_scale=color_scale,  # custom color scale
    scope="usa",  # specify which geographical area to display
    title='2020 U.S. State Population',
    labels={'2020_census':'Population'}
)
# Show the figure
fig.show()
