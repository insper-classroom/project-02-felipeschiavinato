from cities import *
import plotly.graph_objs as go


# Separate the latitudes and longitudes into their own lists
list = [0, 17, 22, 3, 16, 15, 2, 20, 4, 19, 18, 21, 8, 10, 13, 12, 7, 6, 14, 5, 9, 11, 1, 0]
# list = [i for i in (range(23))]
lats = []
lons = []
for i in list:
    lats.append(coordinates[i][0])
    lons.append(coordinates[i][1])

# Create a Scattergeo plot for the points
points = go.Scattergeo(
    lat = lats,
    lon = lons,
    mode = 'markers',
    marker = dict(
        size = 5,
        color = 'rgb(255, 0, 0)',
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )
    ),
)

# Create a Scattergeo plot for the lines
path = go.Scattergeo(
    lat = lats,
    lon = lons,
    mode = 'lines',
    line = dict(
        width = 2,
        color = 'blue',
    ),
)

# Create a layout
layout = go.Layout(
    title_text = 'Path on 3D Globe',
    showlegend = False,
    geo = dict(
        projection_type = 'orthographic',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
    ),
)

# Combine the points and path into a single data object
data = [points, path]

# Create the figure and add the data and layout
fig = go.Figure(data=data, layout=layout)

# Show the figure
fig.show()
