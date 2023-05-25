from utils import *

cities = {
    0: "Sakhir, Bahrain",
    1: "Jeddah, Saudi Arabia",
    2: "Melbourne, Australia",
    3: "Baku, Azerbaijan",
    4: "Miami, USA",
    5: "Imola, Emilia Romagna",
    6: "Monaco, Monaco",
    7: "Barcelona, Spain",
    8: "Montreal, Canada",
    9: "Spielberg, Austria",
    10: "Silverstone, UK",
    11: "Budapest, Hungary",
    12: "Spa-Francorchamps, Belgium",
    13: "Zandvoort, Netherlands",
    14: "Monza, Italy",
    15: "Singapore, Singapore",
    16: "Suzuca, Japan",
    17: "Lusail, Qatar",
    18: "Austin, USA",
    19: "Mexico City, Mexico",
    20: "Sao Paulo, Brazil",
    21: "Las Vegas, USA",
    22: "Yas Marina, Abu Dhabi"
}

coordinates = {
    0: [26.0325, 50.5106],
    1: [21.5433, 39.1728],
    2: [-37.8497, 144.968],
    3: [40.3725, 49.8532],
    4: [25.7617, -80.1918],
    5: [44.3439, 11.7167],
    6: [43.7384, 7.4246],
    7: [41.57, 2.2611],
    8: [45.5017, -73.5673],
    9: [47.2197, 14.7647],
    10: [52.0786, -1.0169],
    11: [47.5839, 19.2486],
    12: [50.4372, 5.9714],
    13: [52.3886, 4.5446],
    14: [45.6156, 9.2811],
    15: [1.2914, 103.864],
    16: [34.8431, 136.541],
    17: [25.4207, 51.4700],
    18: [30.1328, -97.6411],
    19: [19.4326, -99.1332],
    20: [-23.5505, -46.6333],
    21: [36.1699, -115.1398],
    22: [24.4672, 54.6033]
}




list = ['Sakhir, Bahrain', 'Lusail, Qatar', 'Yas Marina, Abu Dhabi', 'Baku, Azerbaijan', 'Suzuca, Japan', 'Singapore, Singapore', 'Melbourne, Australia',
         'Sao Paulo, Brazil', 'Miami, USA', 'Mexico City, Mexico', 'Austin, USA', 'Las Vegas, USA', 'Montreal, Canada', 'Silverstone, UK', 'Zandvoort, Netherlands',
           'Spa-Francorchamps, Belgium', 'Barcelona, Spain', 'Monaco, Monaco', 'Monza, Italy', 'Imola, Emilia Romagna', 'Spielberg, Austria', 'Budapest, Hungary',
             'Jeddah, Saudi Arabia', 'Sakhir, Bahrain']
n_list = [[k for k, v in cities.items() if v == i] for i in list]

flat_list = sum(n_list, [])

distance = 0
for i in range(len(flat_list)-1):
    distance += haversine(coordinates[flat_list[i]], coordinates[flat_list[i+1]])
distance += haversine(coordinates[flat_list[-1]], coordinates[flat_list[0]])
print(distance)