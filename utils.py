import math
import torch

def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # radius of Earth in kilometers

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def haversine_distance(point1, point2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # print(f"Point 1: {point1}")
    # print(f"Point 2: {point2}")
    
    # Convert coordinates from degrees to radians
    lat1, lon1 = torch.deg2rad(point1[0])
    lat2, lon2 = torch.deg2rad(point2[0])
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # Distance
    distance = R * c
    return distance

def nearest_neighbor(state):
    print(f'State: {state}')
    n_cities = state.shape[1]
    print(f'Number of cities: {n_cities}')
    visited = torch.zeros(n_cities, dtype=torch.bool).to(device)
    current_city = 0
    path = [current_city]
    visited[current_city] = 1
    for _ in range(n_cities - 1):
        distances = [haversine_distance(state[0, current_city].unsqueeze(0), state[0, i].unsqueeze(0)) if not visited[i] else float('inf') for i in range(n_cities)]
        next_city = torch.argmin(torch.tensor(distances).to(device))
        visited[next_city] = 1
        path.append(next_city.item())
        current_city = next_city
    print(path)
    return torch.tensor(path).to(device)
