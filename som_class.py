import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import RegularPolygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


class SelfOrganizingMap():
    def __init__(self, map_height, map_width, feature_dim, distance_metric="euclidean"):
        self.map_height = map_height
        self.map_width = map_width
        self.weights = np.random.rand(map_height, map_width, feature_dim)
        self.distance_metric = distance_metric

    def find_bmu(self, input_vector):
        if self.distance_metric == "euclidean":
            distances = np.linalg.norm(self.weights - input_vector, axis=2)
            winner_ind = np.unravel_index(np.argmin(distances), distances.shape)
            return winner_ind

    def update_weights(self, input_vector, bmu_idx, learning_rate, radius):
        for i in range(radius * 2 + 1):
            for j in range(radius * 2 + 1):
                x = bmu_idx[0] - radius + i
                y = bmu_idx[1] - radius + j
                if 0 <= x < self.map_height and 0 <= y < self.map_width:
                    dist = np.sqrt((x - bmu_idx[0])**2 + (y - bmu_idx[1])**2)
                    influence = np.exp(-(dist**2) / (2 * (radius**2 + 1e-8)))
                    self.weights[x, y] += learning_rate * influence * (input_vector - self.weights[x, y])

    def train(self, inputs, num_epochs=1000, learning_rate=0.1, radius=1):
        for epoch in range(num_epochs):
            for input_vector in inputs:
                bmu_ind = self.find_bmu(input_vector)
                self.update_weights(input_vector, bmu_ind, learning_rate, radius)


class SelfOrganizingMapHex():
    def __init__(self, map_height, map_width, feature_dim, distance_metric="euclidean"):
        self.map_height = map_height
        self.map_width = map_width
        self.feature_dim = feature_dim
        self.distance_metric = distance_metric
        self.neurons = {}
        self._initialize_hexagonal_grid()

    def _initialize_hexagonal_grid(self):
        for row in range(self.map_height):
            for col in range(self.map_width):
                q = col - (row - (row & 1)) // 2
                r = row
                self.neurons[(q, r)] = np.random.rand(self.feature_dim)

    def axial_distance(self, coord1, coord2):
        q1, r1 = coord1
        q2, r2 = coord2
        return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2

    def find_bmu(self, input_vector):
        if self.distance_metric == "euclidean":
            min_dist = float('inf')
            bmu_coord = None
            for coord, weights in self.neurons.items():
                dist = np.linalg.norm(weights - input_vector)
                if dist < min_dist:
                    min_dist = dist
                    bmu_coord = coord
            return bmu_coord

    def get_neighbors_in_radius(self, bmu_coord, radius):
        neighbors = []
        for coord in self.neurons.keys():
            if self.axial_distance(bmu_coord, coord) <= radius:
                neighbors.append(coord)
        return neighbors

    def update_weights(self, input_vector, bmu_coord, learning_rate, radius):
        neighbors = self.get_neighbors_in_radius(bmu_coord, radius)
        for coord in neighbors:
            dist = self.axial_distance(bmu_coord, coord)
            influence = np.exp(-(dist**2) / (2 * (radius**2 + 1e-8)))
            self.neurons[coord] += learning_rate * influence * (input_vector - self.neurons[coord])

    def train(self, inputs, num_epochs=1000, learning_rate=0.1, radius=None):
        if radius is None:
            radius = max(self.map_height, self.map_width) // 3
        for epoch in range(num_epochs):
            for input_vector in inputs:
                bmu_coord = self.find_bmu(input_vector)
                self.update_weights(input_vector, bmu_coord, learning_rate, radius)

    def verify_hexagonal_neighbors(self):
        neighbor_counts = {}
        for coord in self.neurons.keys():
            neighbors = self.get_neighbors_in_radius(coord, radius=1)
            neighbors = [n for n in neighbors if n != coord]
            neighbor_counts[coord] = len(neighbors)

        print(f"Total neurons: {len(self.neurons)}")
        print(f"Neighbor count distribution:")
        from collections import Counter
        counts = Counter(neighbor_counts.values())
        for num_neighbors, frequency in sorted(counts.items()):
            print(f"  {num_neighbors} neighbors: {frequency} neurons")

        interior_neurons = [c for c, n in neighbor_counts.items() if n == 6]
        edge_neurons = [c for c, n in neighbor_counts.items() if n < 6]
        print(f"\nInterior neurons (6 neighbors): {len(interior_neurons)}")
        print(f"Edge/corner neurons (<6 neighbors): {len(edge_neurons)}")

        return neighbor_counts


def plot_u_matrix_hex(som_hex, title="U-Matrix (Unified Distance Matrix)", cmap='viridis', figsize=(12, 10)):
    u_matrix = {}

    for coord in som_hex.neurons.keys():
        distances = []
        neighbors = som_hex.get_neighbors_in_radius(coord, radius=1)
        for neighbor_coord in neighbors:
            if neighbor_coord != coord:
                dist = np.linalg.norm(som_hex.neurons[coord] - som_hex.neurons[neighbor_coord])
                distances.append(dist)
        u_matrix[coord] = np.mean(distances) if distances else 0

    fig, ax = plt.subplots(figsize=figsize)
    hex_radius = 0.5
    hex_height = np.sqrt(3) * hex_radius
    values = list(u_matrix.values())
    vmin, vmax = min(values), max(values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for (q, r), value in u_matrix.items():
        col = q + (r - (r & 1)) // 2
        x = col * hex_radius * np.sqrt(3)
        y = r * hex_height
        if r & 1:
            x += hex_radius * np.sqrt(3) / 2
        color = sm.to_rgba(value)
        hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_radius,
                                orientation=0, facecolor=color,
                                edgecolor='white', linewidth=1)
        ax.add_patch(hexagon)

    plt.colorbar(sm, ax=ax, label='Average distance to neighbors')
    ax.set_aspect('equal')
    margin = 1.0
    ax.set_xlim(-margin, som_hex.map_width * hex_radius * np.sqrt(3) + margin)
    ax.set_ylim(-margin, som_hex.map_height * hex_height + margin)
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=15)
    plt.tight_layout()

    return fig, ax, u_matrix


def plot_feature_map_hex(som_hex, input_data, target_values, feature_name=None,
                         cmap='viridis', figsize=(12, 10)):
    feature_map = {coord: [] for coord in som_hex.neurons.keys()}

    for i, sample in enumerate(input_data):
        bmu_coord = som_hex.find_bmu(sample)
        feature_map[bmu_coord].append(target_values[i])

    for coord in feature_map.keys():
        if feature_map[coord]:
            feature_map[coord] = np.mean(feature_map[coord])
        else:
            feature_map[coord] = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    hex_radius = 0.5
    hex_height = np.sqrt(3) * hex_radius
    values = list(feature_map.values())
    vmin, vmax = min(values), max(values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for (q, r), value in feature_map.items():
        col = q + (r - (r & 1)) // 2
        x = col * hex_radius * np.sqrt(3)
        y = r * hex_height
        if r & 1:
            x += hex_radius * np.sqrt(3) / 2
        color = sm.to_rgba(value)
        hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_radius,
                                orientation=0, facecolor=color,
                                edgecolor='gray', linewidth=0.5, alpha=0.9)
        ax.add_patch(hexagon)

    label = feature_name if feature_name else 'Target Value'
    plt.colorbar(sm, ax=ax, label=label)
    ax.set_aspect('equal')
    margin = 1.0
    ax.set_xlim(-margin, som_hex.map_width * hex_radius * np.sqrt(3) + margin)
    ax.set_ylim(-margin, som_hex.map_height * hex_height + margin)
    ax.axis('off')

    title = f'Feature Map: {feature_name}' if feature_name else 'Target Value Feature Map'
    ax.set_title(title, fontsize=14, pad=15)
    plt.tight_layout()

    return fig, ax, feature_map
