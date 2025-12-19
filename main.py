import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import time
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from copy import deepcopy
from collections import deque


class Graph:
    """
    Model:   Responsible ONLY for the graph data structure.
    """

    def __init__(self, vertices):
        self.V = vertices
        self.adj = [[] for _ in range(vertices)]

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)


class GraphVisualizer:
    """View:   Visualizes a colored graph using NetworkX"""

    def __init__(self, graph):
        self.graph_data = graph
        self.palette = [
            'magenta', 'teal', 'red', 'green', 'blue',
            'yellow', 'orange', 'purple', 'cyan', 'pink',
            'gray', 'brown', 'lime', 'indigo', 'gold'
        ]

    def draw_solution(self, color_assignments, chromatic_number, elapsed_time=None, algorithm_name=""):
        G_visual = nx.Graph()
        G_visual.add_nodes_from(range(self.graph_data.V))
        for u in range(self.graph_data.V):
            for v in self.graph_data.adj[u]:
                if u < v:
                    G_visual.add_edge(u, v)

        color_map = []
        for i in color_assignments:
            color_idx = (i - 1) % len(self.palette)
            color_map.append(self.palette[color_idx])

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G_visual, seed=42)
        nx.draw_networkx_nodes(G_visual, pos, node_color=color_map, node_size=700, edgecolors='black')
        nx.draw_networkx_labels(G_visual, pos, font_color='white', font_weight='bold')
        nx.draw_networkx_edges(G_visual, pos, width=2)

        title = f"{algorithm_name} Graph Coloring Solution\nChromatic Number: {chromatic_number}"
        if elapsed_time is not None:
            title += f"\nTime Taken: {elapsed_time:.6f} sec"
        plt.title(title)

        plt.axis('off')
        plt.show()


# ============================================
# BFS COLORING ALGORITHM
# ============================================
class BFSColoring:
    """
    BFS-based greedy graph coloring algorithm.
    Colors vertices in BFS traversal order, using the smallest available color.
    """

    def __init__(self, graph, start_vertex=0):
        self.graph = graph
        self.start_vertex = start_vertex
        self.color_assignment = [-1] * graph.V
        self.chromatic_number = 0
        self.nodes_visited = 0

    def get_available_color(self, vertex):
        """
        Find the smallest color not used by adjacent vertices
        """
        # Get colors of adjacent vertices
        adjacent_colors = set()
        for neighbor in self.graph.adj[vertex]:
            if self.color_assignment[neighbor] != -1:
                adjacent_colors.add(self.color_assignment[neighbor])

        # Find smallest available color (starting from 0)
        color = 0
        while color in adjacent_colors:
            color += 1

        return color

    def solve(self, verbose=True):
        """
        Perform BFS coloring on the graph
        """
        if verbose:
            print(f"\nStarting BFS Coloring from vertex {self.start_vertex}...")

        start_time = time.time()

        # Initialize BFS
        visited = [False] * self.graph.V
        queue = deque([self.start_vertex])
        visited[self.start_vertex] = True

        # BFS traversal and coloring
        while queue:
            vertex = queue.popleft()
            self.nodes_visited += 1

            # Assign the smallest available color
            color = self.get_available_color(vertex)
            self.color_assignment[vertex] = color

            # Update chromatic number
            self.chromatic_number = max(self.chromatic_number, color + 1)

            # Add unvisited neighbors to queue
            for neighbor in self.graph.adj[vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        # Handle disconnected components
        for vertex in range(self.graph.V):
            if self.color_assignment[vertex] == -1:
                color = self.get_available_color(vertex)
                self.color_assignment[vertex] = color
                self.chromatic_number = max(self.chromatic_number, color + 1)
                self.nodes_visited += 1

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Convert 0-indexed colors to 1-indexed for consistency
        self.color_assignment = [c + 1 for c in self.color_assignment]

        if verbose:
            print(f"\n✓ BFS Coloring Complete!")
            print(f"Chromatic Number: {self.chromatic_number}")
            print(f"Color assignments: {self.color_assignment}")
            print(f"Nodes visited: {self.nodes_visited}")
            print(f"Computational time: {elapsed_time:.6f} seconds")

        return self.color_assignment, self.chromatic_number, elapsed_time


# ============================================
# BACKTRACKING ALGORITHM
# ============================================
class BackTracking:
    """
    Controller/Service:   Responsible for the algorithmic logic.
    """

    def __init__(self, graph):
        self.graph = graph
        self.best_color_count = float('inf')
        self.best_solution = None
        self.recursive_calls = 0
        self.pruned_branches = 0

    def is_safe(self, node, c, current_color_assignment):
        for neighbor in self.graph.adj[node]:
            if current_color_assignment[neighbor] == c:
                self.pruned_branches += 1
                return False
        return True

    def rec(self, node, used_colors, current_color_assignment):
        self.recursive_calls += 1

        if node == self.graph.V:
            if used_colors < self.best_color_count:
                self.best_color_count = used_colors
                self.best_solution = list(current_color_assignment)
            return

        if used_colors >= self.best_color_count:
            self.pruned_branches += 1
            return

        for c in range(1, used_colors + 1):
            if self.is_safe(node, c, current_color_assignment):
                current_color_assignment[node] = c
                self.rec(node + 1, used_colors, current_color_assignment)
                current_color_assignment[node] = 0

        if used_colors + 1 < self.best_color_count:
            current_color_assignment[node] = used_colors + 1
            self.rec(node + 1, used_colors + 1, current_color_assignment)
            current_color_assignment[node] = 0

    def solve(self):
        initial_assignment = [0] * self.graph.V
        self.best_color_count = float('inf')
        self.best_solution = None
        self.recursive_calls = 0
        self.pruned_branches = 0

        start_time = time.time()
        self.rec(0, 0, initial_assignment)
        end_time = time.time()

        elapsed_time = end_time - start_time

        if self.best_solution:
            print("\n✓ Valid Solution Found!")
            print(f"Chromatic Number: {self.best_color_count}")
            print(f"Color assignments: {self.best_solution}")
            print(f"Computational time: {elapsed_time:.6f} seconds")

        return self.best_solution, self.best_color_count, elapsed_time


# ============================================
# HILL CLIMBING ALGORITHM
# ============================================
class HillClimbing:
    """Hill Climbing algorithm for graph coloring"""

    def __init__(self, graph, max_colors=None):
        self.graph = graph
        self.max_colors = max_colors or graph.V
        self.current_solution = None
        self.current_conflicts = float('inf')
        self.iterations = 0

    def count_conflicts(self, coloring):
        """Count the number of conflicting edges"""
        conflicts = 0
        for u in range(self.graph.V):
            for v in self.graph.adj[u]:
                if u < v and coloring[u] == coloring[v]:
                    conflicts += 1
        return conflicts

    def get_chromatic_number(self, coloring):
        """Get the number of unique colors used"""
        return len(set(coloring))

    def generate_initial_solution(self):
        """Generate a random initial solution"""
        return [random.randint(1, self.max_colors) for _ in range(self.graph.V)]

    def get_neighbors(self, coloring):
        """
        Generate neighboring solutions by changing one vertex's color
        """
        neighbors = []
        for vertex in range(self.graph.V):
            current_color = coloring[vertex]
            # Try all possible colors for this vertex
            for new_color in range(1, self.max_colors + 1):
                if new_color != current_color:
                    neighbor = coloring[:]
                    neighbor[vertex] = new_color
                    neighbors.append(neighbor)
        return neighbors

    def solve(self, max_iterations=1000, verbose=True):
        """
        Run Hill Climbing algorithm
        """
        if verbose:
            print(f"\nStarting Hill Climbing...")
            print(f"Max colors: {self.max_colors}")
            print(f"Max iterations: {max_iterations}")

        start_time = time.time()

        # Generate initial solution
        self.current_solution = self.generate_initial_solution()
        self.current_conflicts = self.count_conflicts(self.current_solution)

        best_solution = self.current_solution[:]
        best_conflicts = self.current_conflicts
        best_colors = self.get_chromatic_number(best_solution)

        stagnation_counter = 0
        max_stagnation = 100

        for iteration in range(max_iterations):
            self.iterations += 1

            # If we found a valid solution, try to reduce colors
            if self.current_conflicts == 0:
                current_colors = self.get_chromatic_number(self.current_solution)
                if current_colors < self.max_colors:
                    # Try with fewer colors
                    self.max_colors = current_colors
                    self.current_solution = self.generate_initial_solution()
                    self.current_conflicts = self.count_conflicts(self.current_solution)
                    stagnation_counter = 0
                    continue

            # Generate neighbors
            neighbors = self.get_neighbors(self.current_solution)

            # Find the best neighbor
            best_neighbor = None
            best_neighbor_conflicts = self.current_conflicts

            for neighbor in neighbors:
                neighbor_conflicts = self.count_conflicts(neighbor)
                if neighbor_conflicts < best_neighbor_conflicts:
                    best_neighbor = neighbor
                    best_neighbor_conflicts = neighbor_conflicts

            # If we found a better neighbor, move to it
            if best_neighbor is not None and best_neighbor_conflicts < self.current_conflicts:
                self.current_solution = best_neighbor
                self.current_conflicts = best_neighbor_conflicts
                stagnation_counter = 0

                # Update best solution
                if self.current_conflicts < best_conflicts or \
                        (self.current_conflicts == best_conflicts and
                         self.get_chromatic_number(self.current_solution) < best_colors):
                    best_solution = self.current_solution[:]
                    best_conflicts = self.current_conflicts
                    best_colors = self.get_chromatic_number(best_solution)

                    if verbose and (iteration % 100 == 0 or best_conflicts == 0):
                        print(f"Iteration {iteration}:   Conflicts={best_conflicts}, Colors={best_colors}")
            else:
                # No better neighbor found (local optimum)
                stagnation_counter += 1

                if stagnation_counter >= max_stagnation:
                    if verbose:
                        print(f"Stagnation detected at iteration {iteration}.   Restarting...")
                    # Random restart
                    self.current_solution = self.generate_initial_solution()
                    self.current_conflicts = self.count_conflicts(self.current_solution)
                    stagnation_counter = 0

        elapsed_time = time.time() - start_time

        if verbose:
            if best_conflicts == 0:
                print(f"\n✓ Valid Solution Found!")
                print(f"Chromatic Number: {best_colors}")
                print(f"Color assignments: {best_solution}")
            else:
                print(f"\n✗ No valid solution found")
                print(f"Best solution has {best_conflicts} conflicts")
                print(f"Colors used: {best_colors}")
            print(f"Total iterations: {self.iterations}")
            print(f"Computational time:   {elapsed_time:.6f} seconds")

        return best_solution, best_colors if best_conflicts == 0 else None, elapsed_time


# ============================================
# CULTURAL ALGORITHM COMPONENTS
# ============================================
class Individual:
    """Represents a single coloring solution"""

    def __init__(self, graph, coloring=None, max_colors=None):
        self.graph = graph
        self.max_colors = max_colors or graph.V
        if coloring is None:
            self.coloring = [random.randint(1, self.max_colors) for _ in range(graph.V)]
        else:
            self.coloring = coloring
        self.fitness = 0
        self.conflicts = 0
        self.chromatic_number = 0
        self.evaluate()

    def evaluate(self):
        """Calculate fitness based on conflicts and number of colors used"""
        self.conflicts = 0
        for u in range(self.graph.V):
            for v in self.graph.adj[u]:
                if u < v and self.coloring[u] == self.coloring[v]:
                    self.conflicts += 1

        self.chromatic_number = len(set(self.coloring))

        if self.conflicts == 0:
            self.fitness = 10000 - self.chromatic_number
        else:
            self.fitness = -self.conflicts * 100 - self.chromatic_number

    def is_valid(self):
        return self.conflicts == 0


class BeliefSpace:
    """Enhanced Belief Space with three types of knowledge"""

    def __init__(self, graph, size=5):
        self.graph = graph
        self.size = size
        self.best_solution = None
        self.normative = [None] * graph.V
        self.color_frequency = [{} for _ in range(graph.V)]
        self.min_colors_found = float('inf')
        self.successful_patterns = []

    def update(self, population):
        """Update all three knowledge types from population"""
        valid_individuals = [ind for ind in population if ind.is_valid()]

        if not valid_individuals:
            return

        current_best = min(valid_individuals, key=lambda x: x.chromatic_number)
        if self.best_solution is None or current_best.chromatic_number < self.best_solution.chromatic_number:
            self.best_solution = deepcopy(current_best)
            self.min_colors_found = current_best.chromatic_number

        self._update_normative(valid_individuals)
        self._update_domain(valid_individuals)

    def _update_normative(self, valid_individuals):
        """Track which colors work best for each vertex"""
        for ind in valid_individuals[:  self.size]:
            for vertex, color in enumerate(ind.coloring):
                if color not in self.color_frequency[vertex]:
                    self.color_frequency[vertex][color] = 0
                self.color_frequency[vertex][color] += 1

        for vertex in range(self.graph.V):
            if self.color_frequency[vertex]:
                self.normative[vertex] = max(
                    self.color_frequency[vertex].items(),
                    key=lambda x: x[1]
                )[0]

    def _update_domain(self, valid_individuals):
        """Store patterns from successful solutions"""
        sorted_valid = sorted(valid_individuals, key=lambda x: x.chromatic_number)
        self.successful_patterns = [deepcopy(ind) for ind in sorted_valid[: self.size]]

    def influence(self, individual, rate=0.3):
        """Apply belief space knowledge to guide an individual"""
        if random.random() > rate:
            return

        influence_type = random.choice(['situational', 'normative', 'domain'])

        if influence_type == 'situational' and self.best_solution:
            num_genes = random.randint(1, max(1, len(individual.coloring) // 3))
            positions = random.sample(range(len(individual.coloring)), num_genes)
            for pos in positions:
                individual.coloring[pos] = self.best_solution.coloring[pos]

        elif influence_type == 'normative' and any(self.normative):
            for vertex in range(len(individual.coloring)):
                if self.normative[vertex] is not None and random.random() < 0.3:
                    neighbor_colors = {individual.coloring[n] for n in self.graph.adj[vertex]}
                    if self.normative[vertex] not in neighbor_colors:
                        individual.coloring[vertex] = self.normative[vertex]

        elif influence_type == 'domain' and self.successful_patterns:
            pattern = random.choice(self.successful_patterns)
            num_genes = random.randint(1, max(1, len(individual.coloring) // 4))
            positions = random.sample(range(len(individual.coloring)), num_genes)
            for pos in positions:
                individual.coloring[pos] = pattern.coloring[pos]

        individual.evaluate()


class CulturalAlgorithm:
    """Cultural Algorithm for graph coloring with enhanced belief space"""

    def __init__(self, graph, population_size=50, mutation_rate=0.1,
                 influence_rate=0.3, belief_space_size=5, max_colors=None):
        self.graph = graph
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.influence_rate = influence_rate
        self.max_colors = max_colors or graph.V
        self.population = [Individual(graph, max_colors=self.max_colors)
                           for _ in range(population_size)]
        self.belief_space = BeliefSpace(graph, size=belief_space_size)
        self.best_solution = None

    def select_parent(self):
        """Tournament selection"""
        tournament = random.sample(self.population, min(5, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        point = random.randint(1, self.graph.V - 1)
        child_coloring = parent1.coloring[:  point] + parent2.coloring[point:]
        return Individual(self.graph, child_coloring, max_colors=self.max_colors)

    def mutate(self, individual):
        """Mutate by changing random colors intelligently"""
        for i in range(len(individual.coloring)):
            if random.random() < self.mutation_rate:
                neighbor_colors = {individual.coloring[neighbor]
                                   for neighbor in self.graph.adj[i]}
                available = [c for c in range(1, self.max_colors + 1)
                             if c not in neighbor_colors]
                if available:
                    individual.coloring[i] = random.choice(available)
                else:
                    individual.coloring[i] = random.randint(1, self.max_colors)

        individual.evaluate()

    def evolve(self):
        """Create next generation with belief space influence"""
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.belief_space.update(self.population)
        new_population = self.population[:2]

        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            self.belief_space.influence(child, rate=self.influence_rate)
            new_population.append(child)

        self.population = new_population

    def solve(self, max_generations=100, verbose=True):
        """Run the cultural algorithm"""
        if verbose:
            print(f"\nStarting Cultural Algorithm...")
            print(f"Population size: {self.population_size}")

        start_time = time.time()

        for gen in range(max_generations):
            self.evolve()

            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_solution is None or current_best.fitness > self.best_solution.fitness:
                self.best_solution = deepcopy(current_best)

            if verbose and (gen % 20 == 0 or gen == max_generations - 1):
                valid_count = sum(1 for ind in self.population if ind.is_valid())
                print(
                    f"Gen {gen}: Best={self.best_solution.chromatic_number if self.best_solution.is_valid() else 'invalid'}, "
                    f"Valid={valid_count}/{self.population_size}, Conflicts={self.best_solution.conflicts}")

        elapsed = time.time() - start_time

        if verbose:
            if self.best_solution.is_valid():
                print(f"\n✓ Valid Solution Found!")
                print(f"Chromatic Number: {self.best_solution.chromatic_number}")
                print(f"Color assignments:   {self.best_solution.coloring}")
            else:
                print(f"\n✗ No valid solution found")
                print(f"Best solution has {self.best_solution.conflicts} conflicts")
            print(f"Computational time:  {elapsed:.2f} seconds")

        return self.best_solution.coloring, self.best_solution.chromatic_number if self.best_solution.is_valid() else None, elapsed


# ============================================
# FILE LOADING
# ============================================
def load_graph(filename):
    """Load graph from DIMACS format file"""
    graph = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('c'):
                continue
            elif line.startswith('p'):
                parts = line.split()
                vertices = int(parts[2])
                graph = Graph(vertices)
            elif line.startswith('e'):
                parts = line.split()
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                graph.add_edge(u, v)
    return graph


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("=" * 70)
    print("GRAPH COLORING PROBLEM SOLVER".center(70))
    print("Backtracking, BFS, Hill Climbing & Cultural Algorithm".center(70))
    print("=" * 70)

    # INPUT MODE
    print("\nInput Mode:")
    print("1. Manual Input")
    print("2. Load Graph from File (. txt / DIMACS)")

    input_choice = input("\nChoice (1/2): ").strip()

    # OPTION 1: MANUAL INPUT
    if input_choice == '1':
        n = int(input("\nEnter number of vertices:  "))
        my_graph = Graph(n)

        edges_count = int(input("Enter number of edges: "))
        print("Enter edges as:   u v  (1-based vertices)")

        for i in range(edges_count):
            u, v = map(int, input(f"Edge {i + 1}:   ").split())
            my_graph.add_edge(u - 1, v - 1)

    # OPTION 2: LOAD FROM FILE
    elif input_choice == '2':
        filename = input("\nEnter file name: ").strip()
        my_graph = load_graph(filename)
        print(f"✓ Loaded graph with {my_graph.V} vertices")

    else:
        print("Invalid choice.   Exiting...")
        exit()

    # ALGORITHM SELECTION
    print("\n" + "=" * 70)
    print("ALGORITHM SELECTION".center(70))
    print("=" * 70)

    print("1. Backtracking Algorithm")
    print("2. BFS Coloring Algorithm")
    print("3. Hill Climbing Algorithm")
    print("4. Cultural Algorithm")
    print("5. All Algorithms")

    choice = input("\nEnter choice (1/2/3/4/5): ").strip()

    # ===============================================
    # 1) BACKTRACKING ONLY
    # ===============================================
    if choice == '1':
        solver = BackTracking(my_graph)
        solution, chromatic_num, elapsed_time = solver.solve()

        print("\n--- Performance Metrics ---")
        print(f"Recursive Calls: {solver.recursive_calls}")
        print(f"Pruned Branches: {solver.pruned_branches}")

        visualizer = GraphVisualizer(my_graph)
        visualizer.draw_solution(solution, chromatic_num, elapsed_time, "Backtracking")

    # ===============================================
    # 2) BFS COLORING ONLY
    # ===============================================
    elif choice == '2':
        start_vertex = int(input(f"Enter start vertex (0 to {my_graph.V - 1}, default 0): ") or "0")

        bfs_solver = BFSColoring(my_graph, start_vertex=start_vertex)
        solution, chromatic_num, elapsed_time = bfs_solver.solve()

        print("\n--- Performance Metrics ---")
        print(f"Nodes Visited: {bfs_solver.nodes_visited}")

        visualizer = GraphVisualizer(my_graph)
        visualizer.draw_solution(solution, chromatic_num, elapsed_time, "BFS Coloring")

    # ===============================================
    # 3) HILL CLIMBING ONLY
    # ===============================================
    elif choice == '3':
        print("\n--- Hill Climbing Parameters ---")
        max_iter = int(input("Max iterations (default 1000): ") or "1000")
        max_colors = int(input(f"Max colors (default {my_graph.V}): ") or str(my_graph.V))

        hc = HillClimbing(my_graph, max_colors=max_colors)
        solution, chromatic_num, elapsed_time = hc.solve(max_iterations=max_iter)

        visualizer = GraphVisualizer(my_graph)
        if chromatic_num:
            visualizer.draw_solution(solution, chromatic_num, elapsed_time, "Hill Climbing")
        else:
            print("Could not find valid solution.   Displaying best attempt:")
            colors_used = len(set(solution))
            visualizer.draw_solution(solution, colors_used, elapsed_time, "Hill Climbing (Invalid)")

    # ===============================================
    # 4) CULTURAL ALGORITHM ONLY
    # ===============================================
    elif choice == '4':
        print("\n--- Cultural Algorithm Parameters ---")
        pop_size = int(input("Population size (default 50): ") or "50")
        max_gen = int(input("Max generations (default 100): ") or "100")
        mut_rate = float(input("Mutation rate (default 0.1): ") or "0.1")
        inf_rate = float(input("Belief influence rate (default 0.3): ") or "0.3")
        belief_size = int(input("Belief space size (default 5): ") or "5")

        ca = CulturalAlgorithm(
            my_graph,
            population_size=pop_size,
            mutation_rate=mut_rate,
            influence_rate=inf_rate,
            belief_space_size=belief_size
        )

        solution, chromatic_num, elapsed_time = ca.solve(max_generations=max_gen)

        visualizer = GraphVisualizer(my_graph)
        if chromatic_num:
            visualizer.draw_solution(solution, chromatic_num, elapsed_time=elapsed_time,
                                     algorithm_name="Cultural Algorithm")

    # ===============================================
    # 5) ALL ALGORITHMS
    # ===============================================
    elif choice == '5':
        results = {}

        # BFS Coloring (Run first as it's fastest and always finds a solution)
        print("\n" + "=" * 70)
        print("=== BFS COLORING ===")
        print("=" * 70)
        bfs_solver = BFSColoring(my_graph, start_vertex=0)
        bfs_solution, bfs_chromatic, bfs_time = bfs_solver.solve()
        results['BFS Coloring'] = (bfs_solution, bfs_chromatic, bfs_time)

        print("\n--- BFS Performance Metrics ---")
        print(f"Nodes Visited: {bfs_solver.nodes_visited}")

        # Backtracking (Only run if graph is small)
        if my_graph.V <= 50:  # Adjust threshold as needed
            print("\n" + "=" * 70)
            print("=== BACKTRACKING ===")
            print("=" * 70)
            bt_solver = BackTracking(my_graph)
            bt_solution, bt_chromatic, bt_time = bt_solver.solve()
            results['Backtracking'] = (bt_solution, bt_chromatic, bt_time)

            print("\n--- Backtracking Performance Metrics ---")
            print(f"Recursive Calls: {bt_solver.recursive_calls}")
            print(f"Pruned Branches:  {bt_solver.pruned_branches}")
        else:
            print("\n" + "=" * 70)
            print("=== BACKTRACKING ===")
            print("=" * 70)
            print(f"Skipping Backtracking (graph too large:  {my_graph.V} vertices)")

        # Hill Climbing
        print("\n" + "=" * 70)
        print("=== HILL CLIMBING ===")
        print("=" * 70)
        max_iter = int(input("Max iterations (default 1000): ") or "1000")
        max_colors = int(input(f"Max colors (default {my_graph.V}): ") or str(my_graph.V))

        hc = HillClimbing(my_graph, max_colors=max_colors)
        hc_solution, hc_chromatic, hc_time = hc.solve(max_iterations=max_iter)
        results['Hill Climbing'] = (hc_solution, hc_chromatic, hc_time)

        # Cultural Algorithm
        print("\n" + "=" * 70)
        print("=== CULTURAL ALGORITHM ===")
        print("=" * 70)
        pop_size = int(input("Population size (default 50): ") or "50")
        max_gen = int(input("Max generations (default 100): ") or "100")
        mut_rate = float(input("Mutation rate (default 0.1): ") or "0.1")
        inf_rate = float(input("Belief influence rate (default 0.3): ") or "0.3")
        belief_size = int(input("Belief space size (default 5): ") or "5")

        ca = CulturalAlgorithm(
            my_graph,
            population_size=pop_size,
            mutation_rate=mut_rate,
            influence_rate=inf_rate,
            belief_space_size=belief_size
        )

        ca_solution, ca_chromatic, ca_elapsed = ca.solve(max_generations=max_gen)
        results['Cultural Algorithm'] = (ca_solution, ca_chromatic, ca_elapsed)

        # Display all results
        print("\n" + "=" * 70)
        print("COMPARISON OF ALL ALGORITHMS".center(70))
        print("=" * 70)

        visualizer = GraphVisualizer(my_graph)

        for algo_name, (solution, chromatic, elapsed) in results.items():
            print(f"\n{algo_name}:")
            if chromatic:
                print(f"  Chromatic Number: {chromatic}")
                print(f"  Time: {elapsed:.6f} seconds")
                visualizer.draw_solution(solution, chromatic, elapsed, algo_name)
            else:
                print(f"  No valid solution found")

    else:
        print("Invalid choice.  Exiting...")
