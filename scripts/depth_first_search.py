import heapq
import time

def dfs_route_planning(graph, source, target, nr_objectives, lower_bounds_algorithm="reverse_dijkstra"):

    def single_objective_value_iteration(objective):
        """Calculate lower bounds for the objective for each node in the graph."""
        node_values = {target: 0} #  lower_bound per node
        done = False
        while not done:
            done = True
            for node in graph.keys():
                if node == target:
                    continue
                new_cost = min(graph[node][neighbor][objective] + node_values.get(neighbor, float('inf')) for neighbor in graph.get(node, {}))
                if new_cost < node_values.get(node, float('inf')):
                    done = False
                    node_values[node] = new_cost
        return node_values

    def reverse_dijkstra(objective):
        """Calculate lower bounds for the objective for each node in the graph, starting from the target node."""
        node_values = {} #  lower_bound per node
        pq = [(0, target)]
        while pq:
            cost, current = heapq.heappop(pq)
            if current in node_values:
                continue
            node_values[current] = cost
            for neighbor in graph.get(current, {}):
                if neighbor not in node_values:
                    edge_costs = graph[current][neighbor]
                    new_cost = cost + edge_costs[objective]
                    heapq.heappush(pq, (new_cost, neighbor))
        return node_values

    def dijkstra_shortest_path(objective_i):
        """Dijkstra's algorithm for shortest path with current objective."""
        pq = [(0, source, [])]
        visited = set()
        while pq:
            cost, current, path = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            path = path + [current]
            if current == target:
                return path
            for neighbor in graph.get(current, {}):
                if neighbor not in visited:
                    edge_costs = graph[current][neighbor]
                    total_cost = cost + edge_costs[objective_i]  # focus on the given objective to find the lowest cost path
                    heapq.heappush(pq, (total_cost, neighbor, path))
        return []

    def calculate_objectives(path):
        """Calculate the objectives for a given path."""
        totals = [0] * nr_objectives
        for i in range(len(path) - 1):
            edge = graph[path[i]][path[i+1]]
            length = edge[2] + edge[3]  # Recalculate length as walk + bike
            adjusted_edge = (length,) + edge[1:]
            for j in range(len(adjusted_edge)):
                totals[j] += adjusted_edge[j]
        return totals

    def pareto_dominate(objectives1, objectives2):
        """Returns True if objectives1 pareto-dominates objectives2."""
        dominates = False
        for obj1, obj2 in zip(objectives1, objectives2):
            if obj1 > obj2:
                return False
            if obj1 < obj2:
                dominates = True
        return dominates

    def manhattan_distance(objectives1, objectives2):
        """Calculate the Manhattan distance of the values between objective vectors."""
        return sum(abs(obj1 - obj2) for obj1, obj2 in zip(objectives1, objectives2))

    def dfs_path(lower_bnds, target_obj, upper_bnd):
        """Depth first search algorithm for finding a pareto-optimal path between target and upper bound objectives."""
        visited = set()
        current_best_path = []  # current best path
        current_best_cost = upper_bnd
        stack = [([0] * nr_objectives, source, [source])]
        while stack:
            cost, current, path = stack.pop()
            if current in visited:
                continue
            if current == target:
                if pareto_dominate(cost, current_best_cost):  # cost + graph[current][neighbor] instead of just cost?
                    current_best_cost = cost
                    current_best_path = path
                continue
            visited.add(current)
            neighbor_list = []
            for neighbor in graph.get(current, {}):
                if neighbor not in visited:
                    new_edge_cost = graph[current][neighbor]
                    lower_bounds_cost = (lower_bnd[neighbor] for lower_bnd in lower_bnds)
                    result = tuple(c1 + c2 + c3 for c1, c2, c3 in zip(cost, tuple(new_edge_cost), lower_bounds_cost))
                    # Prune paths that will not improve the current upper bound/current_best_cost
                    if pareto_dominate(result, current_best_cost):
                        distance = manhattan_distance(target_obj, result)
                        new_cost = tuple(c1 + c2 for c1, c2 in zip(cost, tuple(new_edge_cost)))
                        neighbor_list.append((new_cost, neighbor, distance))
            neighbor_list_sorted = sorted(neighbor_list, key=lambda x: x[2], reverse=True)
            for cost, neighbor, distance in neighbor_list_sorted:
                new_path = path + [neighbor]
                stack.append((cost, neighbor, new_path))
        return current_best_cost, current_best_path

    start_time = time.time()

    match lower_bounds_algorithm:
        case "reverse_dijkstra":
            lower_bounds = [reverse_dijkstra(objective) for objective in range(nr_objectives)]
        case "single_objective_value_iteration":
            lower_bounds = [single_objective_value_iteration(objective) for objective in range(nr_objectives)]
        case _:
            print("Unknown lower_bounds_algorithm.")
            return [] #  empty path
    print("Value iteration done.")
    objectives_shortest_paths = [calculate_objectives(dijkstra_shortest_path(obj_i)) for obj_i in range(nr_objectives)]
    print("objectives_shortest_paths:", objectives_shortest_paths)
    target_objective = [min(values) for values in zip(*objectives_shortest_paths)]
    upper_bound = [max(values) for values in zip(*objectives_shortest_paths)]
    print(target_objective)
    print(upper_bound)

    objective_cost, final_route = dfs_path(lower_bounds, target_objective, upper_bound)

    duration = time.time() - start_time
    print(f"Finding a pareto-optimal route took {duration:.2f} seconds.")
    print("Objective cost: ", objective_cost)
    print("Route length: ", len(final_route))
    return final_route