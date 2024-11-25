import heapq

def subgradient_multiple_routes(graph, source, target, constraints, weights, max_routes=5, max_iter=100, tol=1e-3):
    def weighted_cost(node1, node2):
        """compute the weighted cost between two nodes."""
        edge = graph[node1][node2]
        # Objective: length = walk + bike
        length = edge[2] + edge[3]
        adjusted_edge = (length,) + edge[1:]  # Recompute edge with updated "length"
        return sum(w * e for w, e in zip(weights, adjusted_edge))
    
    def shortest_path():
        """dijkstra's algorithm for shortest path with current weights."""
        pq = [(0, source, [])]
        visited = set()
        while pq:
            cost, current, path = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            path = path + [current]
            if current == target:
                return cost, path
            for neighbor in graph.get(current, {}):
                if neighbor not in visited:
                    total_cost = cost + weighted_cost(current, neighbor)
                    heapq.heappush(pq, (total_cost, neighbor, path))
        return float('inf'), []
    
    def calculate_objectives(path):
        """calculate the objectives for a given path."""
        totals = [0] * len(constraints)
        for i in range(len(path) - 1):
            edge = graph[path[i]][path[i+1]]
            length = edge[2] + edge[3]  # Recalculate length as walk + bike
            adjusted_edge = (length,) + edge[1:]
            for j in range(len(adjusted_edge)):
                totals[j] += adjusted_edge[j]
        return totals
    
    def check_constraints(totals):
        """check if the path satisfies user-defined constraints."""
        return [totals[i] - constraints[i] for i in range(len(constraints))]
    
    alpha = 1.0  # Initial step size
    feasible_routes = []  # To store feasible routes
    for iteration in range(max_iter):
        cost, path = shortest_path()
        if not path:
            print("No feasible path found.")
            break
        
        objectives = calculate_objectives(path)
        violations = check_constraints(objectives)
        if all(v <= tol for v in violations):
            # Store feasible route if it meets constraints
            feasible_routes.append((cost, path, objectives))
            feasible_routes = sorted(feasible_routes, key=lambda x: x[0])[:max_routes]
        
        # update weights using subgradient
        for i in range(len(weights)):
            weights[i] += alpha * max(0, violations[i])
        
        alpha *= 0.9

        # Stop if we've found enough feasible routes
        if len(feasible_routes) >= max_routes:
            break
    
    if not feasible_routes:
        print("No feasible paths found after optimization.")
        return None
    return feasible_routes
