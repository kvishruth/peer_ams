import networkx as nx
import heapq
import random

class QLearningHeuristicAStar:
    
    def __init__(self, graph, weights, maximize_bike=True, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Q-learning-based heuristic for A* in a directed graph (MultiDiGraph).
        
        :param graph: networkx MultiDiGraph with edge attributes (length, crosswalk, bike, walk)
        :param weights: weights for calculating rewards based on edge attributes
        :param maximize_bike: If True, maximize the bike attribute, otherwise maximize walk
        :param alpha: learning rate (Q-learning)
        :param gamma: discount factor (Q-learning)
        :param epsilon: exploration factor (Q-learning)
        """
        self.graph = graph
        self.weights = weights
        self.maximize_bike = maximize_bike  
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  

    def get_q_value(self, state, action):
        """Retrieve Q-value for a state-action pair."""
        return self.q_table.get((state, action), 0.0)
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value for a state-action pair using Q-learning."""
        best_next_action = max(
            [self.get_q_value(next_state, next_action) for next_action in self.graph.successors(next_state)],
            default=0.0
        )
        new_q_value = self.get_q_value(state, action) + self.alpha * (
            reward + self.gamma * best_next_action - self.get_q_value(state, action)
        )
        self.q_table[(state, action)] = new_q_value

    def compute_heuristic(self, u, v):
        
        reward = self.calculate_reward(u, v)
        q_value = self.get_q_value(u, v)
        return 0.5 * q_value + 0.5 * reward  
    
    def calculate_reward(self, u, v):
      
        if self.graph.has_edge(u, v):
            edge_data = self.graph[u][v]
            length = edge_data.get('length', 1)
            crosswalk = edge_data.get('crossing', 1)
            bike = edge_data.get('bike', 0)
            walk = edge_data.get('walk', 0)
            
            # total length is the sum of bike + walk
            total_length = bike + walk
            

            reward = (
                self.weights['length'] * total_length +  # minimize total length (bike + walk)
                self.weights['crossing'] * crosswalk +  # Mmnimize crosswalk
                self.weights['bike'] * (bike if self.maximize_bike else 0) +  # maximize bike if chosen
                self.weights['walk'] * (walk if not self.maximize_bike else 0)  # maximize walk if chosen
            )
            return reward
        return 0  

    def simulate_random_walks(self, start, goal, num_steps=1000):
        """Simulate random walks to train Q-values."""
        for _ in range(num_steps):
            current = start
            while current != goal:
                neighbors = list(self.graph.successors(current))
                if not neighbors:
                    break
                next_node = random.choice(neighbors)
                reward = self.calculate_reward(current, next_node)
                self.update_q_value(current, next_node, reward, next_node)
                current = next_node

    def a_star(self, start, goal):
 
        open_list = []
        closed_list = set()
        came_from = {}
        g_score = {node: float('inf') for node in self.graph.nodes}
        f_score = {node: float('inf') for node in self.graph.nodes}
        
        g_score[start] = 0
        f_score[start] = self.compute_heuristic(start, goal)
        
        heapq.heappush(open_list, (f_score[start], start))
        
        while open_list:
            _, current = heapq.heappop(open_list)
            
            if current == goal:
                path = []
                total_length = 0
                total_crosswalk = 0
                total_walk = 0
                total_bike = 0
                
                while current in came_from:
                    prev = came_from[current]
                    path.append(current)
                    
                    # Sum up edge attributes
                    edge_data = self.graph[prev][current]
                    total_length += edge_data.get('length', 0)
                    total_crosswalk += edge_data.get('crossing', 0)
                    total_walk += edge_data.get('walk', 0)
                    total_bike += edge_data.get('bike', 0)
                    
                    current = prev
                
                path.append(start)
                path.reverse()
                return {
                    "path": path,
                }
            
            closed_list.add(current)
            
            for neighbor in self.graph.successors(current):  
                if neighbor in closed_list:
                    continue
                
                tentative_g_score = g_score[current] + self.graph[current][neighbor].get('length', 1)
                
                reward = self.calculate_reward(current, neighbor)
                self.update_q_value(current, neighbor, reward, neighbor)
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.compute_heuristic(neighbor, goal)
                    
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
        
        return {
            "path": None,
            "length": float('inf'),
            "crosswalks": 0,
            "walk": 0,
            "bike": 0
        }  
