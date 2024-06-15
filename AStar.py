from queue import PriorityQueue

class Graph:

     def __init__(self):

         self.graph = {
             "A": [(146, ("A", "O")), (140, ("A", "S")), (494, ("A", "C"))],
             "O": [(146, ("O", "A")), (151, ("O", "S"))],
             "S": [(151, ("S", "O")), (140, ("S", "A")), (80, ("S", "R")), (99, ("S", "F"))],
             "C": [(494, ("C", "A")), (146, ("C", "R"))],
             "R": [(80, ("R", "S")), (146, ("R", "C")), (97, ("R", "P"))],
             "F": [(99, ("F", "S")), (211, ("F", "B"))],
             "B": [(211, ("B", "F")), (101, ("B", "P"))],
             "P": [(101, ("P", "B")), (97, ("P", "R")), (138, ("P", "C"))]
         }

         self.edges = {}
         self.weights = {}
         self.heuristics = {
             "A": 10,
             "O": 9,
             "S": 7,
             "C": 8,
             "R": 6,
             "F": 5,
             "P": 3,
             "B": 0
         }

         self.populate_edges()
         self.populate_weights()

     def populate_edges(self):
         for key in self.graph:
             neighbours = []
             for each_tuple in self.graph[key]:
                 neighbours.append(each_tuple[1][1])
             self.edges[key] = neighbours

     def populate_weights(self):
         for key in self.graph:
             neighbours = self.graph[key]
             for each_tuple in neighbours:
                 self.weights[each_tuple[1]] = each_tuple[0]

     def neighbors(self, node):
         return self.edges[node]

     def get_cost(self, from_node, to_node):
         return self.weights[(from_node, to_node)]

def astar(graph, start, goal):
     visited = set()
     queue = PriorityQueue()
     queue.put((0 + graph.heuristics[start], 0, start, []))

     while not queue.empty():
         _, cost, node, path = queue.get()

         if node == goal:
             path.append(node)
             print("Path from", start, "to", goal, ":", path)
             return

         if node in visited:
             continue

         visited.add(node)

         for neighbor in graph.neighbors(node):
             if neighbor not in visited:
                 new_cost = cost + graph.get_cost(node, neighbor)
                 new_path = path + [node]
                 queue.put((new_cost + graph.heuristics[neighbor], new_cost, neighbor, new_path))

     print("Goal node", goal, "isn't reachable from", start)
     return None

print("Traversal: ", astar(Graph(), "A", "B"))

