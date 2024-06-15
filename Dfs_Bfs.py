'''graph = {
            0: [1, 3, 4],
            1: [2, 4],
            2: [5],
            3: [4, 6],
            4: [5, 7],
            5: [],
            6: [4, 7],
            7: [5, 8],
            8: [],
}
'''
graph = {
    'Arad': {'Zerind': 75, 'Timisoara': 118, 'Sibiu': 140},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Pitesti': 97, 'Craiova': 146},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Bucharest': 101, 'Craiova': 138},
    'Craiova': {'Rimnicu Vilcea': 146, 'Pitesti': 138, 'Drobeta': 120},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}
def bfs_connected_component(graph, start, searchnode):
    explored = []
    queue = [start]
    neigbours = []
    while queue:
        node = queue.pop(0)
        if node not in explored:
            explored.append(node)
            if node == searchnode:
                return explored
            neigbours = graph[node]
            for neigbours in neigbours:
                queue.append(neigbours)
    return explored
print("\n\n")
print (bfs_connected_component(graph,'Arad', 'Fagaras'))


def dfs_connected_component(graph, start, searchnode):
    explored = []
    stack= [start]
    neigbours = []
    while stack:
        node = stack.pop()
        if node not in explored:
            explored.append(node)
            if node == searchnode:
                return explored
            neigbours = graph[node]
            for neigbours in reversed (neigbours):
                stack.append(neigbours)
    return explored
print("\n\n")
print (dfs_connected_component(graph,'Arad', 'Lugoj'))

