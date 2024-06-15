class Node:
    def __init__(self,state,parent,actions,totalCost):
        self.state=state
        self.parent=parent
        self.actions=actions
        self.totalCost=totalCost

def actionSequence(graph,initialState,goalState):
     solution=[goalState]
     currentParent=graph[goalState].parent
     while currentParent!=None:
         solution.append(currentParent)
         currentParent=graph[currentParent].parent
     solution.reverse()
     return solution

def DFS():
     initialState='A'
     goalState='F'
     graph = {'A': Node('A', None, ['B', 'E', 'C'], None),
              'B': Node('B', None, ['D', 'E', 'A'], None),
              'C': Node('C', None, ['A', 'F', 'G'], None),
              'D': Node('D', None, ['B', 'E'], None),
              'E': Node('E', None, ['A', 'B', 'D'], None),
              'F': Node('F', None, ['C'], None),
              'G': Node('G', None, ['C'], None)
              }

     frontier=[initialState]
     explored=[]
     while len(frontier)!=0:
         currentNode=frontier.pop(len(frontier)-1)
         print(currentNode)
         explored.append(currentNode)
         currentChildren=0
         for child in graph[currentNode].actions:
             if child not in frontier and child not in explored:
                 graph[child].parent=currentNode
                 if graph[child].state==goalState:
                     print(explored)
                     return  actionSequence(graph, initialState, goalState)
                 currentChildren=currentChildren+1
                 frontier.append(child)
         if currentChildren==0:
             del explored[len(explored)-1]
solution=DFS()
print(solution)