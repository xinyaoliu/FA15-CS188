# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def breadthFirstSearch(problem):      #used for Q5, nearly same with DFS.. though there is no DFS here...
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    "*** YOUR CODE HERE ***"
    explored = set()
    parents = []
    path = []
    
    queue = util.Queue()
    queue.push(problem.getStartState())

    while (not queue.isEmpty()):
        state = queue.pop()
        if(problem.goalTest(state)):
            break
        if(state not in explored):
            explored.add(state)
            successors = problem.getActions(state)
            for successor in successors:
              resultState = problem.getResult(state, successor) 
              queue.push(resultState)
              parents.append([state, resultState, successor])


    currentState = state
    while(currentState != problem.getStartState()):
      for parent in parents:
            if (parent[1] == currentState):
                currentState = parent[0]
                path = [parent[2]] + path
                break

    return path

    

    
    util.raiseNotDefined()

def depthLimitedSearch(problem, limit ):   # not pass finally, for some reasons i don't know..... can run anyway
                                           # fail in some node expand...
    explored = set()                                      
    node = problem.getStartState()
    #path = []

    def recursive_DLS(node, problem, limit ):
        explored.add(node)   
        if problem.goalTest(node):		
            return []     
        elif limit == 0 : 		
            return 'cutoff'
        else:            
            cutoff_occurred = False
            for successor in problem.getActions(node):   
                child = problem.getResult(node,successor)
                if child not in explored : 
                    result = recursive_DLS(child ,problem, limit - 1)    
                    if result == 'cutoff':
                        cutoff_occurred = True
                        explored.remove(child)
                    elif result is not None :
                        return [successor] + result
            if cutoff_occurred:
                    return 'cutoff'
            else:
                    return None
    
    return recursive_DLS(node, problem, limit) 	
                
    

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth.

    Begin with a depth of 1 and increment depth by 1 at every step.
    """
    "*** YOUR CODE HERE ***"
    for depth in xrange(sys.maxint):
        result = depthLimitedSearch(problem, depth)
        if result is not 'cutoff':
            return result    
    util.raiseNotDefined()



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
    

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
   
    explored = set()
    parents = []
    path = []
    
    pQueue = util.PriorityQueue()
    item = [problem.getStartState(), problem.getStartState(), 0, None] #currentState, resultState, cost, action
    priority = 0
    pQueue.push(item,priority)
    
    while (not pQueue.isEmpty()):  
        state = pQueue.pop()  
        parents.append([state[0], state[1], state[3]])  #currentState, resultState, action
                                                        #so for parents[]===> current, result, action
        if problem.goalTest(state[1]):  
            break
        currentState = state[1]
        if currentState not in explored:
            explored.add(currentState)
            for successor in problem.getActions(currentState): 
                resultState = problem.getResult(currentState , successor)
                cost = state[2] + problem.getCost(currentState ,successor)    														#have no idea why use getTotalCost fail.....problem.getCostOfActions(parents[2::2]) ...maybe wrong method...         
                pQueue.push([currentState , resultState, cost, successor], cost + heuristic(resultState, problem))  #push(item. priority)

    currentState = state[1]
    while (currentState != problem.getStartState()):
        for parent in parents:
            if parent[1] == currentState:
                currentState = parent[0]
                path = [parent[2]] + path
                break
  
    return path
  

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
