# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.



        abount three factors: distance food ghost  nearly the same with Q9 coz....
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()                   #Returns a Grid of boolean food indicator variables
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0
        
        foodDistance = []
        foods = newFood.asList()  
        walls = currentGameState.getWalls().asList()
        emptyNeighbors = 0
        for food in foods:  #food is (x,y)
            foodDistance.append(manhattanDistance(newPos,food))  #position, problem, info
            neighbors = [(food[0]-1,food[1]),(food[0]+1,food[1]),(food[0],food[1]-1),(food[0],food[1]+1)]
            for n in neighbors:
                if n not in walls and n not in foods:
                    emptyNeighbors = emptyNeighbors + 2  
            
        
        food_Distance = 0
        if len(foodDistance) > 0:
            food_Distance = 1.0 / (min(foodDistance))
        
        ghostDistance = []
        for ghost in newGhostStates:
            ghostDistance.append(manhattanDistance(ghost.getPosition(),newPos))
        
        for ghost in newScaredTimes:
            score = score + ghost
            
        score = score + (food_Distance**5) * min(ghostDistance)
        score = score + successorGameState.getScore() - float((emptyNeighbors) * 4.0)
        
        return score
        
       #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 7)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        def maximizer(state, depth):
            value = float('-inf')   #negative infinity~
            depth = depth + 1       #Look-ahead agents evaluate future states whereas reflex agents evaluate actions from the current state.

            if state.isWin() or state.isLose() or depth == self.depth :   #depth == self.depth  current gameState using...
                return self.evaluationFunction(state)

            for pacman in state.getLegalActions(0):
                value = max(value, minimizer(state.generateSuccessor(0,pacman), depth, 1)) 
                    
            return value
        
        def minimizer(state, depth, ghostNum):
            value = float('inf')

            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            for pacman in state.getLegalActions(ghostNum):
                if ghostNum == gameState.getNumAgents() - 1:   
                    value = min(value, maximizer(state.generateSuccessor(ghostNum, pacman), depth))
                else:
                    value = min(value, minimizer(state.generateSuccessor(ghostNum, pacman), depth, ghostNum + 1))
            
            return value
         
        def minimax(actions):
            maximum = float('-inf')
            pacmanAction = None  
            for action in actions:   #compare
                tempMax = minimizer(gameState.generateSuccessor(0, action), 0, 1)
                if tempMax > maximum :
                    maximum = tempMax
                    pacmanAction = action
        
            return pacmanAction

        actions = gameState.getLegalActions(0)  #list of action, 0 is pacman
        
        return minimax(actions)
                 
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
           
        def maximizer(state, depth):
            value = float('-inf')
            depth = depth + 1

            if state.isWin() or state.isLose() or depth == self.depth : 
                return self.evaluationFunction(state)
            for pacman in state.getLegalActions(0):
                value = max(value, expectimax(state.generateSuccessor(0,pacman), depth, 1))  
                    
            return value     

        def expectimax(state, depth, ghostNum):
            value = 0
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            for pacman in state.getLegalActions(ghostNum):
                if ghostNum == gameState.getNumAgents() - 1:
                    tempValue = maximizer(state.generateSuccessor(ghostNum, pacman),depth)
                    tempNum = len(state.getLegalActions(ghostNum))
                    value = value + float(tempValue / tempNum)
                else:
                    tempValue = expectimax (state.generateSuccessor(ghostNum, pacman), depth, (ghostNum + 1))
                    tempNum = len(state.getLegalActions(ghostNum))
                    value = value + float(tempValue / tempNum)
            return value
         
        def expecti(actions):
            pacmanAction = None
            maximum = float('-inf')  
        
            for action in actions:          
                tempMax = expectimax(gameState.generateSuccessor(0,action), 0, 1)
                if tempMax > maximum or (tempMax == maximum and random.random()> .5 ):  #50%max 50%min
                    maximum = tempMax
                    pacmanAction = action
            return pacmanAction

        actions = gameState.getLegalActions(0)  #list of action
        return expecti(actions)
   
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 9).
      
      compute values for features about the state that you think are important, 
      and then combine those features by multiplying them by different values and 
      adding the results together
      
      I think ghostDistance is not that important as long as you don't hit it while it
      is not in scared mode..
      ghostScaredTimes is more important than ghostDistance
      it is good to always find foods and not go to walls
      better to be closer to food
      
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = 0
        
    foodDistance = []
    food_Distance = 0
    foods = newFood.asList()
    walls = currentGameState.getWalls().asList()
    emptyNeighbors = 0
    notWalls = 0
    for food in foods:  #food is (x,y)
        neighbors = [(food[0]-1,food[1]),(food[0]+1,food[1]),(food[0],food[1]-1),(food[0],food[1]+1)]
        for n in neighbors:
            if n not in foods:
                emptyNeighbors = emptyNeighbors + 2
            if n not in walls:
                notWalls = notWalls + 1
            foodDistance.append(manhattanDistance(newPos,food))  #position, problem, info
    if len(foodDistance) > 0:
        food_Distance = float (1 / (min(foodDistance)))
        
    ghostDistance = []
    for ghost in newGhostStates:
        ghostDistance.append(manhattanDistance(ghost.getPosition(),newPos))
        
    for ghost in newScaredTimes:
        score = score + float(ghost*2.5)
            
    score = score + float(food_Distance ** 5) * min(ghostDistance)
    score = score + currentGameState.getScore() - float(emptyNeighbors * 4.5) + notWalls
        
    return score
        
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

