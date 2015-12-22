# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game
import pacman
import logic

pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"

    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    
    ex1_t1 = A | B
    ex1_t2 = (~A) % (~B | C)
    ex1_t3 = logic.disjoin((~A),(~B),C)

    return logic.conjoin(ex1_t1, ex1_t2, ex1_t3)

    util.raiseNotDefined()

def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')

    ex2_t1 = C % (B | D)
    ex2_t2 = A >> (~B & ~D)
    ex2_t3 = (~(B & ~C)) >> A
    ex2_t4 = (~D) >> C 

    return logic.conjoin(ex2_t1, ex2_t2, ex2_t3, ex2_t4)

    util.raiseNotDefined()

def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    '''
    Examples:
        >>> red = PropSymbolExpr("R")
        >>> print red
        R
        >>> turnLeft7 = PropSymbolExpr("Left",7)
        >>> print turnLeft7
        Left[7]
        >>> pos_2_3 = PropSymbolExpr("P",2,3)
        >>> print pos_2_3
        P[2,3]
        """
        P[3,4,2] which represent that Pacman is at position (3,4) at time 2
    '''

    alive_0 = logic.PropSymbolExpr("WumpusAlive",0)
    alive_1 = logic.PropSymbolExpr("WumpusAlive",1)
    born_0 = logic.PropSymbolExpr("WumpusBorn",0)
    killed_0 = logic.PropSymbolExpr("WumpusKilled",0)

    ex3_t1 =  (alive_1) % ((alive_0 & ~killed_0) | (~alive_0 & born_0))
    ex3_t2 = ~(alive_0 & born_0)
    ex3_t3 = born_0

    return logic.conjoin(ex3_t1, ex3_t2, ex3_t3)

    util.raiseNotDefined()


A = logic.PropSymbolExpr("A",2,3,4)
B = logic.PropSymbolExpr("B",4,3,2)




def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"
    
    s = logic.to_cnf(sentence)
    model = logic.pycoSAT(s)
    if model is not None:
        return model
    else:
        return False
    util.raiseNotDefined()


def atLeastOne(literals) :
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    if len(literals) == 1:
        return literals[0]
    else:
      logic_atLeastOne = literals[0]
      for l in literals[1:]:
          logic_atLeastOne = logic_atLeastOne | l
      return logic_atLeastOne

    util.raiseNotDefined()


def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"

    
    if len(literals) == 1:
        return literals[0];
    else:
        logic_atMostOne = ~literals[0] | ~literals[1]
        for i in range(len(literals) - 1):
          for j in range(i+1, len(literals)):
              if not (i == 0 and j == 1):
                  logic_atMostOne = logic_atMostOne & (~literals[i] | ~literals[j])
        return logic_atMostOne
    

    util.raiseNotDefined()


def exactlyOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    if len(literals) == 1:
        return literals[0]
    else:
        return atLeastOne(literals) & atMostOne(literals)

    util.raiseNotDefined()

def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    
    P[3,4,1]
    Pacman is at position (3,4) at time 1,
    [1 1 1 1 1]
    [1 1 1 1 1]
    [1 1 1 0 1]
    [1 1 1 1 0]

    t = 3, can go north, can't go west
    t = 1, can go west, can't go east
    t = 2, can go south


    """
    "*** YOUR CODE HERE ***"
    plan = []
    time = {}
    for each in model:
        if model[each] == True:
            action = logic.PropSymbolExpr.parseExpr(each)[0]                                                           
            t = logic.PropSymbolExpr.parseExpr(each)[1]
            if action in actions:
                #print actions
                if time.has_key(t):
                    time[t] = time.get(t) + [action] 
                else:
                    time[t] = [action]  
    for i in range(len(time)):
        plan = plan + time[str(i)] 
    return plan

    util.raiseNotDefined()


def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """

    "*** YOUR CODE HERE ***"
    
    now = logic.PropSymbolExpr("P",x,y,t)
    goSouth = logic.PropSymbolExpr("South",t-1)
    goNorth = logic.PropSymbolExpr("North",t-1)
    goWest = logic.PropSymbolExpr("West",t-1)
    goEast = logic.PropSymbolExpr("East",t-1)

    if walls_grid[x][y-1] == True and walls_grid[x][y+1] == True and walls_grid[x-1][y] == True and walls_grid[x+1][y] == True:
        result_temp = now

    if walls_grid[x][y-1] == True:
        south = None
    else:
        S = logic.PropSymbolExpr("P",x,y-1,t-1)
        south = S & goNorth
        result_temp = south

    if walls_grid[x][y+1] == True:
        north = None
    else:
        N = logic.PropSymbolExpr("P",x,y+1,t-1)
        north = N & goSouth
        if 'result_temp' in locals().keys():
            result_temp = result_temp | north
        else:
            result_temp = north

    if walls_grid[x-1][y] == True:
        west = None
    else:
        W = logic.PropSymbolExpr("P",x-1,y,t-1)
        #go_east = W & now
        west = W & goEast
        if 'result_temp' in locals().keys():
            result_temp = result_temp | west
        else:
            result_temp = west

    if walls_grid[x+1][y] == True:
        east = None
    else:
        E = logic.PropSymbolExpr("P",x+1,y,t-1)
        #go_west = E & now
        east = E & goWest
        if 'result_temp' in locals().keys():
            result_temp = result_temp | east
        else:
            result_temp = east


    return now % (result_temp)# Replace this with your expression


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    >>> Note that STOP is not an available action. <<<
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    
    "*** YOUR CODE HERE ***"
    
    start = problem.getStartState()
    goal = problem.getGoalState()
    startExpr = logic.PropSymbolExpr("P",start[0], start[1], 0) 

    
    ssa = startExpr    
    for x in range(0,width+1):
        for y in range(0,height+1):
            if x==start[0] and y ==start[1]:
                continue
            else:
                ssa = ssa & ~logic.PropSymbolExpr("P",x, y, 0) 
    
   
    
    for t in range(1, 52):         
        goalExpr = logic.PropSymbolExpr("P",goal[0],goal[1],t)
        for x in range(0,width+1):
            for y in range(0,height+1):
                ssa = ssa & logic.to_cnf(pacmanSuccessorStateAxioms(x,y,t,walls))
        
        north = logic.PropSymbolExpr(game.Directions.NORTH, t-1)
        south = logic.PropSymbolExpr(game.Directions.SOUTH,t-1)
        west = logic.PropSymbolExpr(game.Directions.WEST,t-1)
        east = logic.PropSymbolExpr(game.Directions.EAST,t-1)
        oneAction = logic.to_cnf(exactlyOne([north, south, west, east]))
        ssa = ssa & oneAction

        #result = findModel( ssa & goalExpr )
        result = logic.pycoSAT(ssa & goalExpr)
        if result != False:
            return extractActionSequence(result,['North', 'South', 'East', 'West'])                     
        else:
            continue



    util.raiseNotDefined()


 

        #util.raiseNotDefined()

def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    start,foodGrid = problem.getStartState()

    foodList= foodGrid.asList()
    
    start_state = logic.PropSymbolExpr("P",start[0], start[1], 0)

    check = start_state

    for x in range(1,width+1):
        for y in range(1, height+1): 
            if x == start[0] and y == start[1]:
                continue
            else:
                check = check & ~logic.PropSymbolExpr("P", x, y, 0)


    for t in range(1, 52): 
        #print(t)
        for x in range(1,width+1):
            for y in range(1, height+1):
                # print(t)
                check = check & logic.to_cnf(pacmanSuccessorStateAxioms(x, y, t, walls))
        

        foodLeastList = []
        for x,y in foodList:
            food = logic.PropSymbolExpr("P",x,y,1)
            for time in range(2,t+1):
                food  = food | logic.PropSymbolExpr("P",x,y,time)
            foodLeastList.append(food)
        
        f_atLeast = foodLeastList[0]
        for i in range(1,len(foodLeastList)):
            f_atLeast = f_atLeast & foodLeastList[i]
        f_atLeast = logic.to_cnf(f_atLeast)

        South = logic.PropSymbolExpr(game.Directions.SOUTH,t-1) 
        North = logic.PropSymbolExpr(game.Directions.NORTH,t-1)
        West = logic.PropSymbolExpr(game.Directions.WEST,t-1)
        East = logic.PropSymbolExpr(game.Directions.EAST,t-1)
        action = logic.to_cnf(exactlyOne([South, North, East, West]))
        check = check & action 
        # print(check)
        #result = findModel(check & f_atLeast) #& goal_state
        result = logic.pycoSAT(check & f_atLeast)
        if result != False:
            return extractActionSequence(result, ["South", "North", "West", "East"])
        else: 
            continue

        util.raiseNotDefined()
    #You can call foodGrid.asList() to get a list of food coordinates instead.


'''
pos_str = ghost_pos_str+str(1)
east_str = ghost_east_str+str(1)
now = logic.PropSymbolExpr(pos_str,1,2,3)
goEast = logic.PropSymbolExpr(ghost_east_str,2)
goWest = ~goEast
print now
print goEast
print goWest
'''


def ghostPositionSuccessorStateAxioms(x, y, t, ghost_num, walls_grid):
    """
    Successor state axiom for patrolling ghost state (x,y,t) (from t-1).
    Current <==> (causes to stay) | (causes of current)
    
    GE is going east, ~GE is going west 
    """
    pos_str = ghost_pos_str+str(ghost_num)
    east_str = ghost_east_str+str(ghost_num)

    "*** YOUR CODE HERE ***"
    now = logic.PropSymbolExpr(pos_str,x,y,t)
    goEast = logic.PropSymbolExpr(east_str,t-1)

    last0 = logic.PropSymbolExpr(pos_str,x,y,t-1)
    last1 = logic.PropSymbolExpr(pos_str,x-1,y,t-1)
    last2 = logic.PropSymbolExpr(pos_str,x+1,y,t-1)

    if walls_grid[x+1][y] == True and walls_grid[x-1][y] == True:
        return now % last0 

    if walls_grid[x+1][y] == True:
        return now % (last1 & goEast)

    if walls_grid[x-1][y] == True:
        return now % (last2 & ~goEast)


    
    else:
        return now % (last1 & goEast | last2 & ~goEast)        

    # Replace this with your expression

def ghostDirectionSuccessorStateAxioms(t, ghost_num, blocked_west_positions, blocked_east_positions):
    """
    Successor state axiom for patrolling ghost direction state (t) (from t-1).
    west or east walls.
    Current <==> (causes to stay) | (causes of current)
    """
    pos_str = ghost_pos_str+str(ghost_num)
    east_str = ghost_east_str+str(ghost_num)

    "*** YOUR CODE HERE ***"
    goEast = logic.PropSymbolExpr(east_str, t)
    lastGoEast = logic.PropSymbolExpr(east_str,t-1)
    
    '''
    if t == 1:
        for x,y in blocked_east_positions:
            if '(x,y)' in blocked_east_positions:
                ghostLocation = logic.PropSymbolExpr(pos_str,x,y,t)
                return ~goEast 
            else:
                return goEast

    
    else:
    '''      
    result_temp_1 = lastGoEast
    for x1,y1 in blocked_east_positions:        
        ghostLocation = logic.PropSymbolExpr(pos_str,x1,y1,t)
        result_temp_1 = result_temp_1 & ~ghostLocation
    
    
    x,y = blocked_west_positions[1]
    result_temp_2 = logic.PropSymbolExpr(pos_str,x,y,t)
    for x2,y2 in blocked_west_positions:
        ghostLocation = logic.PropSymbolExpr(pos_str,x2,y2,t)
        result_temp_2 = result_temp_2 | ghostLocation

    result_temp_3 = ~lastGoEast & result_temp_2

    
    return goEast % (result_temp_1 | result_temp_3)


    util.raiseNotDefined()
    
    

    
    #return logic.Expr('A')
    #return headEast % (lastHeadEast & blocked_east_positions[] == False | lastHeadWest & blocked_west_positions[] == True)
    #can't get the location??... at least the logic is like this somehow...
    #The arguments blocked_west_positions and blocked_east_positions are 
    #lists of all the positions that are not walls but have walls immediately to their west and east, respectively


    #return  # Replace this with your expression


def pacmanAliveSuccessorStateAxioms(x, y, t, num_ghosts):
    """
    Successor state axiom for patrolling ghost state (x,y,t) (from t-1).
    Current <==> (causes to stay) | (causes of current)
    """
    ghost_strs = [ghost_pos_str+str(ghost_num) for ghost_num in xrange(num_ghosts)]

    "*** YOUR CODE HERE ***"

    pacmanAlive = logic.PropSymbolExpr(pacman_alive_str,t)
    pacmanWasAlive = logic.PropSymbolExpr(pacman_alive_str,t-1)
    
    pacmanPos = logic.PropSymbolExpr("P",x,y,t)
    ghostPos = logic.PropSymbolExpr("G0",x,y,t)
    ghostLastPos = logic.PropSymbolExpr("G0",x,y,t-1)
    
    sentence = (ghostPos | ghostLastPos)
    for i in ghost_strs:
        ghostPos = logic.PropSymbolExpr(i,x,y,t)
        ghostLastPos = logic.PropSymbolExpr(i,x,y,t-1)
        sentence = sentence | ghostPos | ghostLastPos

    #print ~pacmanAlive % (~pacmanWasAlive | sentence & pacmanPos)
    # return  pacmanAlive % (pacmanWasAlive & (sentence | ~pacmanPos) )
    return  ~pacmanAlive % (~pacmanWasAlive | (sentence & pacmanPos) )


    #return logic.Expr('A')
    util.raiseNotDefined()


def foodGhostLogicPlan(problem):
    """
    Given an instance of a FoodGhostPlanningProblem, return a list of actions that help Pacman
    eat all of the food and avoid patrolling ghosts.
    Ghosts only move east and west. They always start by moving East, unless they start next to
    and eastern wall. 
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """


    '''
    By now the sentence contains ghost initial state(position, 
    going east(if not in east_blocked), ghost only one start), 
    pacman's initial state(pacman alive at time 0, pacman only 
    one start), pacman pos ssa, ghost pos ssa, ghost dir ssa, 
    and the goal test is pacman alive and eating all the food...


    '''




    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    start,foodGrid = problem.getStartState()

    foodList= foodGrid.asList()
    
    start_state = logic.PropSymbolExpr("P",start[0], start[1], 0)

    pacmanAliveStart = logic.PropSymbolExpr("PA",0)
    
    #ghost_start = problem.getGhostStartStates()

    ghostList = []

    ghost_state = "None"
    ghost_east = "None"
    for i in problem.getGhostStartStates():
        ghostList+=[i.getPosition()]
    
    blocked_west_positions = []
    blocked_east_positions = []

    for x in range(1,width+1):
        for y in range(1,height+1):
            if walls[x-1][y] == True:
                blocked_west_positions.append((x,y))
            if walls[x+1][y] == True:
                blocked_east_positions.append((x,y))
    ghostNumber = len(ghostList)
    

    num = 0
    for x,y in ghostList:    #ghost initial position
        
        if ghost_state == "None":
            ghost_state = logic.PropSymbolExpr(ghost_pos_str+str(num),x,y,0)
        else:
            ghost_state = ghost_state & logic.PropSymbolExpr(ghost_pos_str+str(num),x,y,0)

        if ghost_east == "None":      #ghost initial direction
            if (x,y) in blocked_east_positions:
                ghost_east = ~logic.PropSymbolExpr(ghost_east_str+str(num),0)
            else:
                ghost_east = logic.PropSymbolExpr(ghost_east_str+str(num),0)
        else:
            if (x,y) in blocked_east_positions:
                ghost_east = ghost_east & ~logic.PropSymbolExpr(ghost_east_str+str(num),0)
            else:
                ghost_east = ghost_east & logic.PropSymbolExpr(ghost_east_str+str(num),0)
        num = num + 1

    #print ghostList

    check = start_state & pacmanAliveStart & ghost_state & ghost_east 
   
    
    
    #only one start for pacman
    for x in range(1,width+1):
        for y in range(1, height+1): 
            if x == start[0] and y == start[1]:
                continue
            else:
                check = check & ~logic.PropSymbolExpr("P", x, y, 0)



    # only one start for every ghost 
    num_ghosts = -1
    for x,y in ghostList: 
        num_ghosts = num_ghosts + 1
        for i in range(1, width + 1):
            for j in range(1, height+1):
                if i == x and j == y :
                    continue
                else:
                    check = check & ~logic.PropSymbolExpr(ghost_pos_str+str(num_ghosts),i,j,0)



    #time loop begins
    for t in range(1, 52): 
        gpssa = "None"
        pa = "None"
        for x in range(1,width+1):
            for y in range(1, height+1):
   
                if pa == "None":
                    pa = logic.to_cnf(pacmanAliveSuccessorStateAxioms(x,y,t,ghostNumber))
                else:
                    pa = pa & logic.to_cnf(pacmanAliveSuccessorStateAxioms(x,y,t,ghostNumber))
                  
                
                for i in range(0,ghostNumber):      #ghost position ssa for every ghost
                    if gpssa == "None":
                        gpssa = logic.to_cnf(ghostPositionSuccessorStateAxioms(x,y,t,i,walls))   #(G0[3,2,1] <=> (G0[2,2,0] & GE0[0]))
                    else:
                        gpssa = gpssa & logic.to_cnf(ghostPositionSuccessorStateAxioms(x,y,t,i,walls))
                   
                check = check & logic.to_cnf(pacmanSuccessorStateAxioms(x, y, t, walls)) & gpssa & pa
      
        gdssa = "None"
        for i in range(0,ghostNumber):
            if gdssa == "None":     #ghost direction ssa for every ghost
                gdssa = logic.to_cnf(ghostDirectionSuccessorStateAxioms(t,i,blocked_west_positions,blocked_east_positions))
            else:
                gdssa = gdssa & logic.to_cnf(ghostDirectionSuccessorStateAxioms(t,i,blocked_west_positions,blocked_east_positions))
                
        check = check & gdssa

        foodLeastList = []

        for x,y in foodList:
            food = logic.PropSymbolExpr("P",x,y,1)
            for time in range(2,t+1):
                food  = food | logic.PropSymbolExpr("P",x,y,time)
            foodLeastList.append(food)
        f_atLeast = foodLeastList[0]
        for i in range(1,len(foodLeastList)):
            f_atLeast = f_atLeast & foodLeastList[i]      #eat all foods 
        f_atLeast = logic.to_cnf(f_atLeast)
        

        South = logic.PropSymbolExpr(game.Directions.SOUTH,t-1) 
        North = logic.PropSymbolExpr(game.Directions.NORTH,t-1)
        West = logic.PropSymbolExpr(game.Directions.WEST,t-1)
        East = logic.PropSymbolExpr(game.Directions.EAST,t-1)
        action = logic.to_cnf(exactlyOne([South, North, East, West]))
        check = check & action 

        result = logic.pycoSAT(check & f_atLeast)

        if result != False:
            return extractActionSequence(result, ["South", "North", "West", "East"])
        else: 
            continue

        util.raiseNotDefined()



# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
fglp = foodGhostLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
    