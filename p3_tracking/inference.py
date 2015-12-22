# inference.py
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


import itertools
import random
import busters
import game
import util 

from util import manhattanDistance


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        total_sum = float(self.total())
        if total_sum == 0:
            return
        for i in self:
            self[i] = self[i] / total_sum
        
       


    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        
        

        total = self.total()
        # if total != 1:
        self.normalize()

        #rad = random.randint(1,total) 
        rad = random.uniform(0,1)
        cur_total = 0
        res = ""
        for u,v in self.items():
            cur_total = cur_total + v
            if rad <= cur_total:
                res = u
                break
        return res
        
dist = DiscreteDistribution()
dist['a'] = 0.24
dist['b'] = 0.1
dist['c'] = 0.16
N = 100
samples = [dist.sample() for _ in range(int(N))]
#number = 1
number = round(samples.count('b'))
print "begin"
print number
print "End"

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        one = 1 
        zero = 0
 
        if noisyDistance == None:
            if ghostPosition == jailPosition:
                return one
            else:
                return zero
        else:
            if ghostPosition == jailPosition:
                return zero
            else:
                trueDistance = manhattanDistance(ghostPosition,pacmanPosition)
                return busters.getObservationProbability(noisyDistance, trueDistance)




    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.update(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def update(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def predict(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def update(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        # P(noisyDistance | pacmanPosition, ghostPosition)
        
        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()

    
        for i in self.allPositions:
            #print i
            p = self.getObservationProb(observation, pacmanPosition, i, jailPosition)
            self.beliefs[i] = self.beliefs[i] * p
        self.beliefs.normalize()
        return self.beliefs


    def predict(self, gameState):

        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
   
        

        "*** YOUR CODE HERE ***"
        predictedPos = util.Counter()
        for oldPos in self.allPositions:
           
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            for newPos,prob in newPosDist.items():
                #newPosDist[newPos] = newPosDist[newPos] + newPosDist[oldPos] * prob
                predictedPos[newPos] = predictedPos[newPos] + self.beliefs[oldPos] * prob
             
        self.beliefs = predictedPos
        self.beliefs.normalize()
        return self.beliefs



    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent);
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles


    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior.
        """
        '''
      
        '''



        "*** YOUR CODE HERE ***"
        
        particleNum = self.numParticles
        legalPositions = self.legalPositions
        legalNum = len(legalPositions)
        particleList = []
        num = 0
        for i in range(particleNum):
            particleList.append(legalPositions[num])
            num = num + 1;
            if num == legalNum:
                num = 0

        self.particles = particleList
        


    def update(self, observation, gameState):
        """


        Update beliefs based on the distance observation and Pacman's position.

        The observation is the estimated Manhattan distance to the ghost you are
        tracking.
        """
        "*** YOUR CODE HERE ***"

        weighted_vals = {}
        total_parts = float(self.numParticles)
        for particle in self.particles:
            if particle not in weighted_vals.keys():
                weighted_vals[particle] = 1
            else:
                weighted_vals[particle] += 1

        len_keys = len(weighted_vals)
      
        dist = {}
        weight = []
        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()
        for p in self.particles:
            dist[p] = self.getObservationProb(observation, pacmanPosition, p, jailPosition) * (weighted_vals[p])
  
        # self.particles = []
        dist_dist = DiscreteDistribution(dist)

        if dist_dist.total() == 0:
            self.initializeUniformly(gameState)

        else:
            self.particles= [dist_dist.sample() for i in range(self.numParticles)]
        
            




    def predict(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        # newPosDist = self.getPositionDistribution(gameState, oldPos)
        predicted_dict = []

        for oldParticle in self.particles:
            newPosDist = self.getPositionDistribution(gameState, oldParticle) # gives a distribution 
            predicted_dict.append(newPosDist.sample())

        self.particles = predicted_dict

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        """


        num_particles = float(self.numParticles)
        ret_val = {}
        for i in self.particles:
            if i in ret_val.keys():
                ret_val[i] += 1
            else:
                ret_val[i] = 1

        ret = DiscreteDistribution(ret_val)
        ret.normalize()
        return ret




class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        "*** YOUR CODE HERE ***"
        legalpos = self.legalPositions
        ret = itertools.product(legalpos, repeat = self.numGhosts)

        ret_val = []
        for i in ret:
            ret_val.append(i)

        random.shuffle(ret_val)

        particle_list = []
        num = 0
        for i in range(self.numParticles):
            particle_list.append(ret_val[num])
            num = num + 1;
            if num == len(ret_val):
                num = 0


        self.particles = particle_list


    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1);

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.update(observation, gameState)

    def update(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the estimated Manhattan distances to all ghosts you
        are tracking.
        """
        "*** YOUR CODE HERE ***"
     

        # weighted_vals = {}

        # for particle in self.particles:
        #     if particle not in weighted_vals.keys():
        #         weighted_vals[particle] = 1
        #     else:
        #         weighted_vals[particle] += 1

           

        dist = {}
        weight = []
        pacmanPosition = gameState.getPacmanPosition()
        newDist = DiscreteDistribution()
        # tempDist = DiscreteDistribution()

        # print "start"
        for p in self.particles:
            # if p not in newDist.keys():
            prob = 1
            for i in range(self.numGhosts):
                jailPosition = self.getJailPosition(i)
                prob *= self.getObservationProb(observation[i], pacmanPosition, p[i], jailPosition) 
            newDist[p] += prob


        if newDist.total() == 0:
            self.initializeUniformly(gameState)

        else:
            self.particles= [newDist.sample() for i in range(self.numParticles)]


    def predict(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.


        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"

            newList = []
            for i in range(self.numGhosts):
                newPosDist = self.getPositionDistribution(gameState, newParticle, i, self.ghostAgents[i])
                newList.append(newPosDist.sample())
                #newParticle[i] = newPosDist
            newParticle = newList
            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles



# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def predict(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.predict(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
