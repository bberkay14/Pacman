# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
import layout
from util import manhattanDistance
import numpy as np

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

def closestCapsule(pos, capsule, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a capsule at this location then exit
        if (pos_x, pos_y) in capsule:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no capsule found
    return None
                                
def closestGhost(pos, ghosts, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a ghost at this location then exit
        if (pos_x, pos_y) in ghosts:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no ghost found
    return None
                                                
def closestScaredGhost(pos, walls, ghostStates):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    ghostStateIndexPosition = {}  
    ghostStateIndexScaredTimer = {} 
    for ghostStateIndex in range(len(ghostStates)):
        ghostStateIndexPosition[ghostStates[ghostStateIndex].getPosition()] = ghostStateIndex  
        ghostStateIndexScaredTimer[ghostStateIndex] = ghostStates[ghostStateIndex].scaredTimer
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a scared ghost at this location then exit
        if (pos_x, pos_y) in ghostStateIndexPosition.keys():
            if ghostStateIndexScaredTimer[ghostStateIndexPosition[(pos_x, pos_y)]] > 0:
                return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no scared ghost found
    return None
                                                                        
def closestNotScaredGhostDirection(pos, walls, ghostStates):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    ghostStateIndexPosition = {}  
    ghostStateIndexDirection = {} 
    ghostStateIndexScaredTimer = {} 
    for ghostStateIndex in range(len(ghostStates)):
        ghostStateIndexPosition[ghostStates[ghostStateIndex].getPosition()] = ghostStateIndex  
        ghostStateIndexDirection[ghostStateIndex] = ghostStates[ghostStateIndex].configuration.getDirection() 
        ghostStateIndexScaredTimer[ghostStateIndex] = ghostStates[ghostStateIndex].scaredTimer
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a ghost at this location then exit
        if (pos_x, pos_y) in ghostStateIndexPosition.keys():   
            if ghostStateIndexScaredTimer[ghostStateIndexPosition[(pos_x, pos_y)]] == 0:
                return ghostStateIndexDirection[ghostStateIndexPosition[(pos_x, pos_y)]]
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no ghost found
    return None  
                                                                                            
def closestScaredGhostDirection(pos, walls, ghostStates):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    ghostStateIndexPosition = {}  
    ghostStateIndexDirection = {} 
    ghostStateIndexScaredTimer = {} 
    for ghostStateIndex in range(len(ghostStates)):
        ghostStateIndexPosition[ghostStates[ghostStateIndex].getPosition()] = ghostStateIndex  
        ghostStateIndexDirection[ghostStateIndex] = ghostStates[ghostStateIndex].configuration.getDirection() 
        ghostStateIndexScaredTimer[ghostStateIndex] = ghostStates[ghostStateIndex].scaredTimer
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a ghost at this location then exit
        if (pos_x, pos_y) in ghostStateIndexPosition.keys():   
            if ghostStateIndexScaredTimer[ghostStateIndexPosition[(pos_x, pos_y)]] > 0:
                return ghostStateIndexDirection[ghostStateIndexPosition[(pos_x, pos_y)]]
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no ghost found
    return None   
                                                                                                                        
def isClosestGhostScared(pos, walls, ghostStates):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    ghostStateIndexPosition = {}  
    ghostStateIndexScaredTimer = {}
    for ghostStateIndex in range(len(ghostStates)):
        ghostStateIndexPosition[ghostStates[ghostStateIndex].getPosition()] = ghostStateIndex  
        ghostStateIndexScaredTimer[ghostStateIndex] = ghostStates[ghostStateIndex].scaredTimer
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a scared ghost at this location then exit
        if (pos_x, pos_y) in ghostStateIndexPosition.keys():
            return ghostStateIndexScaredTimer[ghostStateIndexPosition[(pos_x, pos_y)]] == 0
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no scared ghost found
    return False   
                                                                                                                                            
def numberOfGhosts5StepsAwayInEast(pos, ghosts, walls):
    count = 0
    pos_x, pos_y = pos[0], pos[1]
    for (g_x, g_y) in ghosts:
        if manhattanDistance((g_x, g_y), (pos_x, pos_y)) < 5:
            if g_x > pos_x:
                count = count + 1
    return count    
    
def numberOfGhosts5StepsAwayInWest(pos, ghosts, walls):
    count = 0
    pos_x, pos_y = pos[0], pos[1]
    for (g_x, g_y) in ghosts:
        if manhattanDistance((g_x, g_y), (pos_x, pos_y)) < 5:
            if g_x < pos_x:
                count = count + 1
    return count   
    
def numberOfGhosts5StepsAwayInSouth(pos, ghosts, walls):
    count = 0
    pos_x, pos_y = pos[0], pos[1]
    for (g_x, g_y) in ghosts:
        if manhattanDistance((g_x, g_y), (pos_x, pos_y)) < 5:
            if g_y < pos_y:
                count = count + 1
    return count   
    
def numberOfGhosts5StepsAwayInNorth(pos, ghosts, walls):
    count = 0
    pos_x, pos_y = pos[0], pos[1]
    for (g_x, g_y) in ghosts:
        if manhattanDistance((g_x, g_y), (pos_x, pos_y)) < 5:
            if g_y > pos_y:
                count = count + 1
    return count  

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class MoreFeaturesExtractor(FeatureExtractor):
    """
    Returns features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost is one step away
    - are there scared ghosts
    - how far away the next capsule is
    - how far away the next ghost is
    - how far away the next scared ghost is
    - whether capsule will be eaten
    - number of ghosts pacman is getting closer with this action
    - is closest not scared ghost going south/north/east/west/stop
    - is closest scared ghost going south/north/east/west/stop
    - is closest ghost not scared
    - how many ghosts are max 5 steps away in west/east/south/north
    - how many foods are in west/east/south/north
    """

    def getFeatures(self, state, action):

        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        capsules = state.getCapsules()
        ghostStates = state.getGhostStates()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # is closest ghost not scared
        #features["is-closest-ghost-not-scared"] = isClosestGhostScared((next_x, next_y), walls, ghostStates)

        # 
        features["#-of-ghosts-max-5-steps-away-in-east"] = numberOfGhosts5StepsAwayInEast((next_x, next_y), ghosts, walls)
        features["#-of-ghosts-max-5-steps-away-in-west"] = numberOfGhosts5StepsAwayInWest((next_x, next_y), ghosts, walls)
        features["#-of-ghosts-max-5-steps-away-in-south"] = numberOfGhosts5StepsAwayInSouth((next_x, next_y), ghosts, walls)
        features["#-of-ghosts-max-5-steps-away-in-north"] = numberOfGhosts5StepsAwayInNorth((next_x, next_y), ghosts, walls)

        # is closest not scared ghost going south/north/east/west/stop
        features["is-closest-not-scared-ghost-going-east"] = closestNotScaredGhostDirection((next_x, next_y), walls, ghostStates) == (1, 0)
        features["is-closest-not-scared-ghost-going-west"] = closestNotScaredGhostDirection((next_x, next_y), walls, ghostStates) == (-1, 0)
        features["is-closest-not-scared-ghost-going-south"] = closestNotScaredGhostDirection((next_x, next_y), walls, ghostStates) == (0, -1)
        features["is-closest-not-scared-ghost-going-north"] = closestNotScaredGhostDirection((next_x, next_y), walls, ghostStates) == (0, 1)
        #features["is-closest-not-scared-ghost-going-stop"] = closestNotScaredGhostDirection((next_x, next_y), walls, ghostStates) == (0, 0)

        # is closest scared ghost going south/north/east/west/stop
        features["is-closest-scared-ghost-going-east"] = closestScaredGhostDirection((next_x, next_y), walls, ghostStates) == (1, 0)
        features["is-closest-scared-ghost-going-west"] = closestScaredGhostDirection((next_x, next_y), walls, ghostStates) == (-1, 0)
        features["is-closest-scared-ghost-going-south"] = closestScaredGhostDirection((next_x, next_y), walls, ghostStates) == (0, -1)
        features["is-closest-scared-ghost-going-north"] = closestScaredGhostDirection((next_x, next_y), walls, ghostStates) == (0, 1)
        #features["is-closest-scared-ghost-going-stop"] = closestScaredGhostDirection((next_x, next_y), walls, ghostStates) == (0, 0)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # number of ghosts pacman is getting closer with this action
        features["#-of-ghosts-pacman-getting-closer"] = sum(manhattanDistance((next_x, next_y), (g_x, g_y)) < manhattanDistance((x, y), (g_x, g_y)) for (g_x, g_y) in ghosts)

        # are there scared ghosts
        ghostScaredList = [s.scaredTimer for s in state.getGhostStates()]
        features["exists-scared-ghosts"] = (np.average(ghostScaredList) > 0) * 1.0

        # closest ghost
        distanceOfClosestGhost = closestGhost((next_x, next_y), ghosts, walls)
        if distanceOfClosestGhost is not None:
            features["closest-ghost"] = float(distanceOfClosestGhost) / (walls.width * walls.height)
    
        # closest scared ghost
        distanceOfClosestScaredGhost = closestScaredGhost((next_x, next_y), walls, ghostStates)
        if distanceOfClosestScaredGhost is not None:
            features["closest-scared-ghost"] = float(distanceOfClosestScaredGhost) / (walls.width * walls.height)

        # closest capsule
        distanceOfClosestCapsule = closestCapsule((next_x, next_y), capsules, walls)
        if distanceOfClosestCapsule is not None:
            features["closest-capsule"] = float(distanceOfClosestCapsule) / (walls.width * walls.height)
    
        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
                
        # if there is no danger of ghosts then add the capsule feature
        if not features["#-of-ghosts-1-step-away"] and (next_x, next_y) in capsules:
            features["eats-capsule"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features


class DeepQLearningExtractor(FeatureExtractor):

    def get_state_dim(self,layout):
        pac_ft_size = 2
        ghost_ft_size = 2 * layout.getNumGhosts()
        food_capsule_ft_size = layout.width * layout.height
        return  food_capsule_ft_size * 10

    def get_features(self,state):
        pacman_state = np.array(state.getPacmanPosition())
        ghost_state = np.array(state.getGhostPositions()).astype(np.int)
        capsules = state.getCapsules()
        food_locations = np.array(state.getFood().data).astype(np.float32)
        wall_locations = np.array(state.getWalls().data).astype(np.float32)
        ghost_locations = np.zeros((food_locations.shape[0], food_locations.shape[1]),dtype=np.int)
        capsule_locations = np.zeros((food_locations.shape[0], food_locations.shape[1]),dtype=np.int)
        pacman_locations = np.zeros((food_locations.shape[0], food_locations.shape[1]),dtype=np.int)
     
        for x, y in ghost_state:
            ghost_locations[x][y] = 1
        
        for x, y in capsules:
            capsule_locations[x][y] = 1
            
        pacman_locations[pacman_state[0]][pacman_state[1]] = 1
        food_locations = np.expand_dims(food_locations,0)
        wall_locations = np.expand_dims(wall_locations,0)
        ghost_locations = np.expand_dims(ghost_locations,0)
        capsule_locations = np.expand_dims(capsule_locations,0)
        pacman_locations = np.expand_dims(pacman_locations,0)
    
        matrix = np.concatenate((food_locations,wall_locations,ghost_locations,capsule_locations,pacman_locations), axis=0)
        
        return  matrix
