# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

import layout
import torch
import copy
from collections import deque, namedtuple

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvalue = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qvalue[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return 0.0
        maxQValue = float('-inf')
        for action in legalActions:
            maxQValue = self.getQValue(state, action) if self.getQValue(state,action) > maxQValue
        return maxQValue
        """
        if not self.getLegalActions(state):
            return 0.0
        return max([self.qvalue[(state, action)] for action in self.getLegalActions(state)])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = []
        maxqvalue = float('-inf')
        for action in self.getLegalActions(state):
            if self.getQValue(state,action) > maxqvalue:
                maxqvalue = self.getQValue(state, action)
                bestAction = [action]
            elif self.getQValue(state, action) == maxqvalue:
                bestAction.append(action)
        return random.choice(bestAction)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        nextQValue = reward + self.discount * self.computeValueFromQValues(nextState)
        key = state, action
        currentQValue = self.getQValue(state, action)
        updatedQVvalue = (1 - self.alpha) * currentQValue + self.alpha * nextQValue
        self.qvalue[key] = updatedQVvalue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        
        features = self.featExtractor.getFeatures(state, action)
        qValue = 0
        for feature in features.keys():
            qValue = qValue + self.weights[feature] * features[feature]
        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        maxNextQValue = float('-Inf')
        for actionOfNextState in self.getLegalActions(nextState):
            if (self.getQValue(nextState,actionOfNextState) > maxNextQValue):
                maxNextQValue = self.getQValue(nextState, actionOfNextState)
                
        if (maxNextQValue == float('-Inf')):
            maxNextQValue = 0
        diff = (reward + (self.discount * maxNextQValue)) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        self.qvalue[(state,action)] += self.alpha * diff
        for feature in features.keys():
            self.weights[feature] = self.weights[feature] + self.alpha * diff * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass



class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.cursor = 0
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
    def add(self, state, action, reward, nextState, done):
        if len(self) < self.capacity:
            self.buffer.append(None)
        state = np.array(state).astype("float64")
        nextState = np.array(nextState).astype("float64")
        Transition = namedtuple("Transition", field_names=[
    "state", "action", "reward", "nextState", "done"])
        self.buffer[self.cursor] = Transition(state, action, reward, nextState, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batchSize):
        return random.sample(self.buffer, batchSize)
    

class DeepQNetwork():
    def __init__(self, stateDimension, actionDimension):
        self.numActions = actionDimension
        self.stateSize = stateDimension

        self.learningRate = 0.002
        self.numTrainingGames = 10000
        self.batchSize = 20
        
        self.dQLModel = torch.nn.Sequential(
            torch.nn.Conv2d(5, 10, (3,3),padding=(1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(self.stateSize, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.numActions)
        )
        
        self.lossFunction = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.dQLModel.parameters(), lr=self.learningRate)
        
        
    def getLoss(self, states, qTarget):
        out = self.run(states)

        softmax = torch.nn.Softmax(dim=1)
        return self.lossFunction(out, qTarget)
    
    def set_weights(self, parameters):
        for i in range(len(parameters)):
            self.parameters.append(layers[i])
            self.dQLModel.parameters()[i] = parameters[i]

    def run(self, states):
        try:
            return self.dQLModel(states.float())
        except:
            return self.dQLModel(torch.tensor(states, requires_grad=False).float())
        

    def gradientUpdate(self, states, qTarget):
        self.optim.zero_grad()
        loss = self.getLoss(states, qTarget)
        print(3333333333333)
        print(loss)
        loss.backward()
        self.optim.step()
    
class PacmanDeepQAgent(PacmanQAgent):
    def __init__(self, layout_input="smallGrid", extractor='DeepQLearningExtractor', targetUpdateRate=300, doubleQ=True, **args):
        self.model = None
        self.targetModel = None
        self.targetUpdateRate = targetUpdateRate
        self.updateAmount = 0
        self.epsilonExplore = 1
        self.epsilon0 = 0.05
        self.epsilon = self.epsilon0
        self.discount = 0.9
        self.updateFrequency = 3
        self.counts = None
        self.replayBuffer = ReplayBuffer(100000)
        self.minTransitionsBeforeTraining = 100000
        self.tdErrorClipping = 1
        self.featExtractor = util.lookup(extractor, globals())()

        if isinstance(layout_input, str):
            layout_instantiated = layout.getLayout(layout_input)
        else:
            layout_instantiated = layout_input
        self.state_dim = self.featExtractor.get_state_dim(layout_instantiated)
        self.model = DeepQNetwork(self.state_dim, 5)
        self.targetModel = DeepQNetwork(self.state_dim, 5)

        self.doubleQ = doubleQ
        if self.doubleQ:
            self.targetUpdateRate = -1
            
        PacmanQAgent.__init__(self, **args)

    def getQValue(self, state, action):
        feats = self.featExtractor.get_features(state)    
        legalActions = self.getLegalActions(state)
        action_index = legalActions.index(action)
        
        state = torch.tensor([feats], requires_grad=False) 
        return self.model.run(state)[0][action_index].item()
    
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = []
        maxqvalue = float('-inf')
        legalActions = self.getLegalActions(state)
        for action in legalActions:           
            if self.getQValue(state,action) > maxqvalue:
                maxqvalue = self.getQValue(state, action)
                bestAction = [action]
            elif self.getQValue(state, action) == maxqvalue:
                bestAction.append(action)
        return random.choice(bestAction)
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return 0.0
        maxQValue = float('-inf')
        for action in legalActions:
            maxQValue = self.getQValue(state, action) if self.getQValue(state,action) > maxQValue
        return maxQValue
        """
        legalActions2 = self.getLegalActions(state)
        legalActions = []
        for legalAction in legalActions2:
            if legalAction != 'Stop':
                legalActions.append(legalAction)
        if not legalActions:
            return 0.0
        return max([self.qvalue[(state, action)] for action in legalActions])

    def shapeReward(self, reward):
        if reward > 20:
            reward = 50
        elif reward > 0:
            reward = 20
        elif reward < -10:
            reward = -500
        elif reward < 0:
            reward = -1
        elif not self.getLegalActions(state):
            if reward >= -10:
                reward = 100
        return reward
    

    def computeQTargets(self,stateIndices, minibatch, network = None, targetNetwork=None, doubleQ=False):

        if network is None:
            network = self.model
        if targetNetwork is None:
            targetNetwork = self.targetModel
        states = np.vstack([x.state for x in minibatch])
        actions = np.array([x.action for x in minibatch])      
            
        rewards = np.array([x.reward for x in minibatch])
        nextStates = np.vstack([x.nextState for x in minibatch])
        done = np.array([x.done for x in minibatch])

        qPredict = network.run(states).data
        qTarget = torch.detach(qPredict)
  
        explorationBonus = 1 / (2 * np.sqrt((self.counts[stateIndices] / 1000)))
        replaceIndices = np.arange(actions.shape[0])
        actionIndices = np.argmax(network.run(nextStates).data, axis=1)
        actionIndices2 = actionIndices
        for actionn in range(len(actionIndices)):
            if actionIndices[actionn] == 0:
                actionIndices2[actionn] = -5.0
            else:
                actionIndices2[actionn] = 0.0
        
        done2 = torch.tensor(done, requires_grad=False).float()
        rewards = torch.tensor(rewards, requires_grad=False).float()
        
        explorationBonus = torch.tensor(explorationBonus, requires_grad=False).float()

        target = rewards +actionIndices2+ explorationBonus + (1 - done2) * self.discount * targetNetwork.run(nextStates).data[replaceIndices, actionIndices]
        
        qTarget[replaceIndices, actions] = target.float()

        if self.tdErrorClipping is not None:
            qTarget = qPredict + np.clip(
                     qTarget - qPredict, -self.tdErrorClipping, self.tdErrorClipping)
        return qTarget

    def polyakAverage(self, network, targetNetwork):
        for param, targetParam in zip(network.dQLModel.parameters(), targetNetwork.dQLModel.parameters()):
            targetParam.data.mul_(1-self.polyakFactor)
            targetParam.data.addcmul_(param.data, torch.ones(1, requires_grad=False), value=self.polyakFactor)                   

    def update(self, state, action, nextState, reward):
        legalActions = self.getLegalActions(state)

        action_index = legalActions.index(action)
        done = nextState.isLose() or nextState.isWin()
        reward = self.shapeReward(reward)
        if self.counts is None:
            x, y = np.array(state.getFood().data).shape
            self.counts = np.ones((x, y))

        self.counts[int(state.getPacmanPosition()[0])][int(state.getPacmanPosition()[1])] += 1
        stateIndices = (state.getPacmanPosition()[0], state.getPacmanPosition()[1])
        state = self.featExtractor.get_features(state)
        state = np.expand_dims(state,0)

        nextState = self.featExtractor.get_features(nextState)
        nextState = np.expand_dims(nextState,0)

        transition = (state, action_index, reward, nextState, done)
        self.replayBuffer.add(*transition)

        if len(self.replayBuffer) < self.minTransitionsBeforeTraining:
            self.epsilon = self.epsilonExplore
        else:
            self.epsilon = max(self.epsilon0 * (1 - self.updateAmount / 20000), 0)

        if len(self.replayBuffer) > self.minTransitionsBeforeTraining and self.updateAmount % self.updateFrequency == 0:
            minibatch = self.replayBuffer.pop(self.model.batchSize)

            states = np.vstack([x.state for x in minibatch])
            qTarget1 = self.computeQTargets(stateIndices,minibatch, self.model, self.targetModel, doubleQ=self.doubleQ)

            if self.doubleQ:
                qTarget2 = self.computeQTargets(stateIndices,minibatch, self.targetModel, self.model, doubleQ=self.doubleQ)

            self.model.gradientUpdate(states, torch.nn.Softmax(dim=0)(qTarget1))
            if self.doubleQ:
                self.targetModel.gradientUpdate(states, torch.nn.Softmax(dim=0)(qTarget2))

        self.updateAmount += 1

    def final(self, state):
        PacmanQAgent.final(self, state)