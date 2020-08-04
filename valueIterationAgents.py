# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
        # print(mdp.getStates())

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for k in range(self.iterations):
            temp = util.Counter()
            for state in self.mdp.getStates():
                # print(state)
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)
                    # print(state, actions)
                    value = -float("inf")
                    for action in actions:
                        value = max(value, self.getQValue(state, action))
                        temp[state] = value
                        # print(state, action, self.getQValue(state, action))
            self.values = temp
                # print(self.values)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.getValue(nextState))
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        value, bestAction = max([(self.getQValue(state, action), action) for action in actions])
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for k in range(self.iterations):
            states = self.mdp.getStates()
            state = states[k % len(states)]
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                value = -float("inf")
                for action in actions:
                    value = max(value, self.getQValue(state, action))
                self.values[state] = value


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pred = {}
        pq = util.PriorityQueue()
        for state in self.mdp.getStates():
            pred[state] = set([])
        for predState in self.mdp.getStates():
            qValue = -float("inf")
            for action in self.mdp.getPossibleActions(predState):
                qValue = max(qValue, self.getQValue(predState, action))
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(predState, action):
                    pred[nextState].add(predState)
            if not self.mdp.isTerminal(predState):
                diff = abs(qValue - self.getValue(predState))
                pq.push(predState, -1 * diff)

        for k in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                value = -float("inf")
                for action in actions:
                    value = max(value, self.getQValue(state, action))
                self.values[state] = value
                for state in pred[state]:
                    qValue = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
                    diff = abs(qValue - self.getValue(state))
                    if diff > self.theta:
                        pq.update(state, -1*diff)
