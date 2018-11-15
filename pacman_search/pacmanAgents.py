# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
from heuristics import scoreEvaluation
import random

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write BFS Algorithm instead of returning Directions.STOP
        frontier = [] # FIFO queue to store the states
        frontier.append(state) # add state to the front of the list
        explored = [] # keeps track of the visited states
        track = {} # maps states with the actions needed to get there
        maxState = None #state representing maxscore so far
        maxScore = 0 #maximum score discovered so far

        #loops till the queue is empty
        while frontier:
            tempNode = frontier.pop(0) # pops an element from the front of the list
            if tempNode in explored or tempNode.isWin() or tempNode.isLose():
                continue
            explored.append(tempNode) # marks state as visited
            legalactions = tempNode.getLegalPacmanActions() # possible actions
            for action in legalactions: #iterates through each neighbour
                current = tempNode.generatePacmanSuccessor(action)
                if current == None:
                    break
                if tempNode == state:
                    newState= {current: action} #adds the intial action to get to the state
                    track.update(newState)
                else:
                    newState = {current:track[tempNode]} #adds the intial action to get to the state
                    track.update(newState)
                frontier.append(current)
                score = scoreEvaluation(current)
                if score>maxScore: # gets the best score obtained so far and it's corresponding state
                    maxScore = score
                    maxstate = current
        return track[maxstate]

class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write DFS Algorithm instead of returning Directions.STOP
        frontier = [] # FIFO queue to store the states
        frontier.append(state) # add state to the front of the list
        explored = [] # keeps track of the visited states
        track = {} # maps states with the actions needed to get there
        maxState = None #state representing maxscore so far
        maxScore = 0 #maximum score discovered so far
        #loops till the queue is empty
        while frontier:
            tempNode = frontier.pop() #pops an element from the end of the list
            if tempNode in explored or tempNode.isWin() or tempNode.isLose():
                continue
            explored.append(tempNode) # marks state as visited
            legalactions = tempNode.getLegalPacmanActions()# possible actions
            for action in legalactions:
                current = tempNode.generatePacmanSuccessor(action)
                if current == None:
                    break
                if tempNode == state:
                    newState= {current: action} #adds the intial action to get to the state
                    track.update(newState)
                else:
                    newState = {current:track[tempNode]} #adds the intial action to get to the state
                    track.update(newState)
                frontier.append(current)
                score = scoreEvaluation(current)
                if score>maxScore: # gets the best score obtained so far and it's corresponding state
                    maxScore = score
                    maxstate = current
        return track[maxstate]

class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write A* Algorithm instead of returning Directions.STOP
        #clas that groups a state with its associated cost
        class Node:
            def __init__(self, s, p):
                self.currentstate = s
                self.priority = p
        startNode = Node(state, 0)
        frontier = [] #Queue
        frontier.append(startNode)
        #dictionary that maps state with the action to be taken to reach that state
        startAction = {}
        new = {startNode : None}
        startAction.update(new)
        depth = 0 #counts the depth
        #best cost so far to reach a particular state
        costsofar = {}
        costsofar.update(new)
        #best cost encountered so far
        bestsofar = 9999
        #loops till the queue is empty
        while frontier:
            #sorts the list according to the cost - implements priority queue
            if depth!=0:
                for i in range(len(frontier)):
                    for k in range( len(frontier) - 1, i, -1 ):
                        if ( frontier[k].priority < frontier[k-1].priority ):
                            tempState = frontier[k]
                            frontier[k] = frontier[k-1]
                            frontier[k-1] = frontier[k]
            tempNode = frontier.pop(0) # current Node
            if tempNode.currentstate.isWin() or tempNode.currentstate.isLose():
                continue
            #get ossible actions
            legalactions = tempNode.currentstate.getLegalPacmanActions()
            depth = depth + 1
            #loops for all actions possible
            for action in legalactions:
                current = tempNode.currentstate.generatePacmanSuccessor(action) #neighbour node
                if current == None:
                    break
                #heuristic function to calculate co
                cost = depth - (scoreEvaluation(current)-scoreEvaluation(tempNode.currentstate))
                if current not in costsofar.keys() or costsofar[current]>cost:
                    new = {current : cost}
                    costsofar.update(new)
                    if depth == 1:
                        new = {current : action}
                        startAction.update(new)
                    else:
                        new = {current : startAction[tempNode.currentstate]}
                        startAction.update(new)
                    nodestate = Node(current, cost)
                    frontier.append(nodestate)
                    if cost<bestsofar:
                        bestsofar = cost
                        beststate = current

        return startAction[beststate]
