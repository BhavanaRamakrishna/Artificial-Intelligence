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
from heuristics import *
import random
import math

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

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,5):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        #generate a random action sequene
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];

        #to store the first action from the sequence with highest score
        maxActionList = self.actionList
        maxScore = 0;
        flag =0

        while(1):
            tempState = state;
            #evaluate the action sequence
            for i in range(0,len(self.actionList)):
                if tempState.isWin() + tempState.isLose() == 0:
                    tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
                    #check for terminal state
                    if tempState == None:
                        break
                    else:
                        score = scoreEvaluation(tempState)

                elif tempState.isWin():
                    flag = 1
                    maxActionList = self.actionList
                    break
                else:
                    break;

            if flag == 1 or tempState == None:
                break
            #store the first action from the sequence with highest score
            if score>maxScore:
                maxScore = score
                maxActionList = self.actionList

            #generate new action sequec=nce based on 50% probability
            for i in range(0,len(maxActionList)):
                rand = random.random()
                if(rand)>0.5:
                    self.actionList[i] = possible[random.randint(0,len(possible)-1)]
                else:
                    self.actionList[i] = maxActionList[i]

        #return the first action from the sequence with highest score
        return maxActionList[0]

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,5):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        #population of 8 chromosomes
        possible = state.getAllPossibleActions()
        population = []
        for i in range(0,8):
            population.append([])
        for i in range(0,8):
            for j in range(0,5):
                population[i].append(possible[random.randint(0,len(possible)-1)])
        final = Directions.STOP
        while(1):
            newpopulation = []
            chromosomeScore = []
            newcount = 0
            #evaluate fitness for each chromosome
            for i in range(0,len(population)):
                tempState = state
                for j in range(0,len(population[i])):
                    tempState = tempState.generatePacmanSuccessor(population[i][j])
                    if tempState == None or tempState.isLose()==1:
                        break
                if tempState != None:
                    score = scoreEvaluation(tempState)
                    chromosomeScore.append(score)
                else:
                    break
            if tempState == None:
                break
            #assign ranks
            rank = {}
            newList1 = chromosomeScore
            newList1.sort(reverse=True)
            newList2 = [0.222,0.194,0.167,0.139,0.111,0.083,0.056,0.028]
            chrome = [float(integral) for integral in chromosomeScore]
            for i in range(0,len(chrome)):
                for j in range(0,len(chrome)):
                    if chrome[i] == newList1[j]:
                        chrome[i] = newList2[j]
                        newList1[j] = 0
                        break
            for i in range(0,len(population)):
                temp = {chrome[i] : population[i]}
                rank.update(temp)
            for key, value in rank.iteritems():
                if key > 0.194:
                    finalaction = value
                    break
            final = finalaction[0]
            while newcount <=7: #8 new choromosomes
                #rank selection-select parent
                parent1 = []
                parent2 = []
                rand1 = random.uniform(0.0,0.222)
                for key in sorted(rank):
                    if key >= rand1:
                        parent1 = rank[key]
                        break
                rand2 = random.uniform(0.0,0.222)
                for key in sorted(rank):
                    if key >= rand2:
                        parent2 = rank[key]
                        break
                #generate children
                child1 = []
                child2 = []
                rand = random.random()
                if rand <= 0.7:
                    for m in range(0,5):
                        j = random.random()
                        if j <= 0.5:
                            child1.append(parent1[m])
                        else:
                            child1.append(parent2[m])
                    for m in range(0,5):
                        j = random.random()
                        if j <= 0.5:
                            child2.append(parent1[m])
                        else:
                            child2.append(parent2[m])
                    #mutate children
                    rand1 = random.randint(0,4)
                    r = random.random()
                    if r<=0.1:
                        child1[rand1] = possible[random.randint(0,len(possible)-1)]
                    rand1 = random.randint(0,4)
                    r = random.random()
                    if r<=0.1:
                        child2[rand1] = possible[random.randint(0,len(possible)-1)]
                    #add to new population
                    newpopulation.append(child1)
                    newpopulation.append(child2)
                    newcount = newcount + 2
                else:
                    #keep parent in next generation - mutate
                    rand = random.randint(0,4)
                    r = random.random()
                    if r<=0.1:
                        parent1[rand] = possible[random.randint(0,len(possible)-1)]
                    rand = random.randint(0,4)
                    r = random.random()
                    if r<=0.1:
                        parent2[rand] = possible[random.randint(0,len(possible)-1)]
                    #add to new population
                    newpopulation.append(parent1)
                    newpopulation.append(parent2)
                    newcount = newcount + 2
            population = newpopulation
        return final

class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        root = state
        #class node to sctore parent, reward and visit of each state
        class Node():
            def __init__(self, statespace,incomingaction, parent):
                self.state = statespace
                self.parent = parent
                self.action = incomingaction
                self.visitCount = 1
                self.rewardGained = 0.0
                self.states = [] #node

            #check if the node is fully expanded
            def ExpandedFlag(self,node):
                actions = self.state.getLegalPacmanActions()
                if len(self.states) == len(actions):
                    return True
                return False

        def TreePolicy(node):
            #repeat till node is nonterminal
            while (1):
                if node.ExpandedFlag(node) == False:
                    return Expand(node)
                else:
                    node = BestChild(node)
            return node

        #returns urgent unexpanded node
        def Expand(node):
            triedStates = [child.state for child in node.states]
            actions = node.state.getLegalPacmanActions()
            childnode = None
            for i in range(0,len(actions)):
                child = node.state.generatePacmanSuccessor(actions[i])
                if child == None:
                    break
                if child not in triedStates:
                    childnode = Node(child,actions[i],node)
                    node.states.append(childnode)
                    break
            return childnode

        #returns the best child of the node
        def BestChild(node):
            max = -1 * float('inf')
            maxchild = node
            for child in node.states:
                x = child.rewardGained / child.visitCount
                y = math.sqrt((2.0*math.log(node.visitCount))/child.visitCount)
                score = x + 1*y
                if score > max:
                    max = score
                    maxchild = child
            return maxchild

        #rollout - 5 times to get the reward value
        def DefaultPolicy(node):
            s = node.state
            for i in range(0,5):
                actions = s.getLegalPacmanActions()
                if len(actions) == 0:
                    return None
                a = actions[random.randint(0,len(actions)-1)]
                s = s.generatePacmanSuccessor(a)
                if s == None:
                    return None
            reward = normalizedScoreEvaluation(root,s)
            return reward

        #backpropogate while updating the score and visitcount
        def BackUp(node, reward):
            s = node
            while s!=None:
                s.visitCount = s.visitCount + 1
                s.rewardGained = s.rewardGained + reward
                s = s.parent
            return



        rootnode = Node(root, None, None)
        while(1):
            tempNode = TreePolicy(rootnode)
            if tempNode == None:
                break
            reward = DefaultPolicy(tempNode)
            if reward == None:
                break
            BackUp(tempNode,reward)

        #return the action to the best child of root
        final = BestChild(rootnode)
        finalAction = final.action

        return finalAction
