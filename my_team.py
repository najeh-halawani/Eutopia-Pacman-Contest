# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from capture_agents import CaptureAgent
from distance_calculator import DistanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearest_point
import math

#################
# Team GhostBusters 👻
#################

def create_team(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########



class ReflexCaptureAgent(CaptureAgent):
    '''
    Methods inherited from the baselineTeam.py
    '''
    
    def getSuccessor(self, gameState, action):
        successor = gameState.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
       
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action): 
        features = self.evaluateAttackParameters(gameState, action)
        weights = self.getCostOfAttackParameter(gameState, action)
        return features * weights
 
    def evaluateAttackParameters(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.get_score(successor)
        return features

    def getCostOfAttackParameter(self, gameState, action):
        return {'successorScore': 1.0}

class MCTSNode:
    def __init__(self, state, parent=None, action=None, agent_index=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.action = action
        
        self.untried_actions = state.get_legal_actions(agent_index)
        
        if Directions.STOP in self.untried_actions:
            self.untried_actions.remove(Directions.STOP)
        
        if agent_index is not None:
            current_direction = state.get_agent_state(agent_index).configuration.direction
            reverse_dir = Directions.REVERSE[current_direction]
            if reverse_dir in self.untried_actions:
                self.untried_actions.remove(reverse_dir)

def ucb_score(node, total_visits):
    """
    Calculate the UCB1 score for a node
    """
    if node.visits == 0:
        return float('inf')  # Encourage unexplored nodes
    
    exploration_weight = math.sqrt(2)  # Standard exploration parameter
    exploitation = node.value / node.visits
    exploration = exploration_weight * math.sqrt(math.log(total_visits) / node.visits)
    
    return exploitation + exploration

class OffensiveReflexAgent(ReflexCaptureAgent):
    '''
    Inheriting properties of Base Class
    '''
    def __init__(self, index):
        CaptureAgent.__init__(self, index)        
        self.presentCoordinates = (-5 ,-5)
        self.counter = 0
        self.attack = False
        self.lastFood = []
        self.presentFoodList = []
        self.shouldReturn = False
        self.capsulePower = False
        self.targetMode = None
        self.eatenFood = 0
        self.initialTarget = []
        self.hasStopped = 0
        self.capsuleLeft = 0
        self.prevCapsuleLeft = 0

    def register_initial_state(self, gameState):
        self.currentFoodSize = 9999999
        
        CaptureAgent.register_initial_state(self, gameState)
        self.initPosition = gameState.get_agent_state(self.index).get_position()
        self.initialAttackCoordinates(gameState)

    def initialAttackCoordinates(self ,gameState):
        
        layoutInfo = []
        x = (gameState.data.layout.width - 2) // 2
        if not self.red:
            x +=1
        y = (gameState.data.layout.height - 2) // 2
        layoutInfo.extend((gameState.data.layout.width , gameState.data.layout.height ,x ,y))
       
        self.initialTarget = []

        
        for i in range(1, layoutInfo[1] - 1):
            if not gameState.has_wall(layoutInfo[2], i):
                self.initialTarget.append((layoutInfo[2], i))
        
        noTargets = len(self.initialTarget)
        if(noTargets%2==0):
            noTargets = (noTargets//2) 
            self.initialTarget = [self.initialTarget[noTargets]]
        else:
            noTargets = (noTargets-1)//2
            self.initialTarget = [self.initialTarget[noTargets]] 

    
    def evaluateAttackParameters(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action) 
        position = successor.get_agent_state(self.index).get_position() 
        foodList = self.get_food(successor).as_list() 
        features['successorScore'] = self.get_score(successor) 

        if successor.get_agent_state(self.index).is_pacman:
            features['offence'] = 1
        else:   
            features['offence'] = 0

        if foodList: 
            features['foodDistance'] = min([self.get_maze_distance(position, food) for food in foodList])

        opponentsList = []
       
        disToGhost = []
        opponentsList = self.get_opponents(successor)

        for i in range(len(opponentsList)):
            enemyPos = opponentsList[i]
            enemy = successor.get_agent_state(enemyPos)
            if not enemy.is_pacman and enemy.get_position() != None:
                ghostPos = enemy.get_position()
                disToGhost.append(self.get_maze_distance(position ,ghostPos))


        if len(disToGhost) > 0:
            minDisToGhost = min(disToGhost)
            if minDisToGhost < 5:
                features['distanceToGhost'] = minDisToGhost + features['successorScore']
            else:
                features['distanceToGhost'] = 0


        return features
    
    def getCostOfAttackParameter(self, gameState, action):
        '''
        Setting the weights manually after many iterations
        '''

        if self.attack:
            if self.shouldReturn is True:
                return {'offence' :3010,
                        'successorScore': 202,
                        'foodDistance': -8,
                        'distancesToGhost' :215}
            else:
                return {'offence' :0,
                        'successorScore': 202,
                        'foodDistance': -8,
                        'distancesToGhost' :215}
        else:
            successor = self.getSuccessor(gameState, action) 
            weightGhost = 210
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            invaders = [a for a in enemies if not a.is_pacman and a.get_position() != None]
            if len(invaders) > 0:
                if invaders[-1].scared_timer > 0:
                    weightGhost = 0
                    
            return {'offence' :0,
                    'successorScore': 202,
                    'foodDistance': -8,
                    'distancesToGhost' :weightGhost}

    def getOpponentPositions(self, gameState):
        return [gameState.get_agent_position(enemy) for enemy in self.get_opponents(gameState)]

    def best_possible_action(self ,mcsc):
        ab = mcsc.get_legal_actions(self.index)
        ab.remove(Directions.STOP)

        if len(ab) == 1:
            return ab[0]
        else:
            reverseDir = Directions.REVERSE[mcsc.get_agent_state(self.index).configuration.direction]
            if reverseDir in ab:
                ab.remove(reverseDir)
            return random.choice(ab)

    def monteCarloSimulation(self ,gameState ,depth):
        ss = gameState.deep_copy()
        while depth > 0:
            ss = ss.generate_successor(self.index ,self.best_possible_action(ss))
            depth -= 1
        return self.evaluate(ss ,Directions.STOP)

    def get_best_action(self,legalActions,gameState,possibleActions,distanceToTarget):
        shortestDistance = 9999999999
        for i in range (0,len(legalActions)):    
            action = legalActions[i]
            nextState = gameState.generate_successor(self.index, action)
            nextPosition = nextState.get_agent_position(self.index)
            distance = self.get_maze_distance(nextPosition, self.initialTarget[0])
            distanceToTarget.append(distance)
            if(distance<shortestDistance):
                shortestDistance = distance

        bestActionsList = [a for a, distance in zip(legalActions, distanceToTarget) if distance == shortestDistance]
        bestAction = random.choice(bestActionsList)
        return bestAction
    
    def monte_carlo_tree_search(self, initial_state, max_iterations=100):
            """
            Implement Monte Carlo Tree Search to select the best action
            """
            root = MCTSNode(initial_state, agent_index=self.index)
            
            for _ in range(max_iterations):
                # Selection
                node = self.select_node(root)
                
                # Expansion
                if node.untried_actions:
                    action = random.choice(node.untried_actions)
                    node.untried_actions.remove(action)
                    child_state = node.state.generate_successor(self.index, action)
                    child_node = MCTSNode(child_state, parent=node, action=action, agent_index=self.index)
                    node.children.append(child_node)
                    node = child_node
                
                # Simulation (Rollout)
                result = self.simulate(node.state)
                
                # Backpropagation
                while node is not None:
                    node.visits += 1
                    node.value += result
                    node = node.parent
            
            # Select best action based on most visited child
            best_child = max(root.children, key=lambda n: n.visits)
            return best_child.action

    def select_node(self, node):
        """
        Select the best node to explore using UCB
        """
        while not node.untried_actions and node.children:
            node = max(node.children, key=lambda c: ucb_score(c, node.visits))
        return node

    def simulate(self, state, max_depth=30):
        """
        Perform a simulation (rollout) from the given state
        """
        current_state = state.deep_copy()
        for _ in range(max_depth):
            legal_actions = current_state.get_legal_actions(self.index)
            if not legal_actions:
                break
            
            # Remove STOP and reverse directions
            if Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            
            current_direction = current_state.get_agent_state(self.index).configuration.direction
            reverse_dir = Directions.REVERSE[current_direction]
            if reverse_dir in legal_actions:
                legal_actions.remove(reverse_dir)
            
            action = random.choice(legal_actions)
            current_state = current_state.generate_successor(self.index, action)
        
        return self.evaluate(current_state, Directions.STOP)
    
    def choose_action(self, gameState):
        self.presentCoordinates = gameState.get_agent_state(self.index).get_position()
    
        if self.presentCoordinates == self.initPosition:
            self.hasStopped = 1
        if self.presentCoordinates == self.initialTarget[0]:
            self.hasStopped = 0

        # find next possible best move 
        if self.hasStopped == 1:
            legalActions = gameState.get_legal_actions(self.index)
            legalActions.remove(Directions.STOP)
            possibleActions = []
            distanceToTarget = []
            
            bestAction=self.get_best_action(legalActions,gameState,possibleActions,distanceToTarget)
            
            return bestAction

        if self.hasStopped==0:
            self.presentFoodList = self.get_food(gameState).as_list()
            self.capsuleLeft = len(self.get_capsules(gameState))
            realLastCapsuleLen = self.prevCapsuleLeft
            realLastFoodLen = len(self.lastFood)

            # Set returned = 1 when pacman has secured some food and should to return back home           
            if len(self.presentFoodList) < len(self.lastFood):
                self.shouldReturn = True
            self.lastFood = self.presentFoodList
            self.prevCapsuleLeft = self.capsuleLeft

           
            if not gameState.get_agent_state(self.index).is_pacman:
                self.shouldReturn = False

            # checks the attack situation           
            remainingFoodList = self.get_food(gameState).as_list()
            remainingFoodSize = len(remainingFoodList)
    
        
            if remainingFoodSize == self.currentFoodSize:
                self.counter = self.counter + 1
            else:
                self.currentFoodSize = remainingFoodSize
                self.counter = 0
            if gameState.get_initial_agent_position(self.index) == gameState.get_agent_state(self.index).get_position():
                self.counter = 0
            if self.counter > 20:
                self.attack = True
            else:
                self.attack = False
            
            
            actionsBase = gameState.get_legal_actions(self.index)
            actionsBase.remove(Directions.STOP)

            # distance to closest enemy        
            distanceToEnemy = 999999
            enemies = [gameState.get_agent_state(i) for i in self.get_opponents(gameState)]
            invaders = [a for a in enemies if not a.is_pacman and a.get_position() != None and a.scared_timer == 0]
            if len(invaders) > 0:
                distanceToEnemy = min([self.get_maze_distance(self.presentCoordinates, a.get_position()) for a in invaders])
            
            '''
            Capsule eating:
            -> If there is capsule available then capsulePower is True.
            -> If enemy Distance is less than 5 then capsulePower is False.
            -> If pacman scored a food then return to home capsulePower is False.
            '''
            if self.capsuleLeft < realLastCapsuleLen:
                self.capsulePower = True
                self.eatenFood = 0
            if distanceToEnemy <= 5:
                self.capsulePower = False
            if (len(self.presentFoodList) < len (self.lastFood)):
                self.capsulePower = False

        
            if self.capsulePower:
                if not gameState.get_agent_state(self.index).is_pacman:
                    self.eatenFood = 0

                modeMinDistance = 999999

                if len(self.presentFoodList) < realLastFoodLen:
                    self.eatenFood += 1

                if len(self.presentFoodList )==0 or self.eatenFood >= 5:
                    self.targetMode = self.initPosition
        
                else:
                    for food in self.presentFoodList:
                        distance = self.get_maze_distance(self.presentCoordinates ,food)
                        if distance < modeMinDistance:
                            modeMinDistance = distance
                            self.targetMode = food

                legalActions = gameState.get_legal_actions(self.index)
                legalActions.remove(Directions.STOP)
                possibleActions = []
                distanceToTarget = []
                
                k=0
                while k!=len(legalActions):
                    a = legalActions[k]
                    newpos = (gameState.generate_successor(self.index, a)).get_agent_position(self.index)
                    possibleActions.append(a)
                    distanceToTarget.append(self.get_maze_distance(newpos, self.targetMode))
                    k+=1
                
                minDis = min(distanceToTarget)
                bestActions = [a for a, dis in zip(possibleActions, distanceToTarget) if dis== minDis]
                bestAction = random.choice(bestActions)
                return bestAction
            else:
                # Replace previous Monte Carlo simulation with MCTS
                self.eatenFood = 0
                
                try:
                    # First, try MCTS
                    bestAction = self.monte_carlo_tree_search(gameState, max_iterations=65)
                    return bestAction
                except Exception as e:
                    # Fallback to previous Monte Carlo method if MCTS fails
                    distanceToTarget = []
                    for a in actionsBase:
                        nextState = gameState.generate_successor(self.index, a)
                        value = 0
                        for i in range(1, 24):
                            value += self.monteCarloSimulation(nextState, 20)
                        distanceToTarget.append(value)

                    best = max(distanceToTarget)
                    bestActions = [a for a, v in zip(actionsBase, distanceToTarget) if v == best]
                    bestAction = random.choice(bestActions)
                    return bestAction


class DefensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.previousFood = []
        self.counter = 0

    def register_initial_state(self, gameState):
        CaptureAgent.register_initial_state(self, gameState)
        self.setPatrolPoint(gameState)

    def setPatrolPoint(self ,gameState):
        '''
        Look for center of the maze for patrolling
        '''
        x = (gameState.data.layout.width - 2) // 2
        if not self.red:
            x += 1
        self.patrolPoints = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.has_wall(x, i):
                self.patrolPoints.append((x, i))

        for i in range(len(self.patrolPoints)):
            if len(self.patrolPoints) > 2:
                self.patrolPoints.remove(self.patrolPoints[0])
                self.patrolPoints.remove(self.patrolPoints[-1])
            else:
                break
    

    def get_next_defensive_move(self ,gameState):

        agentActions = []
        actions = gameState.get_legal_actions(self.index)
        
        rev_dir = Directions.REVERSE[gameState.get_agent_state(self.index).configuration.direction]
        actions.remove(Directions.STOP)

        for i in range(0, len(actions)-1):
            if rev_dir == actions[i]:
                actions.remove(rev_dir)


        for i in range(len(actions)):
            a = actions[i]
            new_state = gameState.generate_successor(self.index, a)
            if not new_state.get_agent_state(self.index).is_pacman:
                agentActions.append(a)
        
        if len(agentActions) == 0:
            self.counter = 0
        else:
            self.counter = self.counter + 1
        if self.counter > 4 or self.counter == 0:
            agentActions.append(rev_dir)

        return agentActions

    def choose_action(self, gameState):
        
        position = gameState.get_agent_position(self.index)
        if position == self.target:
            self.target = None
        invaders = []
        nearestInvader = []
        minDistance = float("inf")


        # Look for enemy position in our home        
        opponentsPositions = self.get_opponents(gameState)
        i = 0
        while i != len(opponentsPositions):
            opponentPos = opponentsPositions[i]
            opponent = gameState.get_agent_state(opponentPos)
            if opponent.is_pacman and opponent.get_position() != None:
                opponentPos = opponent.get_position()
                invaders.append(opponentPos)
            i = i + 1

        # if enemy is found chase it and kill it
        if len(invaders) > 0:
            for oppPosition in invaders:
                dist = self.get_maze_distance(oppPosition ,position)
                if dist < minDistance:
                    minDistance = dist
                    nearestInvader.append(oppPosition)
            self.target = nearestInvader[-1]

        # if enemy has eaten some food, then remove it from targets
        else:
            if len(self.previousFood) > 0:
                if len(self.get_food_you_are_defending(gameState).as_list()) < len(self.previousFood):
                    yummy = set(self.previousFood) - set(self.get_food_you_are_defending(gameState).as_list())
                    self.target = yummy.pop()

        self.previousFood = self.get_food_you_are_defending(gameState).as_list()
        
        if self.target == None:
            if len(self.get_food_you_are_defending(gameState).as_list()) <= 4:
                highPriorityFood = self.get_food_you_are_defending(gameState).as_list() + self.get_capsules_you_are_defending(gameState)
                self.target = random.choice(highPriorityFood)
            else:
                self.target = random.choice(self.patrolPoints)
        candAct = self.get_next_defensive_move(gameState)
        awsomeMoves = []
        fvalues = []

        i=0
        
        # find the best move       
        while i < len(candAct):
            a = candAct[i]
            nextState = gameState.generate_successor(self.index, a)
            newpos = nextState.get_agent_position(self.index)
            awsomeMoves.append(a)
            fvalues.append(self.get_maze_distance(newpos, self.target))
            i = i + 1

        best = min(fvalues)
        bestActions = [a for a, v in zip(awsomeMoves, fvalues) if v == best]
        bestAction = random.choice(bestActions)
        return bestAction
