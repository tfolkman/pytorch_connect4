import numpy as np
from utils.logging import setup_logger
from config import RUN_FOLDER, LOGGER_DISABLED, EPSILON, ALPHA, CPUCT

logger_mcts = setup_logger('logger_mcts', RUN_FOLDER + 'logs/logger_mcts.log')
logger_mcts.disabled = LOGGER_DISABLED['mcts']


class Node():
    def __init__(self, state):
        self.state = state
        self.player_turn = state.playerTurn
        self.id = state.id
        self.edges = []

    def is_leaf(self):
        return len(self.edges) <= 0


class Edge():
    def __init__(self, in_node, out_node, prior, action):
        self.id = in_node.state.id + '|' + out_node.state.id
        self.inNode = in_node
        self.outNode = out_node
        self.playerTurn = in_node.state.playerTurn
        self.action = action

        # n is the number of times visited

        # p is the prior from the policy network over probability of making each move

        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior,
        }


class MCTS():
    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.addNode(root)

    def __len__(self):
        return len(self.tree)

    def move_to_leaf(self):
        breadcrumbs = []
        currentNode = self.root

        done = 0
        value = 0

        while not currentNode.is_leaf():

            maxQU = -99999

            # on root node introduce noise to the priors with a weight of epsilon
            if currentNode == self.root:
                epsilon = EPSILON
                nu = np.random.dirichlet([ALPHA] * len(currentNode.edges))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.edges)

            # sum up the total times all the nodes on the edges have been visited
            # will have edges equal to the number of allowed actions
            Nb = 0
            for action, edge in currentNode.edges:
                Nb = Nb + edge.stats['N']

            # iterate over all possible actions
            for idx, (action, edge) in enumerate(currentNode.edges):

                # the exploitation term is the probability of this action from the model
                # if it is the root node, noise is added
                exploitation_term = (1-epsilon) * edge.stats['P'] + epsilon * nu[idx]

                # The numeration is proportional to the total number every edeg node was visited
                # The denominator is the number of times this edge node was visited
                # Thus, gets larger for nodes less visited
                exploration_term = np.sqrt(Nb) / (1 + edge.stats['N'])

                U = CPUCT * exploitation_term * exploration_term
                # this is the average game result across current simulations that took this action
                Q = edge.stats['Q']

                # take the move that maximizes Q+U
                if Q+U > maxQU:
                    maxQU = Q+U
                    simulation_action = action
                    simulation_edge = edge


