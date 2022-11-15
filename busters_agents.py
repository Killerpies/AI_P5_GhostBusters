"""
busters_agents.py
----------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""

from inference import *
import util
from keyboard_agents import KeyboardAgent
from distance_calculator import Distancer
from game import Actions
from game import Directions


class NullGraphics:
    """Placeholder for graphics"""

    def initialize(self, state, is_blue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def update_distributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def __init__(self, ghost_agent):
        super().__init__(ghost_agent)
        self.beliefs = util.Counter()

    def initialize_uniformly(self, game_state):
        """Begin with a uniform distribution over ghost positions."""
        self.beliefs = util.Counter()
        for p in self.legal_positions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, game_state):
        noisy_distance = observation
        emission_model = busters.get_observation_distribution(noisy_distance)
        pacman_position = game_state.get_pacman_position()
        all_possible = util.Counter()
        for p in self.legal_positions:
            true_distance = util.manhattan_distance(p, pacman_position)
            if emission_model[true_distance] > 0:
                all_possible[p] = 1.0
        all_possible.normalize()
        self.beliefs = all_possible

    def elapse_time(self, game_state):
        pass

    def get_belief_distribution(self):
        return self.beliefs


class BustersAgent:
    """An agent that tracks and displays its beliefs about ghost positions."""

    def __init__(self, index=0, inference="ExactInference", ghost_agents=None, observe_enable=True,
                 elapse_time_enable=True):
        if isinstance(inference, list):
            inference = type(inference[0])
        if isinstance(inference, str):
            inference = eval(inference)

        self.inference_modules = [inference(a) for a in ghost_agents]
        self.observeEnable = observe_enable
        self.elapseTimeEnable = elapse_time_enable

    def register_initial_state(self, game_state):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inference_modules:
            inference.initialize(game_state)
        self.ghost_beliefs = [inf.get_belief_distribution() for inf in self.inference_modules]
        self.first_move = True

    def observation_function(self, game_state):
        """Removes the ghost states from the game_state"""
        agents = game_state.data.agent_states
        game_state.data.agent_states = [agents[0]] + [None for _ in range(1, len(agents))]
        return game_state

    def get_action(self, game_state):
        """Updates beliefs, then chooses an action based on updated beliefs."""
        for index, inf in enumerate(self.inference_modules):
            if not self.first_move and self.elapseTimeEnable:
                inf.elapse_time(game_state)
            self.first_move = False
            if self.observeEnable:
                inf.observe_state(game_state)
            self.ghost_beliefs[index] = inf.get_belief_distribution()
        self.display.update_distributions(self.ghost_beliefs)
        return self.choose_action(game_state)

    def choose_action(self, game_state):
        """By default, a BustersAgent just stops.  This should be overridden."""
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index=0, inference="KeyboardInference", ghost_agents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghost_agents)

    def get_action(self, game_state):
        return BustersAgent.get_action(self, game_state)

    def choose_action(self, game_state):
        return KeyboardAgent.get_action(self, game_state)


class GreedyBustersAgent(BustersAgent):
    """An agent that charges the closest ghost."""

    def register_initial_state(self, game_state):
        """Pre-computes the distance between every two points."""
        BustersAgent.register_initial_state(self, game_state)
        self.distancer = Distancer(game_state.data.layout, False)

    def choose_action(self, game_state):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.get_distance(pos1, pos2)

        To find the successor position of a position after an action:
          successor_position = Actions.getSuccessor(position, action)

        living_ghost_position_distributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) game_state.get_living_ghosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             game_state.get_living_ghosts() list.
        """
        pacman_position = game_state.get_pacman_position()
        living_ghosts = game_state.get_living_ghosts()
        living_ghost_position_distributions = \
            [beliefs for i, beliefs in enumerate(self.ghost_beliefs)
             if living_ghosts[i + 1]]
        "*** YOUR CODE HERE ***"
        value = util.Counter()
        actions = game_state.get_legal_pacman_actions()
        # loop pacman positions
        for action in actions:
            # defailt val of 0 for all actions
            value[action] = 0
            # get next pacman positions
            successorPosition = Actions.get_successor(pacman_position, action)
            for ghost in living_ghost_position_distributions:
                # most likely location for ghost is the max val in util.counter()
                ghostLikelyPosition = ghost.arg_max()
                distance = self.distancer.get_distance(ghostLikelyPosition, successorPosition)
                # assign distance to action
                value[action] = distance

        # find and return smallest distance action
        minVal = float("inf")
        bestAction = None
        for action, actionVal in value.items():
            if minVal > actionVal:
                minVal = actionVal
                bestAction = action
        return bestAction

        # print(pacmanActions)
        # print(pacmanActions)
        #self.distancer.get_distance(pos1, pos2)
        #successor_position = Actions.getSuccessor(position, action)
        # util.raise_not_defined()
