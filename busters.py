"""
busters.py
----------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).

Busters.py is a vengeful variant of Pacman where Pacman hunts ghosts, but
cannot see them.  Numbers at the bottom of the display are noisy distance
readings to each remaining ghost.

To play your first game, type 'python pacman.py' from the command line.
The keys are 'a', 's', 'd', and 'w' to move (or arrow keys).  Have fun!
"""

import os
import random
import sys

import graphics_display
import layout
import util
from game import Actions
from game import Configuration
from game import Directions
from game import Game
from game import GameStateData

__observation_distributions__ = {}

########################################
# Parameters for noisy sensor readings #
########################################

SONAR_NOISE_RANGE = 15  # Must be odd
SONAR_MAX = (SONAR_NOISE_RANGE - 1) / 2
SONAR_NOISE_VALUES = [i - SONAR_MAX for i in range(SONAR_NOISE_RANGE)]
SONAR_DENOMINATOR = 2 ** SONAR_MAX + 2 ** (SONAR_MAX + 1) - 2.0
SONAR_NOISE_PROBS = [2 ** (SONAR_MAX - abs(v)) / SONAR_DENOMINATOR for v in SONAR_NOISE_VALUES]


def get_noisy_distance(pos1, pos2):
    if pos2[1] == 1:
        return None
    distance = util.manhattan_distance(pos1, pos2)
    return max(0, distance + util.sample(SONAR_NOISE_PROBS, SONAR_NOISE_VALUES))


def get_observation_distribution(noisy_distance):
    """
    Returns the factor P( noisyDistance | TrueDistances ), the likelihood of the
    provided noisy_distance conditioned upon all the possible true distances that could have
    generated it.
    """
    global __observation_distributions__
    if noisy_distance is None:
        return util.Counter()
    if noisy_distance not in __observation_distributions__:
        distribution = util.Counter()
        for error, prob in zip(SONAR_NOISE_VALUES, SONAR_NOISE_PROBS):
            distribution[max(1, noisy_distance - error)] += prob
        __observation_distributions__[noisy_distance] = distribution
    return __observation_distributions__[noisy_distance]


###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

class GameState:
    """
    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes.

    GameStates are used by the Game object to capture the actual state of the game and
    can be used by agents to reason about the game.

    Much of the information in a GameState is stored in a GameStateData object.  We
    strongly suggest that you access that data via the accessor methods below rather
    than referring to the GameStateData object directly.

    Note that in classic Pacman, Pacman is always agent 0.
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    def get_legal_actions(self, agent_index=0):
        """
        Returns the legal actions for the agent specified.
        """
        if self.is_win() or self.is_lose():
            return []

        if agent_index == 0:  # Pacman is moving
            return PacmanRules.get_legal_actions(self)

        return GhostRules.get_legal_actions(self, agent_index)

    def generate_successor(self, agent_index, action):
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.is_win() or self.is_lose():
            raise Exception('Can\'t generate a successor of a terminal state.')

        # Copy current state
        state = GameState(self)

        # Let agent's logic deal with its action's effects on the board
        if agent_index == 0:  # Pacman is moving
            state.data.eaten = [False for _ in range(state.get_num_agents())]
            PacmanRules.apply_action(state, action)
        else:  # A ghost is moving
            GhostRules.apply_action(state, action, agent_index)

        # Time passes
        if agent_index == 0:
            state.data.score_change += -TIME_PENALTY  # Penalty for waiting around
        else:
            GhostRules.decrement_timer(state.data.agent_states[agent_index])

        # Resolve multi-agent effects
        GhostRules.check_death(state, agent_index)

        # Book keeping
        state.data._agent_moved = agent_index
        state.data.score += state.data.score_change
        p = state.get_pacman_position()
        state.data.ghost_distances = [get_noisy_distance(p, state.get_ghost_position(i)) for i in
                                      range(1, state.get_num_agents())]
        if agent_index == self.get_num_agents() - 1:
            state.num_moves += 1
        return state

    def get_legal_pacman_actions(self):
        return self.get_legal_actions(0)

    def generate_pacman_successor(self, action):
        """
        Generates the successor state after the specified pacman move
        """
        return self.generate_successor(0, action)

    def get_pacman_state(self):
        """
        Returns an AgentState object for pacman (in game.py)

        state.pos gives the current position
        state.direction gives the travel vector
        """
        return self.data.agent_states[0].copy()

    def get_pacman_position(self):
        return self.data.agent_states[0].get_position()

    def get_num_agents(self):
        return len(self.data.agent_states)

    def get_score(self):
        return self.data.score

    def get_capsules(self):
        """
        Returns a list of positions (x,y) of the remaining capsules.
        """
        return self.data.capsules

    def get_num_food(self):
        return self.data.food.count()

    def get_food(self):
        """
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        currentFood = state.get_food()
        if currentFood[x][y] == True: ...
        """
        return self.data.food

    def get_walls(self):
        """
        Returns a Grid of boolean wall indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        walls = state.get_walls()
        if walls[x][y] == True: ...
        """
        return self.data.layout.walls

    def has_food(self, x, y):
        return self.data.food[x][y]

    def has_wall(self, x, y):
        return self.data.layout.walls[x][y]

    ##############################
    # Additions for Busters Pacman #
    ##############################

    def get_living_ghosts(self):
        """
        Returns a list of booleans indicating which ghosts are not yet captured.

        The first entry (a placeholder for Pacman's index) is always False.
        """
        return self.living_ghosts

    def set_ghost_not_living(self, index):
        self.living_ghosts[index] = False

    def is_lose(self):
        return 0 < self.max_moves <= self.num_moves

    def is_win(self):
        return self.living_ghosts.count(True) == 0

    def get_noisy_ghost_distances(self):
        """
        Returns a noisy distance to each ghost.
        """
        return self.data.ghost_distances

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__(self, prev_state=None):
        """
        Generates a new state by copying information from its predecessor.
        """
        if prev_state is not None:
            self.data = GameStateData(prev_state.data)
            self.living_ghosts = prev_state.living_ghosts[:]
            self.num_moves = prev_state.num_moves
            self.max_moves = prev_state.max_moves
        else:  # Initial state
            self.data = GameStateData()
            self.num_moves = 0
            self.max_moves = -1
        self.data.ghost_distances = []

    def deep_copy(self):
        state = GameState(self)
        state.data = self.data.deep_copy()
        state.data.ghost_distances = self.data.ghost_distances
        return state

    def __eq__(self, other):
        """
        Allows two states to be compared.
        """
        return self.data == other.data

    def __hash__(self):
        """
        Allows states to be keys of dictionaries.
        """
        return hash(str(self))

    def __str__(self):

        return str(self.data)

    def initialize(self, this_layout, num_ghost_agents=1000):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.data.initialize(this_layout, num_ghost_agents)
        self.living_ghosts = [False] + [True for _ in range(num_ghost_agents)]
        self.data.ghost_distances = [
            get_noisy_distance(self.get_pacman_position(), self.get_ghost_position(i)) for i in
            range(1, self.get_num_agents())]

    def get_ghost_position(self, agent_index):
        if agent_index == 0:
            raise Exception("Pacman's index passed to get_ghost_position")
        return self.data.agent_states[agent_index].get_position()

    def get_ghost_state(self, agent_index):
        if agent_index == 0:
            raise Exception("Pacman's index passed to get_ghost_position")
        return self.data.agent_states[agent_index]


############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

COLLISION_TOLERANCE = 0.7  # How close ghosts must be to Pacman to kill
TIME_PENALTY = 1  # Number of points lost each round


class BustersGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def new_game(self, this_layout, pacman_agent, ghost_agents, display, max_moves=-1):
        agents = [pacman_agent] + ghost_agents
        init_state = GameState()
        init_state.initialize(this_layout, len(ghost_agents))
        game = Game(agents, display, self)
        game.state = init_state
        game.state.max_moves = max_moves
        return game

    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if state.is_win():
            self.win(state, game)
        if state.is_lose():
            self.lose(state, game)

    def win(self, _, game):
        game.game_over = True

    def lose(self, _, game):
        game.game_over = True


class PacmanRules:
    """
    These functions govern how pacman interacts with his environment under
    the classic game rules.
    """

    @staticmethod
    def get_legal_actions(state):
        """
        Returns a list of possible actions.
        """
        return Actions.get_possible_actions(state.get_pacman_state().configuration,
                                            state.data.layout.walls)

    @staticmethod
    def apply_action(state, action):
        """
        Edits the state to reflect the results of the action.
        """
        legal = PacmanRules.get_legal_actions(state)
        if action not in legal:
            raise Exception("Illegal action", action)

        pacman_state = state.data.agent_states[0]

        # Update Configuration
        vector = Actions.direction_to_vector(action, 1)
        pacman_state.configuration = pacman_state.configuration.generate_successor(vector)


class GhostRules:
    """
    These functions dictate how ghosts interact with their environment.
    """

    @staticmethod
    def get_legal_actions(state, ghost_index):
        conf = state.get_ghost_state(ghost_index).configuration
        return Actions.get_possible_actions(conf, state.data.layout.walls)

    @staticmethod
    def apply_action(state, action, ghost_index):
        legal = GhostRules.get_legal_actions(state, ghost_index)
        if action not in legal:
            raise Exception("Illegal ghost action: " + str(action))

        ghost_state = state.data.agent_states[ghost_index]
        vector = Actions.direction_to_vector(action, 1)
        ghost_state.configuration = ghost_state.configuration.generate_successor(vector)

    @staticmethod
    def decrement_timer(ghost_state):
        timer = ghost_state.scared_timer
        if timer == 1:
            ghost_state.configuration.pos = util.nearest_point(ghost_state.configuration.pos)
        ghost_state.scaredTimer = max(0, timer - 1)

    @staticmethod
    def check_death(state, agent_index):
        pacman_position = state.get_pacman_position()
        if agent_index == 0:  # Pacman just moved; Anyone can kill him
            for index in range(1, len(state.data.agent_states)):
                ghost_state = state.data.agent_states[index]
                ghost_position = ghost_state.configuration.get_position()
                if GhostRules.can_kill(pacman_position, ghost_position):
                    GhostRules.collide(state, ghost_state, index)
        else:
            ghost_state = state.data.agent_states[agent_index]
            ghost_position = ghost_state.configuration.get_position()
            if GhostRules.can_kill(pacman_position, ghost_position):
                GhostRules.collide(state, ghost_state, agent_index)

    @staticmethod
    def collide(state, ghost_state, agent_index):
        state.data.score_change += 200
        GhostRules.place_ghost(ghost_state, agent_index)
        # Added for first-person
        state.data.eaten[agent_index] = True
        state.set_ghost_not_living(agent_index)

    @staticmethod
    def can_kill(pacman_position, ghost_position):
        return util.manhattan_distance(ghost_position, pacman_position) <= COLLISION_TOLERANCE

    @staticmethod
    def place_ghost(ghost_state, agent_index):
        pos = (agent_index * 2 - 1, 1)
        direction = Directions.STOP
        ghost_state.configuration = Configuration(pos, direction)


class RandomGhost:
    def __init__(self, index):
        self.index = index

    def get_action(self, state):
        return random.choice(state.get_legal_actions(self.index))

    def get_distribution(self, state):
        actions = state.get_legal_actions(self.index)
        prob = 1.0 / len(actions)
        return [(prob, action) for action in actions]


#############################
# FRAMEWORK TO START A GAME #
#############################

def default(string):
    return string + ' [Default: %default]'


def parse_agent_args(string):
    if string is None:
        return {}
    pieces = string.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key, val = p, 1
        opts[key] = val
    return opts


def read_command(argv):
    """
    Processes the command used to run pacman from the command line.
    """
    # todo fix with argparser
    from optparse import OptionParser
    usage_str = """
    USAGE:      python busters.py <options>
    EXAMPLE:    python busters.py --layout bigHunt
                  - starts an interactive game on a big board
    """
    parser = OptionParser(usage_str)

    parser.add_option('-n', '--numGames', dest='numGames', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=1)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='oneHunt')
    parser.add_option('-p', '--pacman', dest='pacman',
                      help=default('the agent TYPE in the pacmanAgents module to use'),
                      metavar='TYPE', default='BustersKeyboardAgent')
    parser.add_option('-a', '--agentArgs', dest='agentArgs',
                      help='Comma seperated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_option('-g', '--ghosts', dest='ghost',
                      help=default('the ghost agent TYPE in the ghost_agents module to use'),
                      metavar='TYPE', default='RandomGhost')
    parser.add_option('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('-k', '--numGhosts', type='int', dest='num_ghosts',
                      help=default('The maximum number of ghosts to use'), default=4)
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('Zoom the size of the graphics window'), default=1.0)
    parser.add_option('-f', '--fixRandomSeed', action='store_true', dest='fixRandomSeed',
                      help='Fixes the random seed to always play the same game', default=False)
    parser.add_option('-s', '--showGhosts', action='store_true', dest='showGhosts',
                      help='Renders the ghosts in the display (cheating)', default=False)
    parser.add_option('-t', '--frameTime', dest='frameTime', type='float',
                      help=default('Time to delay between frames; <0 means keyboard'), default=0.1)

    options, other_junk = parser.parse_args()
    if len(other_junk) != 0:
        raise Exception('Command line input not understood: ' + other_junk)
    args = dict()

    # Fix the random seed
    if options.fixRandomSeed:
        random.seed('bustersPacman', version=1)

    # Choose a layout
    args['layout'] = layout.get_layout(options.layout)
    if args['layout'] is None:
        raise Exception("The layout " + options.layout + " cannot be found")

    # Choose a ghost agent
    ghost_type = load_agent(options.ghost, options.quietGraphics)
    args['ghosts'] = [ghost_type(i + 1) for i in range(options.num_ghosts)]

    # Choose a Pacman agent
    no_keyboard = options.quietGraphics
    pacman_type = load_agent(options.pacman, no_keyboard)
    agent_opts = parse_agent_args(options.agentArgs)
    agent_opts['ghost_agents'] = args['ghosts']
    pacman = pacman_type(**agent_opts)  # Instantiate Pacman with agentArgs
    args['pacman'] = pacman

    args['display'] = graphics_display.FirstPersonPacmanGraphics(options.zoom,
                                                                 options.showGhosts,
                                                                 frame_time=options.frameTime)
    args['num_games'] = options.numGames

    return args


def load_agent(pacman, nographics):
    # Looks through all pythonPath Directories for the right module,
    python_path_str = os.path.expandvars("$PYTHONPATH")
    if python_path_str.find(';') == -1:
        python_path_dirs = python_path_str.split(':')
    else:
        python_path_dirs = python_path_str.split(';')
    python_path_dirs.append('.')

    for module_dir in python_path_dirs:
        if not os.path.isdir(module_dir):
            continue
        module_names = [f for f in os.listdir(module_dir) if f.endswith('gents.py')]
        for module_name in module_names:
            try:
                module = __import__(module_name[:-3])
            except ImportError:
                continue
            if pacman in dir(module):
                if nographics and module_name == 'keyboard_agents.py':
                    raise Exception('Using the keyboard requires graphics (not text display)')
                return getattr(module, pacman)
    raise Exception('The agent ' + pacman + ' is not specified in any *Agents.py.')


def run_games(layout, pacman, ghosts, display, num_games, max_moves=-1):
    # Hack for agents writing to the display
    import __main__
    __main__.__dict__['_display'] = display

    rules = BustersGameRules()
    games = []

    for _ in range(num_games):
        game = rules.new_game(layout, pacman, ghosts, display, max_moves)
        game.run()
        games.append(game)

    if num_games > 1:
        scores = [game.state.get_score() for game in games]
        wins = [game.state.is_win() for game in games]
        win_rate = wins.count(True) / float(len(wins))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print('Win Rate:      %d/%d (%.2f)' % (wins.count(True), len(wins), win_rate))
        print('Record:       ', ', '.join([['Loss', 'Win'][int(w)] for w in wins]))

    return games


if __name__ == '__main__':
    # The main function called when pacman.py is run
    # from the command line:
    #
    # > python pacman.py
    #
    # See the usage string for more details.
    #
    # > python pacman.py --help
    args = read_command(sys.argv[1:])  # Get game components based on input
    run_games(**args)
