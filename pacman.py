"""
pacman.py
---------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).

Pacman.py holds the logic for the classic pacman game along with the main
code to run a game.  This file is divided into three sections:

  (i)  Your interface to the pacman world:
          Pacman is a complex environment.  You probably don't want to
          read through all of the code we wrote to make the game runs
          correctly.  This section contains the parts of the code
          that you will need to understand in order to complete the
          project.  There is also some code in game.py that you should
          understand.

  (ii)  The hidden secrets of pacman:
          This section contains all of the logic code that the pacman
          environment uses to decide who can move where, who dies when
          things collide, etc.  You shouldn't need to read this section
          of code, but you can if you want.

  (iii) Framework to start a game:
          The final section contains the code for reading the command
          you use to set up the game, then starting up a new game, along with
          linking in all the external parts (agent functions, graphics).
          Check this section out to see all the options available to you.

To play your first game, type 'python pacman.py' from the command line.
The keys are 'a', 's', 'd', and 'w' to move (or arrow keys).  Have fun!
"""
import os
import pickle
import random
import sys
import time
from argparse import ArgumentParser

import ghost_agents
import graphics_display
import pacman_agents
import text_display
from game import Actions, Directions, Game, GameStateData
from layout import get_layout
from util import manhattan_distance, nearest_point


###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

class GameState:
    """
    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes.

    GameStates are used by the Game this_ojb to capture the actual state of the game and
    can be used by agents to reason about the game.

    Much of the information in a GameState is stored in a GameStateData this_ojb.  We
    strongly suggest that you access that data via the accessor methods below rather
    than referring to the GameStateData this_ojb directly.

    Note that in classic Pacman, Pacman is always agent 0.
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    # static variable keeps track of which states have had get_legal_actions called
    explored = set()

    @staticmethod
    def get_and_reset_explored():
        tmp = GameState.explored.copy()
        GameState.explored = set()
        return tmp

    def get_legal_actions(self, agent_index=0):
        """
        Returns the legal actions for the agent specified.
        """
        #        GameState.explored.add(self)
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
            state.data.score_change -= TIME_PENALTY  # Penalty for waiting around
        else:
            GhostRules.decrement_timer(state.data.agent_states[agent_index])

        # Resolve multi-agent effects
        GhostRules.check_death(state, agent_index)

        # Book keeping
        state.data._agent_moved = agent_index
        state.data.score += state.data.score_change
        GameState.explored.add(self)
        GameState.explored.add(state)
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
        Returns an AgentState this_ojb for pacman (in game.py)

        state.pos gives the current position
        state.direction gives the travel vector
        """
        return self.data.agent_states[0].copy()

    def get_pacman_position(self):
        return self.data.agent_states[0].get_position()

    def get_ghost_states(self):
        return self.data.agent_states[1:]

    def get_ghost_state(self, agent_index):
        if agent_index == 0 or agent_index >= self.get_num_agents():
            raise Exception('Invalid index passed to get_ghost_state')
        return self.data.agent_states[agent_index]

    def get_ghost_position(self, agent_index):
        if agent_index == 0:
            raise Exception('Pacman\'s index passed to get_ghost_position')
        return self.data.agent_states[agent_index].get_position()

    def get_ghost_positions(self):
        return [s.get_position() for s in self.get_ghost_states()]

    def get_num_agents(self):
        return len(self.data.agent_states)

    def get_score(self):
        return float(self.data.score)

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
        if there is a wall at (x,y), just call

        walls = state.get_walls()
        if walls[x][y] == True: ...
        """
        return self.data.layout.walls

    def has_food(self, x, y):
        return self.data.food[x][y]

    def has_wall(self, x, y):
        return self.data.layout.walls[x][y]

    def is_lose(self):
        return self.data._lose

    def is_win(self):
        return self.data._win

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__(self, prev_state=None):
        """
        Generates a new state by copying information from its predecessor.
        """
        if prev_state is not None:  # Initial state
            self.data = GameStateData(prev_state.data)
        else:
            self.data = GameStateData()

    def deep_copy(self):
        state = GameState(self)
        state.data = self.data.deep_copy()
        return state

    def __eq__(self, other):
        """
        Allows two states to be compared.
        """
        return hasattr(other, 'data') and self.data == other.data

    def __hash__(self):
        """
        Allows states to be keys of dictionaries.
        """
        return hash(self.data)

    def __str__(self):

        return str(self.data)

    def initialize(self, layout, num_ghost_agents=1000):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.data.initialize(layout, num_ghost_agents)


############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

SCARED_TIME = 40  # Moves ghosts are scared
COLLISION_TOLERANCE = 0.7  # How close ghosts must be to Pacman to kill
TIME_PENALTY = 1  # Number of points lost each round


def agent_crash(agent_index):
    if agent_index == 0:
        print("Pacman crashed")
    else:
        print("A ghost crashed")


class ClassicGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def __init__(self, timeout=30):
        self.timeout = timeout
        self.initial_state = None
        self.quiet = False

    def new_game(self, layout, pacman, ghosts,
                 display, quiet=False, catch_exceptions=False):
        agents = [pacman] + ghosts[:layout.get_num_ghosts()]
        init_state = GameState()
        init_state.initialize(layout, len(ghosts))
        game = Game(agents, display, self, catch_exceptions=catch_exceptions)
        game.state = init_state
        self.initial_state = init_state.deep_copy()
        self.quiet = quiet
        return game

    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if state.is_win():
            self.win(state, game)
        if state.is_lose():
            self.lose(state, game)

    def win(self, state, game):
        if not self.quiet:
            print(f'Pacman emerges victorious! Score: {state.data.score}')
        game.game_over = True

    def lose(self, state, game):
        if not self.quiet:
            print(f'Pacman died! Score: {state.data.score}')
        game.game_over = True

    def get_progress(self, game):
        return float(game.state.get_num_food()) / self.initial_state.get_num_food()

    def get_max_total_time(self):
        return self.timeout

    def get_max_startup_time(self):
        return self.timeout

    def get_move_warning_time(self):
        return self.timeout

    def get_move_timeout(self):
        return self.timeout

    def get_max_time_warnings(self):
        return 0


class PacmanRules:
    """
    These functions govern how pacman interacts with his environment under
    the classic game rules.
    """
    PACMAN_SPEED = 1

    @staticmethod
    def get_legal_actions(state):
        """
        Returns a list of possible actions.
        """
        legal_actions = Actions.get_possible_actions(
            state.get_pacman_state().configuration, state.data.layout.walls)
        return legal_actions

    @staticmethod
    def apply_action(state, action):
        """
        Edits the state to reflect the results of the action.
        """
        legal = PacmanRules.get_legal_actions(state)
        if action not in legal:
            raise Exception("Illegal action " + str(action))

        pacman_state = state.data.agent_states[0]

        # Update Configuration
        vector = Actions.direction_to_vector(action, PacmanRules.PACMAN_SPEED)
        pacman_state.configuration = pacman_state.configuration.generate_successor(vector)

        # Eat
        next_position = pacman_state.configuration.get_position()
        nearest = nearest_point(next_position)
        if manhattan_distance(nearest, next_position) <= 0.5:
            # Remove food
            PacmanRules.consume(nearest, state)

    @staticmethod
    def consume(position, state):
        x, y = position
        # Eat food
        if state.data.food[x][y]:
            state.data.score_change += 10
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._food_eaten = position
            # TODO: cache num_food?
            num_food = state.get_num_food()
            if num_food == 0 and not state.data._lose:
                state.data.score_change += 500
                state.data._win = True
        # Eat capsule
        if position in state.get_capsules():
            state.data.capsules.remove(position)
            state.data.capsule_eaten = position
            state.data.capsule_eaten = position
            # Reset all ghosts' scared timers
            for index in range(1, len(state.data.agent_states)):
                state.data.agent_states[index].scared_timer = SCARED_TIME


class GhostRules:
    """
    These functions dictate how ghosts interact with their environment.
    """
    GHOST_SPEED = 1.0

    @staticmethod
    def get_legal_actions(state, ghost_index):
        """
        Ghosts cannot stop, and cannot turn around unless they
        reach a dead end, but can turn 90 degrees at intersections.
        """
        conf = state.get_ghost_state(ghost_index).configuration
        possible_actions = Actions.get_possible_actions(conf, state.data.layout.walls)
        reverse = Actions.reverse_direction(conf.direction)
        if Directions.STOP in possible_actions:
            possible_actions.remove(Directions.STOP)
        if reverse in possible_actions and len(possible_actions) > 1:
            possible_actions.remove(reverse)
        return possible_actions

    @staticmethod
    def apply_action(state, action, ghost_index):

        legal = GhostRules.get_legal_actions(state, ghost_index)
        if action not in legal:
            raise Exception("Illegal ghost action " + str(action))

        ghost_state = state.data.agent_states[ghost_index]
        speed = GhostRules.GHOST_SPEED
        if ghost_state.scared_timer > 0:
            speed /= 2.0
        vector = Actions.direction_to_vector(action, speed)
        ghost_state.configuration = ghost_state.configuration.generate_successor(vector)

    @staticmethod
    def decrement_timer(ghost_state):
        timer = ghost_state.scared_timer
        if timer == 1:
            ghost_state.configuration.pos = nearest_point(ghost_state.configuration.pos)
        ghost_state.scared_timer = max(0, timer - 1)

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
        if ghost_state.scared_timer > 0:
            state.data.score_change += 200
            GhostRules.place_ghost(ghost_state)
            ghost_state.scared_timer = 0
            # Added for first-person
            state.data.eaten[agent_index] = True
        else:
            if not state.data._win:
                state.data.score_change -= 500
                state.data._lose = True

    @staticmethod
    def can_kill(pacman_position, ghost_position):
        return manhattan_distance(ghost_position, pacman_position) <= COLLISION_TOLERANCE

    @staticmethod
    def place_ghost(ghost_state):
        ghost_state.configuration = ghost_state.start


#############################
# FRAMEWORK TO START A GAME #
#############################

def default(some_string):
    return some_string + ' [Default: %default]'


def parse_agent_args(arguments):
    if arguments is None:
        return {}
    pieces = arguments.split(',')
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

    usage_str = """
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python pacman.py
                    - starts an interactive game
                (2) python pacman.py --layout smallClassic --zoom 2
                OR  python pacman.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = ArgumentParser(usage_str)

    parser.add_argument('-n',
                        '--num_games',
                        dest='num_games',
                        type=int,
                        help=default('the number of GAMES to play'),
                        metavar='GAMES', default=1)
    parser.add_argument('-l',
                        '--layout',
                        dest='layout',
                        help=default('the LAYOUT_FILE from which to load the map layout'),
                        metavar='LAYOUT_FILE',
                        default='medium_classic')
    parser.add_argument('-p',
                        '--pacman',
                        dest='pacman',
                        help=default('the agent TYPE in the pacmanAgents module to use'),
                        metavar='TYPE',
                        default='KeyboardAgent')
    parser.add_argument('-t',
                        '--text_graphics',
                        action='store_true',
                        dest='text_graphics',
                        help='Display output as text_on_canvas only',
                        default=False)
    parser.add_argument('-q',
                        '--quiet_text_graphics',
                        action='store_true',
                        dest='quiet_graphics',
                        help='Generate minimal output and no graphics',
                        default=False)
    parser.add_argument('-g',
                        '--ghosts',
                        dest='ghost',
                        help=default('the ghost agent TYPE in the ghost_agents module to use'),
                        metavar='TYPE',
                        default='RandomGhost')
    parser.add_argument('-k',
                        '--num_ghosts',
                        type=int,
                        dest='num_ghosts',
                        help=default('The maximum number of ghosts to use'),
                        default=4)
    parser.add_argument('-z',
                        '--zoom',
                        type=float,
                        dest='zoom',
                        help=default('Zoom the size of the graphics window'),
                        default=1.0)
    parser.add_argument('-f',
                        '--fixRandomSeed',
                        action='store_true',
                        dest='fix_random_seed',
                        help='Fixes the random seed to always play the same game',
                        default=False)
    parser.add_argument('-r',
                        '--record_actions',
                        action='store_true',
                        dest='record',
                        help='Writes game histories to a file (named by the time they were played)',
                        default=False)
    parser.add_argument('--replay',
                        dest='game_to_replay',
                        help='A recorded game file (pickle) to replay',
                        default=None)
    parser.add_argument('-a',
                        '--agent_args',
                        dest='agent_args',
                        help='Comma separated values sent to agent. e.g. '
                             '"opt1=val1,opt2,opt3=val3"')
    parser.add_argument('-x',
                        '--num_training',
                        dest='num_training',
                        type=int,
                        help=default('How many episodes are training (suppresses output)'),
                        default=0)
    parser.add_argument('--frame_time',
                        dest='frame_time',
                        type=float,
                        help=default('Time to delay between frames; <0 means keyboard'),
                        default=0.1)
    parser.add_argument('-c',
                        '--catch_exceptions',
                        action='store_true',
                        dest='catch_exceptions',
                        help='Turns on exception handling and timeouts during games',
                        default=False)
    parser.add_argument('--timeout',
                        dest='timeout',
                        type=int,
                        help=default('Maximum length of time an agent can spend '
                                     'computing in a single game'),
                        default=30)

    # Outdated module. New module only passes args. Thus there is no need to check for 'other_junk'
    # options, other_junk = parser.parse_args(argv)
    # if len(other_junk) != 0:
    #     raise Exception('Command line input not understood: ' + str(other_junk))

    options = parser.parse_args(argv)
    arguments = dict()

    # Fix the random seed
    if options.fix_random_seed:
        random.seed('cs188')

    # Choose a layout
    arguments['layout'] = get_layout(options.layout)
    if arguments['layout'] is None:
        raise Exception(f'The layout {options.layout} cannot be found')

    # Choose a Pacman agent
    no_keyboard = options.game_to_replay is None and (options.text_graphics or
                                                      options.quiet_graphics)
    pacman_type = load_agent(options.pacman, no_keyboard)
    agent_opts = parse_agent_args(options.agent_args)
    if options.num_training > 0:
        arguments['num_training'] = options.num_training
        if 'num_training' not in agent_opts:
            agent_opts['num_training'] = options.num_training
    pacman = pacman_type(**agent_opts)  # Instantiate Pacman with agent_args
    arguments['pacman'] = pacman

    # Don't display training games
    if 'numTrain' in agent_opts:
        options.numQuiet = int(agent_opts['numTrain'])
        options.numIgnore = int(agent_opts['numTrain'])

    # Choose a ghost agent
    ghost_type = load_agent(options.ghost, no_keyboard)
    arguments['ghosts'] = [ghost_type(i + 1) for i in range(options.num_ghosts)]

    # Choose a display format
    if options.quiet_graphics:
        # import text_display
        arguments['display'] = text_display.NullGraphics()
    elif options.text_graphics:
        # import text_display
        text_display.SLEEP_TIME = options.frame_time
        arguments['display'] = text_display.PacmanGraphics()
    else:
        # import graphics_display
        arguments['display'] = graphics_display.PacmanGraphics(
            options.zoom, frame_time=options.frame_time)
    arguments['num_games'] = options.num_games
    arguments['record'] = options.record
    arguments['catch_exceptions'] = options.catch_exceptions
    arguments['timeout'] = options.timeout

    # Special case: recorded games don't use the run_games method or args structure
    if options.game_to_replay is not None:
        print('Replaying recorded game %s.' % options.game_to_replay)
        f = open(options.game_to_replay)
        try:
            recorded = pickle.load(f)
        finally:
            f.close()
        recorded['display'] = arguments['display']
        replay_game(**recorded)
        sys.exit(0)

    return arguments


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
                    raise Exception(
                        'Using the keyboard requires graphics (not text_on_canvas display)')

                return getattr(module, pacman)

    raise Exception(f'The agent {pacman} is not specified in any *Agents.py.')


def replay_game(layout, actions, display):
    rules = ClassicGameRules()
    agents = [pacman_agents.GreedyAgent()] + \
             [ghost_agents.RandomGhost(i + 1) for i in range(layout.get_num_ghosts())]
    game = rules.new_game(layout, agents[0], agents[1:], display)
    state = game.state
    display.initialize(state.data)

    for action in actions:
        # Execute the action
        state = state.generate_successor(*action)
        # Change the display
        display.update(state.data)
        # Allow for game specific conditions (winning, losing, etc.)
        rules.process(state, game)

    display.finish()


def run_games(layout, pacman, ghosts, display, num_games,
              record, num_training=0, catch_exceptions=False, timeout=30):
    import __main__
    __main__.__dict__['_display'] = display

    rules = ClassicGameRules(timeout)
    games = []

    for i in range(num_games):
        be_quiet = i < num_training
        if be_quiet:
            # Suppress output and graphics
            game_display = text_display.NullGraphics()
            rules.quiet = True
        else:
            game_display = display
            rules.quiet = False
        game = rules.new_game(layout, pacman, ghosts, game_display, be_quiet, catch_exceptions)
        game.run()
        if not be_quiet:
            games.append(game)

        if record:
            file_name = ('recorded-game-%d' % (i + 1)) + '-'.join(
                [str(t) for t in time.localtime()[1:6]])
            file = open(file_name, 'w')
            components = {'layout': layout, 'actions': game.move_history}
            pickle.dump(components, file)
            file.close()

    if (num_games - num_training) > 0:
        scores = [game.state.get_score() for game in games]
        wins = [game.state.is_win() for game in games]
        win_rate = wins.count(True) / float(len(wins))
        print(f'Average Score: {sum(scores) / len(scores)}\n'
              f'Scores:       ', ', '.join([str(score) for score in scores]),
              f'\nWin Rate:      {wins.count(True)} / {len(wins)} ({win_rate:.2f})',
              '\nRecord:       ', ', '.join([['Loss', 'Win'][int(w)] for w in wins]))

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

    # import cProfile
    # cProfile.run("run_games( **args )")
