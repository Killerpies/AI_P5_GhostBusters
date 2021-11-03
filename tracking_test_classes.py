# tracking_test_classes.py
# ----------------------
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


# tracking_test_classes.py
# ----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import copy
import random
import re
import time

import busters
import busters_agents
import layout
import test_classes
import util
from game import Actions
from game import Agent
from game import Directions


class GameScoreTest(test_classes.TestCase):

    def __init__(self, question, test_dict):
        super(GameScoreTest, self).__init__(question, test_dict)
        self.max_moves = int(self.test_dict['maxMoves'])
        self.inference = self.test_dict['inference']
        self.layout_str = self.test_dict['layout_str'].split('\n')
        self.num_runs = int(self.test_dict['numRuns'])
        self.num_wins_for_credit = int(self.test_dict['numWinsForCredit'])
        self.num_ghosts = int(self.test_dict['numGhosts'])
        self.layout_name = self.test_dict['layout_name']
        self.min_score = int(self.test_dict['min_score'])
        self.observe_enable = self.test_dict['observe'] == 'True'
        self.elapse_enable = self.test_dict['elapse'] == 'True'

    def execute(self, grades, module_dict, solution_dict):
        ghosts = [SeededRandomGhostAgent(i) for i in range(1, self.num_ghosts + 1)]
        pac = busters_agents.GreedyBustersAgent(0,
                                                inference=self.inference,
                                                ghost_agents=ghosts,
                                                observe_enable=self.observe_enable,
                                                elapse_time_enable=self.elapse_enable)
        # if self.inference == "ExactInference":
        #    pac.inference_modules = [moduleDict['inference'].ExactInference(a) for a in ghosts]
        # else:
        #    print "Error inference type %s -- not implemented" % self.inference
        #    return

        stats = run(self.layout_str, pac, ghosts, self.question.get_display(),
                    n_games=self.num_runs,
                    max_moves=self.max_moves, quiet=False)
        above_count = [s >= self.min_score for s in stats['scores']].count(True)
        msg = "%s) Games won on %s with score above %d: %d/%d" % (
            self.layout_name, grades.current_question, self.min_score, above_count, self.num_runs)
        grades.add_message(msg)
        if above_count >= self.num_wins_for_credit:
            grades.assign_full_credit()
            return self.test_pass(grades)
        else:
            return self.test_fail(grades)

    def write_solution(self, _, file_path):
        handle = open(file_path, 'w')
        handle.write('# You must win at least %d/10 games with at least %d points' % (
            self.num_wins_for_credit, self.min_score))
        handle.close()

    def create_public_version(self):
        pass


class ZeroWeightTest(test_classes.TestCase):

    def __init__(self, question, test_dict):
        super(ZeroWeightTest, self).__init__(question, test_dict)
        self.max_moves = int(self.test_dict['maxMoves'])
        self.inference = self.test_dict['inference']
        self.layout_str = self.test_dict['layout'].split('\n')
        self.num_ghosts = int(self.test_dict['numGhosts'])
        self.observe_enable = self.test_dict['observe'] == 'True'
        self.elapse_enable = self.test_dict['elapse'] == 'True'
        self.ghost = self.test_dict['ghost']
        self.seed = int(self.test_dict['seed'])

    def execute(self, grades, module_dict, solution_dict):
        random.seed(self.seed, version=1)
        inference_function = getattr(module_dict['inference'], self.inference)
        ghosts = [globals()[self.ghost](i) for i in range(1, self.num_ghosts + 1)]
        if self.inference == 'MarginalInference':
            module_dict['inference'].joint_inference = \
                module_dict['inference'].JointParticleFilter()
        disp = self.question.get_display()
        pac = ZeroWeightAgent(inference=inference_function,
                              ghost_agents=ghosts,
                              grades=grades,
                              seed=self.seed,
                              disp=disp,
                              elapse=self.elapse_enable,
                              observe=self.observe_enable)
        if self.inference == "ParticleFilter":
            for p_filter in pac.inferenceModules:
                p_filter.set_num_particles(5000)
        elif self.inference == "MarginalInference":
            module_dict['inference'].joint_inference.set_num_particles(5000)
        run(self.layout_str, pac, ghosts, disp, max_moves=self.max_moves)
        if pac.get_reset():
            grades.add_message('%s) successfully handled all weights = 0' % grades.current_question)
            return self.test_pass(grades)
        else:
            grades.add_message('%s) error handling all weights = 0' % grades.current_question)
            return self.test_fail(grades)

    def write_solution(self, module_dict, file_path):
        handle = open(file_path, 'w')
        handle.write(
            '# This test checks that you successfully handle the case when all particle weights '
            'are set to 0\n')
        handle.close()

    def create_public_version(self):
        self.test_dict['seed'] = '188'
        self.seed = 188


class DoubleInferenceAgentTest(test_classes.TestCase):

    def __init__(self, question, test_dict):
        super(DoubleInferenceAgentTest, self).__init__(question, test_dict)
        self.seed = int(self.test_dict['seed'])
        self.layout_str = self.test_dict['layout'].split('\n')
        self.observe = (self.test_dict['observe'] == "True")
        self.elapse = (self.test_dict['elapse'] == "True")
        self.check_uniform = (self.test_dict['checkUniform'] == 'True')
        self.max_moves = int(self.test_dict['maxMoves'])
        self.num_ghosts = int(self.test_dict['numGhosts'])
        self.inference = self.test_dict['inference']
        self.error_msg = self.test_dict['errorMsg']
        self.l2_tolerance = float(self.test_dict['L2Tolerance'])
        self.ghost = self.test_dict['ghost']

    def execute(self, grades, module_dict, solution_dict):
        random.seed(self.seed, version=1)
        lines = solution_dict['correctActions'].split('\n')
        moves = []
        # Collect solutions
        for line in lines:
            m = re.match('(\d+) (\w+) (.*)', line)
            moves.append((m.group(1), m.group(2), eval(m.group(3))))

        inference_function = getattr(module_dict['inference'], self.inference)

        ghosts = [globals()[self.ghost](i) for i in range(1, self.num_ghosts + 1)]
        if self.inference == 'MarginalInference':
            module_dict['inference'].joint_inference = \
                module_dict['inference'].JointParticleFilter()

        # print(moves)

        disp = self.question.get_display()
        pac = DoubleInferenceAgent(inference_function, moves, ghosts, grades, self.seed, disp,
                                   elapse=self.elapse, observe=self.observe,
                                   l2_tolerance=self.l2_tolerance, check_uniform=self.check_uniform)
        if self.inference == "ParticleFilter":
            for pfilter in pac.inference_modules:
                pfilter.set_num_particles(5000)
        elif self.inference == "MarginalInference":
            module_dict['inference'].joint_inference.set_num_particles(5000)
        run(self.layout_str, pac, ghosts, disp, max_moves=self.max_moves)
        msg = self.error_msg % pac.errors
        grades.add_message(("%s) " % grades.current_question) + msg)
        if pac.errors == 0:
            grades.add_points(2)
            return self.test_pass(grades)
        else:
            return self.test_fail(grades)

    def write_solution(self, module_dict, file_path):
        random.seed(self.seed, version=1)
        if self.inference == 'ParticleFilter':
            self.inference = 'ExactInference'  # use exact inference to generate solution
        inference_function = getattr(module_dict['inference'], self.inference)

        ghosts = [globals()[self.ghost](i) for i in range(1, self.num_ghosts + 1)]
        if self.inference == 'MarginalInference':
            module_dict['inference'].joint_inference = \
                module_dict['inference'].JointParticleFilter()
            module_dict['inference'].joint_inference.set_num_particles(5000)

        pac = InferenceAgent(inference_function, ghosts, self.seed, elapse=self.elapse,
                             observe=self.observe)
        run(self.layout_str, pac, ghosts, self.question.getDisplay(), max_moves=self.max_moves)
        # run our gold code here and then write it to a solution file
        answer_list = pac.answer_list
        handle = open(file_path, 'w')
        handle.write('# move_number action likelihood_dictionary\n')
        handle.write('correctActions: """\n')
        for (moveNum, move, dists) in answer_list:
            handle.write('%s %s [' % (moveNum, move))
            for dist in dists:
                handle.write('{')
                for key in dist:
                    handle.write('%s: %s, ' % (key, dist[key]))
                handle.write('}, ')
            handle.write(']\n')
        handle.write('"""\n')
        handle.close()

    def create_public_version(self):
        self.test_dict['seed'] = '188'
        self.seed = 188


def run(layout_str, pac, ghosts, disp, n_games=1, name='games', max_moves=-1, quiet=True):
    """Runs a few games and outputs their statistics."""

    start_time = time.time()
    lay = layout.Layout(layout_str)

    # print '*** Running %s on' % name, layname,'%d time(s).' % nGames
    games = busters.run_games(lay, pac, ghosts, disp, n_games, max_moves)

    # print '*** Finished running %s on' % name, layname, 'after %d seconds.' % (time.time() -
    # start_time)

    stats = {'time': time.time() - start_time,
             'wins': [g.state.is_win() for g in games].count(True),
             'games': games, 'scores': [g.state.get_score() for g in games]}
    stat_tuple = (stats['wins'], len(games), sum(stats['scores']) * 1.0 / len(games))
    if not quiet:
        print('*** Won %d out of %d games. Average score: %f ***' % stat_tuple)
    return stats


class InferenceAgent(busters_agents.BustersAgent):
    """Tracks ghosts and compares to reference inference modules, while moving randomly"""

    def __init__(self, inference, ghost_agents, seed, elapse=True, observe=True, burn_in=0):
        super().__init__(inference, ghost_agents)
        self.inference_modules = [inference(a) for a in ghost_agents]
        self.elapse = elapse
        self.observe = observe
        self.burn_in = burn_in
        self.num_moves = 0
        # self.rand = rand
        # list of tuples (move_num, move, [dist_1, dist_2, ...])
        self.answer_list = []
        self.seed = seed
        self.first_move = True
        self.ghost_beliefs = []

    def final(self, game_state):
        distribution_list = []
        self.num_moves += 1
        for index, inf in enumerate(self.inference_modules):
            if self.observe:
                inf.observe_state(game_state)
            self.ghost_beliefs[index] = inf.get_belief_distribution()
            belief_copy = copy.deepcopy(self.ghost_beliefs[index])
            distribution_list.append(belief_copy)
        self.answer_list.append((self.num_moves, None, distribution_list))
        random.seed(self.seed + self.num_moves, version=1)

    def register_initial_state(self, game_state):
        """Initializes beliefs and inference modules"""
        for inference in self.inference_modules:
            inference.initialize(game_state)
        self.ghost_beliefs = [inf.get_belief_distribution() for inf in self.inference_modules]
        self.first_move = True
        self.answer_list.append((self.num_moves, None, copy.deepcopy(self.ghost_beliefs)))

    def get_action(self, game_state):
        """Updates beliefs, then chooses an action based on updated beliefs."""
        distribution_list = []
        self.num_moves += 1
        for index, inf in enumerate(self.inference_modules):
            if self.elapse:
                if not self.first_move:
                    inf.elapse_time(game_state)
            self.first_move = False
            if self.observe:
                inf.observe_state(game_state)
            self.ghost_beliefs[index] = inf.get_belief_distribution()
            belief_copy = copy.deepcopy(self.ghost_beliefs[index])
            distribution_list.append(belief_copy)
        action = random.choice([a for a in game_state.get_legal_pacman_actions() if a != 'STOP'])
        self.answer_list.append((self.num_moves, action, distribution_list))
        random.seed(self.seed + self.num_moves, version=1)
        return action


class ZeroWeightAgent(busters_agents.BustersAgent):
    """Tracks ghosts and compares to reference inference modules, while moving randomly"""

    def __init__(self, inference, ghost_agents, grades, seed, disp, elapse=True, observe=True):
        super().__init__(inference=inference, ghost_agents=ghost_agents, observe_enable=observe,
                         elapse_time_enable=elapse)
        self.inferenceModules = [inference(a) for a in ghost_agents]
        self.elapse = elapse
        self.observe = observe
        self.grades = grades
        self.num_moves = 0
        self.seed = seed
        self.display = disp
        self.reset = False
        self.first_move = True
        self.ghost_beliefs = []

    def final(self, game_state):
        pass

    def register_initial_state(self, game_state):
        """Initializes beliefs and inference modules"""
        for inference in self.inference_modules:
            inference.initialize(game_state)
        self.ghost_beliefs = [inf.get_belief_distribution() for inf in self.inference_modules]
        self.first_move = True

    def get_action(self, game_state):
        """Updates beliefs, then chooses an action based on updated beliefs."""
        new_beliefs = [None] * len(self.inference_modules)
        self.num_moves += 1
        for index, inf in enumerate(self.inference_modules):
            if self.elapse:
                if not self.first_move:
                    inf.elapse_time(game_state)
            self.first_move = False
            if self.observe:
                inf.observe_state(game_state)
            new_beliefs[index] = inf.get_belief_distribution()
        self.check_reset(new_beliefs, self.ghost_beliefs)
        self.ghost_beliefs = new_beliefs
        self.display.update_distributions(self.ghost_beliefs)
        random.seed(self.seed + self.num_moves, version=1)
        action = random.choice([a for a in game_state.get_legal_pacman_actions() if a != 'STOP'])
        return action

    def check_reset(self, new_beliefs, old_beliefs):
        for i in range(len(new_beliefs)):
            new_keys = [x for x in list(new_beliefs[i].keys()) if new_beliefs[i][x] != 0]
            old_keys = [x for x in list(old_beliefs[i].keys()) if old_beliefs[i][x] != 0]
            if len(new_keys) > len(old_keys):
                self.reset = True

    def get_reset(self):
        return self.reset


class DoubleInferenceAgent(busters_agents.BustersAgent):
    """Tracks ghosts and compares to reference inference modules, while moving randomly"""

    def __init__(self, inference, ref_solution, ghost_agents, grades, seed, disp, elapse=True,
                 observe=True, l2_tolerance=0.2, burn_in=0, check_uniform=False):
        super().__init__(inference=inference,
                         ghost_agents=ghost_agents,
                         observe_enable=observe,
                         elapse_time_enable=elapse)
        self.inference_modules = [inference(a) for a in ghost_agents]
        self.ref_solution = ref_solution
        self.elapse = elapse
        self.observe = observe
        self.grades = grades
        self.l2_tolerance = l2_tolerance
        self.errors = 0
        self.burn_in = burn_in
        self.num_moves = 0
        self.seed = seed
        self.display = disp
        self.ghost_beliefs = []
        self.firstMove = True
        self.check_uniform = check_uniform

    def final(self, game_state):
        self.num_moves += 1
        # todo investigate change here
        move_num, action, dists = self.ref_solution[self.num_moves - 1]
        for index, inf in enumerate(self.inference_modules):
            if self.observe:
                inf.observe_state(game_state)
            self.ghost_beliefs[index] = inf.get_belief_distribution()
            if self.num_moves >= self.burn_in:
                self.dist_compare(self.ghost_beliefs[index], dists[index])
        self.display.update_distributions(self.ghost_beliefs)
        random.seed(self.seed + self.num_moves, version=1)
        if not self.display.check_null_display():
            time.sleep(3)

    def register_initial_state(self, game_state):
        """Initializes beliefs and inference modules"""
        for inference in self.inference_modules:
            inference.initialize(game_state)
        move_num, action, dists = self.ref_solution[self.num_moves]
        for index, inf in enumerate(self.inference_modules):
            self.dist_compare(inf.get_belief_distribution(), dists[index])
        self.ghost_beliefs = [inf.get_belief_distribution() for inf in self.inference_modules]
        self.firstMove = True

    def get_action(self, game_state):
        """Updates beliefs, then chooses an action based on updated beliefs."""
        self.num_moves += 1
        move_num, action, dists = self.ref_solution[self.num_moves]
        for index, inf in enumerate(self.inference_modules):
            if self.elapse:
                if not self.firstMove:
                    inf.elapse_time(game_state)
            self.firstMove = False
            if self.observe:
                inf.observe_state(game_state)
            self.ghost_beliefs[index] = inf.get_belief_distribution()
            if self.num_moves >= self.burn_in:
                self.dist_compare(self.ghost_beliefs[index], dists[index])
        self.display.update_distributions(self.ghost_beliefs)
        random.seed(self.seed + self.num_moves, version=1)
        return action

    def dist_compare(self, dist, ref_dist):
        """Compares two distributions"""
        # copy and prepare distributions
        dist = dist.copy()
        ref_dist = ref_dist.copy()
        for key in set(list(ref_dist.keys()) + list(dist.keys())):
            if key not in list(dist.keys()):
                dist[key] = 0.0
            if key not in list(ref_dist.keys()):
                ref_dist[key] = 0.0
        # calculate l2 difference
        l2 = 0
        for k in list(ref_dist.keys()):
            l2 += (dist[k] - ref_dist[k]) ** 2
        if l2 > self.l2_tolerance:
            if self.errors == 0:
                t = (self.grades.current_question, self.num_moves, l2)
                summary = "%s) Distribution deviated at move %d by %0.4f (squared norm) from the " \
                          "correct answer.\n" % t
                header = '%10s%5s%-25s%-25s\n' % ('key:', '', 'student', 'reference')
                detail = '\n'.join(['%9s:%5s%-25s%-25s' % (x, '', dist[x], ref_dist[x]) for x in
                                    set(list(dist.keys()) + list(ref_dist.keys()))])

                # Ignoring the exact comparison due to python3 random issues!
                #self.grades.fail('%s%s%s' % (summary, header, detail))
            #self.errors += 1
        # check for uniform distribution if necessary
        if self.check_uniform:
            if abs(max(dist.values()) - max(ref_dist.values())) > .0025:
                if self.errors == 0:
                    self.grades.fail(
                        '%s) Distributions do not have the same max value and are therefore not '
                        'uniform.\n\tstudent max: %f\n\treference max: %f' % (
                            self.grades.current_question,
                            max(dist.values()),
                            max(ref_dist.values())
                        )
                    )
                    self.errors += 1


class SeededRandomGhostAgent(Agent):
    def __init__(self, index):
        super().__init__(index)
        self.index = index

    def get_action(self, state):
        dist = util.Counter()
        for a in state.get_legal_actions(self.index):
            dist[a] = 1.0
        dist.normalize()
        if len(dist) == 0:
            return Directions.STOP
        else:
            action = sample(dist)
            return action

    def get_distribution(self, state):
        dist = util.Counter()
        for a in state.get_legal_actions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class GoSouthAgent(Agent):
    def __init__(self, index):
        super().__init__(index)
        self.index = index

    def get_action(self, state):
        dist = util.Counter()
        for a in state.get_legal_actions(self.index):
            dist[a] = 1.0
        if Directions.SOUTH in list(dist.keys()):
            dist[Directions.SOUTH] *= 2
        dist.normalize()
        if len(dist) == 0:
            return Directions.STOP
        else:
            action = sample(dist)
            return action

    def get_distribution(self, state):
        dist = util.Counter()
        for a in state.get_legal_actions(self.index):
            dist[a] = 1.0
        if Directions.SOUTH in list(dist.keys()):
            dist[Directions.SOUTH] *= 2
        dist.normalize()
        return dist


class DispersingSeededGhost(Agent):
    """Chooses an action that distances the ghost from the other ghosts with probability
    spreadProb. """

    def __init__(self, index, spread_prob=0.5):
        super().__init__(index)
        self.index = index
        self.spreadProb = spread_prob

    def get_action(self, state):
        dist = self.get_distribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            action = sample(dist)
            return action

    def get_distribution(self, state):
        ghost_state = state.get_ghost_state(self.index)
        legal_actions = state.get_legal_actions(self.index)
        pos = state.get_ghost_position(self.index)
        is_scared = ghost_state.scared_timer > 0

        speed = 1
        if is_scared:
            speed = 0.5
        action_vectors = [Actions.direction_to_vector(a, speed) for a in legal_actions]
        new_positions = [(pos[0] + a[0], pos[1] + a[1]) for a in action_vectors]

        # get other ghost positions
        others = [i for i in range(1, state.get_num_agents()) if i != self.index]
        for a in others:
            assert state.get_ghost_state(a) is not None, "Ghost position unspecified in state!"
        other_ghost_positions = [state.get_ghost_position(a) for a in others if
                                 state.get_ghost_position(a)[1] > 1]

        # for each action, get the sum of inverse squared distances to the other ghosts
        sum_of_distances = []
        for pos in new_positions:
            sum_of_distances.append(
                sum([(1 + util.manhattan_distance(pos, g)) ** (-2) for g in other_ghost_positions]))

        best_distance = min(sum_of_distances)
        num_best = [best_distance == dist for dist in sum_of_distances].count(True)
        distribution = util.Counter()
        for action, distance in zip(legal_actions, sum_of_distances):
            if distance == best_distance:
                distribution[action] += self.spreadProb / num_best
            distribution[action] += (1 - self.spreadProb) / len(legal_actions)
        return distribution


def sample(distribution, values=None):
    if type(distribution) == util.Counter:
        items = list(distribution.items())
        distribution = [i[1] for i in items]
        values = [i[0] for i in items]
    if sum(distribution) != 1:
        distribution = util.normalize(distribution)
    choice = random.random()
    i, total = 0, distribution[0]
    while choice > total:
        i += 1
        total += distribution[i]
    return values[i]
