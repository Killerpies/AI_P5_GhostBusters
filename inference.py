"""
inference.py
------------
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


import itertools
import random

import busters
import game
import util


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghost_agent):
        """Sets the ghost agent for later access"""
        self.ghostAgent = ghost_agent
        self.index = ghost_agent.index
        self.obs = []  # most recent observation position
        self.legal_positions = []

    def get_jail_position(self):
        return 2 * self.ghostAgent.index - 1, 1

    def get_position_distribution(self, game_state):
        """
        Returns a distribution over successor positions of the ghost from the
        given game_state.

        You must first place the ghost in the game_state, using set_ghost_position
        below.
        """
        ghost_position = game_state.get_ghost_position(self.index)  # The position you set
        action_dist = self.ghostAgent.get_distribution(game_state)
        dist = util.Counter()
        for action, prob in list(action_dist.items()):
            successor_position = game.Actions.get_successor(ghost_position, action)
            dist[successor_position] = prob
        return dist

    def set_ghost_position(self, game_state, ghost_position):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied game_state.

        Note that calling set_ghost_position does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe_state.
        """
        conf = game.Configuration(ghost_position, game.Directions.STOP)
        game_state.data.agent_states[self.index] = game.AgentState(conf, False)
        return game_state

    def observe_state(self, game_state):
        """Collects the relevant noisy distance observation and pass it along."""
        distances = game_state.get_noisy_ghost_distances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, game_state)

    def initialize(self, game_state):
        """Initializes beliefs to a uniform distribution over all positions."""
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legal_positions = [p for p in game_state.get_walls().as_list(False) if p[1] > 1]
        self.initialize_uniformly(game_state)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initialize_uniformly(self, game_state):
        """Sets the belief state to a uniform prior belief over all positions."""
        pass

    def observe(self, observation, game_state):
        """Updates beliefs based on the given distance observation and game_state."""
        pass

    def elapse_time(self, game_state):
        """Updates beliefs for a time step elapsing from a game_state."""
        pass

    def get_belief_distribution(self):
        """
        Returns the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        pass


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward-algorithm updates to
    compute the exact belief function at each time step.
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
        """
        Updates beliefs based on the distance observation and Pacman's position.

        The noisy_distance is the estimated Manhattan distance to the ghost you
        are tracking.

        The emission_model below stores the probability of the noisy_distance for
        any true distance you supply. That is, it stores P(noisy_distance |
        TrueDistance).

        self.legal_positions is a list of the possible ghost positions (you
        should only consider positions that are in self.legal_positions).

        A correct implementation will handle the following special case:
          *  When a ghost is captured by Pacman, all beliefs should be updated
             so that the ghost appears in its prison cell, position
             self.get_jail_position()

             You can check if a ghost has been captured by Pacman by
             checking if it has a noisy_distance of None (a noisy distance
             of None will be returned if, and only if, the ghost is
             captured).
        """
        noisy_distance = observation
        emission_model = busters.get_observation_distribution(noisy_distance)
        pacman_position = game_state.get_pacman_position()

        #" *** YOUR CODE HERE ***"
        # Replace this code with a correct observation update
        # Be sure to handle the "jail" edge case where the ghost is eaten
        # and noisy_distance is None
        all_possible = util.Counter()
        for position in self.legal_positions:
            distance = util.manhattan_distance(position, pacman_position)
            all_possible[position] = 1.0


        #"*** END YOUR CODE HERE ***"

        all_possible.normalize()
        self.beliefs = all_possible

    def elapse_time(self, game_state):
        """
        Update self.beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position (e.g., for DirectionalGhost).  However, this
        is not a problem, as Pacman's current position is known.

        In order to obtain the distribution over new positions for the ghost,
        given its previous position (oldPos) as well as Pacman's current
        position, use this line of code:

          new_pos_dist = self.get_position_distribution(self.set_ghost_position(game_state, oldPos))

        Note that you may need to replace "oldPos" with the correct name of the
        variable that you have used to refer to the previous ghost position for
        which you are computing this distribution. You will need to compute
        multiple position distributions for a single update.

        new_pos_dist is a util.Counter object, where for each position p in
        self.legal_positions,

        newPostDist[p] = Pr( ghost is at position p at time t + 1 | ghost is at position oldPos
        at time t )

        (and also given Pacman's current position).  You may also find it useful
        to loop over key, value pairs in new_pos_dist, like:

          for newPos, prob in new_pos_dist.items():
            ...

        *** GORY DETAIL AHEAD ***

        As an implementation detail (with which you need not concern yourself),
        the line of code at the top of this comment block for obtaining
        new_pos_dist makes use of two helper methods provided in InferenceModule
        above:

          1) self.set_ghost_position(game_state, ghost_position)
              This method alters the game_state by placing the ghost we're
              tracking in a particular position.  This altered game_state can be
              used to query what the ghost would do in this position.

          2) self.get_position_distribution(game_state)
              This method uses the ghost agent to determine what positions the
              ghost will move to from the provided game_state.  The ghost must be
              placed in the game_state with a call to self.set_ghost_position
              above.

        It is worthwhile, however, to understand why these two helper methods
        are used and how they combine to give us a belief distribution over new
        positions after a time update from a particular position.
        """
        # *** YOUR CODE HERE ***
        new_beliefs = util.Counter()
        util.raise_not_defined()


    def get_belief_distribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.

    Useful helper functions will include random.choice, which chooses an element
    from a list uniformly at random, and util.sample, which samples a key from a
    Counter by treating its values as probabilities.
    """

    def __init__(self, ghost_agent, num_particles=300):
        InferenceModule.__init__(self, ghost_agent)
        self.num_particles = num_particles
        self.set_num_particles(num_particles)

    def set_num_particles(self, num_particles):
        self.num_particles = num_particles

    def initialize_uniformly(self, game_state):
        """
        Initializes a list of particles. Use self.num_particles for the number of
        particles. Use self.legal_positions for the legal board positions where a
        particle could be located.  Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        # *** YOUR CODE HERE ***
        util.raise_not_defined()


    def observe(self, observation, game_state):
        """
        Update beliefs based on the given distance observation. Make sure to
        handle the special case where all particles have weight 0 after
        reweighting based on observation. If this happens, resample particles
        uniformly at random from the set of legal positions
        (self.legal_positions).

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell,
             self.get_jail_position()

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisy_distance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initialize_uniformly. The total
             weight for a belief distribution can be found by calling totalCount
             on a Counter object

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.

        You may also want to use util.manhattan_distance to calculate the
        distance between a particle and Pacman's position.
        """
        noisy_distance = observation
        emission_model = busters.get_observation_distribution(noisy_distance)
        pacman_position = game_state.get_pacman_position()
        # *** YOUR CODE HERE ***
        util.raise_not_defined()

    def elapse_time(self, game_state):
        """
        Update beliefs for a time step elapsing.

        As in the elapse_time method of ExactInference, you should use:

          new_pos_dist = self.get_position_distribution(self.set_ghost_position(game_state, oldPos))

        to obtain the distribution over new positions for the ghost, given its
        previous position (oldPos) as well as Pacman's current position.

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.
        """
        # *** YOUR CODE HERE ***
        util.raise_not_defined()


    def get_belief_distribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution (a
        Counter object)
        """
        # *** YOUR CODE HERE ***
        util.raise_not_defined()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """

    def initialize_uniformly(self, game_state):
        """Set the belief state to an initial, prior value."""
        if self.index == 1:
            joint_inference.initialize(game_state, self.legal_positions)
        joint_inference.add_ghost_agent(self.ghostAgent)

    def observe_state(self, game_state):
        """Update beliefs based on the given distance observation and game_state."""
        if self.index == 1:
            joint_inference.observe_state(game_state)

    def elapse_time(self, game_state):
        """Update beliefs for a time step elapsing from a game_state."""
        if self.index == 1:
            joint_inference.elapse_time(game_state)

    def get_belief_distribution(self):
        """Returns the marginal belief over a particular ghost by summing out the others."""
        joint_distribution = joint_inference.get_belief_distribution()
        dist = util.Counter()
        for t, prob in list(joint_distribution.items()):
            dist[t[self.index - 1]] += prob
        return dist


class JointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, num_particles=600):
        self.set_num_particles(num_particles)

    def set_num_particles(self, num_particles):
        self.num_particles = num_particles

    def initialize(self, game_state, legal_positions):
        """Stores information about the game, then initializes particles."""
        self.num_ghosts = game_state.get_num_agents() - 1
        self.ghost_agents = []
        self.legal_positions = legal_positions
        self.initialize_particles()

    def initialize_particles(self):
        """
        Initialize particles to be consistent with a uniform prior.

        Each particle is a tuple of ghost positions. Use self.num_particles for
        the number of particles. You may find the `itertools` package helpful.
        Specifically, you will need to think about permutations of legal ghost
        positions, with the additional understanding that ghosts may occupy the
        same space. Look at the `itertools.product` function to get an
        implementation of the Cartesian product.

        Note: If you use itertools, keep in mind that permutations are not
        returned in a random order; you must shuffle the list of permutations in
        order to ensure even placement of particles across the board. Use
        self.legal_positions to obtain a list of positions a ghost may occupy.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        # *** YOUR CODE HERE ***
        util.raise_not_defined()


    def add_ghost_agent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghost_agents.append(agent)

    def get_jail_position(self, i):
        return 2 * i + 1, 1

    def observe_state(self, game_state):
        """
        Resamples the set of particles using the likelihood of the noisy
        observations.

        To loop over the ghosts, use:

          for i in range(self.num_ghosts):
            ...

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell, position
             self.get_jail_position(i) where `i` is the index of the ghost.

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initialize_particles. After all
             particles are generated randomly, any ghosts that are eaten (have
             noisyDistance of None) must be changed to the jail Position. This
             will involve changing each particle if a ghost has been eaten.

        self.get_particle_with_ghost_in_jail is a helper method to edit a specific
        particle. Since we store particles as tuples, they must be converted to
        a list, edited, and then converted back to a tuple. This is a common
        operation when placing a ghost in jail.
        """
        pacman_position = game_state.get_pacman_position()
        noisy_distances = game_state.get_noisy_ghost_distances()
        if len(noisy_distances) < self.num_ghosts:
            return
        emission_models = [busters.get_observation_distribution(dist) for dist in noisy_distances]

        # *** YOUR CODE HERE ***
        util.raise_not_defined()

    def get_particle_with_ghost_in_jail(self, particle, ghost_index):
        """
        Takes a particle (as a tuple of ghost positions) and returns a particle
        with the ghost_index'th ghost in jail.
        """
        particle = list(particle)
        particle[ghost_index] = self.get_jail_position(ghost_index)
        return tuple(particle)

    def elapse_time(self, game_state):
        """
        Samples each particle's next state based on its current state and the
        game_state.

        To loop over the ghosts, use:

          for i in range(self.num_ghosts):
            ...

        Then, assuming that `i` refers to the index of the ghost, to obtain the
        distributions over new positions for that single ghost, given the list
        (prevGhostPositions) of previous positions of ALL of the ghosts, use
        this line of code:

          new_pos_dist = get_position_distribution_for_ghost(
             set_ghost_positions(game_state, prevGhostPositions), i, self.ghost_agents[i]
          )

        Note that you may need to replace `prevGhostPositions` with the correct
        name of the variable that you have used to refer to the list of the
        previous positions of all of the ghosts, and you may need to replace `i`
        with the variable you have used to refer to the index of the ghost for
        which you are computing the new position distribution.

        As an implementation detail (with which you need not concern yourself),
        the line of code above for obtaining new_pos_dist makes use of two helper
        functions defined below in this file:

          1) set_ghost_positions(game_state, ghostPositions)
              This method alters the game_state by placing the ghosts in the
              supplied positions.

          2) get_position_distribution_for_ghost(game_state, ghost_index, agent)
              This method uses the supplied ghost agent to determine what
              positions a ghost (ghost_index) controlled by a particular agent
              (ghost_agent) will move to in the supplied game_state.  All ghosts
              must first be placed in the game_state using set_ghost_positions
              above.

              The ghost agent you are meant to supply is
              self.ghost_agents[ghost_index-1], but in this project all ghost
              agents are always the same.
        """
        new_particles = []
        for old_particle in self.particles:
            new_particle = list(old_particle)  # A list of ghost positions
            # now loop through and update each entry in new_particle...

            # *** YOUR CODE HERE ***
            util.raise_not_defined()


            # *** END YOUR CODE HERE ***
            new_particles.append(tuple(new_particle))
        self.particles = new_particles

    def get_belief_distribution(self):
        # *** YOUR CODE HERE ***
        util.raise_not_defined()


# One JointInference module is shared globally across instances of MarginalInference
joint_inference = JointParticleFilter()


def get_position_distribution_for_ghost(game_state, ghost_index, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied
    game_state.
    """
    # index 0 is pacman, but the students think that index 0 is the first ghost.
    ghost_position = game_state.get_ghost_position(ghost_index + 1)
    action_dist = agent.get_distribution(game_state)
    dist = util.Counter()
    for action, prob in list(action_dist.items()):
        successor_position = game.Actions.get_successor(ghost_position, action)
        dist[successor_position] = prob
    return dist


def set_ghost_positions(game_state, ghost_positions):
    """Sets the position of all ghosts to the values in ghostPositionTuple."""
    for index, pos in enumerate(ghost_positions):
        conf = game.Configuration(pos, game.Directions.STOP)
        game_state.data.agent_states[index + 1] = game.AgentState(conf, False)
    return game_state
