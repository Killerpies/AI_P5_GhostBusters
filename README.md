## CPSC 471/510 Project 5: Ghostbusters


<center>
Berkeley Project 4
</center>

<center>
Version 1.003\.
</center>

<center>
Last Updated: 10/31/2022
</center>

* * *

### Table of Contents

*   [Introduction](#Introduction)
*   [Welcome](#Welcome)
*   [Q1: Exact Inference: Observation](#Q1)
*   [Q2: Exact Inference: Time Elapse](#Q2)
*   [Q3: Exact Inference: Full System](#Q3)
*   [Q4: Approximate Inference: Observation](#Q4)
*   [Q5: Approximate Inference: Time Elapse](#Q5)
*   [Q6: Joint Particle Filter: Observation](#Q6)
*   [Q7: Joint Particle Filter: Time Elapse](#Q7)


* * *

<center>
 ![Ghost Busters](img/busters.png)
</center>


<center>


I can hear you, ghost.

Running won't save you from my

Particle filter

</center>


***


### <a name="Introduction">Introduction



Pacman spends his life running from ghosts, but things were not always so. Legend has it that many years ago, Pacman's great grandfather Grandpac learned to hunt ghosts for sport. However, he was blinded by his power and could only track ghosts by their banging and clanging.

In this project, you will design Pacman agents that use sensors to locate and eat invisible ghosts. You'll advance from locating single, stationary ghosts to hunting packs of multiple moving ghosts with ruthless efficiency.

The code for this project contains the following files:

<h4> Files you'll edit: </h4>

<table class="intro" border="0" cellpadding="10">

<tbody>

<tr>


</tr>

<tr>

<td>bustersAgents.py</td>

<td>Agents for playing the Ghostbusters variant of Pacman.</td>

</tr>

<tr>

<td>inference.py</td>

<td>Code for tracking ghosts over time using their sounds.</td>

</tr>

<tr>
</table>

<h4> Files you will not edit: </h4>

<table class="intro" border="0" cellpadding="10">

</tr>

<tr>

<td>busters.py</td>

<td>The main entry to Ghostbusters (replacing Pacman.py)</td>

</tr>

<tr>

<td>bustersGhostAgents.py</td>

<td>New ghost agents for Ghostbusters</td>

</tr>

<tr>

<td>distanceCalculator.py</td>

<td>Computes maze distances</td>

</tr>

<tr>

<td>game.py</td>

<td>Inner workings and helper classes for Pacman</td>

</tr>

<tr>

<td>ghostAgents.py</td>

<td>Agents to control ghosts</td>

</tr>

<tr>

<td>graphicsDisplay.py</td>

<td>Graphics for Pacman</td>

</tr>

<tr>

<td>graphicsUtils.py</td>

<td>Support for Pacman graphics</td>

</tr>

<tr>

<td>keyboardAgents.py</td>

<td>Keyboard interfaces to control Pacman</td>

</tr>

<tr>

<td>layout.py</td>

<td>Code for reading layout files and storing their contents</td>

</tr>

<tr>

<td>util.py</td>

<td>Utility functions</td>

</tr>

</tbody>

</table>

**Files to Edit and Submit:** You will fill in portions of `bustersAgents.py` and `inference.py` during the assignment. You should submit these files with your code and comments. Please _do not_ change the other files in this distribution or submit any of our original files other than these files.

**Evaluation:** Your code will be autograded for technical correctness. Please _do not_ change the names of any provided functions or classes within the code, or you will wreak havoc on the autograder. However, the correctness of your implementation -- not the autograder's judgements -- will be the final judge of your score. If necessary, we will review and grade assignments individually to ensure that you receive due credit for your work.

**Academic Dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; _please_ don't let us down. If you do, we will pursue the strongest consequences available to us.

**Getting Help:** You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, section, and the discussion forum are there for your support; please use them. If you can't make our office hours, let us know and we will schedule more. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask.

**Discussion:** Please be careful not to post spoilers.



* * *

### <a name="Welcome">Ghostbusters and BNs

In our version of Ghostbusters, the goal is to hunt down scared but invisible ghosts. Pacman, ever resourceful, is equipped with sonar (ears) that provides noisy readings of the Manhattan distance to each ghost. The game ends when Pacman has eaten all the ghosts. To start, try playing a game yourself using the keyboard.

<pre>python busters.py</pre>

The blocks of color indicate where the each ghost could possibly be, given the noisy distance readings provided to Pacman. The noisy distances at the bottom of the display are always non-negative, and always within 7 of the true distance. The probability of a distance reading decreases exponentially with its difference from the true distance.

Your primary task in this project is to implement inference to track the ghosts. For the keyboard based game above, a crude form of inference was implemented for you by default: all squares in which a ghost could possibly be are shaded by the color of the ghost. Naturally, we want a better estimate of the ghost's position. Fortunately, Bayes' Nets provide us with powerful tools for making the most of the information we have. Throughout the rest of this project, you will implement algorithms for performing both exact and approximate inference using Bayes' Nets. The lab is challenging, so we do encourage you to start early and seek help when necessary.

While watching and debugging your code with the autograder, it will be helpful to have some understanding of what the autograder is doing. There are 2 types of tests in this project, as differentiated by their `*.test` files found in the subdirectories of the `test_cases` folder. For tests of class `DoubleInferenceAgentTest`, your will see visualizations of the inference distributions generated by your code, but all Pacman actions will be preselected according to the actions of the staff implementation. This is necessary in order to allow comparision of your distributions with the staff's distributions. The second type of test is `GameScoreTest`, in which your `BustersAgent` will actually select actions for Pacman and you will watch your Pacman play and win games.

As you implement and debug your code, you may find it useful to run a single test at a time. In order to do this you will need to use the -t flag with the autograder. For example if you only want to run the first test of question 1, use:

<pre>python autograder.py -t test_cases/q1/1-ExactObserve</pre>

In general, all test cases can be found inside test_cases/q*.


* * *


### <a name="Q1"></a>Question 1 (3 points): Exact Inference Observation

In this question, you will update the `observe` method in `ExactInference` class of `inference.py` to correctly update the agent's belief distribution over ghost positions given an observation from Pacman's sensors. A correct implementation should also handle one special case: when a ghost is eaten, you should place that ghost in its prison cell, as described in the comments of `observe`.

To run the autograder for this question and visualize the output:

<pre>python autograder.py -q q1</pre>

As you watch the test cases, be sure that you understand how the squares converge to their final coloring. In test cases where is Pacman boxed in (which is to say, he is unable to change his observation point), why does Pacman sometimes have trouble finding the exact location of the ghost?

_Note:_ your busters agents have a separate inference module for each ghost they are tracking. That's why if you print an observation inside the `observe` function, you'll only see a single number even though there may be multiple ghosts on the board.

Hints:

*   You are implementing the online belief update for observing new evidence. Before any readings, Pacman believes the ghost could be anywhere: a uniform prior (see `initialize_uniformly`). After receiving a reading, the `observe` function is called, which must update the belief at every position.
*   Before typing any code, write down the equation of the inference problem you are trying to solve.
*   Try printing `noisyDistance`, `emissionModel`, and `PacmanPosition` (in the `observe` function) to get started.
*   In the Pacman display, high posterior beliefs are represented by bright colors, while low beliefs are represented by dim colors. You should start with a large cloud of belief that shrinks over time as more evidence accumulates.
*   Beliefs are stored as `util.Counter` objects (like dictionaries) in a field called `self.beliefs`, which you should update.
*   You should not need to store any evidence. The only thing you need to store in `ExactInference` is `self.beliefs`.


* * *


### <a name="Q2"></a>Question 2 (4 points): Exact Inference with Time Elapse

In the previous question you implemented belief updates for Pacman based on his observations. Fortunately, Pacman's observations are not his only source of knowledge about where a ghost may be. Pacman also has knowledge about the ways that a ghost may move; namely that the ghost can not move through a wall or more than one space in one timestep.

To understand why this is useful to Pacman, consider the following scenario in which there is Pacman and one Ghost. Pacman receives many observations which indicate the ghost is very near, but then one which indicates the ghost is very far. The reading indicating the ghost is very far is likely to be the result of a buggy sensor. Pacman's prior knowledge of how the ghost may move will decrease the impact of this reading since Pacman knows the ghost could not move so far in only one move.

In this question, you will implement the `elapse_time` method in `ExactInference`. Your agent has access to the action distribution for any `GhostAgent`. In order to test your `elapse_time` implementation separately from your `observe` implementation in the previous question, this question will not make use of your `observe` implementation.

Since Pacman is not using any observations about the ghost, this means that Pacman will start with a uniform distribution over all spaces, and then update his beliefs according to how he knows the Ghost is able to move. Since Pacman is not observing the ghost, this means the ghost's actions will not impact Pacman's beliefs. Over time, Pacman's beliefs will come to reflect places on the board where he believes ghosts are most likely to be given the geometry of the board and what Pacman already knows about their valid movements.

For the tests in this question we will sometimes use a ghost with random movements and other times we will use the GoSouthGhost. This ghost tends to move south so over time, and without any observations, Pacman's belief distribution should begin to focus around the bottom of the board. To see which ghost is used for each test case you can look in the .test files.

To run the autograder for this question and visualize the output:

<pre>python autograder.py -q q2</pre>

As an example of the GoSouthGhostAgent, you can run

<pre>python autograder.py -t test_cases/q2/2-ExactElapse</pre>

and observe that the distribution becomes concentrated at the bottom of the board.

As you watch the autograder output, remember that lighter squares indicate that pacman believes a ghost is more likely to occupy that location, and darker squares indicate a ghost is less likely to occupy that location. For which of the test cases do you notice differences emerging in the shading of the squares? Can you explain why some squares get lighter and some squares get darker?

Hints:

*   Instructions for obtaining a distribution over where a ghost will go next, given its current position and the `gameState`, appears in the comments of `ExactInference.elapse_time` in `inference.py`.
*   We assume that ghosts still move independently of one another, so while you can develop all of your code for one ghost at a time, adding multiple ghosts should still work correctly.

* * *

### <a name="Q3"></a>Question 3 (3 points): Exact Inference Full Test

Now that Pacman knows how to use both his prior knowledge and his observations when figuring out where a ghost is, he is ready to hunt down ghosts on his own. This question will use your `observe` and `elapse_time` implementations together, along with a simple greedy hunting strategy which you will implement for this question. In the simple greedy strategy, Pacman assumes that each ghost is in its most likely position according to its beliefs, then moves toward the closest ghost. Up to this point, Pacman has moved by randomly selecting a valid action.

Implement the `choose_action` method in `GreedyBustersAgent` in `bustersAgents.py`. Your agent should first find the most likely position of each remaining (uncaptured) ghost, then choose an action that minimizes the distance to the closest ghost. If correctly implemented, your agent should win the game in `q3/3-gameScoreTest` with a score greater than 700 at least 8 out of 10 times. _Note:_ the autograder will also check the correctness of your inference directly, but the outcome of games is a reasonable sanity check.

To run the autograder for this question and visualize the output:

<pre>python autograder.py -q q3</pre>

_Note:_ If you want to run this test (or any of the other tests) without graphics you can add the following flag:

<pre>python autograder.py -q q3 --no-graphics</pre>

Hints:

*   When correctly implemented, your agent will thrash around a bit in order to capture a ghost.
*   The comments of `choose_action` provide you with useful method calls for computing maze distance and successor positions.
*   Make sure to only consider the living ghosts, as described in the comments.


* * *


### <a name="Q4"></a>Question 4 (3 points): Approximate Inference Observation

Approximate inference is very trendy among ghost hunters this season. Next, you will implement a particle filtering algorithm for tracking a single ghost.

Implement the functions `initialize_uniformly`, `get_belief_distribution`, and `observe` for the `ParticleFilter` class in `inference.py`. A correct implementation should also handle two special cases. (1) When all your particles receive zero weight based on the evidence, you should resample all particles from the prior to recover. (2) When a ghost is eaten, you should update all particles to place that ghost in its prison cell, as described in the comments of `observe`. When complete, you should be able to track ghosts nearly as effectively as with exact inference.

To run the autograder for this question and visualize the output:

<pre>python autograder.py -q q4</pre>

Hints:

*   A particle (sample) is a ghost position in this inference problem.
*   The belief cloud generated by a particle filter will look noisy compared to the one for exact inference.
*   `util.sample` or `util.nSample` will help you obtain samples from a distribution. If you use `util.sample` and your implementation is timing out, try using `util.nSample`.


* * *


### <a name="Q5"></a>Question 5 (4 points): Approximate Inference with Time Elapse

Implement the `elapse_time` function for the `ParticleFilter` class in `inference.py`. When complete, you should be able to track ghosts nearly as effectively as with exact inference.

Note that in this question, we will test both the `elapse_time` function in isolation, as well as the full implementation of the particle filter combining `elapse_time` and `observe`.

To run the autograder for this question and visualize the output:

<pre>python autograder.py -q q5</pre>

For the tests in this question we will sometimes use a ghost with random movements and other times we will use the GoSouthGhost. This ghost tends to move south so over time, and without any observations, Pacman's belief distribution should begin to focus around the bottom of the board. To see which ghost is used for each test case you can look in the .test files. As an example, you can run

<pre>python autograder.py -t test_cases/q5/2-ParticleElapse</pre>

and observe that the distribution becomes concentrated at the bottom of the board.


* * *


### <a name="Q6"></a>Question 6 (4 points): Joint Particle Filter Observation

So far, we have tracked each ghost independently, which works fine for the default `RandomGhost` or more advanced `DirectionalGhost`. However, the prized `DispersingGhost` chooses actions that avoid other ghosts. Since the ghosts' transition models are no longer independent, all ghosts must be tracked jointly in a dynamic Bayes net!

The Bayes net has the following structure, where the hidden variables G represent ghost positions and the emission variables E are the noisy distances to each ghost. This structure can be extended to more ghosts, but only two (a and b) are shown below.

<center>
![Dynamic Bayes Net](img/dbn.png)
</center>

You will now implement a particle filter that tracks multiple ghosts simultaneously. Each particle will represent a tuple of ghost positions that is a sample of where all the ghosts are at the present time. The code is already set up to extract marginal distributions about each ghost from the joint inference algorithm you will create, so that belief clouds about individual ghosts can be displayed.

Complete the `initialize_particles`, `get_belief_distribution`, and `observe_state` method in `JointParticleFilter` to weight and resample the whole list of particles based on new evidence. As before, a correct implementation should also handle two special cases. (1) When all your particles receive zero weight based on the evidence, you should resample all particles from the prior to recover. (2) When a ghost is eaten, you should update all particles to place that ghost in its prison cell, as described in the comments of `observe_state`.

You should now effectively track dispersing ghosts. To run the autograder for this question and visualize the output:

<pre>python autograder.py -q q6</pre>


* * *


### <a name="Q7"></a>Question 7 (4 points): Joint Particle Filter with Elapse Time

Complete the `elapse_time` method in `JointParticleFilter` in `inference.py` to resample each particle correctly for the Bayes net. In particular, each ghost should draw a new position conditioned on the positions of all the ghosts at the previous time step. The comments in the method provide instructions for support functions to help with sampling and creating the correct distribution.

Note that completing this question involves removing the call to util.raiseNotDefined(). This means that the autograder will now grade both question 6 and question 7\. Since these questions involve joint distributions, they require more computational power (and time) to grade, so please be patient!

As you run the autograder note that `q7/1-JointParticleElapse` and `q7/2-JointParticleElapse` test your `elapse_time` implementations only, and `q7/3-JointParticleElapse` tests both your `elapse_time` and `observe` implementations. Notice the difference between test 1 and test 3\. In both tests, pacman knows that the ghosts will move to the sides of the gameboard. What is different between the tests, and why?

<center>
![Disperse](img/disperse.png)
</center>

To run the autograder for this question use:

<pre>python autograder.py -q q7</pre>


* * *

### Submission

You're not done yet!
Make sure your modified code is pushed to Gitlab.
You are expected to make regular commits and pushes to GitLab, and will be graded
on making at least 5 regular commits in addition to the autograder points.
