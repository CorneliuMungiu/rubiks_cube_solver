from __future__ import annotations

import time

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation
from math import sqrt, log
import random
import heapq
import matplotlib.pyplot as plt
import numpy as np
import copy

from constants import MOVES, CORNERS, COLORS, LETTERS
from moves import Move, MoveSequence, Moves


class Cube:

    def __init__(self, moves: Moves | None = None, scrambled: bool = True):
        self.goal_state = np.repeat(np.arange(6), 4)
        self.state = np.repeat(np.arange(6), 4)

        if moves or scrambled:
            self.scramble(moves)

    def scramble(self, moves: Moves | None = None):

        if moves is None:
            num_of_moves = np.random.randint(5, 11)
            moves = list(np.random.randint(len(MOVES), size=num_of_moves))

        self.state = Cube.move_state(self.state, moves)

    def move(self, move: Moves) -> Cube:
        cube = Cube()
        cube.state = Cube.move_state(self.clone_state(), move)
        return cube

    @staticmethod
    def move_state(state: np.ndarray, move: Moves) -> np.ndarray:
        move = Move.parse(move)

        if isinstance(move, list):
            for m in move:
                state = state[MOVES[m.value]]
        else:
            state = state[MOVES[move.value]]

        return state

    def clone_state(self) -> np.ndarray:
        return np.copy(self.state)

    def clone(self) -> Cube:
        cube = Cube()
        cube.state = self.clone_state()
        return cube

    def hash(self) -> str:
        return Cube.hash_state(self.state)

    @staticmethod
    def hash_state(state: np.ndarray) -> str:
        return ''.join(map(str, state))

    @staticmethod
    def _draw_corner(ax, position, colors):

        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]) + position

        indices = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                   (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]

        faces = [[vertices[idx] for idx in face] for face in indices]

        ax.add_collection3d(Poly3DCollection(faces, facecolors=colors, linewidths=1, edgecolors='black'))

    @staticmethod
    def _draw_cube(state: np.ndarray, ax):

        for corner, (state_idxs, color_idxs) in CORNERS.items():
            colors = ["gray"] * 6

            for sticker_idx, color_idx in zip(state_idxs, color_idxs):
                colors[color_idx] = COLORS[state[sticker_idx]]

            Cube._draw_corner(ax, corner, colors)

    @staticmethod
    def render_state(state):
        fig, ax = plt.subplots(figsize=(7, 5))
        base_coords = np.array([(0, 1), (1, 1), (0, 0), (1, 0)])
        offsets = np.array([[0, 0], [1, 0], [2, 0], [-1, 0], [0, 1], [0, -1]]) * 2

        idx = 0

        for offset in offsets:
            for coords in base_coords:
                rect = plt.Rectangle(coords + offset, 1, 1, edgecolor='black', linewidth=1)
                rect.set_facecolor(COLORS[state[idx]])
                ax.add_patch(rect)

                idx += 1

        ax.set_xlim(-2.1, 6.1)
        ax.set_ylim(-2.1, 4.1)
        ax.axis('off')
        plt.show()

    def render(self):
        Cube.render_state(self.state)

    def render3D(self):

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')

        Cube._draw_cube(self.state, ax)

        ax.axis('off')
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 2])
        ax.set_zlim([0, 2])
        plt.show()

    @staticmethod
    def render3D_moves(initial_state: np.ndarray, moves: MoveSequence, save: bool = False):
        moves = Move.parse(moves)

        original_state = np.copy(initial_state)
        state = initial_state

        fig = plt.figure(figsize=(4, 4), frameon=False)
        ax = fig.add_subplot(111, projection='3d')

        Cube._draw_cube(state, ax)

        ax.axis('off')
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 2])
        ax.set_zlim([0, 2])

        move_index = 0

        def init():
            Cube._draw_cube(state, ax)
            return ax

        def animate(i):
            nonlocal move_index

            if i == 0:  # For the initial frame, show the original state
                state[:] = np.copy(original_state)
                Cube._draw_cube(state, ax)

            else:
                if move_index < len(moves):  # Check if there are more moves to perform
                    state[:] = Cube.move_state(state, moves[move_index])
                    ax.clear()

                    Cube._draw_cube(state, ax)
                    move_index += 1
                    ax.axis('off')
                    ax.set_xlim([0, 2])
                    ax.set_ylim([0, 2])
                    ax.set_zlim([0, 2])
                else:

                    move_index = 0
                    state[:] = np.copy(original_state)
                    Cube._draw_cube(state, ax)

        ani = animation.FuncAnimation(fig, animate, frames=len(moves) + 2, init_func=init,
                                      interval=1000, blit=False)

        if save:
            ani.save('rubiks_cube_animation.gif', writer='pillow', fps=1)

        plt.show()
        return ani

    def render_text(self):
        lines = [
            [None, None, 16, 17],
            [None, None, 18, 19],
            [12, 13, 0, 1, 4, 5, 8, 9],
            [14, 15, 2, 3, 6, 7, 10, 11],
            [None, None, 20, 21],
            [None, None, 22, 23]
        ]

        for line in lines:
            print("".join(LETTERS[self.state[idx]] if idx is not None else " " for idx in line))


def wrong_placed_cubes_h1(cube):
    # This function calculates the heuristic h1 for a given Rubik's Cube state.
    # The heuristic h1 is defined as the number of cubes that are in the wrong position
    # when comparing the current state of the cube with the goal state.
    #
    # Parameters:
    # - cube: The Rubik's Cube object containing the current state.
    #
    # Returns:
    # - incorrect_cubes: The number of cubes that are not in their correct position
    #   when comparing the current state with the goal state.
    #
    # Explanation:
    # The function uses a zip operation to iterate over corresponding elements
    # in the current state and goal state. It creates a generator expression to count
    # the number of pairs (i, j) where i is the color in the current state,
    # and j is the color in the goal state, and i is not equal to j.
    # The sum function is then used to calculate the total count of incorrect cubes,
    # which is returned as the heuristic value.

    cube_state = None
    if isinstance(cube, Cube):
        cube_state = cube.state
    else:
        cube_state = cube

    incorrect_cubes = sum(1 for i, j in zip(cube_state, Cube().goal_state) if i != j)
    return incorrect_cubes


def opposite_move(move):
    # This function calculates the opposite move index for a given move index.
    # The cube has six possible moves represented by indices 0 to 5.
    # Opposite moves are such that applying the original move and its opposite
    # will undo each other's effects and bring the cube back to the original state.

    # Parameters:
    # - move: The move index for which the opposite move needs to be determined.

    # Returns:
    # - opposite: The opposite move index.

    # Explanation:
    # If the original move index is less than 3, then the opposite move is obtained
    # by adding 3 to the original move index. If the original move index is 3 or greater,
    # then the opposite move is obtained by subtracting 3 from the original move index.
    # This logic ensures that the opposite move undoes the effects of the original move.

    if move < 3:
        return move + 3
    return move - 3


def a_star(cube, heuristic):
    # This function performs the A* search algorithm to find the optimal sequence
    # of moves to solve a Rubik's Cube.

    # Parameters:
    # - cube: The Rubik's Cube object representing the initial state.
    # - heuristic: The heuristic function to evaluate the state of the cube.

    # Returns:
    # - path: A list of move indices representing the optimal sequence of moves to
    #   solve the Rubik's Cube.

    # Explanation:
    # The A* algorithm uses a priority queue to explore states in the order of their
    # estimated cost (priority). The cost of a state is the sum of the cost to reach
    # that state from the initial state and an estimated heuristic cost.
    # The priority queue is initially populated with the start state, its priority,
    # and an empty path. The costs dictionary keeps track of the cost to reach each state.
    # The algorithm continues until the priority queue is empty or the goal state is reached.
    # In each iteration, the state with the lowest priority is dequeued, and its neighbors
    # (resulting from applying each possible move) are added to the queue if they have not
    # been visited or if a lower cost path to them is discovered.

    priority_queue = [(0, tuple(cube.state), [])]
    costs = {tuple(cube.state): 0}
    state = 0
    while len(priority_queue) != 0:
        _, current_node, path = heapq.heappop(priority_queue)

        if np.array_equal(np.array(current_node), cube.goal_state):
            return path, len(costs)

        for move in range(6):
            child = Cube()
            child.state = cube.move_state(np.array(current_node), move)

            if tuple(child.state) not in costs or costs[current_node] + 1 < costs[tuple(child.state)]:
                costs[tuple(child.state)] = costs[current_node] + 1
                state += 1
                priority = costs[current_node] + 1 + heuristic(child)
                heapq.heappush(priority_queue, (priority, tuple(child.state), path + [move]))

    return []


def process_state(state, move, visited, queue, state2moves):
    # This function processes a state in the breadth-first search (BFS) bidirectional algorithm.
    # It generates child states resulting from applying each possible move to the current state.
    # If a child state has not been visited, it is added to the visited set, the queue, and
    # the dictionary state2moves is updated to store the path to this state from the initial state.

    # Parameters:
    # - state: The current state of the Rubik's Cube.
    # - move: The move index to be applied to the current state.
    # - visited: The list of visited states.
    # - queue: The BFS queue containing states to be processed.
    # - state2moves: A dictionary mapping state representations to the corresponding move sequences.

    child_state = tuple(Cube.move_state(np.array(state), move))
    if child_state not in visited:
        visited.append(child_state)
        queue.append((child_state, move))
        state2moves[str(child_state)] = copy.deepcopy(state2moves[str(state)])
        state2moves[str(child_state)].append(move)


def return_path(goal, state2moves_from_goal, state2moves_from_start):
    # This function constructs the final path from the initial state to the goal state
    # using move sequences stored in dictionaries state2moves_from_goal and state2moves_from_start.
    # It reverses the move sequence from the goal to the common state, converts each move to its
    # opposite move, and appends it to the move sequence from the initial state to the common state.

    # Parameters:
    # - goal: The goal state reached during bidirectional BFS.
    # - state2moves_from_goal: A dictionary mapping state representations to move sequences
    #   from the goal state to the common state.
    # - state2moves_from_start: A dictionary mapping state representations to move sequences
    #   from the initial state to the common state.

    path_from_state_to_goal = state2moves_from_goal[str(goal[0])]
    path_from_start_to_state = state2moves_from_start[str(goal[0])]
    path_from_state_to_goal.reverse()
    path_from_state_to_goal = [opposite_move(x) for x in path_from_state_to_goal]
    path_from_start_to_state.extend(path_from_state_to_goal)
    return path_from_start_to_state


def bfs_bidirectional(cube):
    # This function performs bidirectional BFS to find the optimal solution to the Rubik's Cube.
    # It explores states from both the start and goal simultaneously until they meet at a common state.
    # The function maintains two queues, start_queue and goal_queue, and two dictionaries,
    # state2moves_from_start and state2moves_from_goal, to store move sequences from the initial
    # state and goal state to each state. When a common state is found, the final path is constructed.

    # Parameters:
    # - cube: The Rubik's Cube object representing the initial state.

    # Returns:
    # - path: A list of move indices representing the optimal sequence of moves to solve the Rubik's Cube.

    start_queue = [(tuple(cube.state), None)]
    goal_queue = [(tuple(cube.goal_state), None)]
    start_visited = [tuple(cube.state)]
    goal_visited = [tuple(cube.goal_state)]
    state2moves_from_start = {}
    state2moves_from_goal = {}
    state2moves_from_start[str((tuple(cube.state)))] = []
    state2moves_from_goal[str((tuple(cube.goal_state)))] = []

    while (len(start_queue) != 0) and (len(goal_queue) != 0):
        start = start_queue[0]
        start_queue = start_queue[1:]

        goal = goal_queue[0]
        goal_queue = goal_queue[1:]

        if str(goal[0]) in state2moves_from_start.keys():
            return return_path(goal, state2moves_from_goal, state2moves_from_start), len(start_visited) + len(
                goal_visited)

        if str(start[0]) in state2moves_from_goal.keys():
            return return_path(start, state2moves_from_goal, state2moves_from_start), len(start_visited) + len(
                goal_visited)

        for move in range(6):
            process_state(start[0], move, start_visited, start_queue, state2moves_from_start)
            process_state(goal[0], move, goal_visited, goal_queue, state2moves_from_goal)
    return [], len(start_visited) + len(goal_visited)


N = 'N'
Q = 'Q'
PARENT = 'parent'
ACTIONS = 'actions'
current_state_node = "current_state_node"
how_to_go_to_parent = "how_to_go_to_parent"


def init_node(parent=None):
    return {N: 0, Q: 0, PARENT: parent, ACTIONS: {}, current_state_node: tuple, how_to_go_to_parent: None}


def select_action(node, c):
    # This function selects an action (move) based on the Upper Confidence Bound (UCB) formula.
    # It explores child nodes of the given parent node and calculates the UCB score for each child.
    # The action with the highest UCB score is selected.

    # Parameters:
    # - node: The parent node containing information about the current state and its child nodes.
    # - c: The exploration-exploitation trade-off parameter.

    # Returns:
    # - result: The selected action (move) based on the UCB formula.

    # Explanation:
    # The function iterates through each child node of the given parent node (node[ACTIONS]).
    # For each child, it calculates the UCB score using the formula:
    # UCB = Q(child) / N(child) + c * sqrt(2 * log(N(parent)) / N(child))
    # where Q(child) is the cumulative quality (reward) of the child,
    # N(child) is the number of visits to the child, and N(parent) is the number of visits to the parent.
    # The action with the highest UCB score is selected and returned.

    n_node = node[N]
    result = None
    best_quality = -1

    for move, child in node[ACTIONS].items():
        quality = child[Q] / child[N] + c * sqrt(2 * log(n_node) / child[N])
        if best_quality < quality:
            best_quality = quality
            result = move

    return result


def get_available_actions(node):
    # This function determines the available actions (moves) that have not been explored
    # in the given node of the search tree.

    # Parameters:
    # - node: The current node in the search tree containing information about explored actions.

    # Returns:
    # - result: A list of available actions (moves) that have not been explored in the current node.

    # Explanation:
    # The function initializes a list of all possible moves (available_moves).
    # It then iterates through the explored moves in the given node and appends them to the list
    # of explored_moves. The result is obtained by filtering out the explored_moves from available_moves,
    # leaving only the moves that have not been explored.

    available_moves = [0, 1, 2, 3, 4, 5]
    explored_moves = []
    for move, action in node[ACTIONS].items():
        explored_moves.append(move)
    result = [x for x in available_moves if x not in explored_moves]
    return result


def manhattan_distance_h2(state):
    # This function calculates the Manhattan distance heuristic (h2) for a given state of a 2x2x2 Rubik's Cube.
    # The Manhattan distance represents the sum of distances between each cubelet in the current state
    # and its corresponding position in the goal state.

    # Parameters:
    # - state: The current state of the 2x2x2 Rubik's Cube.

    # Returns:
    # - distance: The Manhattan distance heuristic value normalized by dividing it by 8.

    # Explanation:
    # The function defines the goal state for the 2x2x2 Rubik's Cube.
    # It then iterates through each cubelet in the current state and calculates the Manhattan distance
    # for that cubelet by summing the absolute differences in the x (mod 4) and y (integer division by 4) coordinates
    # between the current state and the goal state. The total Manhattan distance is obtained by summing
    # these distances for all cubelets. Finally, the total distance is normalized by dividing it by 8 and returned
    # as the heuristic value.

    goal_state = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    cube_state = None
    if isinstance(state, Cube):
        cube_state = state.state
    else:
        cube_state = state

    distance = 0
    for i in range(len(cube_state)):
        distance += abs(cube_state[i] % 4 - goal_state[i] % 4) + abs(
            cube_state[i] // 4 - goal_state[i] // 4)

    return distance // 8


def generate_moves(state, depth):
    # This function generates a sequence of random moves applied to the initial state of a 2x2x2 Rubik's Cube.
    # It continues generating moves until either the specified depth is reached or the cube reaches the solved state.

    # Parameters:
    # - state: The initial state of the 2x2x2 Rubik's Cube.
    # - depth: The maximum depth of moves to generate.

    # Returns:
    # - final_state: The final state of the cube after generating the specified
    #   number of moves or reaching the solved state.
    # - solved: A boolean indicating whether the cube is solved after generating the moves.
    # - moves: The sequence of moves applied to the cube.

    # Explanation:
    # The function initializes the initial hash of the state and maintains a list 'all_hash' to keep track of
    # the hash values and the remaining possible moves for each state in the generation process.
    # It generates random moves until either the specified depth is reached or the cube reaches the solved state
    # (the goal state is predefined).
    # The function returns the final state, a boolean indicating whether the cube is solved,
    # and the sequence of moves applied to the cube.

    initial_hash = Cube.hash_state(state)
    all_hash = [(initial_hash, [0, 1, 2, 3, 4, 5])]
    moves = []
    states_calc = [state]
    j = 0
    while j < depth:
        if j < 0:
            # Theoretically it should not reach here for a depth that does not exceed
            # the total number of states that the 2x2x2 cube can have
            raise Exception("There are not \"depth\" distinct states "
                            "starting from the same state using all possible moves")

        hash2moves = all_hash[j]
        # If no movement brings you to a new state, return to the last visited state
        if len(hash2moves[1]) == 0:
            all_hash.pop()
            moves.pop()
            states_calc.pop()
            j -= 1
            continue

        choice_number = random.choice(hash2moves[1])
        hash2moves[1].remove(choice_number)
        new_state = Cube.move_state(np.array(states_calc[j]), choice_number)

        if tuple(new_state) == (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5):
            moves.append(choice_number)
            return tuple(new_state), True, moves

        # New state found (add to all_hash)
        new_state_hash = Cube.hash_state(new_state)
        if new_state_hash not in [x[0] for x in all_hash]:
            all_hash.append((new_state_hash, [0, 1, 2, 3, 4, 5]))
            moves.append(choice_number)
            states_calc.append(new_state)
            j += 1

    return tuple(states_calc[-1]), False, moves


def mcts(budget, cube, constant, depth, heuristic, max_heuristic_value):
    # This function performs the Monte Carlo Tree Search (MCTS) algorithm to find an optimal sequence of moves
    # for solving a Rubik's Cube within a specified budget.

    # Parameters:
    # - budget: The maximum number of iterations or simulations to perform in the search.
    # - cube: The Rubik's Cube object representing the initial state.
    # - constant: The exploration-exploitation trade-off constant for the UCB formula.
    # - depth: The depth of moves to generate in case an unexplored action is encountered.
    # - heuristic: The heuristic function to evaluate the state of the cube.
    # - max_heuristic_value: The maximum heuristic value used for reward calculation.

    # Returns:
    # - path: The optimal sequence of moves obtained from the MCTS search.

    # Explanation:
    # The function initializes the goal state and the root of the search tree.
    # It iteratively performs simulations (iterations) to explore the tree and update the node values.
    # For each iteration, the function selects actions based on the UCB formula and explores the tree
    # until a leaf node is reached or the cube is solved. If the leaf node corresponds to an unexplored state,
    # it generates random moves to explore the tree further. The function then back-propagates the rewards obtained
    # during the simulations to update the quality and visit counts of the nodes in the tree.
    # The optimal sequence of moves is determined based on the highest quality values during backpropagation.

    goal_state = tuple(cube.goal_state)
    tree = init_node()
    path = []
    number_of_states = 0
    for x in range(budget):
        state = tuple(cube.state)
        node = tree

        while state != goal_state:
            available_actions = get_available_actions(node)
            actions_exist_in_node = all(action in node[ACTIONS] for action in available_actions)

            # UCB
            if actions_exist_in_node:
                new_action = select_action(node, constant)
                state = tuple(Cube.move_state(np.array(state), new_action))
                node = node[ACTIONS][new_action]
            else:
                break

        # Create new node
        if state != goal_state:
            unexplored_actions = get_available_actions(node)
            if unexplored_actions:
                new_action = random.choice(unexplored_actions)
                state = tuple(Cube.move_state(np.array(state), new_action))
                node = init_node(node)
                node[how_to_go_to_parent] = new_action
                node[current_state_node] = state
                node[PARENT][ACTIONS][new_action] = node
                number_of_states += 1

        # Simulation
        moves_as_fuck = generate_moves(state, depth)
        state = moves_as_fuck[0]
        if state == goal_state:
            before_moves = []
            new_node = node
            # Reconstruct path
            while not (new_node is None):
                before_moves.append(new_node[how_to_go_to_parent])
                new_node = new_node[PARENT]

            # Remove None (because parent of root is None)
            before_moves.pop()
            before_moves.reverse()
            before_moves.extend(moves_as_fuck[2])
            return before_moves, number_of_states

        # Calculate reward
        cube1 = Cube()
        cube1.state = state
        cube1.goal_state = goal_state
        reward = max_heuristic_value - heuristic(cube1.state)
        if reward < 0:
            raise Exception("Reward < 0")

        # Backpropagation
        current_node = node
        while current_node:
            current_node[N] += 1
            current_node[Q] += reward
            current_node = current_node[PARENT]

    # If you enter here, the solution was not found, but it will display the best solution depending on the reward
    while tree[ACTIONS]:
        q_max = -1
        move = -1
        for k, v in tree[ACTIONS].items():
            if v[Q] > q_max:
                q_max = v[Q]
                move = k

        path.append(move)
        tree = tree[ACTIONS][move]

    return path, number_of_states


def process_state_database(state, move, visited, queue, state2moves):
    child_state = tuple(Cube.move_state(np.array(state), move))
    if child_state not in visited:
        visited.append(child_state)
        queue.append((child_state, move))
        state2moves[str(child_state)] = copy.deepcopy(state2moves[str(state)])
        state2moves[str(child_state)].insert(0, opposite_move(move))


database = {}


def pattern_database(state_final):
    start_queue = [(tuple(state_final), None)]
    global database
    database[str(tuple(state_final))] = []
    visited = [tuple(state_final)]
    while len(start_queue) != 0:
        start = start_queue[0]
        start_queue = start_queue[1:]

        for move in range(6):
            if len(database[str(start[0])]) < 7:
                process_state_database(start[0], move, visited, start_queue, database)


def find_in_database(state):
    try:
        global database
        return database[str(tuple(state))]
    except Exception:
        return []


def h3(state):
    try:
        if isinstance(state, Cube):
            cube_state = state.state
        else:
            cube_state = state
        global database
        return len(database[str(tuple(cube_state))])
    except KeyError:
        if isinstance(state, Cube):
            cube_state = state.state
        else:
            cube_state = state
        return wrong_placed_cubes_h1(cube_state)


case1 = "R U' R' F' U"
case2 = "F' R U R U F' U'"
case3 = "F U U F' U' R R F' R"
case4 = "U' R U' F' R F F U' F U U"

pattern_database(Cube().goal_state)


# print(bfs_bidirectional(cube1))
# print("Algorithm")
# for i in range(4):
#     if i == 0:
#         cube1 = Cube(case1)
#     if i == 1:
#         cube1 = Cube(case2)
#     if i == 2:
#         cube1 = Cube(case3)
#     if i == 3:
#         cube1 = Cube(case4)
#     for i in range(10):
#         start_time = time.time()
#         solution = mcts(20000, cube1, 0.5, 14, h3, 31)
#         end_time = time.time()
#         print("state = ", cube1.move(solution[0]).state, " nr_states = ", solution[1], " time = ", end_time - start_time)
#         if (cube1.move(solution[0]).state == cube1.goal_state).all():
#             count += 1
#     count = 0
#     print(count)



def run_algorithm(algorithm, cube, heuristic=None):
    start_time = time.time()
    if heuristic:
        solution = algorithm(cube, heuristic)
    else:
        solution = algorithm(cube)
    end_time = time.time()
    execution_time = end_time - start_time
    return solution[0], solution[1], execution_time


def compare_algorithms(num_trials, heuristic, case):
    a_star_times = []
    bfs_times = []
    a_star_states_discovered = []
    bfs_states_discovered = []
    a_star_path_lengths = []
    bfs_path_lengths = []

    for i in range(num_trials):
        print(i)
        cube = Cube(case)

        # A* algorithm
        a_star_path, a_star_states, a_star_time = run_algorithm(a_star, cube, heuristic)
        a_star_times.append(a_star_time)
        a_star_states_discovered.append(a_star_states)
        a_star_path_lengths.append(len(a_star_path))

        # BFS algorithm
        bfs_path, bfs_states, bfs_time = run_algorithm(bfs_bidirectional, cube)
        bfs_times.append(bfs_time)
        bfs_states_discovered.append(bfs_states)
        bfs_path_lengths.append(len(bfs_path))

    # Plotting
    plt.figure(figsize=(10, 6))

    # Execution time
    plt.subplot(3, 1, 1)
    plt.plot(range(num_trials), a_star_times, label='A*')
    plt.plot(range(num_trials), bfs_times, label='BFS')
    plt.title('Execution Time Comparison')
    plt.xlabel('Trial')
    plt.ylabel('Time (seconds)')
    plt.legend()

    # States discovered
    plt.subplot(3, 1, 2)
    plt.plot(range(num_trials), a_star_states_discovered, label='A*')
    plt.plot(range(num_trials), bfs_states_discovered, label='BFS')
    plt.title('States Discovered Comparison')
    plt.xlabel('Trial')
    plt.ylabel('Number of States')
    plt.legend()

    # Path lengths
    plt.subplot(3, 1, 3)
    plt.plot(range(num_trials), a_star_path_lengths, label='A*')
    plt.plot(range(num_trials), bfs_path_lengths, label='BFS')
    plt.title('Path Length Comparison')
    plt.xlabel('Trial')
    plt.ylabel('Path Length')
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_mcts(budget, constant, depth, heuristic, max_heuristic_value, cube):
    start_time = time.time()
    solution = mcts(budget, cube, constant, depth, heuristic, max_heuristic_value)
    end_time = time.time()
    execution_time = end_time - start_time
    return solution[0], solution[1], execution_time


def compare_mcts():
    num_trials = 3
    budgets = [1000, 5000, 10000, 20000]
    constants = [0.1, 0.5]
    heuristics = [manhattan_distance_h2, wrong_placed_cubes_h1]

    plt.figure(figsize=(30, 20))

    for i, constant in enumerate(constants):
        print(i)
        for j, budget in enumerate(budgets):
            for k, heuristic in enumerate(heuristics):
                mcts_times = []
                mcts_states_discovered = []
                mcts_path_lengths = []

                for _ in range(num_trials):
                    cube = Cube(case1)
                    if heuristic == manhattan_distance_h2:
                        max_heuristic_value = 10
                    if heuristic == wrong_placed_cubes_h1:
                        max_heuristic_value = 24
                    path, states, time_taken = run_mcts(budget, constant, 14, heuristic, max_heuristic_value, cube)
                    mcts_times.append(time_taken)
                    mcts_states_discovered.append(states)
                    mcts_path_lengths.append(len(path))

                # Plotting
                plt.subplot(len(constants), len(budgets), i * len(budgets) + j + 1)
                plt.plot(range(num_trials), mcts_times,
                         label=f'MCTS (Budget={budget}, Constant={constant}, Heuristic={heuristic.__name__}, Max Heuristic Value={max_heuristic_value})')
                plt.title(f'Comparison (Constant={constant}, Budget={budget}, Heuristic={heuristic.__name__})')
                plt.xlabel('Trial')
                plt.ylabel('Time (seconds)')
                plt.legend()

    plt.tight_layout()
    plt.show()

# compare_algorithms(10, h3, case1)
# compare_algorithms(10, h3, case2)
# compare_algorithms(10, h3, case3)
# compare_algorithms(10, h3, case4)

