import numpy as np
import os
import copy
import time
import datetime
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from tools import encode_board, can_move, default_policy, get_time_delta, get_time_str


class TreeNode:
    def __init__(self, color, board = None, parent = None):
        self.color = color
        self.board = board
        self.parent = parent
        self.children = {}
        self.q = 0
        self.n = 0

    def expand(self, action_color_board):
        for action, color, board in action_color_board:
            self.children[action] = TreeNode(color, board, self)

    def select(self, c_puct, p):
        non_tried_subnode = []
        best_node = None
        best_value = None
        all_tried = True
        for act, node in self.children.items():
            if node.n == 0:
                non_tried_subnode.append((act, node))
                all_tried = False
            else:
                current_value = node.get_value(c_puct)
                if all_tried and (best_value is None or current_value > best_value):
                    best_node = (act, node)
                    best_value = current_value
        if not all_tried:
            best_node = non_tried_subnode[np.random.choice(len(non_tried_subnode))]
        else:
            if np.random.random() < p:
                items = list(self.children.items())
                best_node = items[np.random.choice(len(items))]
        return best_node

    def get_value(self, c_puct = 0):
        return self.q + c_puct * np.sqrt(2 * np.log(self.parent.n) / self.n)
        # return self.q + c_puct * self.p * np.sqrt(self.parent.n) / (1 + self.n)

    def update(self, value):
        self.n += 1
        self.q += (value - self.q) / self.n

    def update_recursive(self, value):
        if not self.is_root():
            root_color = self.parent.update_recursive(value)
            if root_color != self.parent.color:
                value = -value
        else:
            root_color = self.color

        self.update(value)
        return root_color

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None


def policy_value_function(state, color, eval_color, times, executor, limit = 1000):
    value = default_policy(state, color, eval_color, times, executor, limit)

    action_color_board = []
    actions = list(state.get_legal_actions(color))
    for i, act in enumerate(actions):
        new_state = copy.deepcopy(state)
        new_state._move(act, color)
        new_color = 'X' if color == 'O' else 'O'
        if not can_move(new_state, new_color):
            new_color = color
        action_color_board.append((act, new_color, new_state))

    return action_color_board, value


class MCTS:
    def __init__(self, color, terminal_condition, c_puct, p, times, thread_nums):
        self.color = color
        self.root = TreeNode(color)
        if isinstance(terminal_condition, int):
            self.terminal_condition = datetime.timedelta(seconds = terminal_condition)
        else:
            self.terminal_condition = get_time_delta(terminal_condition)
        self.c_puct = c_puct
        self.p = p
        self.times = times
        if thread_nums is None:
            thread_nums = multiprocessing.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers = thread_nums)

    def __getstate__(self):
        return self.color, self.root, self.terminal_condition, self.c_puct, self.p, self.times, self.executor._max_workers

    def __setstate__(self, state):
        self.color, self.root, self.terminal_condition, self.c_puct, self.p, self.times, thread_nums = state
        self.executor = ProcessPoolExecutor(max_workers = thread_nums)

    def update_param(self, color, terminal_condition, c_puct, p, times, thread_nums, **kwargs):
        if color is not None and color != self.color:
            self.color = color
            self.root = TreeNode(color)
        if terminal_condition is not None:
            if isinstance(terminal_condition, int):
                self.terminal_condition = datetime.timedelta(seconds = terminal_condition)
            else:
                self.terminal_condition = get_time_delta(terminal_condition)
        if c_puct is not None:
            self.c_puct = c_puct
        if p is not None:
            self.p = p
        if times is not None:
            self.times = times
        if thread_nums is not None:
            self.executor = ProcessPoolExecutor(max_workers = thread_nums)

    def play_out(self, state):
        node = self.root
        color = node.color

        while not node.is_leaf():
            action, node = node.select(self.c_puct, self.p)
            state._move(action, color)
            color = node.color

        end = not can_move(state, 'X') and not can_move(state, 'O')

        policy_time = time.time()
        if end:
            winner, value = state.get_winner()
            if winner != 2 and ['X', 'O'][winner] != self.root.color:
                value = -value
        else:
            action_color_board, value = policy_value_function(state, color, self.root.color, self.times, self.executor)
            node.expand(action_color_board)
        policy_time = time.time() - policy_time

        node.update_recursive(value)
        return policy_time

    def get_move(self, state, reset):
        start_time = datetime.datetime.now()
        if reset:
            self.reset()
        self.root.board = copy.deepcopy(state)
        avg_policy_time = 0
        self.i = 0
        while datetime.datetime.now() - start_time < self.terminal_condition:
            play_out_time = time.time()
            state_copy = copy.deepcopy(state)
            policy_time = self.play_out(state_copy)
            avg_policy_time += policy_time
            self.i += 1
            print(' ' * 130, end = '\r')
            format_str = 'play out: {}\tplay out time: {:.2f}s\tpolicy time: {:.2f}s\ttotal time: {}'
            print(format_str.format(self.i, time.time() - play_out_time, policy_time,
                                    get_time_str(datetime.datetime.now() - start_time)), end = '\r')

        format_str = 'play out times: {}\tavg play out time: {}\tavg policy time: {:.2f}s'
        print()
        print(format_str.format(self.i, get_time_str((datetime.datetime.now() - start_time) / self.i),
                                avg_policy_time / self.i))

        act_q = [(act, node.n) for act, node in self.root.children.items()]

        return max(act_q, key = lambda item: item[1])[0]

    def reset(self):
        self.root = TreeNode(self.color)


class MctsPlayer:
    """
        AI player use policy_value_function to control it
        use policy_value_function = None to use pure player
    """

    def __init__(self, color, terminal_condition = 59, c_puct = 1, p = 0.25, times = 1, thread_nums = None,
                 look_up_table = None, **kwargs):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color
        self.backend = MCTS(color, terminal_condition, c_puct, p, times, thread_nums)
        if look_up_table is not None:
            look_up_table = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), look_up_table),
                                         'manual.pkl')
            self.look_up_table = pickle.load(open(look_up_table, 'rb'))
        else:
            self.look_up_table = None

    def update_param(self, color = None, terminal_condition = None, c_puct = None, p = None, times = None,
                     thread_nums = None, look_up_table = None):
        if color is not None:
            self.color = color
        if look_up_table is not None:
            look_up_table = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), look_up_table),
                                         'manual.pkl')
            self.look_up_table = pickle.load(open(look_up_table, 'rb'))
        else:
            self.look_up_table = None
        self.backend.update_param(color, terminal_condition, c_puct, p, times, thread_nums)

    def reset(self):
        self.backend.reset()

    def get_move(self, board, reset = True):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------

        if self.look_up_table is not None:
            encode, _ = encode_board(board, self.color)
            if encode in self.look_up_table:
                action = self.look_up_table[encode]
                if isinstance(action, list):
                    action = np.random.choice(action)
            else:
                action = self.backend.get_move(board, reset)
        else:
            action = self.backend.get_move(board, reset)
        # ------------------------------------------------------------------------

        return action
