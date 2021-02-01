import numpy as np
import os
import copy
import datetime
import time
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from tools import encode_board, can_move, default_policy, get_time_str


class TreeNode:
    def __init__(self, color, board = None, parent = None):
        self.color = color
        self.parent = parent
        self.children = {}
        self.board = board
        self.q = 0

    def best_child(self):
        return max(self.children.items(), key = lambda item: item[1].q)

    def worst_child(self):
        return min(self.children.items(), key = lambda item: item[1].q)

    def expand(self, action_color_board):
        for action, color, board in action_color_board:
            if action not in self.children:
                self.children[action] = TreeNode(color, board, self)

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None


def expand(state, node, color, player_color, cut_off_depth, times, executor, start_time, create_manual, alpha = None,
           beta = None, depth = 0, process = 0., weight = 1., policy_duration = 0.25):
    end = not can_move(state, 'X') and not can_move(state, 'O')
    if end:
        cut_off = True
    else:
        # if process < 0.1:
        #     cut_off = depth >= cut_off_depth
        # else:
        #     cut_off = ((datetime.datetime.now() - start_time) / process).total_seconds() > 50

        cut_off = depth >= cut_off_depth

        # actions = list(state.get_legal_actions(color))
        # n = len(actions)
        # cut_off = weight / n < 0.001

        # if policy_duration is None:
        #     cut_off = weight / n < 0.001
        # else:
        #     cut_off = n / weight > (55 / policy_duration) ** 1

    if cut_off:
        if end:
            policy_time = None
            winner, value = state.get_winner()
            if winner != 2 and ['X', 'O'][winner] != player_color:
                value = -value
        else:
            policy_time = time.time()
            value = default_policy(state, color, player_color, times, executor)
            policy_time = time.time() - policy_time
        node.q = value

        if create_manual:
            process += weight
            now_time = datetime.datetime.now()
            used_time = now_time - start_time
            total_time = used_time / process
            rest_time = total_time - used_time

            print(' ' * 130, end = '\r')
            format_str = '{:.2f}%\tused time: {}\trest time: {}\ttotal time: {}\texcept end time: {}'
            print(format_str.format(process * 100, get_time_str(used_time), get_time_str(rest_time),
                                    get_time_str(total_time), (now_time + rest_time).strftime('%Y-%m-%d %H:%M:%S')),
                  end = '\r')
        # else:
        #     n = len(list(state.get_legal_actions(color)))
        #     print('n: {}\tcurrent weight: {}\tnext weight: {}'.format(n, weight, weight / n if n != 0 else 0))
    else:
        actions = list(state.get_legal_actions(color))
        if not create_manual:
            np.random.shuffle(actions)
        n = len(actions)
        policy_time = policy_duration
        policy_times = 0
        for i, act in enumerate(actions):
            new_state = copy.deepcopy(state)
            new_state._move(act, color)
            new_color = 'X' if color == 'O' else 'O'
            if not can_move(new_state, new_color):
                new_color = color
            sub_node = TreeNode(new_color, new_state, node)
            node.children[act] = sub_node
            current_policy_time = expand(new_state, sub_node, new_color, player_color, cut_off_depth, times, executor,
                                         start_time, create_manual, alpha, beta, depth + 1, process + (i / n) * weight,
                                         weight / n, policy_time)
            if current_policy_time is not None:
                policy_times += 1
                if policy_time is None:
                    policy_time = current_policy_time
                else:
                    policy_time += (current_policy_time - policy_time) / policy_times
            if color == player_color:
                if alpha is None or sub_node.q > alpha:
                    alpha = sub_node.q
                if not create_manual and beta is not None and sub_node.q >= beta:
                    break
            else:
                if beta is None or sub_node.q < beta:
                    beta = sub_node.q
                if not create_manual and alpha is not None and sub_node.q <= alpha:
                    break

        if color == player_color:
            node.q = node.best_child()[1].q
        else:
            node.q = node.worst_child()[1].q

    return policy_time


class MiniMax:
    def __init__(self, color, cut_off_depth, times, thread_nums, create_manual, **kwargs):
        self.color = color
        self.root = TreeNode(color)
        self.cut_off_depth = cut_off_depth
        self.times = times
        self.create_manual = create_manual
        if thread_nums is None:
            thread_nums = multiprocessing.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers = thread_nums)

    def __getstate__(self):
        return self.color, self.root, self.cut_off_depth, self.times, self.create_manual, self.executor._max_workers

    def __setstate__(self, state):
        self.color, self.root, self.cut_off_depth, self.times, self.create_manual, thread_nums = state
        self.executor = ProcessPoolExecutor(max_workers = thread_nums)

    def update_param(self, color, cut_off_depth, times, thread_nums, create_manual, **kwargs):
        if color is not None and color != self.color:
            self.color = color
            self.root = TreeNode(color)
        if cut_off_depth is not None:
            self.cut_off_depth = cut_off_depth
        if times is not None:
            self.times = times
        if thread_nums is not None:
            self.executor = ProcessPoolExecutor(max_workers = thread_nums)
        if create_manual is not None:
            self.create_manual = create_manual

    def reset(self):
        self.root = TreeNode(self.color)

    def get_move(self, state):
        self.reset()
        self.root.board = copy.deepcopy(state)
        policy_time = expand(state, self.root, self.color, self.color, self.cut_off_depth, self.times, self.executor,
                             datetime.datetime.now(), self.create_manual)
        print()
        if policy_time is None:
            print('q: {:.2f}\tpolicy_time: 0s'.format(self.root.q))
        else:
            print('q: {:.2f}\tpolicy_time: {:.2f}s'.format(self.root.q, policy_time))
        return self.root.best_child()[0]


class MiniMaxPlayer:
    """
        AI player use policy_value_function to control it
        use policy_value_function = None to use pure player
    """

    def __init__(self, color, cut_off_depth = 4, times = 3, thread_nums = None, look_up_table = None,
                 create_manual = False):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color
        self.cut_off_depth = cut_off_depth
        self.backend = MiniMax(color, cut_off_depth, times, thread_nums, create_manual)
        if look_up_table is not None:
            look_up_table = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), look_up_table),
                                         'manual.pkl')
            self.look_up_table = pickle.load(open(look_up_table, 'rb'))
        else:
            self.look_up_table = None
        self.n = 0
        self.avg_duration = 0
        self.max_duration = 0

    def update_param(self, color = None, cut_off_depth = None, times = None, thread_nums = None, look_up_table = None,
                     create_manual = None):
        if color is not None:
            self.color = color
        if cut_off_depth is not None:
            self.cut_off_depth = cut_off_depth
        if look_up_table is not None:
            look_up_table = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), look_up_table),
                                         'manual.pkl')
            self.look_up_table = pickle.load(open(look_up_table, 'rb'))
        else:
            self.look_up_table = None
        self.backend.update_param(color, cut_off_depth, times, thread_nums, create_manual)

    def reset(self):
        self.backend.reset()
        self.max_duration = 0
        self.avg_duration = 0

    def get_move(self, board, reset = True):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        start_time = time.time()
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------
        if self.look_up_table is not None:
            if board.count('.') >= 59 - 8 + self.cut_off_depth:
                encode, _ = encode_board(board, self.color)
                action = self.look_up_table[encode // 2]
                if isinstance(action, list):
                    action = np.random.choice(action)
            else:
                action = self.backend.get_move(board)
        else:
            action = self.backend.get_move(board)
        # ------------------------------------------------------------------------
        duration = time.time() - start_time

        self.n += 1
        self.avg_duration += (duration - self.avg_duration) / self.n
        if duration > self.max_duration:
            self.max_duration = duration
        format_str = 'duration: {:.2f}s\tavg duration: {:.2f}s\tmax duration: {:.2f}s'
        print(format_str.format(duration, self.avg_duration, self.max_duration))

        return action
