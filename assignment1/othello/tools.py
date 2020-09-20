import requests
import datetime
import time
import hmac
import base64
import urllib
import json
import copy
import os
import numpy as np
from hashlib import sha256
from concurrent.futures import as_completed

from board import Board

import logging
import yaml
import sys


def parse_config(file, display = True):
    config_str = ''
    for line in open(file, 'r').readlines():
        if '#' in line:
            config_str += line[:line.find('#')]
        else:
            config_str += line
    cfg = yaml.load(config_str, Loader = yaml.FullLoader)
    res_str = 'config file from {}\n'.format(file) + 'config file start\n' + config_str
    if res_str[-1] != '\n':
        res_str += '\n'
    res_str += 'config file end\n'
    if display:
        print(res_str)
        sys.stdout.flush()
    return cfg, res_str
    # return DictObj(cfg), res_str


def build_logger(path):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Log等级总开关

    # 第二步，创建一个handler，用于写入日志文件
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    fh = logging.FileHandler(log_path, mode = 'a')
    fh.setLevel(logging.INFO)  # 输出到file的log等级的开关

    # 第三步，再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 输出到console的log等级的开关

    # 第四步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def clear_log(cfg):
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['log']['path'])
    if os.path.exists(log_path):
        lines = open(log_path, 'r').readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'Start new evaluation' in line:
                j = i + 1
                while j < len(lines) and 'Start new evaluation' in lines[j]:
                    j += 1
                for x in range(j - i - 1):
                    del lines[i]
                if i == len(lines) - 1 and 'Start new evaluation' in lines[i]:
                    del lines[i]
            i += 1
        f = open(log_path, 'w')
        for line in lines:
            f.write(line)


def get_time_str(time):
    import math
    time = time.total_seconds()
    unit = 0
    factor = 1000
    for i in range(2):
        if time > 60:
            time /= 60
            unit += 1
            factor = 60
        else:
            break
    if unit == 2 and time > 24:
        time /= 24
        unit += 1
        factor = 24
    res = '{0:}{2} {1:.3g}{3}'.format(int(time), math.modf(time)[0] * factor, ['s', 'min', 'h', 'd'][unit],
                                      ['s', 'min', 'h', 'd', 'ms'][unit - 1])
    return res


def get_time_delta(time_str):
    if 'd' in time_str:
        days, time_str = time_str.split('d')
        days = float(days)
    else:
        days = 0
    if 'h' in time_str:
        hours, time_str = time_str.split('h')
        hours = float(hours)
    else:
        hours = 0
    if 'min' in time_str:
        index = time_str.find('min')
        minutes, time_str = float(time_str[:index]), time_str[index:]
    else:
        minutes = 0
    if 's' in time_str:
        seconds, time_str = time_str.split('s')
        seconds = float(seconds)
    else:
        seconds = 0
    if time_str:
        try:
            micro_seconds = float(time_str)
        except ValueError:
            micro_seconds = 0
    else:
        micro_seconds = 0

    return datetime.timedelta(days = days, hours = hours, minutes = minutes, seconds = seconds,
                              microseconds = micro_seconds)


def report_res(res):
    header = {
        "Content-Type": "application/json"
    }
    data = {
        "msgtype": "text",
        "text": {
            "content": res
        },
        "at": {
            "isAtAll": False
        }
    }

    timestamp = int(round(time.time() * 1000))
    private_key = 'SEC16a378f2e7577c068e8409857eb0eb7e2dbc753f62ffb2c6f6685771102e39bb'
    sign_str = '{}\n{}'.format(timestamp, private_key).encode('utf-8')
    private_key = private_key.encode('utf-8')
    sign = urllib.parse.quote_plus(base64.b64encode(hmac.new(private_key, sign_str, digestmod = sha256).digest()))

    r = requests.post(
        'https://oapi.dingtalk.com/robot/send?access_token=4e8440ea345a8ad99bab3b4e91af33ee119193c25981b8f3c9c48baf1f9403fa&timestamp={}&sign={}'.format(
            timestamp, sign), data = json.dumps(data), headers = header)

    print(r.text)


def can_move(board, color):
    return len(list(board.get_legal_actions(color))) > 0


def board_equal(board_fir, board_sec):
    equal = True
    for i in range(8):
        for j in range(8):
            if board_fir[i][j] != board_sec[i][j]:
                equal = False
                break
    if equal:
        return True
    else:
        equal = True
    for i in range(8):
        for j in range(8):
            if board_fir[i][j] != board_sec[7 - i][7 - j]:
                equal = False
                break
    if equal:
        return True
    else:
        equal = True
    for i in range(8):
        for j in range(8):
            if board_fir[i][j] != board_sec[j][i]:
                equal = False
                break
    if equal:
        return True
    else:
        equal = True
    for i in range(8):
        for j in range(8):
            if board_fir[i][j] != board_sec[7 - j][7 - i]:
                equal = False
                break
    return equal


def store_value(manual_item, value_dict, color):
    current_board = copy.deepcopy(manual_item['board'])
    current_board._move(value_dict['action'], color)

    for j in range(len(manual_item['action'])):
        store_board = copy.deepcopy(manual_item['board'])
        if isinstance(manual_item['action'][j], list):
            store_board._move(manual_item['action'][j][0], color)
            if board_equal(current_board, store_board):
                for key in value_dict:
                    manual_item[key][j].append(value_dict[key])
                break
        else:
            store_board._move(manual_item['action'][j], color)
            if board_equal(current_board, store_board):
                for key in value_dict:
                    manual_item[key][j] = [manual_item[key][j], value_dict[key]]
                break
    else:
        for key in value_dict:
            manual_item[key].append(value_dict[key])


def policy_value_function_worker(state, color, eval_color, times, limit = 1000):
    value = 0
    for t in range(times):
        current_state = copy.deepcopy(state)
        current_color = color
        for i in range(limit):
            end = not can_move(current_state, 'X') and not can_move(current_state, 'O')
            if end:
                winner, current_value = current_state.get_winner()
                break
            if can_move(current_state, current_color):
                # np.random.seed(1)
                act = np.random.choice(list(current_state.get_legal_actions(current_color)))
                current_state._move(act, current_color)
            other_color = 'X' if current_color == 'O' else 'O'
            if can_move(current_state, other_color):
                current_color = other_color
        else:
            winner = 2
            current_value = 0
            print('roll out reach limit')

        if winner != 2 and eval_color != ['X', 'O'][winner]:
            current_value = -current_value
        value += current_value

    return value


def default_policy(state, color, eval_color, times, executor, limit = 1000):
    thread_nums = executor._max_workers
    tasks = [executor.submit(policy_value_function_worker, state, color, eval_color, times, limit) for i in
             range(thread_nums)]
    value = 0
    for i, res in enumerate(as_completed(tasks)):
        value += res.result()
    n = thread_nums * times

    value /= n
    return value


def worker(arg_list, done_list, semaphore, make_step, index):
    while not all(done_list):
        run_str = ''
        for i in range(len(done_list)):
            if not done_list[i]:
                run_str += str(i) + ' '
        if run_str == '':
            print('thread {}: no thread still running'.format(index))
        else:
            print('thread {}: threads {}still running'.format(index, run_str))
        print('thread {}: acquire a param'.format(index))
        if semaphore.acquire(timeout = np.random.random()):
            args, kwargs = arg_list.pop()
            done_list[index] = False
            print('thread {}: get one param, set undone'.format(index))
            have_arg = True
            while have_arg:
                print('thread {}: start make step'.format(index))
                try:
                    current_arg_list = make_step(*args, **kwargs)
                except Exception as e:
                    print('thread {}: make step error, terminal'.format(index))
                    print(e)
                    done_list[index] = True
                    return
                n = len(current_arg_list)
                print('thread {}: end make step return {} params'.format(index, n))
                if n == 0:
                    have_arg = False
                    done_list[index] = True
                    print('thread {}: set done'.format(index))
                else:
                    args, kwargs = current_arg_list.pop()
                    n -= 1
                    if n != 0:
                        arg_list.extend(current_arg_list)
                        for i in range(n):
                            semaphore.release()
                        print('thread {}: put {} params'.format(index, n))
        else:
            done_list[index] = True
            print('thread {}: set done'.format(index))


# def parallel(make_step):
#     def wrapper(threads, manager, *args, **kwargs):
#         pool = multiprocessing.Pool(threads)
#         threads = pool._processes
#         semaphore = manager.Semaphore()
#         arg_list = manager.list()
#         done_list = manager.list([False for i in range(threads)])
#         arg_list.append((args, kwargs))
#         pool.starmap(worker, [(arg_list, done_list, semaphore, make_step, i) for i in range(threads)])
#         pool.close()
#         pool.join()
#
#     return wrapper
#
#
# class TraverseTreeParallel:
#     def __init__(self, make_step):
#         self.make_step = make_step
#
#     def __call__(self, manual, node, color = None, depth = 1):
#         self.make_step(manual, node, color, depth)
#         args_list = []
#         for item in node.children.values():
#             if not item.is_leaf():
#                 args_list.append(((manual, item, color, depth + 1), {}))
#         return args_list


def traverse_tree(make_step):
    def wrapper(manual, node, color = None, depth = 1):
        make_step(manual, node, color, depth)
        for item in node.children.values():
            if not item.is_leaf():
                wrapper(manual, item, color, depth + 1)

    return wrapper


def make_manual(make_step):
    def wrapper(debug_manual, manual, depth_dict):
        for encode, item in debug_manual.items():
            act = make_step(item, depth_dict)
            if act is not None:
                manual[encode] = act

    return wrapper


@traverse_tree
def make_mcts_debug_manual(manual, node, color, depth):
    encodes, boards = encode_board(node.board, node.color, True)
    for act, sub_node in node.children.items():
        acts = []
        x, y = node.board.board_num(act)
        for item in get_equal_action(x, y):
            acts.append(node.board.num_board(item))
        for i in range(len(encodes)):
            if encodes[i] not in manual:
                manual[encodes[i]] = {'action': [acts[i]], 'q': [sub_node.q * sub_node.n], 'n': [sub_node.n],
                                      'parent_n': [sub_node.n + 1], 'node': [node], 'board': boards[i],
                                      'color': node.color, 'depth': depth}
            else:
                store_value(manual[encodes[i]], {'action': acts[i], 'node': sub_node, 'q': node.q * node.n,
                                                 'n': sub_node.n, 'parent_n': sub_node.n}, node.color)


@make_manual
def make_mcts_manual(debug_manual_item, depth_dict):
    q = copy.deepcopy(debug_manual_item['q'])
    n = copy.deepcopy(debug_manual_item['n'])
    for i in range(len(q)):
        if isinstance(q[i], list):
            q[i] = sum(q[i])
            n[i] = sum(n[i])
            debug_manual_item['parent_n'][i] = sum(debug_manual_item['parent_n'][i])

    debug_manual_item['parent_n'] = sum(debug_manual_item['parent_n'])
    # if debug_manual_item['parent_n'] > depth_dict[debug_manual_item['depth']]['res']:
    if debug_manual_item['parent_n'] > 0:
        # index = np.argmax([a / b if b != 0 else 0 for a, b in zip(q, n)])
        index = np.argmax(n)
        action = debug_manual_item['action'][index]
        if isinstance(action, list):
            temp = set(action)
            if len(temp) > 1:
                action = list(temp)
            else:
                action = action[0]

        debug_manual_item['action_res'] = action
        debug_manual_item['q_res'] = q
        debug_manual_item['n_res'] = n
        return action
    else:
        return None


@traverse_tree
def make_minimax_debug_manual(manual, node, color, depth):
    encodes, boards = encode_board(node.board, node.color, True)
    acts = []
    if node.color == color:
        act = node.best_child()[0]
    else:
        act = node.worst_child()[0]
    x, y = node.board.board_num(act)
    for item in get_equal_action(x, y):
        acts.append(node.board.num_board(item))
    for i in range(len(encodes)):
        if encodes[i] not in manual:
            manual[encodes[i]] = {'action': [acts[i]], 'node': [node], 'q': [node.q], 'board': boards[i],
                                  'color': node.color, 'depth': depth}
        else:
            store_value(manual[encodes[i]], {'action': acts[i], 'node': node, 'q': node.q}, node.color)


@make_manual
def make_minimax_manual(debug_manual_item, *args):
    q = copy.deepcopy(debug_manual_item['q'])
    for i in range(len(q)):
        if isinstance(q[i], list):
            q[i] = np.mean(q[i])

    # TODO index = np.argmax(q) if debug_manual_item['color'] == player.color else np.argmin(q)

    index = np.argmax(q)
    action = debug_manual_item['action'][index]
    if isinstance(action, list):
        if len(set(action)) > 1:
            action = list(set(action))
        else:
            action = action[0]

    debug_manual_item['action_res'] = action
    debug_manual_item['q_res'] = q
    return action


mcts_default_dict = {'action': [['D3', 'C4', 'F5', 'E6']], 'q': [[0, 0, 0, 0]],
                     'n': [[1e3, 1e3, 1e3, 1e3]], 'parent_n': [[1e3 + 1, 1e3, 1e3, 1e3]],
                     'node': [None for i in range(4)], 'board': Board(), 'color': 'X', 'depth': 0}
minimax_default_dict = {'action': [['D3', 'C4', 'F5', 'E6']], 'q': [[0, 0, 0, 0]],
                        'node': [None for i in range(4)], 'board': Board(), 'color': 'X', 'depth': 0}
type_map = {'mcts': [make_mcts_debug_manual, make_mcts_manual, mcts_default_dict],
            'minimax': [make_minimax_debug_manual, make_minimax_manual, minimax_default_dict]}


def get_equal_action(x, y):
    return [(x, y), (7 - y, 7 - x), (y, x), (7 - x, 7 - y)]


def encode_board(board, color, equal = False):
    def get_item(piece):
        return {'X': 0, 'O': 1, '.': 2}[piece]

    n = len(get_equal_action(3, 4))
    encode = [0 for i in range(n)] if equal else 0
    boards = [Board() for i in range(n)] if equal else None

    for i in range(8):
        for j in range(8):
            if equal:
                for k, (x, y) in enumerate(get_equal_action(i, j)):
                    boards[k]._board[i][j] = board[x][y]
                    encode[k] *= 3
                    encode[k] += get_item(board[x][y])
            else:
                encode *= 3
                encode += get_item(board[i][j])

    if equal:
        for k in range(n):
            encode[k] *= 2
            encode[k] += get_item(color)
    else:
        encode *= 2
        encode += get_item(color)

    return encode, boards


def main():
    b = Board()
    n = 10000
    start_time = time.time()
    for i in range(n):
        encode_board(b, 'X')
    print(time.time() - start_time)


if __name__ == '__main__':
    main()
