import os
import datetime
import pickle
import numpy as np

from board import Board
from game import Game

from tools import encode_board, build_logger, get_time_str, can_move, type_map
from mcts import MctsPlayer
from minimax import MiniMaxPlayer
from player import RandomPlayer, HumanPlayer


def worker(cfg, index, logger):
    import datetime

    def build_players(cfg):
        players = []
        players_note = ['X', 'O']
        players_map = {'random': RandomPlayer, 'human': HumanPlayer, 'mcts': MctsPlayer, 'minimax': MiniMaxPlayer}
        for i, player_dict in enumerate(cfg['player'][:2]):
            players.append(players_map[player_dict['kind']](**player_dict['param'], color = players_note[i]))
        return players

    def handle_result(index, result, diff, data):
        res = None
        if result == 'draw':
            res = 'turn {}, draw'.format(index)
        elif result == 'black_win':
            res = 'turn {}, black win diff: {}'.format(index, diff)
            data['black_win_times'] += 1
            data['black_average_diff'] += diff
            data['white_average_diff'] -= diff
            data['black_win_diff'] += diff
        elif result == 'white_win':
            res = 'turn {}, white win diff: {}'.format(index, diff)
            data['white_win_times'] += 1
            data['white_average_diff'] += diff
            data['black_average_diff'] -= diff
            data['white_win_diff'] += diff
        return res

    players = build_players(cfg)
    n = cfg['game']['times'] // cfg['thread']
    if index < cfg['game']['times'] % cfg['thread']:
        n += 1

    vars_name = ['black_win_times', 'white_win_times', 'black_average_diff', 'white_average_diff', 'black_win_diff',
                 'white_win_diff', 'times']
    data = {var: 0 for var in vars_name}
    data['times'] = n

    start_time = datetime.datetime.now()
    for i in range(n):
        current_start_time = datetime.datetime.now()
        for player in players:
            player.reset()
        g = Game(*players, debug = cfg['debug'])
        result, diff = g.run()
        res = handle_result(i, result, diff, data)
        now = datetime.datetime.now()
        current_duration = now - current_start_time
        duration = now - start_time
        logger.info(res + '\ttime: {}\tavg time: {}\ttotal time: {}\texcept end time: {}'.format(
            get_time_str(current_duration), get_time_str(duration / (i + 1)), get_time_str(duration),
            (start_time + duration / (i + 1) * n).strftime('%Y-%m-%d %H:%M:%S')))
    return data


def evaluation(cfg):
    def update_date(data_sum, data):
        for key in data:
            data_sum[key] += data[key]

    def parse_data(data):
        black_win_rate = data['black_win_times'] / data['times'] * 100
        white_win_rate = data['white_win_times'] / data['times'] * 100
        draw_rate = 100 - black_win_rate - white_win_rate
        black_average_diff = data['black_average_diff'] / data['times']
        white_average_diff = data['white_average_diff'] / data['times']
        black_win_diff = 0 if data['black_win_times'] == 0 else data['black_win_diff'] / data['black_win_times']
        white_win_diff = 0 if data['white_win_times'] == 0 else data['white_win_diff'] / data['white_win_times']
        return data['times'], data['black_win_times'], data[
            'white_win_times'], black_win_rate, white_win_rate, draw_rate, black_average_diff, white_average_diff, black_win_diff, white_win_diff

    logger = build_logger(cfg['log']['path'])
    logger.info('Start new evaluation')
    if cfg['thread'] == 1:
        data = worker(cfg, 1, logger)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        executor = ProcessPoolExecutor(max_workers = cfg['thread'])
        tasks = [executor.submit(worker, cfg, i, logger) for i in range(cfg['thread'])]

        vars_name = ['black_win_times', 'white_win_times', 'black_average_diff', 'white_average_diff', 'black_win_diff',
                     'white_win_diff', 'times']
        data = {var: 0 for var in vars_name}
        for i, res in enumerate(as_completed(tasks)):
            current_data = res.result()
            update_date(data, current_data)
    format_string = 'Total times: {1}{0}black win times: {2:}{0}white win times: {3:}{0}black win rate: {4:.2f}%{0}white win rate: {5:.2f}%{0}draw rate: {6:.2f}%{0}black avg diff: {7:.2f}{0}white avg diff: {8:.2f}{0}black win diff: {9:.2f}{0}white win diff: {10:.2f}'
    data = parse_data(data)
    logger.info(format_string.format('\t', *data))
    return format_string.format('\n', *data)


def create_manual(cfg):
    manual_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['manual']['path'])
    if not os.path.exists(manual_dir):
        os.makedirs(manual_dir)

    data_path = os.path.join(manual_dir, 'data.pkl')

    b = Board()
    b._move('D3', 'X')
    if cfg['manual']['player'] is None:
        player_map = {'mcts': MctsPlayer, 'minimax': MiniMaxPlayer}
        player = player_map[cfg['manual']['kind']]('O', **cfg[cfg['manual']['kind']], create_manual = True)
    else:
        player = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['manual']['player'])
        player = pickle.load(open(player, 'rb'))
        player.update_param(**cfg[cfg['manual']['kind']])
    player.get_move(b, False)

    f = open(data_path, 'wb')
    pickle.dump(player, f, 2)
    f.close()

    res_str = 'data created at {}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(res_str, end = '')

    return res_str


def read_manual(cfg):
    make_debug_manual, make_manual, default_dict = type_map[cfg['manual']['kind']]

    manual_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['manual']['path'])

    data_path = os.path.join(manual_dir, 'data.pkl')
    depth_path = os.path.join(manual_dir, 'depth.pkl')
    manual_path = os.path.join(manual_dir, 'manual.pkl')
    debug_manual_path = os.path.join(manual_dir, 'debug_manual.pkl')

    # read for debug
    # manual = pickle.load(open(manual_path, 'rb'))
    # debug_manual = pickle.load(open(debug_manual_path, 'rb'))
    # multi_debug_manual = {key: value for key, value in debug_manual.items()}

    print('start load player')
    duration = datetime.datetime.now()
    player = pickle.load(open(data_path, 'rb'))
    duration = datetime.datetime.now() - duration
    print('end load player in {:.2f}s'.format(duration.total_seconds()))

    print('start make debug manual...')
    duration = datetime.datetime.now()
    debug_manual = {}
    default_encode, _ = encode_board(Board(), 'X')
    debug_manual[default_encode] = default_dict
    make_debug_manual(debug_manual, player.backend.root, player.color)
    duration = datetime.datetime.now() - duration
    print('end make debug manual in {}'.format(get_time_str(duration)))

    print('start save debug manual...')
    duration = datetime.datetime.now()
    f = open(debug_manual_path, 'wb')
    pickle.dump(debug_manual, f, 2)
    f.close()
    duration = datetime.datetime.now() - duration
    print('end save debug manual in {}'.format(get_time_str(duration)))

    print('start make manual...')
    duration = datetime.datetime.now()
    manual = {}
    depth_dict = pickle.load(open(depth_path, 'rb')) if cfg['manual']['kind'] == 'mcts' else None
    make_manual(debug_manual, manual, depth_dict)
    duration = datetime.datetime.now() - duration
    print('end make manual in {}'.format(get_time_str(duration)))

    print('start save manual...')
    duration = datetime.datetime.now()
    f = open(manual_path, 'wb')
    pickle.dump(manual, f, 2)
    f.close()
    duration = datetime.datetime.now() - duration
    print('end save manual in {}'.format(get_time_str(duration)))

    res = 'manual created at {}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    print(res, end = '')
    return res


def play_out(cfg):
    def generate_board(n):
        b = Board()
        color = 'X'
        for i in range(n):
            actions = list(b.get_legal_actions(color))
            act = np.random.choice(actions)
            b._move(act, color)
            other_color = 'X' if color == 'O' else 'O'
            if can_move(b, other_color):
                color = other_color
        return b, color

    if isinstance(cfg['play_out']['depth'], int):
        depths = [cfg['play_out']['depth'], ]
    else:
        depths = range(*cfg['play_out']['depth'])

    res = {}

    for depth in depths:
        for i in range(cfg['play_out']['times']):
            b, color = generate_board(depth)
            player = MctsPlayer(color, **cfg['mcts'])
            player.get_move(b)
            if depth in res:
                res[depth]['i'].append(player.backend.i)
            else:
                res[depth] = {'i': [player.backend.i]}
        res[depth]['res'] = np.mean(res[depth]['i'])
        res[depth]['std'] = np.std(res[depth]['i'])

    return str(res)


def do_work(cfg):
    work_map = {'eval': evaluation, 'manual': create_manual, 'read': read_manual, 'play_out': play_out}
    return work_map[cfg['kind']](cfg)
