# -*- coding: utf-8 -*-

import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8
PASS = 64
ACTION_SIZE = 65

EMPTY = 0
BLACK = 1
WHITE = -1
CHARS = ['-', 'x', 'o']
DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
       (0, 1), (1, -1), (1, 0), (1, 1))  # 方向向量

INF = 100000
MARGIN = 10000
START_TIME = 0
TIME_LIMIT = 3.5
WEIGHTS = np.array([
    [20, -3, 11, 8, 8, 11, -3, 20],
    [-3, -7, -4, 1, 1, -4, -7, -3],
    [11, -4, 2, 2, 2, 2, -4, 11],
    [8, 1, 2, -3, -3, 2, 1, 8],
    [8, 1, 2, -3, -3, 2, 1, 8],
    [11, -4, 2, 2, 2, 2, -4, 11],
    [-3, -7, -4, 1, 1, -4, -7, -3],
    [20, -3, 11, 8, 8, 11, -3, 20]
])


def is_timeout():
    return time.perf_counter() - START_TIME > TIME_LIMIT


def alpha_beta(game, depth, alpha, beta):
    best_action = -1, -1
    #!!!!!!!!!!!!!!weight
    if game.is_game_ended():
        value = np.sum(game.board)
        if game.color == WHITE:
            value = -value
        value += MARGIN if value > 0 else -MARGIN
        return best_action, value

    if depth == 0 or is_timeout():
        value = np.sum(game.board * WEIGHTS)
        if game.color == WHITE:
            value = -value
        return best_action, value

    board = game.board
    color = game.color

    has_legal_move = False
    for i in range(8):
        for j in range(8):
            if board[i][j] != 0 or not game.place(i, j, color, check_only=True):
                continue
            has_legal_move = True
            new_game = game.copy()
            new_game.apply_moveXY(i, j)
            value = -alpha_beta(new_game, depth - 1, -beta, -alpha)[1]
            if value > alpha:
                alpha = value
                best_action = i, j
                if beta <= alpha:
                    return best_action, value

    if not has_legal_move:
        new_game = game.copy()
        new_game.apply_moveXY(-1, -1)
        return best_action, -alpha_beta(new_game, depth, -beta, -alpha)[1]

    return best_action, alpha


class NNet(nn.Module):
    def __init__(self, dropout, num_channels, hidden_size):
        super().__init__()

        self.conv1 = nn.Conv2d(1, num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        self.fc1 = nn.Linear(num_channels * 16, hidden_size)
        self.fc_bn1 = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_bn2 = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size, ACTION_SIZE)

        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, s):
        s = s.view(-1, 1, 8, 8)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.fc1.in_features)

        s = self.dropout(F.relu(self.fc_bn1(self.fc1(s))))
        s = self.dropout(F.relu(self.fc_bn2(self.fc2(s))))

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, board):
        board = torch.from_numpy(board.astype(np.float32))
        board = board.view(1, 8, 8)

        self.eval()
        with torch.no_grad():
            pi, v = self(board)

        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0]


def xy2pos(x, y):
    return chr(ord('A') + x) + str(y + 1)


class GameBoard:
    def __init__(self):
        board = np.zeros((8, 8), dtype=np.int8)

        board[3, 4] = board[4, 3] = BLACK
        board[3, 3] = board[4, 4] = WHITE

        self.color = BLACK
        self.board = board

    def key(self):
        return self.board * self.color

    def copy(self):
        clone = GameBoard()
        clone.color = self.color
        clone.board = self.board.copy()
        return clone

    def valid_moves(self):
        color = self.color
        board = self.board
        count = 0
        valids = np.zeros((ACTION_SIZE,), dtype=np.float64)
        for i in range(8):
            for j in range(8):
                if board[i][j] == 0:
                    if self.place(i, j, color, check_only=True):
                        count += 1
                        valids[i * 8 + j] = 1.0
        if count == 0:
            valids[PASS] = 1.0
        return valids

    def place(self, x, y, color, check_only=False):
        board = self.board
        if not check_only:
            board[x][y] = color
        valid = False
        for d in range(8):
            i = x + DIR[d][0]
            j = y + DIR[d][1]
            while 0 <= i and i < 8 and 0 <= j and j < 8 and \
                    board[i][j] == -color:
                i += DIR[d][0]
                j += DIR[d][1]
            if 0 <= i and i < 8 and 0 <= j and j < 8 and \
               board[i][j] == color:
                while True:
                    i -= DIR[d][0]
                    j -= DIR[d][1]
                    if i == x and j == y:
                        break
                    valid = True
                    if check_only:
                        return True
                    board[i][j] = color
        return valid

    def apply_moveXY(self, x, y):
        if x != -1:
            assert self.place(x, y, self.color)
        self.color = -self.color

    def apply_move(self, move):
        if move != PASS:
            # assert self.place(x, y, color), f'Invalid move {xy2pos(x, y)}'
            assert self.place(move // 8, move % 8, self.color)
        self.color = -self.color

    def has_legal_move(self, color):
        board = self.board
        for i in range(8):
            for j in range(8):
                if board[i][j] == 0:
                    if self.place(i, j, color, check_only=True):
                        return True
        return False

    def evaluate(self):
        diff = np.sum(self.board)
        if self.color == WHITE:
            diff = -diff
        if diff > 0:
            return 1
        return -1

    def count(self, color):
        count = 0
        board = self.board
        for i in range(8):
            for j in range(8):
                if board[i][j] == color:
                    count += 1
        return count

    def show(self):
        board = self.board
        print(f'x: {self.count(BLACK)} o: {self.count(WHITE)}')
        print('  ', end='')
        for i in range(8):
            print(f'{i + 1} ', end='')
        print()
        for y in range(8):
            print(f'{chr(ord("A") + y)} ', end='')
            for x in range(8):
                print(f'{CHARS[board[x][y]]} ', end='')
            print()

    def is_game_ended(self):
        return not self.has_legal_move(BLACK) \
            and not self.has_legal_move(WHITE)


class MCTS:

    def __init__(self, nnet, cpuct=1.0):
        self.nnet = nnet
        self.cpuct = cpuct

        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}
        self.Vs = {}        # stores game.getValidMoves for board s

    def best_move(self, game, timeout, temp=1):
        start_time = time.perf_counter()
        count = 0
        while time.perf_counter() - start_time < timeout:
            count += 1
            self._search(game.copy())

        best_action = -1
        best_value = 0
        s = game.key().tostring()
        for a in range(ACTION_SIZE):
            nsa = self.Nsa.get((s, a), 0)
            if nsa > best_value:
                best_value = nsa
                best_action = a
            # debugging
            # if nsa > 0:
            #     print(xy2pos(a % 8, a // 8), nsa)

        return count, best_action

    def _search(self, game):
        canonical_board = game.key()
        s = canonical_board.tostring()

        result = self.Es.get(s)
        if result is None:
            result = 0
            if game.is_game_ended():
                result = game.evaluate()
            self.Es[s] = result

        if result != 0:  # 游戏结束
            return -result

        if s not in self.Ps:  # leaf node
            ps, v = self.nnet.predict(canonical_board)
            valids = game.valid_moves()
            ps = ps * valids
            ps_sum = np.sum(ps)
            if ps_sum > 0:
                ps /= ps_sum    # renormalize
            else:  # ???
                # print("All valid moves were masked, do workaround.")
                ps = ps + valids
                ps /= np.sum(ps)

            self.Ps[s] = ps
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        ps = self.Ps[s]
        ns = self.Ns[s]
        cpuct = self.cpuct

        best_uct = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(ACTION_SIZE):
            if not valids[a]:
                continue
            qsa = self.Qsa.get((s, a), None)
            if qsa is not None:
                u = qsa + cpuct * ps[a] * math.sqrt(ns) / (1 + self.Nsa[s, a])
            else:
                u = cpuct * ps[a] * math.sqrt(ns + EPS)     # Q = 0 ?
            if u > best_uct:
                best_uct = u
                best_act = a

        a = best_act
        game.apply_move(a)
        v = self._search(game)

        qsa = self.Qsa.get((s, a), None)
        if qsa is not None:
            nsa = self.Nsa[s, a]
            self.Qsa[s, a] = (nsa * qsa + v) / (1 + nsa)
            self.Nsa[s, a] = nsa + 1
        else:
            self.Qsa[s, a] = v
            self.Nsa[s, a] = 1

        self.Ns[s] += 1
        return -v


def main():
    global START_TIME

    saved_state = torch.load('data/cnn_4l_simple.pt', map_location='cpu')
    nnet = NNet(0, 128, 256)
    nnet.load_state_dict(saved_state['state_dict'])

    game = GameBoard()
    mcts = MCTS(nnet)

    turn_ID = 0
    while not game.is_game_ended():
        line = input().strip()
        if line == '':
            continue
        full_input = json.loads(line)

        if turn_ID == 0:
            requests = full_input['requests']
            x = requests[0]['x']
            y = requests[0]['y']
            if x >= 0:
                game.apply_moveXY(x, y)
        else:
            x = full_input['x']
            y = full_input['y']
            game.apply_moveXY(x, y)

        if np.count_nonzero(game.board) >= 56:
            depth = 4
            x, y = -1, -1
            START_TIME = time.perf_counter()
            while depth < 32:
                result = alpha_beta(game, depth, -INF, INF)
                if is_timeout():
                    depth -= 1
                    break
                x, y = result[0]
                depth += 1
            debug = depth
        else:
            debug, move = mcts.best_move(game, TIME_LIMIT, 0)
            if move == PASS:
                x, y = -1, -1
            else:
                x, y = move // 8, move % 8

        print(json.dumps({'response': {'x': x, 'y': y, 'debug': debug}}))
        print('\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n', flush=True)

        game.apply_moveXY(x, y)
        turn_ID += 1


if __name__ == '__main__':
    main()
