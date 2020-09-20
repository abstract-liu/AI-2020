import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import Board


CP = 1.0
EPS = 1e-8
ACTIONS_SIZE = 65
ACTIONS_LIMIT = 50
BOARD_TYPE = 8
TIME_LIMIT = 59


DROPOUT = 0.1
NUM_CHANNELS = 512
IS_CUDA_AVAILABLE = torch.cuda.is_available()

class AlphaBeta:
    """
    AI 玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color
        
        self.INFINITY = 100
        
        #tune this hyparameter
        self.max_depth = 15
        
        if color == 'X':
            self.color_oppo = 'O'
        else:
            self.color_oppo = 'X'
        
        
    def game_over(self,board):
        b_list = list(board.get_legal_actions('X'))
        w_list = list(board.get_legal_actions('O'))
        is_over =  len(b_list) == 0 and len(w_list) == 0
        return is_over 
    
    def alpha_beta_search(self, board):
        best_action = None
        self.start_time = time.time()
        best_action, value = self.max_value(board, -self.INFINITY, self.INFINITY, 0)
        return best_action
        
    def max_value(self, board, alpha, beta, depth):
           
        value = -self.INFINITY
        best_action = None
        
        #terminal state
        if self.game_over(board) or depth == self.max_depth:    
            winner, value = board.get_winner()
            if (winner == 0 and self.color == 'X') or (winner == 1 and self.color == 'O'):
                return best_action,value
            else:
                return best_action,-value
        
        depth += 1
        action_list = list(board.get_legal_actions(self.color))
        
        #can move     
        for action in action_list:
            temp_board = copy.deepcopy(board)
            temp_board._move(action, self.color)
      
            _, temp_value = self.min_value(temp_board, alpha, beta, depth)
            if temp_value > value:
                value = temp_value
                best_action = action
                
            if time.time() - self.start_time > TIME_LIMIT:
                return best_action, value
                
            if value > beta:
                return best_action, value
            alpha = max(alpha, value)
            
        
        #cannot move
        if len(action_list) == 0 :
            temp_board = copy.deepcopy(board)
            _, value = self.min_value(temp_board, alpha, beta, depth)
        
        return best_action, value
        
        
    def min_value(self, board, alpha, beta, depth):
        
        value = self.INFINITY
        best_action = None
        
        if self.game_over(board) or depth == self.max_depth:         
            winner, value = board.get_winner()
            if (winner == 0 and self.color == 'X') or (winner == 1 and self.color == 'O'):
                return best_action, value
            else:
                return best_action, -value
        
        depth += 1
        action_list = list(board.get_legal_actions(self.color_oppo))
        
        #can move
        for action in action_list:
            temp_board = copy.deepcopy(board)
            temp_board._move(action, self.color_oppo)

            _, temp_value = self.max_value(temp_board, alpha, beta, depth)
            if value > temp_value:
                value = temp_value
                best_action = action
                
            if time.time() - self.start_time > TIME_LIMIT:
                return best_action, value
                
            if value < alpha :
                return best_action, value
            beta = min(beta, value)
            
        
        #cannot move
        if len(action_list) == 0 :
            temp_board = copy.deepcopy(board)
            _, value = self.max_value(temp_board, alpha, beta, depth)
        
        return best_action, value
        
        

    def get_move(self, board):
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
        action = self.alpha_beta_search(board)
        # ------------------------------------------------------------------------

        return action


class NNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, NUM_CHANNELS, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, 3, stride=1)
        self.conv4 = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(NUM_CHANNELS)
        self.bn2 = nn.BatchNorm2d(NUM_CHANNELS)
        self.bn3 = nn.BatchNorm2d(NUM_CHANNELS)
        self.bn4 = nn.BatchNorm2d(NUM_CHANNELS)

        self.fc1 = nn.Linear(NUM_CHANNELS*(BOARD_TYPE-4)*(BOARD_TYPE-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, ACTIONS_SIZE)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, BOARD_TYPE, BOARD_TYPE)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, NUM_CHANNELS*(BOARD_TYPE-4)*(BOARD_TYPE-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=DROPOUT, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=DROPOUT, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    
class AlphaOne():
    
    def __init__(self, color,load): 
        self.color = color
        self.oppseColor = 'X' if color == 'O' else 'O'
        self.alpha_beta = AlphaBeta(color)
        
        self.nnet = NNet()
        map_location = 'cpu'
        if IS_CUDA_AVAILABLE:
            self.nnet.cuda()
            map_location = None 
        if load == 1:
            checkpoint = torch.load('./8x8_100checkpoints_best.pth.tar', map_location=map_location)
        else:
            checkpoint = torch.load('./best.pth.tar', map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)
        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def get_move(self, board):
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        
        if board.count('X') + board.count('O') > ACTIONS_LIMIT:
            start_time = time.time()
            action = self.alpha_beta.get_move(board)
            print('alpha beta', time.time()-start_time)
        else:
            canonicalBoard = self.getCanonicalBoard(board)
            start_time = time.time()
            while time.time() - start_time < TIME_LIMIT:
                self.search(canonicalBoard)
        
            s = canonicalBoard.tostring()
            counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(65)]
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            action = self.getAction(bestA)
            print('mcts',time.time()-start_time)

        return action
        

    def search(self, canonicalBoard):

        s = canonicalBoard.tostring()

        if s not in self.Es:
            self.Es[s] = self.getGameEnded(canonicalBoard)
        if self.Es[s]!=0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.predict(canonicalBoard)
            valids = self.getValidMoves(canonicalBoard)
            self.Ps[s] = self.Ps[s]*valids  
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s   
            else:
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(ACTIONS_SIZE):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + CP*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = CP*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)   

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.getNextState(canonicalBoard, a)


        v = self.search(next_s)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v

    def predict(self, canonicalBoard):
        # preparing input
        canonicalBoard = torch.FloatTensor(canonicalBoard.astype(np.float64))
        if IS_CUDA_AVAILABLE: canonicalBoard = canonicalBoard.contiguous().cuda()
        canonicalBoard = canonicalBoard.view(1, BOARD_TYPE, BOARD_TYPE)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(canonicalBoard)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def getGameEnded(self, canonicalBoard):
        board = self.getBoard(canonicalBoard)
        x_list = list(board.get_legal_actions('X'))
        o_list = list(board.get_legal_actions('O'))
        if len(x_list) == 0 and len(o_list) == 0:
            return 1 if board.count('X') > board.count('O') else -1
        return 0


    def getValidMoves(self, canonicalBoard):
        board = self.getBoard(canonicalBoard)
        x_list = list(board.get_legal_actions('X'))
        valids = [0]*ACTIONS_SIZE
        
        if len(x_list) == 0:
            valids[-1] = 1
            return np.array(valids)
        
        for action in x_list:
            canonicalAction = self.getCanonicalAction(action)
            valids[canonicalAction] = 1
        return np.array(valids)

        
    def getNextState(self, canonicalBoard, canonicalAction):
        action = self.getAction(canonicalAction)
        if action == None:
            return (-1) * canonicalBoard 
        else:
            board = self.getBoard(canonicalBoard)
            board._move(action, 'X')
            return self.getCanonicalBoard(board)

       
    
    def getBoard(self, canonicalBoard):
        board = Board()
        for i in range(BOARD_TYPE):
            for j in range(BOARD_TYPE):
                if canonicalBoard[i][j] == 1:
                    board[j][i] = 'X'
                elif canonicalBoard[i][j] == 0:
                    board[j][i] = '.'
                else:
                    board[j][i] = 'O'
        return board


    def getCanonicalBoard(self,board):    
        canonicalBoard = np.zeros((BOARD_TYPE,BOARD_TYPE))
        for i in range(BOARD_TYPE):
            for j in range(BOARD_TYPE):
                if board[i][j] == self.color:
                    canonicalBoard[j][i] = 1
                elif board[i][j] == self.oppseColor:
                    canonicalBoard[j][i] = -1
        return canonicalBoard
    

    def getAction(self, canonicalAction):
        action = chr(int(canonicalAction/8)+ord('A'))+chr(ord('1')+canonicalAction%8) if canonicalAction != BOARD_TYPE ** 2 else None
        return action


    def getCanonicalAction(self, action):
        canonicalAction = 8*(ord(action[0]) - ord('A')) + int(action[1]) - 1
        return canonicalAction
