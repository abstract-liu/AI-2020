import copy

class AIPlayerMiniMax:
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
        if color == 'X':
            self.color_oppo = 'O'
        else:
            self.color_oppo = 'X'
        
        
    def game_over(self,board):
        b_list = list(board.get_legal_actions('X'))
        w_list = list(board.get_legal_actions('O'))
        is_over =  len(b_list) == 0 and len(w_list) == 0
        return is_over 
    
    def minimax_decision(self, board):
        best_value = 0
        best_action = None
        
        for action in list(board.get_legal_actions(self.color)):
            temp_board = copy.deepcopy(board)
            temp_board._move(action, self.color)      
            temp_value = self.min_value(temp_board)
            
            if temp_value > best_value:
                best_action = action
                best_value = temp_value
        return best_action
        
    def max_value(self, board):
        if self.game_over(board):
          
            winner, temp_value = board.get_winner()
            if (winner == 0 and self.color == 'X') or (winner == 1 and self.color == 'O'):
                return temp_value
            else:
                return -temp_value
        
        value = -self.INFINITY
        for action in list(board.get_legal_actions(self.color)):
            temp_board = copy.deepcopy(board)
            temp_board._move(action, self.color)
            
            temp_value = self.min_value(temp_board)
            value = max(value, temp_value)
        
        if len(list(board.get_legal_actions(self.color))) == 0:
            temp_board = copy.deepcopy(board)
            value = self.min_value(temp_board)
        
        return value
        
        
    def min_value(self, board):
        if self.game_over(board):
          
            winner, temp_value = board.get_winner()
            if (winner == 0 and self.color == 'X') or (winner == 1 and self.color == 'O'):
                return temp_value
            else:
                return -temp_value
        
        value = self.INFINITY
        for action in list(board.get_legal_actions(self.color_oppo)):
            temp_board = copy.deepcopy(board)
            temp_board._move(action, self.color_oppo)
            
            temp_value = self.max_value(temp_board)
            value = min(value, temp_value)
            
        if len(list(board.get_legal_actions(self.color_oppo))) == 0:
            temp_board = copy.deepcopy(board)
            value = self.max_value(temp_board)
                
        return value
        
        

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
        action = self.minimax_decision(board)
        # ------------------------------------------------------------------------

        return action
