import copy

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
        self.max_depth = 5
        
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
        
        best_action, _ = self.max_value(board, -self.INFINITY, self.INFINITY, 0)
     
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
