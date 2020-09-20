from copy import deepcopy
from treelib import Tree, Node
import math
import time

TIME_LIMIT = 5

class MCTS:
    """
    AI 玩家
    """
    
    def is_timeout(self):
        return time.perf_counter() - self.START_TIME > TIME_LIMIT
        

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color
        self.Cp = 1
        self.root = 'root'
        self.tree = Tree()
        self.max_deep = 15

        
    def cal_color(self, color):
        if color == 'X':
            return 'O'
        else:
            return 'X'
        
    def game_over(self,board):
        b_list = list(board.get_legal_actions('X'))
        w_list = list(board.get_legal_actions('O'))
        is_over =  len(b_list) == 0 and len(w_list) == 0
        return is_over 
    
    
    def mcts_search(self, board):
        
        new_board = deepcopy(board)
        self.tree.create_node(identifier=self.root, data=(0,0, new_board, self.color))
        
        #explore and exploit
        self.START_TIME = time.perf_counter()
        while self.is_timeout() == False:
            temp_node_id = self.tree_policy(self.root)
            winner = self.defautl_policy(temp_node_id)
            self.backup(temp_node_id, winner)
        
        #modify tree    
        #without memory version
        
        #new_root, best_action = best_child(v0, 0)
        best_child_id = self.best_child(self.root, 0)
        best_action = self.tree.get_node(best_child_id).tag
        self.tree = Tree()
        
        return best_action
            
    
    
    def tree_policy(self, node_id):
        
        #initialize
        node = self.tree.get_node(node_id)
        Q,N,board,color = node.data
        action_list = list(board.get_legal_actions(color))
        child_list = self.tree.children(node_id)
        
        while not self.game_over(board) :
          
            if len(action_list) > len(child_list) or (len(action_list) == 0 and len(child_list) == 0):
                return self.expand(node_id)
            else:
                node_id = self.best_child(node_id, self.Cp)
                
                node = self.tree.get_node(node_id)
                Q,N,board,color = node.data
                action_list = list(board.get_legal_actions(color))
                child_list = self.tree.children(node_id)
        
        return node_id
        
    
    
    def expand(self, node_id):
        
        #initialize
        node = self.tree.get_node(node_id)
        Q,N,board,color = node.data
        action_list = list(board.get_legal_actions(color))
        child_list = self.tree.children(node_id)
        tried_list = []
        
        for child in child_list:
            tried_list.append(child.tag)
        
        action = None
        while len(action_list)>0:
            action = random.choice(action_list)
            if action not in tried_list: break
                
        new_board = deepcopy(board)
        if action != None:
            new_board._move(action, color)
        
        new_node = self.tree.create_node(parent=node_id, tag=action, data=(0,0,new_board,self.cal_color(color)))
        
        return new_node.identifier
        
    
    
    def best_child(self, node_id, c):
        
        best_value = -float('inf')
        best_child = None
        
        _,N_all,_,_ = self.tree.get_node(node_id).data
        child_list = self.tree.children(node_id)
        
        for child in child_list:
            
            Q,N,board,color = child.data
            temp_value = Q/N + c*math.sqrt(2*math.log(N_all)/N)
            if temp_value > best_value:
                best_value = temp_value
                best_child = child
                
        return best_child.identifier
        
    
    def defautl_policy(self, node_id):
        
        node = self.tree.get_node(node_id)
        _,_,board,color = node.data
        
        temp_board = deepcopy(board)
        
        cnt = 0
        while not self.game_over(temp_board) and cnt<self.max_deep :
            cnt+=1
            action_list = list(temp_board.get_legal_actions(color))
            if len(action_list) != 0:
                action = random.choice(action_list)
                temp_board = deepcopy(temp_board)
                temp_board._move(action, color)
                color = self.cal_color(color)
            else:
                color = self.cal_color(color)
        
            
        winner,_ = temp_board.get_winner()
        return winner
    
    
    def backup(self, node_id, winner):
        
        temp_node_id = node_id
        while True:
            temp_node = self.tree.get_node(temp_node_id)
            Q,N,board,color = temp_node.data
            
            if (winner == 0 and color == 'O') or (winner == 1 and color == 'X'):
                Q += 1  
           
            temp_node.data = (Q, N+1, board, color)
            if temp_node_id == self.root:
                break
            temp_node_id = self.tree.parent(temp_node_id).identifier
        
        
            
        
    

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
        action = self.mcts_search(board)
        # ------------------------------------------------------------------------

        return action
