from game import Game

piece_values = {'P':1, 'N':3, 'B':3, 'R':5, 'Q':9, 'K':0}

def evaluate_board(game):
    score = 0
    for row in range(8):
        for col in range(8):
            piece = game.board[row][col]
            if piece:
                val = piece_values[piece.name]
                score += val if piece.colour == 'w' else -val
    return score

def minimax(game, depth, maximizing, original_depth=None):
    if original_depth is None:
        original_depth = depth

    if depth == 0 or game.state is not None:
        return evaluate_board(game)

    best_move = None

    if maximizing:
        max_eval = float('-inf')
        for r in range(8):
            for c in range(8):
                piece = game.board[r][c]
                if piece and piece.colour == 'w':
                    for move in game.get_moves(r, c):
                        promotions = ['Q','R','B','N'] if piece.name=='P' and move[0]==0 else [None]
                        for promotion in promotions:
                            copy = game.copy()
                            copy.make_move((r,c), move, promotion) if promotion else copy.make_move((r,c), move)
                            eval_score = minimax(copy, depth-1, False, original_depth)
                            if eval_score > max_eval:
                                max_eval = eval_score
                                best_move = ((r,c), move, promotion)
        return best_move if depth==original_depth else (max_eval if best_move else 0)

    else:
        min_eval = float('inf')
        for r in range(8):
            for c in range(8):
                piece = game.board[r][c]
                if piece and piece.colour == 'b':
                    for move in game.get_moves(r, c):
                        promotions = ['Q','R','B','N'] if piece.name=='P' and move[0]==7 else [None]
                        for promotion in promotions:
                            copy= game.copy()
                            copy.make_move((r,c), move, promotion) if promotion else copy.make_move((r,c), move)
                            eval_score = minimax(copy, depth-1, True, original_depth)
                            if eval_score < min_eval:
                                min_eval = eval_score
                                best_move = ((r,c), move, promotion)
        return best_move if depth==original_depth else (min_eval if best_move else 0)




