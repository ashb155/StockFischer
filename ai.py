piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0}       #material values

pawn_table = [                                                  #tables for positional heuristics
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

knight_table = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
]


bishop_table = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
]

rook_table = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
]

queen_table = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
]

king_table = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]



def evaluate_board(game):
    score = 0
    white_pawns = []
    black_pawns = []
    for row in range(8):
        for col in range(8):
            piece = game.board[row][col]
            if piece:
                val = piece_values[piece.name]
                index = row * 8 + col if piece.colour == 'w' else (7 - row) * 8 + col

                if piece.name == 'P':
                    val += pawn_table[index]
                elif piece.name == 'N':
                    val += knight_table[index]
                elif piece.name == 'B':
                    val += bishop_table[index]
                elif piece.name == 'R':
                    val += rook_table[index]
                elif piece.name == 'Q':
                    val += queen_table[index]
                elif piece.name == 'K':
                    val += king_table[index]

                if piece.name == 'P':
                    if piece.colour == 'w':
                        white_pawns.append((row, col))
                    else:
                        black_pawns.append((row, col))

                score += val if piece.colour == 'w' else -val


    def pawn_structure(pawns, enemy_pawns, colour):                     #considers pawn structure
        bonus=0
        files=[0]*8
        for r,c in pawns:
            files[c]+=1

        for f in range(8):
            if files[f]>1:                                              #doubled pawns
                bonus-=20*(files[f]-1)                                  #isolated pawns
            if f == 0 and files[f+1]==0:
                bonus-=15
            elif f == 7 and files[f-1]==0:
                bonus-=15
            elif 0 < f < 7 and files[f - 1] == 0 and files[f + 1] == 0:
                bonus -= 15

        for r, c in pawns:                                              #passed pawns, checks if any enemy pawns
            blocked = False
            for er, ec in enemy_pawns:
                if ec in [c - 1, c, c + 1]:
                    if (colour == 'w' and er < r) or (colour == 'b' and er > r):
                        blocked = True
                        break

            if not blocked:
                rank = r if colour == 'b' else 7 - r
                bonus += 10 + rank * 5

        return bonus
    score += pawn_structure(white_pawns, black_pawns,'w')
    score -= pawn_structure(black_pawns, white_pawns,'b')

    #mobility - more legal moves
    white_moves = sum(len(game.get_moves(r, c)) for r in range(8) for c in range(8) if game.board[r][c] and game.board[r][c].colour == 'w')
    black_moves = sum(len(game.get_moves(r, c)) for r in range(8) for c in range(8) if game.board[r][c] and game.board[r][c].colour == 'b')
    score += 10 * (white_moves - black_moves)

    #bishop pair
    white_bishops = sum(1 for r in range(8) for c in range(8) if
                        game.board[r][c] and game.board[r][c].name == 'B' and game.board[r][c].colour == 'w')
    black_bishops = sum(1 for r in range(8) for c in range(8) if
                        game.board[r][c] and game.board[r][c].name == 'B' and game.board[r][c].colour == 'b')
    if white_bishops >= 2: score += 50
    if black_bishops >= 2: score -= 50

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




