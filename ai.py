piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}       #material values

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
   -30,  5, 10, 15,  15, 10,  5,-30,
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

def generate_attackers_map(game, move_cache):                               #generates attacker for a map
    attackers_map = {}
    for r in range(8):
        for c in range(8):
            piece = game.board[r][c]
            if not piece:
                continue
            if (r, c) in move_cache:
                moves = move_cache[(r, c)]
            else:
                moves = game.get_moves(r, c)
                move_cache[(r, c)] = moves
            for target in moves:
                if target not in attackers_map:
                    attackers_map[target] = {'w': [], 'b': []}
                attackers_map[target][piece.colour].append((r, c))
    return attackers_map

def get_attackers(attackers_map, square, colour):
    if square not in attackers_map:
        return []
    return attackers_map[square][colour]

def static_exchange_eval(game, target, colour, move_cache, attackers_map, depth=0):
    max_steps = 32                              # safety cap for long sequences
    max_depth = 8                               # prevent infinite recursion
    if depth > max_depth:
        return 0

    if not game.board[target[0]][target[1]]:
        return 0


    copyg = game.light_copy()                   #game copy


    local_cache = move_cache.copy() if move_cache is not None else {} #local move cache

    attackers_map_local = attackers_map if attackers_map is not None else generate_attackers_map(copyg, local_cache)


    target_piece = copyg.board[target[0]][target[1]]
    gains = [piece_values[target_piece.name]]

    side = colour                               #starting colour
    steps = 0

    while steps < max_steps:
        steps += 1


        attackers = get_attackers(attackers_map_local, target, side)

        attackers = [sq for sq in attackers if copyg.board[sq[0]][sq[1]]]

        if not attackers:
            break


        best_attacker = min(attackers, key=lambda sq: piece_values[copyg.board[sq[0]][sq[1]].name]) #least valuable piece

        copyg.make_move(best_attacker, target)              #move made on copy

        side = 'b' if side == 'w' else 'w'

        attackers_map_local = generate_attackers_map(copyg, local_cache.copy())

        # value now on target (0 if empty)
        if copyg.board[target[0]][target[1]]:
            gains.append(piece_values[copyg.board[target[0]][target[1]].name] - gains[-1])
        else:
            gains.append(-gains[-1])


    if len(gains) == 1:
        return gains[0]

    # reduce gains from the back: gains[i] = max(-gains[i+1], gains[i])
    for i in range(len(gains)-2, -1, -1):
        gains[i] = max(-gains[i+1], gains[i])

    # gains[0] is net for the side that initiated the capture
    return gains[0]


def evaluate_board(game, move_cache=None):
    if move_cache is None:
        move_cache = {}
    score = 0
    white_pawns, black_pawns = [], []

    for row in range(8):
        for col in range(8):
            piece = game.board[row][col]
            if not piece:
                continue
            val = piece_values[piece.name]
            index = row*8 + col if piece.colour == 'w' else (7-row)*8 + (7-col)
            if piece.name == 'P': val += pawn_table[index]
            elif piece.name == 'N': val += knight_table[index]
            elif piece.name == 'B': val += bishop_table[index]
            elif piece.name == 'R': val += rook_table[index]
            elif piece.name == 'Q': val += queen_table[index]
            elif piece.name == 'K': val += king_table[index]
            if piece.name == 'P':
                (white_pawns if piece.colour=='w' else black_pawns).append((row,col))
            score += val if piece.colour=='w' else -val

            # Hanging pieces
            defenders, attackers = 0,0
            for r1 in range(8):
                for c1 in range(8):
                    piece2 = game.board[r1][c1]
                    if not piece2:
                        continue
                    if (r1,c1) in move_cache:
                        moves = move_cache[(r1,c1)]
                    else:
                        moves = game.get_moves(r1,c1)
                        move_cache[(r1,c1)] = moves
                    if (row,col) in moves:
                        if piece2.colour == piece.colour:
                            defenders += 1
                        else:
                            attackers += 1
            if attackers > defenders:
                score += -30 if piece.colour=='w' else 30

    # Pawn structure
    def pawn_structure(pawns, enemy_pawns, colour):
        bonus = 0
        files = [0]*8
        for r,c in pawns:
            files[c] += 1
        for f in range(8):
            if files[f]>1: bonus -= 20*(files[f]-1)
            if f==0 and files[f+1]==0: bonus -= 15
            elif f==7 and files[f-1]==0: bonus -= 15
            elif 0<f<7 and files[f-1]==0 and files[f+1]==0: bonus -= 15
        for r,c in pawns:
            blocked = False
            for er,ec in enemy_pawns:
                if ec in [c-1,c,c+1] and ((colour=='w' and er<r) or (colour=='b' and er>r)):
                    blocked = True
                    break
            if not blocked:
                rank = r if colour=='b' else 7-r
                bonus += 10 + rank*5
        return bonus

    score += pawn_structure(white_pawns, black_pawns,'w')
    score -= pawn_structure(black_pawns, white_pawns,'b')

    # Mobility using move_cache
    white_moves = sum(len(move_cache[(r,c)]) if (r,c) in move_cache else len(game.get_moves(r,c))
                      for r in range(8) for c in range(8)
                      if game.board[r][c] and game.board[r][c].colour=='w')
    black_moves = sum(len(move_cache[(r,c)]) if (r,c) in move_cache else len(game.get_moves(r,c))
                      for r in range(8) for c in range(8)
                      if game.board[r][c] and game.board[r][c].colour=='b')
    score += 10*(white_moves - black_moves)

    # Bishop pair
    white_bishops = sum(1 for r in range(8) for c in range(8)
                        if game.board[r][c] and game.board[r][c].name=='B' and game.board[r][c].colour=='w')
    black_bishops = sum(1 for r in range(8) for c in range(8)
                        if game.board[r][c] and game.board[r][c].name=='B' and game.board[r][c].colour=='b')
    if white_bishops >= 2: score += 50
    if black_bishops >= 2: score -= 50

    # King safety
    def king_safety(game, colour, move_cache):
        kingr, kingc = None, None
        for r in range(8):
            for c in range(8):
                piece = game.board[r][c]
                if piece and piece.name=='K' and piece.colour==colour:
                    kingr, kingc = r, c
                    break
            if kingr is not None: break
        danger = 0
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0: continue
                nr, nc = kingr+dr, kingc+dc
                if 0<=nr<8 and 0<=nc<8:
                    for r1 in range(8):
                        for c1 in range(8):
                            piece2 = game.board[r1][c1]
                            if piece2 and piece2.colour != colour:
                                if (r1,c1) in move_cache:
                                    attacker_moves = move_cache[(r1,c1)]
                                else:
                                    attacker_moves = game.get_moves(r1,c1)
                                    move_cache[(r1,c1)] = attacker_moves
                                if (nr,nc) in attacker_moves:
                                    danger += 1
        shield_bonus = 0
        directions = -1 if colour=='w' else 1
        for dc in [-1,0,1]:
            file = kingc+dc
            for r in [kingr+directions, kingr+2*directions]:
                if 0<=r<8 and 0<=file<8:
                    p = game.board[r][file]
                    if p and p.name=='P' and p.colour==colour:
                        shield_bonus += 15 if r==kingr+directions else 10
        return -30*danger + shield_bonus

    score += king_safety(game,'w', move_cache)
    score -= king_safety(game,'b', move_cache)

    central_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]          #control of central squares
    for r, c in central_squares:
        piece = game.board[r][c]
        if piece:
            if piece.colour == 'w':
                score += 15
            else:
                score -= 15
    return score

def static_exchange_eval_local(game, target, colour, move_cache):

    if not game.board[target[0]][target[1]]:
        return 0

    copyg = game.light_copy()
    gains = [piece_values[copyg.board[target[0]][target[1]].name]]
    side = colour
    MAX_STEPS = 32
    steps = 0

    while steps < MAX_STEPS:
        steps += 1
        # find all attackers of target square
        attackers = []
        for r in range(8):
            for c in range(8):
                piece = copyg.board[r][c]
                if not piece or piece.colour != side:
                    continue
                moves = move_cache.get((r,c), copyg.get_moves(r,c))
                if target in moves:
                    attackers.append((r,c))
        if not attackers:
            break

        # select least valuable attacker
        best_attacker = min(attackers, key=lambda sq: piece_values[copyg.board[sq[0]][sq[1]].name])
        copyg.make_move(best_attacker, target)

        side = 'b' if side == 'w' else 'w'
        # updated value on target
        if copyg.board[target[0]][target[1]]:
            gains.append(piece_values[copyg.board[target[0]][target[1]].name] - gains[-1])
        else:
            gains.append(-gains[-1])

    # reduce from back
    for i in range(len(gains)-2, -1, -1):
        gains[i] = max(-gains[i+1], gains[i])

    return gains[0]


def minimax_sse(game, depth, alpha, beta, maximizing, original_depth=None, move_cache=None):
    if original_depth is None: original_depth = depth
    if move_cache is None: move_cache = {}

    if depth == 0 or game.state is not None:
        return evaluate_board(game, move_cache)

    best_move = None
    cutoff = False

    if maximizing:
        max_eval = float('-inf')
        for r in range(8):
            for c in range(8):
                if cutoff: break
                piece = game.board[r][c]
                if piece and piece.colour == 'w':
                    moves = move_cache.get((r,c), game.get_moves(r,c))
                    move_cache[(r,c)] = moves
                    for move in moves:
                        if cutoff: break

                        # SSE check for captures (local only)
                        if game.board[move[0]][move[1]]:
                            net_gain = static_exchange_eval_local(game, move, piece.colour, move_cache)
                            if net_gain < 0:
                                continue

                        promotions = ['Q','R','B','N'] if piece.name == 'P' and move[0] == 0 else [None]
                        for promotion in promotions:
                            copy = game.light_copy()
                            if promotion:
                                copy.make_move((r,c), move, promotion)
                            else:
                                copy.make_move((r,c), move)
                            eval_score = minimax_sse(copy, depth-1, alpha, beta, False, original_depth, move_cache.copy())
                            if eval_score > max_eval:
                                max_eval = eval_score
                                best_move = ((r,c), move, promotion)
                            alpha = max(alpha, eval_score)
                            if beta <= alpha:
                                cutoff = True
                                break
        return best_move if depth == original_depth else max_eval
    else:
        min_eval = float('inf')
        for r in range(8):
            for c in range(8):
                if cutoff: break
                piece = game.board[r][c]
                if piece and piece.colour == 'b':
                    moves = move_cache.get((r,c), game.get_moves(r,c))
                    move_cache[(r,c)] = moves
                    for move in moves:
                        if cutoff: break

                        # SSE check for captures (local only)
                        if game.board[move[0]][move[1]]:
                            net_gain = static_exchange_eval_local(game, move, piece.colour, move_cache)
                            if net_gain < 0:
                                continue

                        promotions = ['Q','R','B','N'] if piece.name == 'P' and move[0] == 7 else [None]
                        for promotion in promotions:
                            copy = game.light_copy()
                            if promotion:
                                copy.make_move((r,c), move, promotion)
                            else:
                                copy.make_move((r,c), move)
                            eval_score = minimax_sse(copy, depth-1, alpha, beta, True, original_depth, move_cache.copy())
                            if eval_score < min_eval:
                                min_eval = eval_score
                                best_move = ((r,c), move, promotion)
                            beta = min(beta, eval_score)
                            if beta <= alpha:
                                cutoff = True
                                break
        return best_move if depth == original_depth else min_eval
