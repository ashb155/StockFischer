import time
import random

# --- FUTILITY PRUNING MARGIN (1 Pawn value = 100 centipawns) ---
FUTILITY_MARGIN = 100

# --- HEURISTIC TABLES ---
KILLER_MOVES = [[None, None] for _ in range(64)]
# History Moves: Stores a score for every possible start->end square move (64x64)
HISTORY_MOVES = [[0 for _ in range(64)] for _ in range(64)]

# --- ZOBRIST HASHING SETUP ---
ZOBRIST_SIZE = 64 * 12 + 1 + 4 + 8
random.seed(42)
ZOBRIST_KEYS = [random.randint(1, 2 ** 64 - 1) for _ in range(ZOBRIST_SIZE)]

PIECE_TO_INDEX = {
    ('w', 'P'): 0, ('w', 'N'): 1, ('w', 'B'): 2, ('w', 'R'): 3, ('w', 'Q'): 4, ('w', 'K'): 5,
    ('b', 'P'): 6, ('b', 'N'): 7, ('b', 'B'): 8, ('b', 'R'): 9, ('b', 'Q'): 10, ('b', 'K'): 11,
}

# TT Constants
TT_EXACT = 0
TT_ALPHA = 1
TT_BETA = 2
TT = {}
TT_MAX_SIZE = 500000

# --- PIECE VALUES AND TABLES ---
piece_values = {'P': 100, 'N': 320, 'B': 330, 'R'
: 500, 'Q': 900, 'K': 20000}
PHASE_MATERIAL = {'Q': 4, 'R': 2, 'B': 1, 'N': 1}
MAX_PHASE_MATERIAL = sum(PHASE_MATERIAL.values()) * 2

pawn_table = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
]

knight_table = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -50, 5, 10, 15, 15, 10, 5, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]

bishop_table = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

rook_table = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0
]

queen_table = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
]

king_table = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20
]

king_endgame_table = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50
]


def calculate_zobrist_hash(game):
    h = 0
    for r in range(8):
        for c in range(8):
            piece = game.board[r][c]
            if piece:
                index = r * 8 + c
                if (piece.colour, piece.name) in PIECE_TO_INDEX:
                    piece_index = PIECE_TO_INDEX[(piece.colour, piece.name)]
                    h ^= ZOBRIST_KEYS[index * 12 + piece_index]

    offset = 64 * 12
    if game.turn == 'w': h ^= ZOBRIST_KEYS[offset]

    offset += 1
    if game.castling.get('wKR'): h ^= ZOBRIST_KEYS[offset + 0]
    if game.castling.get('wQR'): h ^= ZOBRIST_KEYS[offset + 1]
    if game.castling.get('bKR'): h ^= ZOBRIST_KEYS[offset + 2]
    if game.castling.get('bQR'): h ^= ZOBRIST_KEYS[offset + 3]

    offset += 4
    if game.enpassant:
        file_index = game.enpassant[1]
        h ^= ZOBRIST_KEYS[offset + file_index]

    return h


def evaluate_board(game, move_cache=None):
    if move_cache is None: move_cache = {}
    score = 0
    white_pawns, black_pawns = [], []

    all_moves_cache = move_cache if move_cache is not None else {}

    current_material = sum(
        PHASE_MATERIAL.get(game.board[r][c].name, 0) for r in range(8) for c in range(8) if game.board[r][c])
    phase = 1.0 - (current_material / MAX_PHASE_MATERIAL)
    phase = max(0.0, min(1.0, phase))

    for row in range(8):
        for col in range(8):
            piece = game.board[row][col]
            if not piece: continue
            val = piece_values[piece.name]
            index = row * 8 + col if piece.colour == 'w' else (7 - row) * 8 + (7 - col)

            if piece.name == 'P':
                val += pawn_table[index]
            elif piece.name == 'N':
                val += knight_table[index]
                # knight outpost
                is_supported = False
                is_attackable = False
                if piece.colour == 'w':
                    support_r, attack_r = row + 1, row - 1
                else:
                    support_r, attack_r = row - 1, row + 1

                # Check for friendly pawn support
                if 0 <= support_r < 8:
                    for dc in [-1, 1]:
                        support_c = col + dc
                        if 0 <= support_c < 8:
                            p = game.board[support_r][support_c]
                            if p and p.name == 'P' and p.colour == piece.colour:
                                is_supported = True
                                break

                # Check for enemy pawn attacks
                if 0 <= attack_r < 8:
                    for dc in [-1, 1]:
                        attack_c = col + dc
                        if 0 <= attack_c < 8:
                            p = game.board[attack_r][attack_c]
                            if p and p.name == 'P' and p.colour != piece.colour:
                                is_attackable = True
                                break

                if is_supported and not is_attackable:
                    val += 20


            elif piece.name == 'B':
                val += bishop_table[index]
            elif piece.name == 'R':
                val += rook_table[index]

                if (piece.colour == 'w' and row == 1) or (piece.colour == 'b' and row == 6):
                    val += 25


            elif piece.name == 'Q':
                val += queen_table[index]
            elif piece.name == 'K':
                mg_value = king_table[index]
                eg_value = king_endgame_table[index]
                val += (1 - phase) * mg_value + phase * eg_value

            if piece.name == 'P': (white_pawns if piece.colour == 'w' else black_pawns).append((row, col))
            score += val if piece.colour == 'w' else -val

            defenders, attackers = 0, 0
            for r1 in range(8):
                for c1 in range(8):
                    piece2 = game.board[r1][c1]
                    if not piece2: continue

                    # NOTE: This will now only check for attackers/defenders
                    # from the *active* player, as that's all all_moves_cache contains.
                    # This is a reasonable trade-off for the massive speed boost.
                    moves = all_moves_cache.get((r1, c1), [])

                    if (row, col) in moves:
                        if piece2.colour == piece.colour:
                            defenders += 1
                        else:
                            attackers += 1
            if attackers > defenders: score += -30 * (1 if piece.colour == 'w' else -1)

            # Rook on Open/Semi-Open File Bonus
            if piece.name == 'R':
                has_friendly_pawn = False
                for r_check in range(8):
                    if game.board[r_check][col] and game.board[r_check][col].name == 'P' and game.board[r_check][
                        col].colour == piece.colour: has_friendly_pawn = True; break

                if not has_friendly_pawn:
                    has_enemy_pawn = any(
                        game.board[r_check][col] and game.board[r_check][col].name == 'P' for r_check in range(8))
                    score += (25 if not has_enemy_pawn else 10) * (1 if piece.colour == 'w' else -1)

    # Pawn structure and Passed Pawn logic
    def pawn_structure(pawns, enemy_pawns, colour):
        bonus = 0
        files = [0] * 8
        for r, c in pawns: files[c] += 1
        for f in range(8):
            if files[f] > 1: bonus -= 20 * (files[f] - 1)
            if f == 0 and files[f + 1] == 0:  # Corrected from files[f-1]
                bonus -= 15
            elif f == 7 and files[f - 1] == 0:
                bonus -= 15
            elif 0 < f < 7 and files[f - 1] == 0 and files[f + 1] == 0:
                bonus -= 15

        for r, c in pawns:

            # backward pawns
            is_backward = True
            direction = 1 if colour == 'w' else -1  # 1 moves "behind" white, -1 moves "behind" black

            for dc in [-1, 1]:
                if not (0 <= c + dc < 8): continue
                check_r = r
                while (0 <= check_r < 8) if colour == 'w' else (0 <= check_r < 8):
                    p = game.board[check_r][c + dc]
                    if p and p.name == 'P' and p.colour == colour:
                        is_backward = False  # Found support
                        break
                    if colour == 'w':
                        check_r += 1  # Check ranks behind
                    else:
                        check_r -= 1  # Check ranks behind
                if not is_backward: break

            if is_backward:
                front_r = r - direction
                is_stoppable = False
                if 0 <= front_r < 8:
                    p_front = game.board[front_r][c]
                    # Stopped by any enemy piece
                    if p_front and p_front.colour != colour:
                        is_stoppable = True
                    else:
                        # Or attackable by enemy pawn
                        for dc in [-1, 1]:
                            attack_c = c + dc
                            if 0 <= attack_c < 8:
                                p = game.board[front_r][attack_c]
                                if p and p.name == 'P' and p.colour != colour:
                                    is_stoppable = True
                                    break  # Attackable, no need to check other side

                if is_stoppable:
                    bonus -= 10

            blocked = False
            for er, ec in enemy_pawns:
                if ec in [c - 1, c, c + 1] and ((colour == 'w' and er < r) or (colour == 'b' and er > r)):
                    blocked = True;
                    break
            if not blocked:
                rank = r if colour == 'b' else 7 - r
                bonus += 10 + rank * 5

            is_passed = True
            direction = -1 if colour == 'w' else 1
            current_r = r + direction
            while 0 <= current_r < 8:
                for dc in [-1, 0, 1]:
                    current_c = c + dc
                    if 0 <= current_c < 8:
                        p_check = game.board[current_r][current_c]
                        if p_check and p_check.name == 'P' and p_check.colour != colour:
                            is_passed = False;
                            break
                    if not is_passed: break
                if not is_passed: break
                current_r += direction

            if is_passed:
                if colour == 'w':
                    rank_bonus = 6 - r
                else:
                    rank_bonus = r - 1
                passed_pawn_bonus = [0, 50, 100, 200, 350, 500, 0][rank_bonus] if 0 <= rank_bonus < 7 else 0
                bonus += passed_pawn_bonus
        return bonus

    score += pawn_structure(white_pawns, black_pawns, 'w')
    score -= pawn_structure(black_pawns, white_pawns, 'b')

    white_moves = sum(len(all_moves_cache.get((r, c), [])) for r in range(8) for c in range(8) if
                      game.board[r][c] and game.board[r][c].colour == 'w')
    black_moves = sum(len(all_moves_cache.get((r, c), [])) for r in range(8) for c in range(8) if
                      game.board[r][c] and game.board[r][c].colour == 'b')

    if white_moves > 0 and black_moves == 0:
        score += 10 * white_moves
    elif black_moves > 0 and white_moves == 0:
        score -= 10 * black_moves

    white_bishops = sum(1 for r in range(8) for c in range(8) if
                        game.board[r][c] and game.board[r][c].name == 'B' and game.board[r][c].colour == 'w')
    black_bishops = sum(1 for r in range(8) for c in range(8) if
                        game.board[r][c] and game.board[r][c].name == 'B' and game.board[r][c].colour == 'b')
    if white_bishops >= 2: score += 50
    if black_bishops >= 2: score -= 50

    def king_safety(game, colour, move_cache):
        attack_weights = {'Q': 10, 'R': 5, 'B': 3, 'N': 3, 'P': 1, 'K': 0}
        kingr, kingc = None, None
        for r in range(8):
            for c in range(8):
                piece = game.board[r][c]
                if piece and piece.name == 'K' and piece.colour == colour:
                    kingr, kingc = r, c
                    break
            if kingr is not None: break

        if kingr is None: return 0

        danger = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = kingr + dr, kingc + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    for r1 in range(8):
                        for c1 in range(8):
                            piece2 = game.board[r1][c1]
                            if piece2 and piece2.colour != colour:
                                # This will now only check for attackers from the active player
                                attacker_moves = move_cache.get((r1, c1), [])
                                if (nr, nc) in attacker_moves:
                                    danger += attack_weights.get(piece2.name, 0)

        shield_bonus = 0
        directions = -1 if colour == 'w' else 1
        for dc in [-1, 0, 1]:
            file = kingc + dc
            for r in [kingr + directions, kingr + 2 * directions]:
                if 0 <= r < 8 and 0 <= file < 8:
                    p = game.board[r][file]
                    if p and p.name == 'P' and p.colour == colour:
                        shield_bonus += 15 if r == kingr + directions else 10
        return -30 * danger + shield_bonus

    # Only apply king safety for the active (cached) player
    if game.turn == 'w':
        score += king_safety(game, 'w', all_moves_cache) * (1 - phase)
        score -= king_safety(game, 'b', {}) * (1 - phase)  # Pass empty cache for inactive
    else:
        score += king_safety(game, 'w', {}) * (1 - phase)  # Pass empty cache for inactive
        score -= king_safety(game, 'b', all_moves_cache) * (1 - phase)

    white_rooks = []
    black_rooks = []
    for r in range(8):
        for c in range(8):
            p = game.board[r][c]
            if p and p.name == 'R':
                if p.colour == 'w':
                    white_rooks.append((r, c))
                else:
                    black_rooks.append((r, c))

    if len(white_rooks) >= 2:
        r1, c1 = white_rooks[0]
        r2, c2 = white_rooks[1]

        if r1 == r2:
            if all(game.board[r1][c] is None for c in range(min(c1, c2) + 1, max(c1, c2))):
                score += 20

    if len(black_rooks) >= 2:
        r1, c1 = black_rooks[0]
        r2, c2 = black_rooks[1]
        if r1 == r2:
            if all(game.board[r1][c] is None for c in range(min(c1, c2) + 1, max(c1, c2))):
                score -= 20

    # Central control
    central_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    for r, c in central_squares:
        piece = game.board[r][c]
        if piece:
            score += 15 * (1 if piece.colour == 'w' else -1)
    return score


def static_exchange_eval_local(game, start, end, move_cache):
    attacker_piece = game.board[start[0]][start[1]]
    target_piece = game.board[end[0]][end[1]]

    if not attacker_piece or not target_piece:
        return 0  # Not a capture

    # 'gains' will store the value of *each piece captured* in the sequence
    gains = [piece_values[target_piece.name]]

    copyg = game.light_copy()
    copyg.make_move(start, end)

    side = copyg.turn
    target_square = end

    while True:
        attackers = []
        for r in range(8):
            for c in range(8):
                piece = copyg.board[r][c]
                if not piece or piece.colour != side:
                    continue

                # Use simple attack_moves for SEE, it's faster
                moves = copyg.attack_moves(r, c)

                if target_square in moves:
                    attackers.append((r, c))

        if not attackers:
            break  # No more attackers, sequence ends

        best_attacker_pos = min(attackers, key=lambda sq: piece_values[copyg.board[sq[0]][sq[1]].name])

        if not copyg.board[target_square[0]][target_square[1]]:
            break

        victim_value = piece_values[copyg.board[target_square[0]][target_square[1]].name]
        gains.append(victim_value)

        # Simulate this next capture
        copyg.make_move(best_attacker_pos, target_square)

        # Swap sides for the next recapture
        side = 'b' if side == 'w' else 'w'

    score = 0
    # Iterate backwards
    for i in range(len(gains) - 1, -1, -1):
        score = max(0, gains[i] - score)

    return gains[0] - score


def quiescence_search(game, alpha, beta, maximizing, move_cache):

    eval_score = evaluate_board(game, {})

    if maximizing:
        if eval_score >= beta:
            return beta
        alpha = max(alpha, eval_score)
    else:
        if eval_score <= alpha:
            return alpha
        beta = min(beta, eval_score)

    forcing_moves = []
    colour = 'w' if maximizing else 'b'

    if move_cache:  # Only if move_cache was provided
        for start_pos, moves in move_cache.items():
            piece = game.board[start_pos[0]][start_pos[1]]
            if not piece: continue

            for end_pos in moves:
                is_capture = game.board[end_pos[0]][end_pos[1]] is not None
                if is_capture:
                    victim = game.board[end_pos[0]][end_pos[1]]
                    victim_val = piece_values.get(victim.name, 0)
                    attacker_val = piece_values.get(piece.name, 1)
                    priority = 10000 + victim_val * 10 - attacker_val

                    # Check for promotion captures
                    if piece.name == 'P' and (end_pos[0] == 0 or end_pos[0] == 7):
                        promos = ['Q', 'R', 'B', 'N']
                    else:
                        promos = [None]

                    for p in promos:
                        forcing_moves.append((priority, start_pos, end_pos, p))

    # Sort moves: Best captures first
    forcing_moves.sort(key=lambda x: x[0], reverse=True)

    for priority, start, end, promotion in forcing_moves:

        # Run SEE on the *legal* capture.
        net_gain = static_exchange_eval_local(game, start, end, {})
        if net_gain < 0:
            continue

        # Makes the move and searches deeper
        copy = game.light_copy()
        copy.make_move(start, end, promotion)

        recursive_move_cache = {}
        for r in range(8):
            for c in range(8):
                p = copy.board[r][c]
                if p and p.colour == copy.turn:
                    recursive_move_cache[(r, c)] = copy.get_moves(r, c)

        q_eval = quiescence_search(copy, alpha, beta, not maximizing, recursive_move_cache)

        if maximizing:
            if q_eval >= beta:
                return beta
            alpha = max(alpha, q_eval)
        else:
            if q_eval <= alpha:
                return alpha
            beta = min(beta, q_eval)

    # Return the best score found
    return alpha if maximizing else beta




def minimax_sse(game, depth, alpha, beta, maximizing, original_depth=None,
                start_time=None, time_limit=None, principal_variation=None):
    if original_depth is None: original_depth = depth

    move_cache = {}

    ply = original_depth - depth  # Current depth from root

    best_move = None

    if time_limit and time.time() - start_time > time_limit:
        return None

        # --- 1. Transposition Table Lookup ---
    alpha_orig = alpha
    tt_key = calculate_zobrist_hash(game)
    tt_entry = TT.get(tt_key)

    tt_best_move = None
    if tt_entry:
        tt_best_move = tt_entry.get('best_move')

        if tt_entry['depth'] >= depth:
            tt_score = tt_entry['score']
            tt_flag = tt_entry['flag']
            if tt_flag == TT_EXACT:
                # Returns move if top-level, else the score
                return tt_best_move if depth == original_depth else tt_score
            elif tt_flag == TT_ALPHA:
                alpha = max(alpha, tt_score)
            elif tt_flag == TT_BETA:
                beta = min(beta, tt_score)

            if alpha >= beta:
                # Return move if top-level, else the score
                return tt_best_move if depth == original_depth else tt_score

    # ---new Check Extensions ---
    is_in_check = game.is_check(game.turn)
    if is_in_check:
        depth += 1  # Extends the search if in check

    R = 2 + (1 if depth >= 5 else 0)
    if depth >= 3 and not is_in_check:

        copy_null = game.light_copy()
        copy_null.turn = 'b' if copy_null.turn == 'w' else 'w'
        copy_null.enpassant = None

        null_eval = minimax_sse(copy_null, depth - 1 - R, -beta, -alpha, not maximizing, original_depth=original_depth,
                                start_time=start_time, time_limit=time_limit)

        if null_eval is None: return None

        if -null_eval >= beta:
            return tt_best_move if depth == original_depth else beta


    if depth == 0 or game.state is not None:
        #generates moves for the *active* player to pass to quiescence
        for r in range(8):
            for c in range(8):
                p = game.board[r][c]
                if p and p.colour == game.turn:
                    move_cache[(r, c)] = game.get_moves(r, c)

        score = quiescence_search(game, alpha, beta, maximizing, move_cache)
        TT[tt_key] = {'depth': 0, 'score': score, 'flag': TT_EXACT, 'best_move': None}
        return best_move if depth == original_depth else score


    # --- Move List and Ordering Setup ---
    all_moves = []
    current_colour = 'w' if maximizing else 'b'

    def calculate_priority(start, end, promotion, tt_best_move=None, principal_variation=None):
        priority = 0
        move_tuple = (start, end, promotion)
        start_idx = start[0] * 8 + start[1]
        end_idx = end[0] * 8 + end[1]

        if principal_variation and move_tuple == principal_variation:
            priority = 100000
        elif tt_best_move == move_tuple:
            priority = 90000
        elif game.board[end[0]][end[1]]:  # Captures (MVV-LVA)
            victim_val = piece_values.get(game.board[end[0]][end[1]].name, 0)
            attacker_val = piece_values.get(game.board[start[0]][start[1]].name, 1)
            priority = 10000 + victim_val * 10 - attacker_val
        else:  # Quiet Moves (Killer and History Heuristics)
            if move_tuple in KILLER_MOVES[ply]:
                priority = 8000 + (KILLER_MOVES[ply].index(move_tuple) * 250)
            else:
                priority = HISTORY_MOVES[start_idx][end_idx]

        return priority

    pv_move_tuple = principal_variation

    for r in range(8):
        for c in range(8):
            piece = game.board[r][c]
            if piece and piece.colour == current_colour:

                # --- new local cache population ---
                moves = game.get_moves(r, c)
                move_cache[(r, c)] = moves

                for move in moves:
                    promotion_options = ['Q', 'R', 'B', 'N'] if piece.name == 'P' and (
                            move[0] == 0 or move[0] == 7) else [None]
                    for p in promotion_options:
                        priority = calculate_priority((r, c), move, p, tt_best_move, pv_move_tuple)
                        all_moves.append((priority, (r, c), move, p))

    all_moves.sort(key=lambda x: x[0], reverse=True)

    # --- Check for stalemate/checkmate ---
    if not all_moves:
        if is_in_check:
            return -999999 if maximizing else 999999  # Checkmate
        else:
            return 0  # Stalemate

    move_index = 0  # For PVS/LMR

    if maximizing:
        max_eval = float('-inf')
        # Pass local cache to eval
        static_eval = evaluate_board(game, move_cache)

        for priority, start, end, promotion in all_moves:

            is_capture = game.board[end[0]][end[1]]

            # futility pruning
            if not is_in_check and priority < 90000 and not is_capture and depth <= 2 and depth < original_depth:
                if static_eval + FUTILITY_MARGIN * depth < alpha:
                    continue

            # SSE pruning
            if is_capture:
                net_gain = static_exchange_eval_local(game, start, end, move_cache)
                if net_gain < 0: continue

            copy = game.light_copy()
            copy.make_move(start, end, promotion)

            # --- LMR (Late Move Reductions) ---
            reduction = 0
            if depth >= 3 and move_index >= 3 and not is_capture and not is_in_check and priority < 8000:
                reduction = 1

            eval_score = None

            # --- PVS (Principal Variation Search) ---
            if move_index == 0:  # PVS: Full window search for the first move
                eval_score = minimax_sse(copy, depth - 1, alpha, beta, False, original_depth,
                                         start_time=start_time, time_limit=time_limit,
                                         principal_variation=principal_variation)
            else:
                # PVS: Zero-window search for subsequent moves (with LMR applied)
                eval_score = minimax_sse(copy, depth - 1 - reduction, alpha, alpha + 1, False, original_depth,
                                         start_time=start_time, time_limit=time_limit,
                                         principal_variation=principal_variation)

                if eval_score is None:
                    return None

                if eval_score > alpha and eval_score < beta:
                    eval_score = minimax_sse(copy, depth - 1, alpha, beta, False, original_depth,
                                             start_time=start_time, time_limit=time_limit,
                                             principal_variation=principal_variation)

            move_index += 1  # Increments move counter

            if eval_score is None:
                return None

            if eval_score > max_eval:
                max_eval = max_eval
                best_move = (start, end, promotion)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                max_eval = beta
                if not is_capture:
                    KILLER_MOVES[ply][1] = KILLER_MOVES[ply][0]
                    KILLER_MOVES[ply][0] = (start, end, promotion)
                    start_idx = start[0] * 8 + start[1]
                    end_idx = end[0] * 8 + end[1]
                    HISTORY_MOVES[start_idx][end_idx] += depth * depth
                break
        score = max_eval

    else:
        min_eval = float('inf')
        static_eval = evaluate_board(game, move_cache)

        move_index = 0

        for priority, start, end, promotion in all_moves:

            is_capture = game.board[end[0]][end[1]]

            # futility pruning
            if not is_in_check and priority < 90000 and not is_capture and depth <= 2 and depth < original_depth:
                if static_eval - FUTILITY_MARGIN * depth > beta:
                    continue

            # SSE pruning
            if is_capture:
                net_gain = static_exchange_eval_local(game, start, end, move_cache)
                if net_gain < 0: continue

            copy = game.light_copy()
            copy.make_move(start, end, promotion)

            # --- LMR (Late Move Reductions) ---
            reduction = 0
            if depth >= 3 and move_index >= 3 and not is_capture and not is_in_check and priority < 8000:
                reduction = 1

            eval_score = None  # Initialize

            # --- PVS (Principal Variation Search) ---
            if move_index == 0:
                eval_score = minimax_sse(copy, depth - 1, alpha, beta, True, original_depth,
                                         start_time=start_time, time_limit=time_limit,
                                         principal_variation=principal_variation)
            else:
                # PVS: Zero-window search
                eval_score = minimax_sse(copy, depth - 1 - reduction, beta - 1, beta, True, original_depth,
                                         start_time=start_time, time_limit=time_limit,
                                         principal_variation=principal_variation)

                if eval_score is None:
                    return None

                # PVS: Re-search if failed high (low for minimizing)
                if eval_score < beta and eval_score > alpha:
                    # Re-search at full depth (no reduction)
                    eval_score = minimax_sse(copy, depth - 1, alpha, beta, True, original_depth,
                                             start_time=start_time, time_limit=time_limit,
                                             principal_variation=principal_variation)

            move_index += 1  # Increment move counter

            if eval_score is None:
                return None

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = (start, end, promotion)
            beta = min(beta, eval_score)
            if beta <= alpha:
                min_eval = alpha
                if not is_capture:
                    KILLER_MOVES[ply][1] = KILLER_MOVES[ply][0]
                    KILLER_MOVES[ply][0] = (start, end, promotion)
                    start_idx = start[0] * 8 + start[1]
                    end_idx = end[0] * 8 + end[1]
                    HISTORY_MOVES[start_idx][end_idx] += depth * depth
                break
        score = min_eval

    # --- 2. Transposition Table Store ---
    tt_flag = TT_EXACT
    if score <= alpha_orig:
        tt_flag = TT_ALPHA
    elif score >= beta:
        tt_flag = TT_BETA

    # TT Size Management
    if len(TT) >= TT_MAX_SIZE:
        keys_to_delete = list(TT.keys())[:int(TT_MAX_SIZE * 0.1)]  # Delete 10% of the oldest entries
        for key in keys_to_delete:
            del TT[key]

    if best_move:
        TT[tt_key] = {'depth': depth, 'score': score, 'flag': tt_flag, 'best_move': best_move}

    return best_move if depth == original_depth else score