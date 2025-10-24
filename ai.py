import time
import random
import math # Needed for dynamic LMR

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

# King attack weights for king safety/attack
king_attack_weights = {'Q': 5, 'R': 3, 'B': 2, 'N': 2, 'P': 1}
# Global array to store attack counts near each king (index 0=white, 1=black)
king_zone_attack_count = [0] * 2


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
    white_bishops, black_bishops = [], []
    white_king_pos, black_king_pos = game.w_king_pos, game.b_king_pos

    all_moves_cache = move_cache if move_cache is not None else {}

    current_material = sum(
        PHASE_MATERIAL.get(game.board[r][c].name, 0) for r in range(8) for c in range(8) if game.board[r][c])
    phase = 1.0 - (current_material / MAX_PHASE_MATERIAL)
    phase = max(0.0, min(1.0, phase))

    white_pawn_attacks = set()
    black_pawn_attacks = set()
    for r in range(8):
        for c in range(8):
            p = game.board[r][c]
            if p and p.name == 'P':
                if p.colour == 'w':
                    if r - 1 >= 0:
                        if c - 1 >= 0: white_pawn_attacks.add((r - 1, c - 1))
                        if c + 1 < 8: white_pawn_attacks.add((r - 1, c + 1))
                else:
                    if r + 1 < 8:
                        if c - 1 >= 0: black_pawn_attacks.add((r + 1, c - 1))
                        if c + 1 < 8: black_pawn_attacks.add((r + 1, c + 1))

    king_zone_attack_count[0] = 0
    king_zone_attack_count[1] = 0

    for row in range(8):
        for col in range(8):
            piece = game.board[row][col]
            if not piece: continue
            val = piece_values[piece.name]
            # Use mirrored index for black pieces in piece-square tables only if needed by your tables
            # Assuming your tables are defined from white's perspective, black needs mirroring.
            index = row * 8 + col if piece.colour == 'w' else (7 - row) * 8 + col

            # Apply piece-square table bonuses/penalties
            if piece.name == 'P': val += pawn_table[index]
            elif piece.name == 'N': val += knight_table[index]
            elif piece.name == 'B': val += bishop_table[index]
            elif piece.name == 'R': val += rook_table[index]
            elif piece.name == 'Q': val += queen_table[index]
            elif piece.name == 'K':
                 # Tapered king eval: Use king_table in opening/midgame, king_endgame_table in endgame
                 mg_value = king_table[row * 8 + col] # Use non-mirrored index for king safety calcs maybe? Or ensure tables are consistent
                 eg_value = king_endgame_table[row * 8 + col]
                 val += int((1 - phase) * mg_value + phase * eg_value)

            # --- Specific Piece Bonuses/Penalties ---
            if piece.name == 'N': # Knight outpost
                is_supported, is_attackable = False, False
                support_r, attack_r = (row + 1, row - 1) if piece.colour == 'w' else (row - 1, row + 1)
                if 0 <= support_r < 8:
                    for dc in [-1, 1]:
                        support_c = col + dc
                        if 0 <= support_c < 8:
                            p = game.board[support_r][support_c]
                            if p and p.name == 'P' and p.colour == piece.colour: is_supported = True; break
                if 0 <= attack_r < 8:
                    for dc in [-1, 1]:
                        attack_c = col + dc
                        if 0 <= attack_c < 8:
                            p = game.board[attack_r][attack_c]
                            if p and p.name == 'P' and p.colour != piece.colour: is_attackable = True; break
                if is_supported and not is_attackable: val += 20

            elif piece.name == 'B': # Add to list for Bad Bishop eval
                if piece.colour == 'w': white_bishops.append((row, col))
                else: black_bishops.append((row, col))

            elif piece.name == 'R': # Rook on 7th rank (relative)
                if (piece.colour == 'w' and row == 1) or \
                   (piece.colour == 'b' and row == 6):
                     val += 25
                # Rook on Open/Semi-Open File
                has_friendly_pawn = False
                for r_check in range(8):
                    p_check = game.board[r_check][col]
                    if p_check and p_check.name == 'P' and p_check.colour == piece.colour: has_friendly_pawn = True; break
                if not has_friendly_pawn:
                    has_enemy_pawn = any(game.board[r_check][col] and game.board[r_check][col].name == 'P' and game.board[r_check][col].colour != piece.colour for r_check in range(8))
                    val += (25 if not has_enemy_pawn else 10) # Bigger bonus for fully open file

            if piece.name == 'P': # Add pawns to lists
                 (white_pawns if piece.colour == 'w' else black_pawns).append((row, col))

            score += val if piece.colour == 'w' else -val # Add piece value to total score

            # --- Tapered King Attack Score Logic ---
            if piece.name != 'K': # Don't count the king itself
                enemy_king_pos = black_king_pos if piece.colour == 'w' else white_king_pos
                if enemy_king_pos:
                    distance = max(abs(row - enemy_king_pos[0]), abs(col - enemy_king_pos[1]))
                    if distance <= 3: # Piece is near the enemy king
                         attack_bonus = king_attack_weights.get(piece.name, 0) * (4 - distance) * (1 + phase / 2)
                         score += int(attack_bonus * (1 if piece.colour == 'w' else -1))
                         king_index = 1 if piece.colour == 'w' else 0 # Attacking black king (index 1) if white piece, vice versa
                         king_zone_attack_count[king_index] += king_attack_weights.get(piece.name, 0)

    # --- Pawn Structure Evaluation (including advanced passed pawns) ---
    def pawn_structure(pawns, enemy_pawns, colour):
        bonus = 0
        files = [0] * 8
        for r, c in pawns: files[c] += 1
        for f in range(8):
            if files[f] > 1: bonus -= 20 * (files[f] - 1) # Doubled pawn penalty
            # Isolated pawn penalty
            left_neighbor = files[f - 1] if f > 0 else 0
            right_neighbor = files[f + 1] if f < 7 else 0
            if files[f] > 0 and left_neighbor == 0 and right_neighbor == 0:
                 bonus -= 15

        for r, c in pawns:
            # --- Backward Pawn Check ---
            direction = 1 if colour == 'w' else -1 # Direction "behind" pawn

            # >>> FIX: Initialize is_backward correctly HERE <<<
            is_backward = True

            for dc in [-1, 1]: # Check adjacent files for support
                if not (0 <= c + dc < 8): continue
                check_r = r + direction
                found_support = False
                while 0 <= check_r < 8:
                    p = game.board[check_r][c + dc]
                    if p and p.name == 'P' and p.colour == colour:
                        found_support = True; break
                    check_r += direction
                if found_support:
                    is_backward = False; break # Correctly breaks the 'for dc' loop
            # >>> END FIX LOCATION <<<

            if is_backward:
                stoppable_by_pawn = False
                attack_r = r - direction # Rank in front
                if 0 <= attack_r < 8:
                     for dc_attack in [-1, 1]:
                          attack_c = c + dc_attack
                          if 0 <= attack_c < 8:
                               p = game.board[attack_r][attack_c]
                               if p and p.name == 'P' and p.colour != colour:
                                    stoppable_by_pawn = True; break
                     if stoppable_by_pawn: bonus -= 10

            # --- Passed Pawn Logic ---
            is_passed = True
            pass_direction = -1 if colour == 'w' else 1
            current_r = r + pass_direction
            while 0 <= current_r < 8:
                for dc_pass in [-1, 0, 1]:
                    current_c = c + dc_pass
                    if 0 <= current_c < 8:
                        p_check = game.board[current_r][current_c]
                        if p_check and p_check.name == 'P' and p_check.colour != colour:
                            is_passed = False; break
                if not is_passed: break
                current_r += pass_direction

            if is_passed:
                rank = 6 - r if colour == 'w' else r - 1
                passed_pawn_bonus = [0, 10, 30, 60, 100, 150, 200][rank] # Use rank 0-6
                bonus += passed_pawn_bonus

                # Advanced passed pawn bonuses
                is_path_clear = True
                check_r_safety = r + pass_direction
                while 0 <= check_r_safety < 8:
                    p_check_safety = game.board[check_r_safety][c]
                    if p_check_safety:
                         if p_check_safety.colour == colour and abs(check_r_safety - r) > 2: pass
                         else: is_path_clear = False; break
                    check_r_safety += pass_direction
                if is_path_clear: bonus += rank * 15

                support_r = r + direction
                is_connected = False
                if 0 <= support_r < 8:
                    for dc_connect in [-1, 1]:
                        support_c = c + dc_connect
                        if 0 <= support_c < 8:
                            p_connect = game.board[support_r][support_c]
                            if p_connect and p_connect.name == 'P' and p_connect.colour == colour: is_connected = True; break
                if is_connected: bonus += 20

                blockade_r = r + pass_direction
                if 0 <= blockade_r < 8:
                    blocker = game.board[blockade_r][c]
                    if blocker and blocker.colour != colour:
                        if blocker.name == 'N': bonus -= 40
                        elif blocker.name == 'B': bonus -= 30
                        else: bonus -= 15

                check_r_rook = r + direction
                while 0 <= check_r_rook < 8:
                    p_check_rook = game.board[check_r_rook][c]
                    if p_check_rook and p_check_rook.name == 'R' and p_check_rook.colour == colour: bonus += 25; break
                    elif p_check_rook: break
                    check_r_rook += direction
        return bonus

    score += pawn_structure(white_pawns, black_pawns, 'w')
    score -= pawn_structure(black_pawns, white_pawns, 'b')

    # --- Safe Mobility Scoring ---
    safe_mobility_bonus = 0
    if all_moves_cache:
        opponent_pawn_attacks = black_pawn_attacks if game.turn == 'w' else white_pawn_attacks
        for start_pos, moves in all_moves_cache.items():
             piece_mob = game.board[start_pos[0]][start_pos[1]]
             mob_factor = 1.0
             if piece_mob:
                  if piece_mob.name == 'P': mob_factor = 0.5
                  elif piece_mob.name == 'K': mob_factor = 0.7
             for end_pos in moves:
                 if end_pos not in opponent_pawn_attacks: safe_mobility_bonus += mob_factor
    score += int(safe_mobility_bonus * 3 * (1 if game.turn == 'w' else -1))

    # --- Bad Bishop Penalty ---
    white_pawns_on_light_central, white_pawns_on_dark_central = 0, 0
    black_pawns_on_light_central, black_pawns_on_dark_central = 0, 0
    central_files = {2, 3, 4, 5}
    for r, c in white_pawns:
        if c in central_files:
            if (r + c) % 2 == 0: white_pawns_on_light_central += 1
            else: white_pawns_on_dark_central += 1
    for r, c in black_pawns:
        if c in central_files:
            if (r + c) % 2 == 0: black_pawns_on_light_central += 1
            else: black_pawns_on_dark_central += 1

    bad_bishop_penalty = 0
    for r, c in white_bishops:
        if (r + c) % 2 == 0: bad_bishop_penalty -= white_pawns_on_light_central * 5
        else: bad_bishop_penalty -= white_pawns_on_dark_central * 5
    for r, c in black_bishops:
        if (r + c) % 2 == 0: bad_bishop_penalty += black_pawns_on_light_central * 5
        else: bad_bishop_penalty += black_pawns_on_dark_central * 5
    score += bad_bishop_penalty

    # --- Bishop Pair Bonus ---
    white_bishops_count = len(white_bishops)
    black_bishops_count = len(black_bishops)
    if white_bishops_count >= 2: score += 50
    if black_bishops_count >= 2: score -= 50

    # --- King Safety ---
    def king_safety(colour, king_pos, num_attackers):
        kingr, kingc = king_pos if king_pos else (None, None)
        if kingr is None: return 0
        shield_bonus = 0
        shield_direction = -1 if colour == 'w' else 1
        for dc in [-1, 0, 1]:
            check_c = kingc + dc
            if 0 <= check_c < 8:
                 r1 = kingr + shield_direction
                 if 0 <= r1 < 8:
                      p1 = game.board[r1][check_c]
                      if p1 and p1.name == 'P' and p1.colour == colour: shield_bonus += 15
                 r2 = kingr + 2 * shield_direction
                 if 0 <= r2 < 8:
                      p2 = game.board[r2][check_c]
                      if p2 and p2.name == 'P' and p2.colour == colour: shield_bonus += 10
        attack_danger_penalty = num_attackers * num_attackers * 2
        return shield_bonus - attack_danger_penalty

    score += int(king_safety('w', white_king_pos, king_zone_attack_count[0]) * (1 - phase))
    score -= int(king_safety('b', black_king_pos, king_zone_attack_count[1]) * (1 - phase))

    # --- Connected Rooks Bonus ---
    white_rooks, black_rooks = [], []
    for r in range(8):
        for c in range(8):
            p = game.board[r][c]
            if p and p.name == 'R':
                if p.colour == 'w': white_rooks.append((r, c))
                else: black_rooks.append((r, c))
    if len(white_rooks) >= 2:
        r1, c1 = white_rooks[0]; r2, c2 = white_rooks[1]
        if r1 == r2 and all(game.board[r1][c] is None for c in range(min(c1, c2) + 1, max(c1, c2))): score += 20
        elif c1 == c2 and all(game.board[r][c1] is None for r in range(min(r1, r2) + 1, max(r1, r2))): score += 20
    if len(black_rooks) >= 2:
        r1, c1 = black_rooks[0]; r2, c2 = black_rooks[1]
        if r1 == r2 and all(game.board[r1][c] is None for c in range(min(c1, c2) + 1, max(c1, c2))): score -= 20
        elif c1 == c2 and all(game.board[r][c1] is None for r in range(min(r1, r2) + 1, max(r1, r2))): score -= 20

    # --- Central Control ---
    central_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    for r, c in central_squares:
        piece = game.board[r][c]
        if piece: score += 10 * (1 if piece.colour == 'w' else -1)

    return score


def static_exchange_eval_local(game, start, end):
    attacker_piece = game.board[start[0]][start[1]]
    target_piece = game.board[end[0]][end[1]]
    if not attacker_piece or not target_piece: return 0
    gains = [piece_values[target_piece.name]]
    copyg = game.light_copy()
    if not copyg._force_move(start, end): return 0
    side = copyg.turn
    target_square = end
    while True:
        attackers = []
        best_attacker_val = float('inf')
        best_attacker_pos = None
        for r in range(8):
            for c in range(8):
                piece = copyg.board[r][c]
                if not piece or piece.colour != side: continue
                if target_square in copyg.attack_moves(r, c):
                     attacker_val = piece_values[piece.name]
                     if attacker_val < best_attacker_val:
                          best_attacker_val = attacker_val
                          best_attacker_pos = (r,c)
        if best_attacker_pos is None: break
        captured_piece = copyg.board[target_square[0]][target_square[1]]
        if not captured_piece: break
        gains.append(piece_values[captured_piece.name])
        if not copyg._force_move(best_attacker_pos, target_square): break
        side = copyg.turn
    score = 0
    for i in range(len(gains) - 1, -1, -1): score = max(0, gains[i] - score)
    return gains[0] - score


def quiescence_search(game, alpha, beta, maximizing, move_cache):
    stand_pat = evaluate_board(game)
    if maximizing:
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha: return alpha
        beta = min(beta, stand_pat)

    forcing_moves = []
    colour = game.turn

    current_move_cache = move_cache if move_cache else {}
    if not move_cache:
        for r in range(8):
            for c in range(8):
                p = game.board[r][c]
                if p and p.colour == colour:
                     current_move_cache[(r, c)] = game.get_moves(r, c)

    for start_pos, moves in current_move_cache.items():
        piece = game.board[start_pos[0]][start_pos[1]]
        if not piece: continue
        for end_pos in moves:
            is_capture = game.board[end_pos[0]][end_pos[1]] is not None
            is_check = False
            if not is_capture:
                 temp_copy = game.light_copy()
                 if temp_copy._force_move(start_pos, end_pos):
                      if temp_copy.is_check(temp_copy.turn): is_check = True

            if is_capture or is_check:
                promos = ['Q']
                if piece.name == 'P' and (end_pos[0] == 0 or end_pos[0] == 7):
                    if is_check and not is_capture: promos = ['Q', 'N']
                else: promos = [None]

                for p in promos:
                    victim = game.board[end_pos[0]][end_pos[1]]
                    victim_val = piece_values.get(victim.name, 0) if victim else 0
                    attacker_val = piece_values.get(piece.name, 1)
                    priority = 0
                    if is_capture: priority = 10000 + victim_val * 10 - attacker_val
                    elif is_check: priority = 5000
                    forcing_moves.append((priority, start_pos, end_pos, p))

    forcing_moves.sort(key=lambda x: x[0], reverse=True)

    for priority, start, end, promotion in forcing_moves:
        # Delta Pruning
        capture_val = 0; victim = game.board[end[0]][end[1]]
        if victim: capture_val = piece_values.get(victim.name, 0)
        promo_val = piece_values.get(promotion, 0) if promotion else 0
        delta_margin = 150
        if maximizing:
             if stand_pat + capture_val + promo_val + delta_margin < alpha: continue
        else:
             if stand_pat - capture_val - promo_val - delta_margin > beta: continue

        # SEE Pruning for captures
        if game.board[end[0]][end[1]]:
            net_gain = static_exchange_eval_local(game, start, end)
            if net_gain < 0: continue

        copy = game.light_copy()
        if not copy.make_move(start, end, promotion): continue

        # Recursive call requires moves for the next state
        recursive_move_cache = {}
        next_colour = copy.turn
        for r in range(8):
            for c in range(8):
                p_next = copy.board[r][c]
                if p_next and p_next.colour == next_colour:
                    try: recursive_move_cache[(r, c)] = copy.get_moves(r, c)
                    except: pass

        q_eval = quiescence_search(copy, alpha, beta, not maximizing, recursive_move_cache)

        if maximizing:
            if q_eval >= beta: return beta
            alpha = max(alpha, q_eval)
        else:
            if q_eval <= alpha: return alpha
            beta = min(beta, q_eval)

    return alpha if maximizing else beta


def minimax_sse(game, depth, alpha, beta, maximizing, original_depth=None,
                start_time=None, time_limit=None, principal_variation=None):
    if original_depth is None: original_depth = depth
    move_cache = {}
    ply = original_depth - depth
    best_move = None
    score = -float('inf') if maximizing else float('inf')

    if time_limit and time.time() - start_time > time_limit: return None

    alpha_orig = alpha
    tt_key = calculate_zobrist_hash(game)
    tt_entry = TT.get(tt_key)
    tt_best_move = None
    if tt_entry:
        tt_best_move = tt_entry.get('best_move')
        if tt_entry['depth'] >= depth:
            tt_score = tt_entry['score']
            tt_flag = tt_entry['flag']
            # Basic Mate Score Adjustment (Commented out for now, can be complex)
            # if abs(tt_score) > 900000: tt_score += (ply - (original_depth - tt_entry['depth'])) * (1 if tt_score > 0 else -1)
            if tt_flag == TT_EXACT: return tt_best_move if depth == original_depth else tt_score
            elif tt_flag == TT_ALPHA: alpha = max(alpha, tt_score)
            elif tt_flag == TT_BETA: beta = min(beta, tt_score)
            if alpha >= beta: return tt_best_move if depth == original_depth else tt_score

    if game.state in ["Draw (Threefold repetition)", "Draw (50-move rule)"]: return 0

    is_in_check = game.is_check(game.turn)
    if is_in_check: depth += 1

    # Base Case Check (Moved before Null Move)
    if depth <= 0 or game.state in ["Checkmate", "Stalemate"]:
         q_move_cache = {}
         for r in range(8):
             for c in range(8):
                 p = game.board[r][c]
                 if p and p.colour == game.turn:
                      try: q_move_cache[(r, c)] = game.get_moves(r, c)
                      except: pass
         q_score = quiescence_search(game, alpha, beta, maximizing, q_move_cache)
         # Adjust mate score by ply (Commented out)
         # if abs(q_score) > 900000: q_score += ply * (1 if q_score > 0 else -1)
         return q_score # Return score directly


    R = 3
    has_major_pieces = any(p and p.name in ['R', 'Q', 'B', 'N'] for row in game.board for p in row if p and p.colour == game.turn)
    if depth >= 3 and not is_in_check and has_major_pieces:
        copy_null = game.light_copy()
        copy_null.turn = 'b' if copy_null.turn == 'w' else 'w'
        copy_null.enpassant = None
        null_eval = minimax_sse(copy_null, depth - 1 - R, -beta, -alpha, not maximizing, original_depth, start_time, time_limit)
        if null_eval is None: return None
        if -null_eval >= beta:
             # Basic Null Move Verification (Optional)
             # ... (verification logic can be added here) ...
             return tt_best_move if depth == original_depth else beta

    all_moves = []
    current_colour = game.turn

    # >>> Re-add calculate_priority function definition <<<
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
            if ply < len(KILLER_MOVES) and move_tuple in KILLER_MOVES[ply]:
                 priority = 8000 + (1 - KILLER_MOVES[ply].index(move_tuple)) * 250
            else:
                priority = HISTORY_MOVES[start_idx][end_idx]
        return priority
    # >>> End re-add <<<


    for r in range(8):
        for c in range(8):
            piece = game.board[r][c]
            if piece and piece.colour == current_colour:
                moves = game.get_moves(r, c)
                move_cache[(r, c)] = moves
                for move in moves:
                    promotion_options = ['Q', 'R', 'B', 'N'] if piece.name == 'P' and (move[0] == 0 or move[0] == 7) else [None]
                    for p in promotion_options:
                        priority = calculate_priority((r, c), move, p, tt_best_move, principal_variation)
                        all_moves.append((priority, (r, c), move, p))
    all_moves.sort(key=lambda x: x[0], reverse=True)

    if not all_moves:
        if is_in_check: return -999999 + ply if maximizing else 999999 - ply
        else: return 0

    move_index = 0
    static_eval = evaluate_board(game, move_cache)

    for priority, start, end, promotion in all_moves:
        is_capture = game.board[end[0]][end[1]] is not None

        # Futility Pruning
        if not is_in_check and not is_capture and depth <= 3 and abs(alpha) < 900000 and abs(beta) < 900000:
             futility_margin_val = FUTILITY_MARGIN * depth
             if maximizing and static_eval + futility_margin_val <= alpha: continue
             if not maximizing and static_eval - futility_margin_val >= beta: continue

        # SEE Pruning
        if is_capture:
            see_threshold = -50
            net_gain = static_exchange_eval_local(game, start, end)
            if net_gain < see_threshold: continue

        copy = game.light_copy()
        if not copy.make_move(start, end, promotion): continue

        # Dynamic LMR
        reduction = 0
        if depth >= 3 and move_index >= 3 and not is_capture and not is_in_check:
            try:
                reduction = int(math.log(depth) * math.log(move_index) / 2.5)
                if priority < 1000: reduction += 1
                reduction = max(0, min(reduction, depth - 2))
            except ValueError: reduction = 0

        # PVS
        eval_score = None
        search_depth = depth - 1 - reduction
        current_alpha = alpha
        current_beta = beta

        if move_index == 0:
            eval_score = minimax_sse(copy, depth - 1, alpha, beta, not maximizing, original_depth, start_time, time_limit, principal_variation)
        else:
            if maximizing:
                 eval_score = minimax_sse(copy, search_depth, alpha, alpha + 1, not maximizing, original_depth, start_time, time_limit, principal_variation)
            else:
                 eval_score = minimax_sse(copy, search_depth, beta - 1, beta, not maximizing, original_depth, start_time, time_limit, principal_variation)

            if eval_score is not None:
                 needs_research = False
                 if maximizing and eval_score > alpha: needs_research = True
                 elif not maximizing and eval_score < beta: needs_research = True

                 if needs_research and (eval_score < beta if maximizing else eval_score > alpha):
                      eval_score = minimax_sse(copy, depth - 1, current_alpha, current_beta, not maximizing, original_depth, start_time, time_limit, principal_variation)

        move_index += 1
        if eval_score is None: return None # Timeout

        # Repetition Penalty
        rep_count = copy.position_count.get(copy.board_key(), 0)
        if rep_count == 2: eval_score += -10 if maximizing else 10

        # Update best score and alpha/beta
        if maximizing:
            if eval_score > score: score = eval_score; best_move = (start, end, promotion)
            alpha = max(alpha, score)
            if alpha >= beta:
                score = beta # Use beta score for cutoff
                if not is_capture and ply < 64:
                    if best_move and best_move != KILLER_MOVES[ply][0]: # Check if best_move is valid
                        KILLER_MOVES[ply][1] = KILLER_MOVES[ply][0]; KILLER_MOVES[ply][0] = best_move
                    start_idx=start[0]*8+start[1]; end_idx=end[0]*8+end[1]
                    HISTORY_MOVES[start_idx][end_idx] = min(HISTORY_MOVES[start_idx][end_idx] + depth*depth, 32000)
                break
        else: # Minimizing
            if eval_score < score: score = eval_score; best_move = (start, end, promotion)
            beta = min(beta, score)
            if beta <= alpha:
                score = alpha # Use alpha score for cutoff
                if not is_capture and ply < 64:
                     if best_move and best_move != KILLER_MOVES[ply][0]:
                         KILLER_MOVES[ply][1] = KILLER_MOVES[ply][0]; KILLER_MOVES[ply][0] = best_move
                     start_idx=start[0]*8+start[1]; end_idx=end[0]*8+end[1]
                     HISTORY_MOVES[start_idx][end_idx] = min(HISTORY_MOVES[start_idx][end_idx] + depth*depth, 32000)
                break
    # End of move loop

    # --- Store in Transposition Table ---
    tt_flag = TT_EXACT
    if score <= alpha_orig: tt_flag = TT_ALPHA
    elif score >= beta: tt_flag = TT_BETA

    if len(TT) >= TT_MAX_SIZE:
        keys_to_delete = list(TT.keys())[:int(TT_MAX_SIZE * 0.1)]
        for key in keys_to_delete: del TT[key]

    # Store if score is valid (not initial worst value) and depth > 0
    if depth > 0 and ((maximizing and score > -float('inf')) or (not maximizing and score < float('inf'))):
         # Adjust mate scores relative to root before storing (Commented out)
         # store_score = score
         # if abs(score) > 900000: store_score += ply * (-1 if score > 0 else 1)
         TT[tt_key] = {'depth': depth, 'score': score, 'flag': tt_flag, 'best_move': best_move}


    # Adjust mate score for return value (Commented out)
    # if abs(score) > 900000: score += ply * (1 if score > 0 else -1)

    return best_move if depth == original_depth else score