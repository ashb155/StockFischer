import time
import random
import math
import chess
import numpy as np
from pieces import Piece
import os

# --- FUTILITY PRUNING MARGIN ---
FUTILITY_MARGIN = 100

# --- HEURISTIC TABLES ---
KILLER_MOVES = [[None, None] for _ in range(64)]
HISTORY_MOVES = [[0 for _ in range(64)] for _ in range(64)]

# --- ZOBRIST HASHING SETUP ---
ZOBRIST_SIZE = 64 * 12 + 1 + 4 + 8
random.seed(42)
ZOBRIST_KEYS = [random.randint(1, 2 ** 64 - 1) for _ in range(ZOBRIST_SIZE)]

PIECE_TO_INDEX = {
    ('w', 'P'): 0, ('w', 'N'): 1, ('w', 'B'): 2, ('w', 'R'): 3, ('w', 'Q'): 4, ('w', 'K'): 5,
    ('b', 'P'): 6, ('b', 'N'): 7, ('b', 'B'): 8, ('b', 'R'): 9, ('b', 'Q'): 10, ('b', 'K'): 11,
}

# --- TT Constants ---
TT_EXACT = 0
TT_ALPHA = 1
TT_BETA = 2
TT = {}
TT_MAX_SIZE = 500000

# --- MATE SCORE CONSTANT ---
MATE_VALUE = 100000  # Large score indicating mate
MATE_SCORE_THRESHOLD = MATE_VALUE - 200  # Buffer for detecting mate scores

# --- PIECE VALUES AND TABLES (Default / Tunable Parameters) ---
piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}
PHASE_MATERIAL = {'Q': 4, 'R': 2, 'B': 1, 'N': 1}
_piece_counts = {'Q': 2, 'R': 4, 'B': 4, 'N': 4, 'P': 16}
MAX_PHASE_MATERIAL = sum(PHASE_MATERIAL.get(p, 0) * _piece_counts.get(p, 0) for p in PHASE_MATERIAL)

# PSTs (Lists for modification)
pawn_table = list([
    0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10, 5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0
])
knight_table = list([
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
])
bishop_table = list([
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20,
])
rook_table = list([
    0, 0, 0, 5, 5, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0
])
queen_table = list([
    -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10, -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5, -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20
])
king_table = list([  # Middlegame
    20, 30, 10, 0, 0, 10, 30, 20, 20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10, -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30
])
king_endgame_table = list([  # Endgame
    -50, -30, -30, -30, -30, -30, -30, -50, -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -20, -10, 0, 0, -10, -20, -30, -50, -30, -30, -30, -30, -30, -30, -50
])

# --- Scalar Bonuses (Tunable Parameters) ---
MOBILITY_BONUS = 2.0
CENTER_CONTROL_BONUS = 15
TEMPO_BONUS = 10
KNIGHT_OUTPOST_BONUS = 20
BISHOP_FIANCHETTO_BONUS = 15
ROOK_ON_SEVENTH_BONUS = 25
ROOK_OPEN_FILE_BONUS = 25
ROOK_SEMI_OPEN_FILE_BONUS = 10
DOUBLED_PAWN_PENALTY = -20
ISOLATED_PAWN_PENALTY = -15
PAWN_CHAIN_BONUS = 5
BACKWARD_PAWN_PENALTY = -10
PASSED_PAWN_BONUS_RANKS = list([0, 10, 30, 60, 100, 150, 200])
PASSED_PAWN_PATH_CLEAR_FACTOR = 15.0
PASSED_PAWN_CONNECTED_BONUS = 20
PASSED_PAWN_BLOCKER_N_PENALTY = -40
PASSED_PAWN_BLOCKER_B_PENALTY = -30
PASSED_PAWN_BLOCKER_OTHER_PENALTY = -15
ROOK_BEHIND_PASSED_PAWN_BONUS = 30
ENEMY_ROOK_IN_FRONT_PASSED_PAWN_PENALTY = -25
BAD_BISHOP_FACTOR = -5.0
BISHOP_PAIR_BONUS = 50
KING_SHIELD_RANK1_BONUS = 15
KING_SHIELD_RANK2_BONUS = 10
KING_STORM_FACTOR = -5.0
KING_SEMI_OPEN_PENALTY = -10
KING_OPEN_PENALTY = -20
KING_ATTACK_FACTOR = -2.0
CONNECTED_ROOKS_BONUS = 20
UNDEVELOPED_MINOR_PENALTY = -15
UNDEVELOPED_ROOK_PENALTY = -10
KING_NOT_CASTLED_PENALTY = -25
KING_STUCK_CENTER_PENALTY = -40
PIGS_ON_SEVENTH_BONUS = 75
BATTERY_BONUS = 20
SPACE_BONUS_FACTOR = 2.0
INITIATIVE_FACTOR = 3.0
INITIATIVE_FLAT_BONUS = 20
CASTLING_BONUS = 30  # <<< --- ADDED CASTLING BONUS --- <<<
KNIGHT_SYNERGY_BONUS = 25  # <<< --- ADDED KNIGHT SYNERGY BONUS --- <<<

# King attack weights (Tunable)
king_attack_weights = {'Q': 5, 'R': 3, 'B': 2, 'N': 2, 'P': 1}
# Global array for king attack counts (reset internally)
king_zone_attack_count = [0] * 2

# --- Parameter Metadata (For Loading Weights) ---
_param_order = []
_param_refs = {}


def _register_param_meta(name, value, original_ref=None, index=None):
    global _param_order, _param_refs
    if name not in _param_order:
        _param_order.append(name)
        if original_ref is not None:
            _param_refs[name] = (original_ref, index)


def _build_parameter_metadata():
    _param_order.clear();
    _param_refs.clear();
    current_globals = globals()
    tables = {"pawn": pawn_table, "knight": knight_table, "bishop": bishop_table, "rook": rook_table,
              "queen": queen_table, "king_mg": king_table, "king_eg": king_endgame_table}
    for name, table_ref in tables.items():
        if isinstance(table_ref, list) and len(table_ref) == 64:
            for i in range(64): _register_param_meta(f"{name}_pst_{i}", table_ref[i], table_ref, i)
    if isinstance(piece_values, dict):
        for piece_name in ['P', 'N', 'B', 'R', 'Q']:
            if piece_name in piece_values: _register_param_meta(f"value_{piece_name}", piece_values[piece_name],
                                                                piece_values, piece_name)

    # *** ADDED KNIGHT_SYNERGY_BONUS and CASTLING_BONUS TO LIST ***
    scalar_constants = [
        "MOBILITY_BONUS", "CENTER_CONTROL_BONUS", "TEMPO_BONUS", "KNIGHT_OUTPOST_BONUS",
        "BISHOP_FIANCHETTO_BONUS", "ROOK_ON_SEVENTH_BONUS", "ROOK_OPEN_FILE_BONUS",
        "ROOK_SEMI_OPEN_FILE_BONUS", "DOUBLED_PAWN_PENALTY", "ISOLATED_PAWN_PENALTY",
        "PAWN_CHAIN_BONUS", "BACKWARD_PAWN_PENALTY", "PASSED_PAWN_PATH_CLEAR_FACTOR",
        "PASSED_PAWN_CONNECTED_BONUS", "PASSED_PAWN_BLOCKER_N_PENALTY",
        "PASSED_PAWN_BLOCKER_B_PENALTY", "PASSED_PAWN_BLOCKER_OTHER_PENALTY",
        "ROOK_BEHIND_PASSED_PAWN_BONUS", "ENEMY_ROOK_IN_FRONT_PASSED_PAWN_PENALTY",
        "BAD_BISHOP_FACTOR", "BISHOP_PAIR_BONUS", "KING_SHIELD_RANK1_BONUS",
        "KING_SHIELD_RANK2_BONUS", "KING_STORM_FACTOR", "KING_SEMI_OPEN_PENALTY",
        "KING_OPEN_PENALTY", "KING_ATTACK_FACTOR", "CONNECTED_ROOKS_BONUS",
        "UNDEVELOPED_MINOR_PENALTY", "UNDEVELOPED_ROOK_PENALTY",
        "KING_NOT_CASTLED_PENALTY", "KING_STUCK_CENTER_PENALTY",
        "PIGS_ON_SEVENTH_BONUS", "BATTERY_BONUS", "SPACE_BONUS_FACTOR",
        "INITIATIVE_FACTOR", "INITIATIVE_FLAT_BONUS", "CASTLING_BONUS",
        "KNIGHT_SYNERGY_BONUS"  # <-- ADDED
    ]
    for const_name in scalar_constants:
        if const_name in current_globals: _register_param_meta(const_name, current_globals[const_name])

    if 'PASSED_PAWN_BONUS_RANKS' in current_globals and isinstance(PASSED_PAWN_BONUS_RANKS, list) and len(
            PASSED_PAWN_BONUS_RANKS) == 7:
        for i in range(7): _register_param_meta(f"PASSED_PAWN_BONUS_RANK_{i}", PASSED_PAWN_BONUS_RANKS[i],
                                                PASSED_PAWN_BONUS_RANKS, i)
    if 'king_attack_weights' in current_globals and isinstance(king_attack_weights, dict):
        for piece_name, value in king_attack_weights.items(): _register_param_meta(f"king_attack_{piece_name}", value,
                                                                                   king_attack_weights, piece_name)


# --- Function to Apply Weights ---
def _apply_weights(theta):
    if len(theta) != len(_param_order): return False
    current_globals = globals()
    for i, name in enumerate(_param_order):
        value = theta[i]
        if name in _param_refs:
            original_ref, index = _param_refs[name]
            try:
                if isinstance(original_ref, list) and 0 <= index < len(original_ref):
                    original_ref[index] = int(round(value))
                elif isinstance(original_ref, dict) and index in original_ref:
                    original_ref[index] = int(round(value))
            except Exception:
                pass
        elif name in current_globals:
            is_float = name in ["MOBILITY_BONUS", "PASSED_PAWN_PATH_CLEAR_FACTOR", "SPACE_BONUS_FACTOR",
                                "INITIATIVE_FACTOR", "KING_ATTACK_FACTOR", "BAD_BISHOP_FACTOR", "KING_STORM_FACTOR"]
            current_globals[name] = float(value) if is_float else int(round(value))
    return True


# --- Load Optimized Weights at Startup ---
OPTIMIZED_WEIGHTS_FILE = 'optimized_weights.npy'
try:
    _build_parameter_metadata()
    if os.path.exists(OPTIMIZED_WEIGHTS_FILE):
        optimized_theta = np.load(OPTIMIZED_WEIGHTS_FILE)
        print(f"INFO: Loading optimized weights from '{OPTIMIZED_WEIGHTS_FILE}'...")
        if _apply_weights(optimized_theta):
            print(f"INFO: Successfully applied {len(optimized_theta)} optimized weights.")
        else:
            print("WARNING: Failed to apply optimized weights. Using defaults.")
    else:
        print(f"INFO: '{OPTIMIZED_WEIGHTS_FILE}' not found. Using default weights.")
except Exception as e:
    print(f"ERROR: Could not load or apply weights: {type(e).__name__} {e}. Using defaults.")
# Ensure metadata is built even if loading fails
if not _param_order:
    _build_parameter_metadata()


# --- Helper Function: Convert python-chess board to internal representation ---
def board_to_internal_representation(board):
    internal_board = [[None for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            square = chess.square(c, 7 - r)
            piece = board.piece_at(square)
            if piece:
                color = 'w' if piece.color == chess.WHITE else 'b'
                name = piece.symbol().upper()
                internal_board[r][c] = Piece(color, name)
    return internal_board


# --- MODIFIED evaluate_board ---
def evaluate_board(internal_board, turn, castling_rights, enpassant_square,
                   move_count=15):
    score = 0;
    white_pawns, black_pawns = [], [];
    white_bishops, black_bishops = [], []
    white_king_pos, black_king_pos = None, None

    # --- Pre-evaluation setup (no change) ---
    for r_k in range(8):
        for c_k in range(8):
            p_k = internal_board[r_k][c_k]
            if p_k and p_k.name == 'K':
                if p_k.colour == 'w':
                    white_king_pos = (r_k, c_k)
                else:
                    black_king_pos = (r_k, c_k)

    current_material = sum(PHASE_MATERIAL.get(p.name, 0) for row in internal_board for p in row if p)
    phase = max(0.0, min(1.0, 1.0 - (current_material / MAX_PHASE_MATERIAL))) if MAX_PHASE_MATERIAL > 0 else 0.5

    white_pawn_attacks, black_pawn_attacks = set(), set()
    for r in range(8):
        for c in range(8):
            p = internal_board[r][c];
            if p and p.name == 'P':
                dr, attack_set = (-1, white_pawn_attacks) if p.colour == 'w' else (1, black_pawn_attacks)
                attack_r = r + dr
                if 0 <= attack_r < 8:
                    if c > 0: attack_set.add((attack_r, c - 1))
                    if c < 7: attack_set.add((attack_r, c + 1))

    global king_zone_attack_count;
    king_zone_attack_count = [0] * 2
    white_queen_pos, black_queen_pos = None, None
    white_rooks_bishops, black_rooks_bishops = [], []
    white_knights, black_knights = [], []  # <<< Added to collect Knight positions

    # --- Main Evaluation Loop ---
    for row in range(8):
        for col in range(8):
            piece = internal_board[row][col];
            if not piece: continue
            val = piece_values.get(piece.name, 0)
            index = (row * 8 + col, (7 - row) * 8 + col)[piece.colour == 'b']
            if not (0 <= index < 64): continue
            try:
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
                    val += int((1 - phase) * king_table[index] + phase * king_endgame_table[index])
            except IndexError:
                pass

            if piece.name == 'N':
                # --- Original Knight Outpost Logic ---
                is_s, is_a = False, False;
                s_r, a_r = (row + (-1, 1)[piece.colour == 'b'], row + (1, -1)[piece.colour == 'b'])
                if 0 <= s_r < 8:
                    for dc in (-1, 1):
                        sc = col + dc;
                        if 0 <= sc < 8 and (
                                p := internal_board[s_r][
                                    sc]) and p.name == 'P' and p.colour == piece.colour: is_s = True; break
                if 0 <= a_r < 8:
                    for dc in (-1, 1):
                        ac = col + dc;
                        if 0 <= ac < 8 and (
                                p := internal_board[a_r][
                                    ac]) and p.name == 'P' and p.colour != piece.colour: is_a = True; break
                if is_s and not is_a: val += KNIGHT_OUTPOST_BONUS

                # --- Store Knight position for synergy check later ---
                (white_knights, black_knights)[piece.colour == 'b'].append((row, col))

            elif piece.name == 'B':
                (white_bishops, black_bishops)[piece.colour == 'b'].append((row, col))
                (white_rooks_bishops, black_rooks_bishops)[piece.colour == 'b'].append((piece.name, (row, col)))
                if (piece.colour == 'w' and (row, col) in ((6, 1), (6, 6))) or (
                        piece.colour == 'b' and (row, col) in ((1, 1), (1, 6))): val += BISHOP_FIANCHETTO_BONUS
            elif piece.name == 'R':
                (white_rooks_bishops, black_rooks_bishops)[piece.colour == 'b'].append((piece.name, (row, col)))
                if (piece.colour == 'w' and row == 1) or (
                        piece.colour == 'b' and row == 6): val += ROOK_ON_SEVENTH_BONUS
                ffp = any(
                    (p := internal_board[r][col]) and p.name == 'P' and p.colour == piece.colour for r in range(8))
                if not ffp:
                    fep = any(
                        (p := internal_board[r][col]) and p.name == 'P' and p.colour != piece.colour for r in range(8))
                    val += (ROOK_SEMI_OPEN_FILE_BONUS, ROOK_OPEN_FILE_BONUS)[not fep]
            elif piece.name == 'Q':
                if piece.colour == 'w':
                    white_queen_pos = (row, col)
                else:
                    black_queen_pos = (row, col)
            if piece.name == 'P': (white_pawns, black_pawns)[piece.colour == 'b'].append((row, col))

            score += val * (1, -1)[piece.colour == 'b']

            if piece.name != 'K':
                ekp = (black_king_pos, white_king_pos)[piece.colour == 'b']
                if ekp:
                    dist = max(abs(row - ekp[0]), abs(col - ekp[1]))
                    if dist <= 3:
                        aw = king_attack_weights.get(piece.name, 0)
                        ab = aw * (4 - dist) * (1 + phase / 2)
                        score += int(ab * (1, -1)[piece.colour == 'b'])
                        ki = (1, 0)[piece.colour == 'b'];
                        king_zone_attack_count[ki] += aw

    # --- Pawn Structure (no change) ---
    def pawn_structure(pawns, colour):
        bonus = 0;
        files = [0] * 8
        try:
            for r, c in pawns:
                if 0 <= c < 8: files[c] += 1
        except (TypeError, ValueError):
            return 0
        for f in range(8):
            if files[f] > 1: bonus += DOUBLED_PAWN_PENALTY * (files[f] - 1)
            ln = files[f - 1] if f > 0 else 0;
            rn = files[f + 1] if f < 7 else 0
            if files[f] > 0 and ln == 0 and rn == 0: bonus += ISOLATED_PAWN_PENALTY
        for r, c in pawns:
            s_dir, s_r = (1, -1)[colour == 'b'], r + (1, -1)[colour == 'b']
            if 0 <= s_r < 8:
                if c > 0 and (
                        p := internal_board[s_r][
                            c - 1]) and p.name == 'P' and p.colour == colour: bonus += PAWN_CHAIN_BONUS
                if c < 7 and (
                        p := internal_board[s_r][
                            c + 1]) and p.name == 'P' and p.colour == colour: bonus += PAWN_CHAIN_BONUS
            b_dir, is_bwd = s_dir, True
            for dc in (-1, 1):
                if not 0 <= c + dc < 8: continue
                cr, fs = r + b_dir, False
                while 0 <= cr < 8:
                    if (p := internal_board[cr][c + dc]) and p.name == 'P' and p.colour == colour: fs = True; break
                    cr += b_dir
                if fs: is_bwd = False; break
            if is_bwd:
                stop = False;
                ar = r - b_dir
                if 0 <= ar < 8:
                    for da in (-1, 1):
                        if 0 <= c + da < 8 and (
                                p := internal_board[ar][
                                    c + da]) and p.name == 'P' and p.colour != colour: stop = True; break
                if stop: bonus += BACKWARD_PAWN_PENALTY
            is_pass, p_dir, curr_r = True, (-1, 1)[colour == 'b'], r + (-1, 1)[colour == 'b']
            while 0 <= curr_r < 8:
                for dp in (-1, 0, 1):
                    if 0 <= c + dp < 8 and (p := internal_board[curr_r][
                        c + dp]) and p.name == 'P' and p.colour != colour: is_pass = False; break
                if not is_pass: break
                curr_r += p_dir
            if is_pass:
                rank = (6 - r, r - 1)[colour == 'b']
                if 0 <= rank < len(PASSED_PAWN_BONUS_RANKS): bonus += PASSED_PAWN_BONUS_RANKS[rank]
                clear = all(internal_board[cr][c] is None for cr in range(r + p_dir, (-1, 8)[colour == 'b'], p_dir))
                if clear: bonus += rank * PASSED_PAWN_PATH_CLEAR_FACTOR
                s_r = r + b_dir;
                conn = False
                if 0 <= s_r < 8:
                    for dc in (-1, 1):
                        if 0 <= c + dc < 8 and (
                                p := internal_board[s_r][
                                    c + dc]) and p.name == 'P' and p.colour == colour: conn = True; break
                if conn: bonus += PASSED_PAWN_CONNECTED_BONUS
                bl_r = r + p_dir
                if 0 <= bl_r < 8 and (b := internal_board[bl_r][c]) and b.colour != colour:
                    bonus += {'N': PASSED_PAWN_BLOCKER_N_PENALTY, 'B': PASSED_PAWN_BLOCKER_B_PENALTY}.get(b.name,
                                                                                                          PASSED_PAWN_BLOCKER_OTHER_PENALTY)
                r_r = r + b_dir
                while 0 <= r_r < 8:
                    if p := internal_board[r_r][c]:
                        if p.name == 'R' and p.colour == colour: bonus += ROOK_BEHIND_PASSED_PAWN_BONUS
                        break
                    r_r += b_dir
                rf_r = r + p_dir
                while 0 <= rf_r < 8:
                    if p := internal_board[rf_r][c]:
                        if p.name == 'R' and p.colour != colour: bonus += ENEMY_ROOK_IN_FRONT_PASSED_PAWN_PENALTY
                        break
                    rf_r += p_dir
        return bonus

    score += pawn_structure(white_pawns, 'w')
    score -= pawn_structure(black_pawns, 'b')

    # --- Bad Bishop ---
    wp_lc, wp_dc, bp_lc, bp_dc = 0, 0, 0, 0;
    cf = {2, 3, 4, 5}
    for r, c in white_pawns:
        if c in cf:
            if (r + c) % 2 == 0:
                wp_lc += 1
            else:
                wp_dc += 1
    for r, c in black_pawns:
        if c in cf:
            if (r + c) % 2 == 0:
                bp_lc += 1
            else:
                bp_dc += 1
    bbp = 0
    for r, c in white_bishops:
        idx = (r + c) % 2
        if idx in (0, 1): bbp += (wp_lc, wp_dc)[idx] * BAD_BISHOP_FACTOR
    for r, c in black_bishops:
        idx = (r + c) % 2
        if idx in (0, 1): bbp -= (bp_lc, bp_dc)[idx] * BAD_BISHOP_FACTOR
    score += bbp

    # --- Bishop Pair (no change) ---
    if len(white_bishops) >= 2: score += BISHOP_PAIR_BONUS
    if len(black_bishops) >= 2: score -= BISHOP_PAIR_BONUS

    # --- NEW: Knight Synergy/Mutual Support ---
    def check_knight_synergy(knights):
        bonus = 0
        if len(knights) < 2: return 0

        # Knight Deltas for checking attack
        knight_deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

        for i in range(len(knights)):
            for j in range(i + 1, len(knights)):
                r1, c1 = knights[i]
                r2, c2 = knights[j]

                # Check if N1 attacks N2 (since they are the same color, this is mutual defense)
                dr, dc = abs(r1 - r2), abs(c1 - c2)
                is_mutual_support = (dr == 1 and dc == 2) or (dr == 2 and dc == 1)

                # Check for central influence (r=2 to 5, c=2 to 5 -> squares c3/f6 to f3/c6)
                is_central = r1 in range(2, 6) and c1 in range(2, 6) and \
                             r2 in range(2, 6) and c2 in range(2, 6)

                if is_mutual_support and is_central:
                    bonus += KNIGHT_SYNERGY_BONUS

        return bonus

    score += check_knight_synergy(white_knights)
    score -= check_knight_synergy(black_knights)

    # --- King Safety (no change) ---
    def king_safety(colour, king_pos, num_attackers):
        if king_pos is None: return 0
        kr, kc = king_pos;
        sb, sp = 0, 0;
        sd = (-1, 1)[colour == 'b']
        for dc in (-1, 0, 1):
            cc = kc + dc;
            if not 0 <= cc < 8: continue
            fpp, epp = False, False;
            r1 = kr + sd
            if 0 <= r1 < 8 and (p := internal_board[r1][
                cc]) and p.name == 'P' and p.colour == colour: sb += KING_SHIELD_RANK1_BONUS; fpp = True
            r2 = kr + 2 * sd
            if 0 <= r2 < 8 and (p := internal_board[r2][
                cc]) and p.name == 'P' and p.colour == colour: sb += KING_SHIELD_RANK2_BONUS; fpp = True
            for r_chk in range(8):
                if (p := internal_board[r_chk][cc]) and p.name == 'P' and p.colour != colour:
                    epp = True;
                    rank = (6 - r_chk, r_chk - 1)[p.colour == 'b']
                    if rank >= 3: sp += rank * KING_STORM_FACTOR
            if not fpp: sp += (KING_SEMI_OPEN_PENALTY, KING_OPEN_PENALTY)[not epp]
        adp = num_attackers ** 2 * KING_ATTACK_FACTOR
        return sb + min(0, sp) + min(0, adp)

    if white_king_pos: score += int(king_safety('w', white_king_pos, king_zone_attack_count[0]) * (1 - phase))
    if black_king_pos: score -= int(king_safety('b', black_king_pos, king_zone_attack_count[1]) * (1 - phase))

    # --- Connected Rooks (no change) ---
    wr = [(r, c) for r in range(8) for c in range(8) if
          (p := internal_board[r][c]) and p.name == 'R' and p.colour == 'w']
    br = [(r, c) for r in range(8) for c in range(8) if
          (p := internal_board[r][c]) and p.name == 'R' and p.colour == 'b']
    if len(wr) >= 2:
        r1, c1 = wr[0];
        r2, c2 = wr[1]
        if (r1 == r2 and all(internal_board[r1][c] is None for c in range(min(c1, c2) + 1, max(c1, c2)))) or \
                (c1 == c2 and all(internal_board[r][c1] is None for r in
                                  range(min(r1, r2) + 1, max(r1, r2)))): score += CONNECTED_ROOKS_BONUS
    if len(br) >= 2:
        r1, c1 = br[0];
        r2, c2 = br[1]
        if (r1 == r2 and all(internal_board[r1][c] is None for c in range(min(c1, c2) + 1, max(c1, c2)))) or \
                (c1 == c2 and all(internal_board[r][c1] is None for r in
                                  range(min(r1, r2) + 1, max(r1, r2)))): score -= CONNECTED_ROOKS_BONUS

    # --- Undeveloped Pieces & Castling Bonus (no change) ---
    if move_count > 4:
        up = 0
        # White
        if internal_board[7][1] and internal_board[7][1].name == 'N': up += UNDEVELOPED_MINOR_PENALTY
        if internal_board[7][2] and internal_board[7][2].name == 'B': up += UNDEVELOPED_MINOR_PENALTY
        if internal_board[7][5] and internal_board[7][5].name == 'B': up += UNDEVELOPED_MINOR_PENALTY
        if internal_board[7][6] and internal_board[7][6].name == 'N': up += UNDEVELOPED_MINOR_PENALTY

        # Check castling status
        white_castled = False
        if castling_rights.get('wKR') or castling_rights.get('wQR'):
            # Still has rights, penalize
            if internal_board[7][7] and internal_board[7][7].name == 'R': up += UNDEVELOPED_ROOK_PENALTY
            if internal_board[7][0] and internal_board[7][0].name == 'R': up += UNDEVELOPED_ROOK_PENALTY
            up += KING_NOT_CASTLED_PENALTY
        elif white_king_pos in [(7, 6), (7, 2)]:  # Rights are gone AND king is on castled square
            white_castled = True
            score += CASTLING_BONUS  # *** ADDED BONUS ***

        # Black
        if internal_board[0][1] and internal_board[0][1].name == 'N': up -= UNDEVELOPED_MINOR_PENALTY
        if internal_board[0][2] and internal_board[0][2].name == 'B': up -= UNDEVELOPED_MINOR_PENALTY
        if internal_board[0][5] and internal_board[0][5].name == 'B': up -= UNDEVELOPED_MINOR_PENALTY
        if internal_board[0][6] and internal_board[0][6].name == 'N': up -= UNDEVELOPED_MINOR_PENALTY

        black_castled = False
        if castling_rights.get('bKR') or castling_rights.get('bQR'):
            if internal_board[0][7] and internal_board[0][7].name == 'R': up -= UNDEVELOPED_ROOK_PENALTY
            if internal_board[0][0] and internal_board[0][0].name == 'R': up -= UNDEVELOPED_ROOK_PENALTY
            up -= KING_NOT_CASTLED_PENALTY
        elif black_king_pos in [(0, 6), (0, 2)]:  # Rights are gone AND king is on castled square
            black_castled = True
            score -= CASTLING_BONUS  # *** ADDED BONUS (negative for black) ***

        score += up

        # --- King Stuck in Center Penalty (Modified) ---
        # Only apply if *not* castled and still has rights
        if move_count > 10:
            if white_king_pos == (7, 4) and not white_castled and (
                    castling_rights.get('wKR') or castling_rights.get('wQR')):
                score += int(KING_STUCK_CENTER_PENALTY * (1 - phase))
            if black_king_pos == (0, 4) and not black_castled and (
                    castling_rights.get('bKR') or castling_rights.get('bQR')):
                score -= int(KING_STUCK_CENTER_PENALTY * (1 - phase))

    # --- Central Control (no change) ---
    center = [(r, c) for r in range(2, 6) for c in range(2, 6)]
    for r, c in center:
        if p := internal_board[r][c]:
            bonus = (CENTER_CONTROL_BONUS, CENTER_CONTROL_BONUS // 2)[not (r in (3, 4) and c in (3, 4))]
            score += bonus * (1, -1)[p.colour == 'b']

    # --- Pigs on 7th (no change) ---
    if sum(1 for c in range(8) if
           (p := internal_board[1][c]) and p.name == 'R' and p.colour == 'w') >= 2: score += PIGS_ON_SEVENTH_BONUS
    if sum(1 for c in range(8) if
           (p := internal_board[6][c]) and p.name == 'R' and p.colour == 'b') >= 2: score -= PIGS_ON_SEVENTH_BONUS

    # --- Battery Bonus (no change) ---
    bb = 0
    if white_queen_pos:
        qr, qc = white_queen_pos
        for pn, (pr, pc) in white_rooks_bishops:
            is_bat = False
            if pn == 'R' and (qr == pr or qc == pc):
                lc = (qr == pr and all(internal_board[qr][c] is None for c in range(min(qc, pc) + 1, max(qc, pc)))) or \
                     (qc == pc and all(internal_board[r][qc] is None for r in range(min(qr, pr) + 1, max(qr, pr))))
                if lc: is_bat = True
            elif pn == 'B' and abs(qr - pr) == abs(qc - pc):
                dr, dc = (1, -1)[pr < qr], (1, -1)[pc < qc];
                rr, cc = qr + dr, qc + dc;
                path_clear = True
                while (rr, cc) != (pr, pc):
                    if not (0 <= rr < 8 and 0 <= cc < 8): path_clear = False; break
                    if internal_board[rr][cc]: path_clear = False; break
                    rr += dr;
                    cc += dc
                if path_clear: is_bat = True
            if is_bat: bb += BATTERY_BONUS
    if black_queen_pos:
        qr, qc = black_queen_pos
        for pn, (pr, pc) in black_rooks_bishops:
            is_bat = False
            if pn == 'R' and (qr == pr or qc == pc):
                lc = (qr == pr and all(internal_board[qr][c] is None for c in range(min(qc, pc) + 1, max(qc, pc)))) or \
                     (qc == pc and all(internal_board[r][qc] is None for r in range(min(qr, pr) + 1, max(qr, pr))))
                if lc: is_bat = True
            elif pn == 'B' and abs(qr - pr) == abs(qc - pc):
                dr, dc = (1, -1)[pr < qr], (1, -1)[pc < qc];
                rr, cc = qr + dr, qc + dc;
                path_clear = True
                while (rr, cc) != (pr, pc):
                    if not (0 <= rr < 8 and 0 <= cc < 8): path_clear = False; break
                    if internal_board[rr][cc]: path_clear = False; break
                    rr += dr;
                    cc += dc
                if path_clear: is_bat = True
            if is_bat: bb -= BATTERY_BONUS
    score += bb

    # --- Space Control (no change) ---
    sb = 0
    for r in (4, 5, 6):
        for c in (2, 3, 4, 5):
            if (r, c) not in black_pawn_attacks: sb += SPACE_BONUS_FACTOR
    for r in (1, 2, 3):
        for c in (2, 3, 4, 5):
            if (r, c) not in white_pawn_attacks: sb -= SPACE_BONUS_FACTOR
    score += sb

    # --- Initiative (no change) ---
    init_diff = king_zone_attack_count[0] - king_zone_attack_count[1]
    score += int(init_diff * INITIATIVE_FACTOR * (1 - phase))
    if init_diff > 0:
        score += int(INITIATIVE_FLAT_BONUS * (1 - phase))
    elif init_diff < 0:
        score -= int(INITIATIVE_FLAT_BONUS * (1 - phase))

    # --- Tempo (no change) ---
    score += TEMPO_BONUS * (1, -1)[turn == 'b']

    return int(score)


# --- NEW FUNCTION: evaluate_board_fen (no change) ---
def evaluate_board_fen(fen_str):
    """
    Parses a FEN string, converts to internal representation, and calls evaluate_board.
    """
    try:
        board = chess.Board(fen_str)
        if board.is_insufficient_material(): return 0

        internal_board = board_to_internal_representation(board)
        turn = ('w', 'b')[board.turn == chess.BLACK]
        castling = {
            'wKR': board.has_kingside_castling_rights(chess.WHITE),
            'wQR': board.has_queenside_castling_rights(chess.WHITE),
            'bKR': board.has_kingside_castling_rights(chess.BLACK),
            'bQR': board.has_queenside_castling_rights(chess.BLACK)
        }
        enp = None
        if board.ep_square:
            rank = chess.square_rank(board.ep_square)
            file = chess.square_file(board.ep_square)
            enp = (7 - rank, file)
        mc = board.fullmove_number

        score = evaluate_board(internal_board, turn, castling, enp, mc)
        return score  # Return score relative to White

    except (ValueError, AttributeError) as e:
        return 0


# --- Functions below still rely on the Game object (no change) ---

# (calculate_zobrist_hash function - complete)
def calculate_zobrist_hash(game):
    """Calculates Zobrist hash for a Game object state."""
    h = 0
    if not all(hasattr(game, attr) for attr in ['board', 'turn', 'castling', 'enpassant']): return 0

    board_repr, turn, castling, enpassant = game.board, game.turn, game.castling, game.enpassant

    for r in range(8):
        for c in range(8):
            piece = board_repr[r][c]
            if piece:
                index = r * 8 + c
                if (piece.colour, piece.name) in PIECE_TO_INDEX:
                    piece_index = PIECE_TO_INDEX[(piece.colour, piece.name)]
                    key_index = index * 12 + piece_index
                    if 0 <= key_index < ZOBRIST_SIZE: h ^= ZOBRIST_KEYS[key_index]

    offset = 64 * 12
    if 0 <= offset < ZOBRIST_SIZE:
        if turn == 'w': h ^= ZOBRIST_KEYS[offset]

    offset += 1
    if 0 <= offset + 3 < ZOBRIST_SIZE:
        if castling.get('wKR'): h ^= ZOBRIST_KEYS[offset + 0]
        if castling.get('wQR'): h ^= ZOBRIST_KEYS[offset + 1]
        if castling.get('bKR'): h ^= ZOBRIST_KEYS[offset + 2]
        if castling.get('bQR'): h ^= ZOBRIST_KEYS[offset + 3]

    offset += 4
    if enpassant and isinstance(enpassant, tuple) and len(enpassant) == 2 and \
            0 <= enpassant[0] < 8 and 0 <= enpassant[1] < 8:
        file_index = enpassant[1]
        key_index = offset + file_index
        if 0 <= key_index < ZOBRIST_SIZE: h ^= ZOBRIST_KEYS[key_index]

    return h


# (static_exchange_eval_local function - complete with checks)
def static_exchange_eval_local(game, start, end):
    """Performs SEE on a Game object."""
    if not (isinstance(start, tuple) and len(start) == 2 and 0 <= start[0] < 8 and 0 <= start[1] < 8 and
            isinstance(end, tuple) and len(end) == 2 and 0 <= end[0] < 8 and 0 <= end[1] < 8): return 0
    if not (hasattr(game, 'board') and isinstance(game.board, list) and len(game.board) == 8 and isinstance(
            game.board[0], list) and len(game.board[0]) == 8): return 0

    attacker_piece = game.board[start[0]][start[1]]
    target_piece = game.board[end[0]][end[1]]
    if not attacker_piece or not target_piece: return 0
    target_val = piece_values.get(target_piece.name, 0)
    if target_val == 0: return 0

    gains = [target_val]
    try:
        if not all(hasattr(game, m) for m in ['light_copy', '_force_move', 'attack_moves']): return 0
        copyg = game.light_copy()
        if not copyg._force_move(start, end): return 0
    except Exception:
        return 0

    side, target_square = copyg.turn, end

    for _ in range(32):  # Limit iterations
        best_attacker_val, best_attacker_pos, found_attacker = float('inf'), None, False
        try:
            if not hasattr(copyg, 'attack_moves'): return 0
            for r in range(8):
                for c in range(8):
                    piece = copyg.board[r][c]
                    if not piece or piece.colour != side: continue
                    attacked_squares = copyg.attack_moves(r, c)
                    if isinstance(attacked_squares, (list, set)) and target_square in attacked_squares:
                        current_attacker_val = piece_values.get(piece.name, 0)
                        if current_attacker_val < best_attacker_val:
                            best_attacker_val = current_attacker_val;
                            best_attacker_pos = (r, c);
                            found_attacker = True
            if not found_attacker: break
        except Exception:
            return 0

        captured_piece_obj = copyg.board[target_square[0]][target_square[1]]
        if not captured_piece_obj: break
        gains.append(piece_values.get(captured_piece_obj.name, 0))

        try:
            if not hasattr(copyg, '_force_move'): return 0
            if not copyg._force_move(best_attacker_pos, target_square): break
        except Exception:
            return 0

        side = ('b', 'w')[side == 'b']

    see_score, next_capture_sign = 0, -1
    for i in range(len(gains) - 1, 0, -1):
        see_score = max(0, gains[i] + next_capture_sign * see_score);
        next_capture_sign *= -1
    return gains[0] + next_capture_sign * see_score


# (quiescence_search function - complete with checks)
def quiescence_search(game, alpha, beta, maximizing, move_cache=None):
    """Performs quiescence search using the Game object."""
    try:
        if not all(hasattr(game, a) for a in
                   ['board', 'turn', 'castling', 'enpassant', 'move_count']): return alpha if maximizing else beta
        stand_pat = evaluate_board(game.board, game.turn, game.castling, game.enpassant, game.move_count)
    except Exception:
        return alpha if maximizing else beta

    if maximizing:
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha: return alpha
        beta = min(beta, stand_pat)

    forcing_moves, colour = [], game.turn
    try:
        if not all(hasattr(game, m) for m in
                   ['get_moves', 'light_copy', 'make_move', 'is_check', 'board']): return alpha if maximizing else beta
        for r in range(8):
            for c in range(8):
                piece = game.board[r][c]
                if not piece or piece.colour != colour: continue
                legal_moves_q = game.get_moves(r, c)
                for end_pos in legal_moves_q:
                    is_capture = game.board[end_pos[0]][end_pos[1]] is not None;
                    is_check = False
                    if not is_capture:
                        try:
                            temp_copy = game.light_copy()
                            if temp_copy.make_move((r, c), end_pos): is_check = temp_copy.is_check(temp_copy.turn)
                        except Exception:
                            pass
                    if is_capture or is_check:
                        promos = [None]
                        if piece.name == 'P' and (end_pos[0] == 0 or end_pos[0] == 7): promos = ['Q', 'R', 'B', 'N']
                        for p_promo in promos:
                            victim = game.board[end_pos[0]][end_pos[1]]
                            victim_val = piece_values.get(victim.name, 0) if victim else 0
                            attacker_val = piece_values.get(piece.name, 1)
                            priority = (10000 + victim_val * 10 - attacker_val) if is_capture else 5000
                            forcing_moves.append((priority, (r, c), end_pos, p_promo))
    except Exception:
        return alpha if maximizing else beta

    forcing_moves.sort(key=lambda x: x[0], reverse=True)

    for priority, start, end, promotion in forcing_moves:
        capture_val = 0;
        victim = game.board[end[0]][end[1]]
        if victim: capture_val = piece_values.get(victim.name, 0)
        promo_val = piece_values.get(promotion, 0) if promotion and promotion in piece_values else 0
        delta_margin = piece_values.get('P', 100)
        potential_score = stand_pat + capture_val + promo_val
        if (maximizing and potential_score + delta_margin < alpha) or \
                (not maximizing and potential_score - delta_margin > beta): continue

        is_capture_move = game.board[end[0]][end[1]] is not None
        if is_capture_move:
            see_score = static_exchange_eval_local(game, start, end)
            if (maximizing and see_score < 0) or (not maximizing and see_score > 0): continue

        try:
            if not all(hasattr(game, m) for m in ['light_copy', 'make_move']): continue
            copy = game.light_copy()
            if not copy.make_move(start, end, promotion): continue
            q_eval = quiescence_search(copy, alpha, beta, not maximizing)
            if maximizing:
                if q_eval >= beta: return beta
                alpha = max(alpha, q_eval)
            else:
                if q_eval <= alpha: return alpha
                beta = min(beta, q_eval)
        except Exception:
            continue

    return alpha if maximizing else beta


# (minimax_sse function - complete with checks and MATE_VALUE fix)
def minimax_sse(game, depth, alpha, beta, maximizing, original_depth=None,
                start_time=None, time_limit=None, principal_variation=None):
    """Main search function, uses the Game object."""
    if original_depth is None: original_depth = depth
    ply = original_depth - depth

    if time_limit is not None and start_time is not None and ply > 0 and (ply % 3 == 0):
        if time.time() - start_time > time_limit: return None

    alpha_orig = alpha
    tt_key = calculate_zobrist_hash(game)
    tt_entry = TT.get(tt_key) if tt_key is not None else None
    tt_best_move, tt_hit = None, False

    if tt_entry:
        tt_depth = tt_entry.get('depth', -1)
        if tt_depth >= depth:
            tt_score = tt_entry.get('score');
            tt_flag = tt_entry.get('flag');
            tt_best_move = tt_entry.get('best_move')
            if tt_score is not None and tt_flag is not None:
                if abs(tt_score) > MATE_SCORE_THRESHOLD:
                    sign = np.sign(tt_score)
                    mate_ply_stored = MATE_VALUE - abs(tt_score)
                    tt_score = sign * (MATE_VALUE - (mate_ply_stored + ply))
                if tt_flag == TT_EXACT:
                    tt_hit = True;
                    return tt_best_move if depth == original_depth else tt_score
                elif tt_flag == TT_ALPHA and tt_score >= beta:
                    tt_hit = True;
                    return tt_best_move if depth == original_depth else beta
                elif tt_flag == TT_BETA and tt_score <= alpha:
                    tt_hit = True;
                    return tt_best_move if depth == original_depth else alpha
                if tt_flag == TT_ALPHA:
                    alpha = max(alpha, tt_score)
                elif tt_flag == TT_BETA:
                    beta = min(beta, tt_score)
                if tt_best_move: tt_hit = True

    game_state = getattr(game, 'state', None)
    if hasattr(game, 'position_count') and hasattr(game, 'board_key'):
        try:
            current_board_key = game.board_key()
        except Exception:
            current_board_key = None
        if current_board_key and game.position_count.get(current_board_key, 0) >= 3: return 0
    if hasattr(game, 'move_clock') and game.move_clock >= 100: return 0
    if game_state in ["Draw (Threefold repetition)", "Draw (50-move rule)"]: return 0

    try:
        is_in_check = game.is_check(game.turn)
    except AttributeError:
        is_in_check = False
    effective_depth = depth + 1 if is_in_check else depth

    if effective_depth <= 0 or game_state in ["Checkmate", "Stalemate"]:
        q_score = quiescence_search(game, alpha, beta, maximizing)
        if abs(q_score) > MATE_SCORE_THRESHOLD:
            sign = np.sign(q_score)
            mate_ply_leaf = MATE_VALUE - abs(q_score)
            q_score = sign * (MATE_VALUE - (mate_ply_leaf + ply))
        return q_score

    best_score_so_far, best_move_found = (-float('inf'), None) if maximizing else (float('inf'), None)

    has_majors = any(
        p and p.name not in ['P', 'K'] for r in game.board for p in r if p and p.colour == game.turn) if hasattr(game,
                                                                                                                 'board') else False
    can_null_move = (not is_in_check and depth >= 3 and has_majors and ply > 0)

    if can_null_move:
        try:
            if not all(hasattr(game, m) for m in ['light_copy', 'turn', 'enpassant']): raise AttributeError
            copy_null = game.light_copy();
            copy_null.turn = ('b', 'w')[copy_null.turn == 'w'];
            copy_null.enpassant = None
            R = 3
            null_eval = minimax_sse(copy_null, depth - 1 - R, -beta, -alpha, not maximizing,
                                    original_depth, start_time, time_limit)
            if null_eval is None: return None
            if maximizing and -null_eval >= beta: return beta
            if not maximizing and -null_eval <= alpha: return alpha
        except Exception:
            pass

    all_moves = []
    current_colour = game.turn

    # --- Nested calculate_priority function ---
    def calculate_priority(start, end, promotion, tt_best_move=None, principal_variation=None):
        priority = 0;
        move_tuple = (start, end, promotion)
        start_idx, end_idx = start[0] * 8 + start[1], end[0] * 8 + end[1]
        if principal_variation and move_tuple == principal_variation: return 1000000
        if tt_best_move == move_tuple: return 900000

        # *** ADDED: Check for castling ***
        piece = game.board[start[0]][start[1]]
        if piece and piece.name == 'K' and abs(start[1] - end[1]) == 2:
            return 750000  # Give castling high priority

        target = game.board[end[0]][end[1]]
        if target:  # MVV-LVA Captures
            victim = piece_values.get(target.name, 0)
            attacker = piece_values.get(game.board[start[0]][start[1]].name, 1)
            priority = 800000 + victim * 10 - attacker
            see = static_exchange_eval_local(game, start, end)
            priority += max(-50000, min(50000, int(see)))
            return priority

        # Killer Moves
        if ply < len(KILLER_MOVES):
            killers = KILLER_MOVES[ply]
            if move_tuple == killers[0]: return 650000
            if move_tuple == killers[1]: return 600000

        # History Heuristic
        if 0 <= start_idx < 64 and 0 <= end_idx < 64:
            priority = HISTORY_MOVES[start_idx][end_idx]
        else:
            priority = 0

        if end[0] in [3, 4] and end[1] in [3, 4]: priority += 500
        return priority

    # --- End of nested function ---

    try:
        if not hasattr(game, 'get_moves') or not hasattr(game, 'board'): return 0
        for r in range(8):
            for c in range(8):
                piece = game.board[r][c]
                if piece and piece.colour == current_colour:
                    legal_ends = game.get_moves(r, c)
                    for move in legal_ends:
                        if not (isinstance(move, tuple) and len(move) == 2): continue
                        promos = ('Q', 'R', 'B', 'N') if piece.name == 'P' and (move[0] == 0 or move[0] == 7) else (
                            None,)
                        for p_promo in promos:
                            prio = calculate_priority((r, c), move, p_promo, tt_best_move, principal_variation)
                            all_moves.append((prio, (r, c), move, p_promo))
    except Exception:
        return 0

    all_moves.sort(key=lambda x: x[0], reverse=True)

    if not all_moves:
        return (-MATE_VALUE + ply) if is_in_check else 0

    move_index = 0
    current_static_eval = evaluate_board(game.board, game.turn, game.castling, game.enpassant, game.move_count)

    for priority, start, end, promotion in all_moves:
        if time_limit and start_time and (move_index > 0 and move_index % 5 == 0):
            if time.time() - start_time > time_limit: return None

        is_capture = game.board[end[0]][end[1]] is not None

        # Futility Pruning
        can_futility_prune = (depth <= 3 and not is_in_check and not is_capture and not promotion and
                              abs(alpha) < MATE_SCORE_THRESHOLD and abs(beta) < MATE_SCORE_THRESHOLD)
        if can_futility_prune:
            futility_margin = FUTILITY_MARGIN * depth
            if maximizing and current_static_eval + futility_margin <= alpha: continue
            if not maximizing and current_static_eval - futility_margin >= beta: continue

        # SEE Pruning (moved to priority calculation, but check again for non-positive)
        if is_capture:
            see_score = static_exchange_eval_local(game, start, end)
            if see_score < 0:
                continue  # Prune likely losing captures # Prune likely losing captures

        try:
            if not all(hasattr(game, m) for m in ['light_copy', 'make_move']): continue
            copy = game.light_copy()
            if not copy.make_move(start, end, promotion): continue
        except Exception:
            continue

        reduction = 0
        try:
            is_check_move = hasattr(copy, 'is_check') and copy.is_check(copy.turn)
        except Exception:
            is_check_move = False

        can_reduce = (depth >= 3 and move_index >= (2 if is_in_check else 4) and
                      not is_capture and not promotion and not is_check_move)
        if can_reduce:
            reduction = int(math.log(depth) * math.log(move_index + 1) / 1.8)
            if priority < 600000: reduction += 1
            reduction = max(0, min(reduction, depth - 2))

        eval_score = None;
        search_depth = depth - 1 - reduction
        is_pv_move = (principal_variation == (start, end, promotion))

        if move_index == 0 or reduction == 0 or is_pv_move:
            eval_score = minimax_sse(copy, depth - 1, alpha, beta, not maximizing,
                                     original_depth, start_time, time_limit,
                                     principal_variation if is_pv_move else None)
        else:
            zw_alpha, zw_beta = (alpha, alpha + 1) if maximizing else (beta - 1, beta)
            eval_score = minimax_sse(copy, search_depth, zw_alpha, zw_beta, not maximizing,
                                     original_depth, start_time, time_limit)
            if eval_score is not None and (eval_score > alpha if maximizing else eval_score < beta):
                if (eval_score < beta if maximizing else eval_score > alpha):
                    eval_score = minimax_sse(copy, depth - 1, alpha, beta, not maximizing,
                                             original_depth, start_time, time_limit)

        move_index += 1
        if eval_score is None: return None

        if maximizing:
            if eval_score > best_score_so_far:
                best_score_so_far = eval_score;
                best_move_found = (start, end, promotion)
            alpha = max(alpha, best_score_so_far)
            if alpha >= beta:
                if not is_capture and ply < 64:
                    move_tuple = (start, end, promotion);
                    current_killers = KILLER_MOVES[ply]
                    if move_tuple != current_killers[0]: KILLER_MOVES[ply][1] = current_killers[0]; KILLER_MOVES[ply][
                        0] = move_tuple
                    s_idx, e_idx = start[0] * 8 + start[1], end[0] * 8 + end[1]
                    if 0 <= s_idx < 64 and 0 <= e_idx < 64:
                        HISTORY_MOVES[s_idx][e_idx] = min(HISTORY_MOVES[s_idx][e_idx] + depth * depth, 32000)
                break
        else:
            if eval_score < best_score_so_far:
                best_score_so_far = eval_score;
                best_move_found = (start, end, promotion)
            beta = min(beta, best_score_so_far)
            if beta <= alpha:
                if not is_capture and ply < 64:
                    move_tuple = (start, end, promotion);
                    current_killers = KILLER_MOVES[ply]
                    if move_tuple != current_killers[0]: KILLER_MOVES[ply][1] = current_killers[0]; KILLER_MOVES[ply][
                        0] = move_tuple
                    s_idx, e_idx = start[0] * 8 + start[1], end[0] * 8 + end[1]
                    if 0 <= s_idx < 64 and 0 <= e_idx < 64:
                        HISTORY_MOVES[s_idx][e_idx] = min(HISTORY_MOVES[s_idx][e_idx] + depth * depth, 32000)
                break

    if tt_key is not None:
        score_to_store = best_score_so_far;
        tt_flag = TT_EXACT
        if abs(score_to_store) > MATE_SCORE_THRESHOLD:
            sign = np.sign(score_to_store)
            score_to_store = sign * (MATE_VALUE - ply)
        if score_to_store <= alpha_orig:
            tt_flag = TT_BETA
        elif score_to_store >= beta:
            tt_flag = TT_ALPHA
        if len(TT) >= TT_MAX_SIZE:
            keys_to_delete = list(TT.keys())[:int(TT_MAX_SIZE * 0.1)]
            for key in keys_to_delete: TT.pop(key, None)
        if depth > 0:
            move_to_store = best_move_found if (tt_flag == TT_EXACT or tt_flag == TT_ALPHA) else None
            if abs(score_to_store) < MATE_VALUE * 2:
                TT[tt_key] = {'depth': depth, 'score': score_to_store, 'flag': tt_flag, 'best_move': move_to_store}

    final_score = best_score_so_far
    if abs(final_score) > MATE_SCORE_THRESHOLD:
        sign = np.sign(final_score)
        final_score = sign * (MATE_VALUE - ply)

    return best_move_found if depth == original_depth else final_score