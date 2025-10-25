import numpy as np
import chess  # For FEN parsing
import random
import time
import math
import ai  # Import your ai module directly

# --- 1. Parameter Management ---
param_indices = {}
param_list = []
param_names = []
param_original_refs = {}  # To store references to original lists/dicts


def add_param(name, value, original_ref=None, index=None):
    """Registers a parameter, storing its name, index, and original reference if it's part of a list/dict."""
    global param_list, param_names, param_indices
    current_idx = len(param_list)
    param_indices[name] = current_idx
    param_list.append(float(value))  # Ensure float for calculations
    param_names.append(name)
    if original_ref is not None and index is not None:
        param_original_refs[name] = (original_ref, index)  # Store list/dict reference and index/key


def register_parameters():
    """Builds the initial parameter vector from ai.py globals."""
    print("Registering parameters...")
    global param_list, param_names, param_indices, param_original_refs
    param_list = []
    param_names = []
    param_indices = {}
    param_original_refs = {}

    # Register PSTs (Piece Square Tables)
    tables = {
        "pawn": ai.pawn_table, "knight": ai.knight_table, "bishop": ai.bishop_table,
        "rook": ai.rook_table, "queen": ai.queen_table, "king_mg": ai.king_table,
        "king_eg": ai.king_endgame_table
    }
    for name, table_ref in tables.items():
        if isinstance(table_ref, list) and len(table_ref) == 64:
            for i in range(64):
                add_param(f"{name}_pst_{i}", table_ref[i], original_ref=table_ref, index=i)
        else:
            print(f"Warning: PST '{name}' in ai.py is not a list or not size 64. Skipping.")

    # Register piece values (excluding King)
    if isinstance(ai.piece_values, dict):
        for piece_name in ['P', 'N', 'B', 'R', 'Q']:
            if piece_name in ai.piece_values:
                add_param(f"value_{piece_name}", ai.piece_values[piece_name],
                          original_ref=ai.piece_values, index=piece_name)
    else:
        print("Warning: ai.piece_values not found or not a dict. Skipping.")

    # Register scalar bonuses (including NEW PIN parameters)
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
        "KNIGHT_SYNERGY_BONUS",
        "RELATIVE_PIN_PENALTY", "PIN_NEAR_KING_PENALTY", "KING_EXPOSURE_PIN_PENALTY"
    ]

    for const_name in scalar_constants:
        if hasattr(ai, const_name):
            add_param(const_name, getattr(ai, const_name))
        else:
            print(f"Warning: Constant {const_name} not found in ai.py")

    # Register Passed Pawn Bonus Ranks list
    if hasattr(ai, 'PASSED_PAWN_BONUS_RANKS') and isinstance(ai.PASSED_PAWN_BONUS_RANKS, list) and len(
            ai.PASSED_PAWN_BONUS_RANKS) == 7:
        for i in range(len(ai.PASSED_PAWN_BONUS_RANKS)):
            add_param(f"PASSED_PAWN_BONUS_RANK_{i}", ai.PASSED_PAWN_BONUS_RANKS[i],
                      original_ref=ai.PASSED_PAWN_BONUS_RANKS, index=i)
    else:
        print("Warning: PASSED_PAWN_BONUS_RANKS not found, not a list, or not size 7 in ai.py. Skipping.")

    # Register King Attack Weights dictionary
    if hasattr(ai, 'king_attack_weights') and isinstance(ai.king_attack_weights, dict):
        for piece_name, value in ai.king_attack_weights.items():
            add_param(f"king_attack_{piece_name}", value,
                      original_ref=ai.king_attack_weights, index=piece_name)
    else:
        print("Warning: king_attack_weights not found or not a dict in ai.py. Skipping.")

    print(f"Registered {len(param_list)} parameters.")
    return np.array(param_list, dtype=float)


def load_params_into_engine(theta):
    """Updates the variables in the imported ai module based on the theta vector."""
    if len(theta) != len(param_names):
        raise ValueError(f"Theta length ({len(theta)}) does not match registered parameter count ({len(param_names)})")

    for i, name in enumerate(param_names):
        value = theta[i]

        if name in param_original_refs:
            original_ref, index = param_original_refs[name]
            try:
                if isinstance(original_ref, list) and 0 <= index < len(original_ref):
                    original_ref[index] = int(round(value))
                elif isinstance(original_ref, dict) and index in original_ref:
                    original_ref[index] = int(round(value))
            except Exception as e:
                print(f"Warning while loading {name}: {e}")
        else:
            if hasattr(ai, name):
                is_float_param = name in [
                    "MOBILITY_BONUS", "PASSED_PAWN_PATH_CLEAR_FACTOR",
                    "SPACE_BONUS_FACTOR", "INITIATIVE_FACTOR", "KING_ATTACK_FACTOR",
                    "BAD_BISHOP_FACTOR", "KING_STORM_FACTOR"
                ]
                if is_float_param:
                    setattr(ai, name, float(value))
                else:
                    setattr(ai, name, int(round(value)))


# --- 3. Evaluation and Loss ---
EVAL_CACHE = {}
CACHE_HITS = 0
CACHE_MISSES = 0


def get_eval_with_params(fen_str, theta):
    """Evaluates a FEN using the provided parameters (theta)."""
    global CACHE_HITS, CACHE_MISSES
    theta_tuple = tuple(theta)
    cache_key = (fen_str, theta_tuple)

    if cache_key in EVAL_CACHE:
        CACHE_HITS += 1
        return EVAL_CACHE[cache_key]
    else:
        CACHE_MISSES += 1
        load_params_into_engine(theta)
        score = ai.evaluate_board_fen(fen_str)
        EVAL_CACHE[cache_key] = score
        if len(EVAL_CACHE) > 500000:
            keys_to_remove = random.sample(list(EVAL_CACHE.keys()), k=int(len(EVAL_CACHE) * 0.1))
            for key in keys_to_remove:
                EVAL_CACHE.pop(key, None)
        return score


def sigmoid(score, K=1.13):
    """Maps evaluation score (centipawns) to winning probability (0-1)."""
    clamped_score = np.clip(score, -2000, 2000)
    exponent = -math.log(10) * K * clamped_score / 400.0
    exponent = np.clip(exponent, -700, 700)
    return 1.0 / (1.0 + np.exp(exponent))


def calculate_batch_loss(theta, data_batch):
    """Calculates Mean Squared Error for a batch of (FEN, Result) pairs."""
    total_squared_error = 0.0
    valid_samples = 0
    for fen, result_str in data_batch:
        if result_str == '1-0':
            actual_score = 1.0
        elif result_str == '0-1':
            actual_score = 0.0
        elif result_str == '1/2-1/2':
            actual_score = 0.5
        else:
            continue

        engine_score_cp = get_eval_with_params(fen, theta)
        predicted_prob = sigmoid(engine_score_cp)
        total_squared_error += (predicted_prob - actual_score) ** 2
        valid_samples += 1

    if valid_samples == 0:
        return 0.0
    return total_squared_error / valid_samples


# --- 4. SPSA Implementation ---
def spsa_tune(initial_theta, data, iterations=50, batch_size=1024,
              a=0.5, c=0.05, A_factor=0.1, alpha=0.602, gamma=0.101):
    """Performs SPSA optimization."""
    theta = initial_theta.copy()
    num_params = len(theta)
    A = max(10.0, float(iterations * A_factor))

    print(f"Starting SPSA: {iterations} iterations, Batch Size: {batch_size}")
    print(f"Hyperparameters: a={a}, c={c}, A={A}, alpha={alpha}, gamma={gamma}")

    best_loss = float('inf')
    best_theta = theta.copy()
    start_tune_time = time.time()

    gradient_clip_norm = 10.0

    for k in range(iterations):
        iter_start_time = time.time()

        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma

        delta_k = np.random.choice([-1.0, 1.0], size=num_params)
        theta_plus = theta + ck * delta_k
        theta_minus = theta - ck * delta_k

        if not data:
            print("Error: Tuning data is empty.")
            return best_theta

        actual_batch_size = min(batch_size, len(data))
        data_batch = random.sample(data, actual_batch_size)

        loss_plus = calculate_batch_loss(theta_plus, data_batch)
        loss_minus = calculate_batch_loss(theta_minus, data_batch)

        epsilon = 1e-10
        gradient_estimate = (loss_plus - loss_minus) / (2 * ck * delta_k + epsilon)

        gradient_norm = np.linalg.norm(gradient_estimate)
        if gradient_norm > gradient_clip_norm and gradient_norm > 0:
            gradient_estimate = gradient_estimate * (gradient_clip_norm / gradient_norm)

        theta = theta - ak * gradient_estimate

        # Progress Monitoring
        if (k + 1) % 5 == 0 or k == iterations - 1:
            current_loss = calculate_batch_loss(theta, random.sample(data, actual_batch_size))
            iteration_time = time.time() - iter_start_time
            if current_loss < best_loss:
                best_loss = current_loss
                best_theta = theta.copy()
                print(f"Iter {k + 1}/{iterations} | *New Best Loss*: {current_loss:.6f} | ak: {ak:.4e} | ck: {ck:.4e} | Time: {iteration_time:.2f}s")
                np.save('best_theta_interim.npy', best_theta)
            else:
                print(f"Iter {k + 1}/{iterations} | Current Loss: {current_loss:.6f} | ak: {ak:.4e} | ck: {ck:.4e} | Time: {iteration_time:.2f}s")

    total_time = time.time() - start_tune_time
    print(f"\nSPSA Tuning finished in {total_time:.2f} seconds.")
    if (CACHE_HITS + CACHE_MISSES) > 0:
        hit_rate = CACHE_HITS / (CACHE_HITS + CACHE_MISSES) * 100
        print(f"Cache efficiency: Hits={CACHE_HITS}, Misses={CACHE_MISSES} ({hit_rate:.1f}% hit rate)")
    else:
        print("Cache efficiency: No cache usage recorded.")
    print(f"Best recorded loss during tuning: {best_loss:.6f}")
    return best_theta


# --- 5. Main Execution Logic ---
if __name__ == "__main__":
    DATA_FILE = 'tuning_data.txt'
    OUTPUT_WEIGHTS_FILE = 'optimized_weights.npy'

    print(f"Loading data from {DATA_FILE}...")
    tuning_data = []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split('|')
                if len(parts) == 2:
                    fen, result = parts[0].strip(), parts[1].strip()
                    if len(fen.split()) == 6 and result in ['1-0', '0-1', '1/2-1/2']:
                        tuning_data.append((fen, result))
        print(f"Loaded {len(tuning_data)} valid positions.")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please run the PGN parsing script first.")
        exit()
    except Exception as e:
        print(f"Error loading data file: {e}")
        exit()

    if not tuning_data:
        print("Error: No valid data loaded from file.")
        exit()

    initial_theta = register_parameters()

    # --- Run SPSA ---
    ITERATIONS_TO_RUN = 200

    optimized_theta = spsa_tune(initial_theta, tuning_data,
                                iterations=ITERATIONS_TO_RUN,
                                batch_size=4096,
                                a=0.5,
                                c=0.05
                                )

    np.save(OUTPUT_WEIGHTS_FILE, optimized_theta)
    print(f"\nBest optimized weights (after {ITERATIONS_TO_RUN} iterations) saved to {OUTPUT_WEIGHTS_FILE}")

    print("\n--- Next Steps ---")
    print(f"1. Your 'ai.py' file is already set up to load '{OUTPUT_WEIGHTS_FILE}' at startup.")
    print(f"2. Run 'main.py' to play against the tuned engine.")
    print(f"3. Run 'main.py' in 'auto' mode for SPRT self-play tests to verify Elo gain.")
