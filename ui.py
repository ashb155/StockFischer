import pygame
import os
import sys
import time
import random
import threading
from game import Game
from ai import minimax_sse
from opening_book import get_polyglot_book_move

# --- Configuration (Hardcoded for Hard/30s) ---
MAX_DEPTH = 6
TIME_LIMIT = 30.0
HUMAN_COLOR = 'w'
AI_COLOR = 'b'
ANTICIPATION_DELAY = 4.0  # <--- NEW: Delay BEFORE the AI move is applied

# --- Shared Variables for Threading and Delay ---
AI_RESULT = [None, None]  # [move_tuple, move_source]
AI_THREAD = None
AI_START_TIME = 0.0
# We only need one timer for when the move should be applied
AI_APPLY_TIME = 0.0  # Timestamp when the AI move should be visually applied

# --- Setup ---
pygame.init()

SQUARE_SIZE = 75
BOARD_WIDTH, BOARD_HEIGHT = 8 * SQUARE_SIZE, 8 * SQUARE_SIZE
SCREEN = pygame.display.set_mode((BOARD_WIDTH, BOARD_HEIGHT))
pygame.display.set_caption("StockFischer 2.0 GUI - Board & Pieces")

BOARD_BACKGROUND = None
PIECE_IMAGES = {}

# --- Colors (for highlights) ---
COLOR_HIGHLIGHT_SELECTED = pygame.Color(255, 255, 51, 100)
COLOR_HIGHLIGHT_LAST_MOVE = pygame.Color(205, 210, 106, 150)
COLOR_HIGHLIGHT_LEGAL = pygame.Color(0, 0, 0, 40)

FILENAME_MAP = {
    'wP': 'wp.png', 'wR': 'wr.png', 'wN': 'wn.png', 'wB': 'wb.png', 'wQ': 'wq.png', 'wK': 'wk.png',
    'bP': 'bp.png', 'bR': 'br.png', 'bN': 'bn.png', 'bB': 'bb.png', 'bQ': 'bq.png', 'bK': 'bk.png',
}


def index_to_notation_local(pos):
    col_map = 'abcdefgh'
    row, col = pos
    return col_map[col] + str(8 - row)


# --- Asset Loading Function (Unchanged) ---
def load_assets(image_folder="images"):
    global BOARD_BACKGROUND, PIECE_IMAGES
    try:
        board_image_path = os.path.join(image_folder, "200.png")
        if not os.path.exists(board_image_path):
            print(f"Error: Board background '200.png' not found in folder: {image_folder}")
            BOARD_BACKGROUND = None
        else:
            board_img = pygame.image.load(board_image_path)
            BOARD_BACKGROUND = pygame.transform.scale(board_img, (BOARD_WIDTH, BOARD_HEIGHT))
            print(f"Loaded board background: {board_image_path}")
    except Exception as e:
        print(f"Error loading board background: {e}")
        BOARD_BACKGROUND = None

    missing_files = []
    for piece_key, filename in FILENAME_MAP.items():
        path = os.path.join(image_folder, filename)
        try:
            if not os.path.exists(path):
                missing_files.append(filename)
                continue
            image = pygame.image.load(path).convert_alpha()
            PIECE_IMAGES[piece_key] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
        except pygame.error as e:
            print(f"Error loading image {path}: {e}")
            missing_files.append(f"{filename} (Error)")

    if missing_files:
        print("\n--- WARNING: MISSING PIECE IMAGES ---")
        print(f"The following files were not found in your '{image_folder}' folder:")
        for f in missing_files: print(f"- {f}")
        print("--------------------------------------\n")


def draw_board(screen):
    if BOARD_BACKGROUND:
        screen.blit(BOARD_BACKGROUND, (0, 0))
    else:
        colors = [pygame.Color("#EAF0CE"), pygame.Color("#7B955D")]
        for r in range(8):
            for c in range(8):
                color = colors[((r + c) % 2)]
                pygame.draw.rect(screen, color, pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def draw_pieces(screen, board_state):
    for r in range(8):
        for c in range(8):
            piece = board_state[r][c]
            if piece is not None:
                piece_key = str(piece)
                if piece_key in PIECE_IMAGES:
                    rect = PIECE_IMAGES[piece_key].get_rect(
                        center=(c * SQUARE_SIZE + SQUARE_SIZE // 2,
                                r * SQUARE_SIZE + SQUARE_SIZE // 2)
                    )
                    screen.blit(PIECE_IMAGES[piece_key], rect)


def draw_highlights(screen, selected_square, legal_moves, last_move):
    highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)

    # 1. Highlight Last Move
    if last_move:
        start_pos, end_pos = last_move
        highlight_surface.fill(COLOR_HIGHLIGHT_LAST_MOVE)
        screen.blit(highlight_surface, (start_pos[1] * SQUARE_SIZE, start_pos[0] * SQUARE_SIZE))
        screen.blit(highlight_surface, (end_pos[1] * SQUARE_SIZE, end_pos[0] * SQUARE_SIZE))

    # 2. Highlight Selected Piece
    if selected_square:
        r, c = selected_square
        highlight_surface.fill(COLOR_HIGHLIGHT_SELECTED)
        screen.blit(highlight_surface, (c * SQUARE_SIZE, r * SQUARE_SIZE))

    # 3. Highlight Legal Moves (circles)
    circle_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    circle_pos = (SQUARE_SIZE // 2, SQUARE_SIZE // 2)
    circle_radius = SQUARE_SIZE // 6
    pygame.draw.circle(circle_surface, COLOR_HIGHLIGHT_LEGAL, circle_pos, circle_radius)

    for r, c in legal_moves:
        screen.blit(circle_surface, (c * SQUARE_SIZE, r * SQUARE_SIZE))


def draw_game_state(screen, game, selected_square, legal_moves, last_move):
    draw_board(screen)
    draw_highlights(screen, selected_square, legal_moves, last_move)
    draw_pieces(screen, game.board)
    pygame.display.flip()


# --- AI Search Worker Function (Runs in Thread) ---
def ai_search_worker(game_copy, max_depth, time_limit):
    """Executes the AI search logic and stores result in the global AI_RESULT."""
    global AI_RESULT

    # 1. Opening Book Check
    book_move = None
    if game_copy.move_count < 20:
        try:
            book_move = get_polyglot_book_move(game_copy)
        except Exception:
            pass

    if book_move:
        (start, end, promotion) = book_move
        try:
            if game_copy.board[start[0]][start[1]].colour == game_copy.turn and end in game_copy.get_moves(start[0],
                                                                                                           start[1]):
                AI_RESULT[0], AI_RESULT[1] = book_move, "Book"
                return
        except Exception:
            pass

    # 2. Search
    start_time = time.time()
    best_move_so_far = None

    for depth in range(1, max_depth + 1):
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining < 0.2: break

        pv_move_tuple = best_move_so_far
        search_result = minimax_sse(game_copy, depth, float('-inf'), float('inf'), game_copy.turn == 'w',
                                    original_depth=depth, start_time=start_time, time_limit=time_remaining,
                                    principal_variation=pv_move_tuple)

        if search_result is None: break
        if isinstance(search_result, tuple) and len(search_result) == 3: best_move_so_far = search_result
        if time_limit - (time.time() - start_time) < 0.2: break

    elapsed_time = time.time() - start_time
    AI_RESULT[0] = best_move_so_far
    AI_RESULT[1] = "Search"
    print(f"AI Debug: Search finished at D{depth - 1}/{max_depth}. Time: {elapsed_time:.2f}s")


# --- Main Loop ---
def main_gui():
    global AI_THREAD, AI_RESULT, AI_START_TIME, AI_APPLY_TIME

    load_assets(image_folder="images")
    if not BOARD_BACKGROUND: print("Continuing with fallback colored squares.")

    game = Game()
    selected_square, legal_moves, last_move = None, [], None
    ai_thinking = False
    ai_move_ready = False  # NEW: Flag set when thread finishes

    running = True
    clock = pygame.time.Clock()

    while running:

        # --- 1. CHECK FOR AI THREAD COMPLETION (Non-blocking) ---
        if AI_THREAD and not AI_THREAD.is_alive():
            # Thread finished: record time, set flag, calculate apply time
            AI_THREAD = None
            AI_APPLY_TIME = time.time() + ANTICIPATION_DELAY  # Set the delay timer
            ai_thinking = False  # Search is done
            ai_move_ready = True  # Move is ready, just waiting for delay
            print(f"AI Search Complete. Applying move in {ANTICIPATION_DELAY} seconds...")

        # --- 2. APPLY AI MOVE AFTER DELAY ---
        if ai_move_ready and time.time() >= AI_APPLY_TIME:

            ai_move, move_source = AI_RESULT[0], AI_RESULT[1]
            ai_move_ready = False  # Clear flag after processing

            if ai_move:
                start, end, promotion = ai_move
                piece_to_move = game.board[start[0]][start[1]]

                if game.make_move(start, end, promotion):
                    last_move = (start, end)

                    # Safe print notation
                    start_sq_str = index_to_notation_local(start);
                    end_sq_str = index_to_notation_local(end)
                    move_str = f"{piece_to_move.name if piece_to_move else '?'}{start_sq_str}{end_sq_str}"
                    if promotion: move_str += f"={promotion}"

                    print(f"AI Move ({move_source}): {move_str}\n")
                else:
                    print(f"FATAL ERROR: AI returned an illegal move.")
            else:
                print("Game End: AI found no legal moves.")

        # --- 3. INPUT & UI LOGIC ---
        # Block user input ONLY if the AI move is waiting to be applied or thinking
        allow_input = not ai_move_ready and not ai_thinking

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            # --- UNDO FEATURE ('U' Key) ---
            if e.type == pygame.KEYDOWN and e.key == pygame.K_u:
                if AI_THREAD and AI_THREAD.is_alive():
                    print("Cannot undo: AI is currently searching (wait or exit).")
                    continue
                # If undoing while AI move is ready (but not applied), clear the delay timer
                if ai_move_ready:
                    ai_move_ready = False
                    AI_APPLY_TIME = 0.0

                if game.history:
                    game.full_unmake_move()  # Undo AI move
                    if game.turn == AI_COLOR and game.history:
                        game.full_unmake_move()  # Undo Human move

                    last_move = game.history[-1][:2] if game.history else None
                    selected_square = None
                    legal_moves = []
                    print("\n--- UNDO SUCCESSFUL. Your turn. ---")

            # --- HUMAN MOVE EXECUTION ---
            if allow_input and game.turn == HUMAN_COLOR and e.type == pygame.MOUSEBUTTONDOWN:
                col = e.pos[0] // SQUARE_SIZE
                row = e.pos[1] // SQUARE_SIZE
                clicked_square = (row, col)

                # --- Piece Selection/Execution Logic ---
                if selected_square:
                    start_r, start_c = selected_square
                    piece_to_move = game.board[start_r][start_c]

                    if clicked_square == selected_square:  # Deselect
                        selected_square, legal_moves = None, []

                    elif clicked_square in legal_moves:  # Execute Move
                        end_r, end_c = clicked_square
                        promotion = 'Q' if piece_to_move and piece_to_move.name == 'P' and (
                                    end_r == 0 or end_r == 7) else None

                        if game.make_move((start_r, start_c), clicked_square, promotion):
                            last_move = (selected_square, clicked_square)
                            start_sq_str = index_to_notation_local(selected_square);
                            end_sq_str = index_to_notation_local(clicked_square)
                            move_str = f"{piece_to_move.name if piece_to_move else '?'}{start_sq_str}{end_sq_str}"
                            if promotion: move_str += f"={promotion}"
                            print(f"Human Move: {move_str}")

                            selected_square, legal_moves = None, []

                            # --- START AI THREAD ---
                            AI_SEARCH_RESULT = [None, None]
                            AI_START_TIME = time.time()
                            # Pass a *copy* of the game state to the thread
                            AI_THREAD = threading.Thread(target=ai_search_worker,
                                                         args=(game.light_copy(), MAX_DEPTH, TIME_LIMIT))
                            AI_THREAD.start()
                            ai_thinking = True
                        else:
                            print("Error: Illegal move detected.")

                    else:  # Reselect/Deselect
                        new_piece = game.board[row][col]
                        if new_piece and new_piece.colour == HUMAN_COLOR:
                            selected_square, legal_moves = clicked_square, list(game.get_moves(row, col))
                        else:
                            selected_square, legal_moves = None, []

                else:  # Initial Selection
                    piece = game.board[row][col]
                    if piece is not None and piece.colour == HUMAN_COLOR:
                        selected_square = clicked_square
                        try:
                            legal_moves = list(game.get_moves(row, col))
                        except Exception:
                            legal_moves = []
                    else:
                        selected_square, legal_moves = None, []

        # --- 4. DRAW ---
        draw_game_state(SCREEN, game, selected_square, legal_moves, last_move)

        clock.tick(30)  # Ensure a smooth FPS for the GUI

    # Clean up upon exit
    if AI_THREAD and AI_THREAD.is_alive(): print("Stopping AI thread...")
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main_gui()