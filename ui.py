import pygame
import os
import sys
import time
import threading
from game import Game
from ai import minimax_sse
from opening_book import get_polyglot_book_move

# --- Configuration ---
MAX_DEPTH = 6
TIME_LIMIT = 30.0
HUMAN_COLOR = 'w'
AI_COLOR = 'b'
BOOK_MOVE_DELAY = 4.0

# --- Shared Variables ---
AI_RESULT = [None, None, 'Search']  # [move_tuple, move_source, delay_mode]
AI_THREAD = None
AI_START_TIME = 0.0
AI_APPLY_TIME = 0.0
AI_STATUS = "Idle"
AI_TIME_USED = 0.0
MOVE_HISTORY = []

# --- Pygame Setup ---
pygame.init()
SQUARE_SIZE = 75
BOARD_WIDTH, BOARD_HEIGHT = 8 * SQUARE_SIZE, 8 * SQUARE_SIZE
SIDEBAR_WIDTH = 200
WINDOW_WIDTH = BOARD_WIDTH + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("StockFischer 2.0 GUI - Chess.com Style")

# --- Colors ---
COLOR_HIGHLIGHT_SELECTED = pygame.Color(255, 255, 51, 100)
COLOR_HIGHLIGHT_LAST_MOVE = pygame.Color(205, 210, 106, 150)
COLOR_HIGHLIGHT_LEGAL = pygame.Color(0, 0, 0, 40)
SIDEBAR_BG_COLOR = pygame.Color(0, 0, 0)

# --- Images ---
BOARD_BACKGROUND = None
PIECE_IMAGES = {}
BOT_IMAGE = None

FILENAME_MAP = {
    'wP': 'wp.png', 'wR': 'wr.png', 'wN': 'wn.png', 'wB': 'wb.png', 'wQ': 'wq.png', 'wK': 'wk.png',
    'bP': 'bp.png', 'bR': 'br.png', 'bN': 'bn.png', 'bB': 'bb.png', 'bQ': 'bq.png', 'bK': 'bk.png',
}


def index_to_notation_local(pos):
    col_map = 'abcdefgh'
    row, col = pos
    return col_map[col] + str(8 - row)


# --- NEW: Function to generate (Simplified) Standard Algebraic Notation ---
def generate_simple_san(game_instance, start_sq, end_sq, promotion):
    """
    Generates a simplified algebraic notation string (e.g., e2e4, Nf3).
    This does NOT handle full SAN features like disambiguation (Nbd2 vs Nfd2),
    captures ('x'), check/checkmate ('+/#'), or castling ('O-O').

    For full, correct SAN, you must implement logic based on the move's effect
    or use a proper chess library's SAN generator.
    """
    start_str = index_to_notation_local(start_sq)
    end_str = index_to_notation_local(end_sq)

    start_r, start_c = start_sq
    piece = game_instance.board[start_r][start_c]

    # Simple move notation based on piece and destination
    if piece:
        piece_char = piece.name if piece.name != 'P' else ''
        move_str = f"{piece_char}{end_str}"

        # Add a simple 'x' if the destination is occupied (not perfectly accurate for SAN)
        if game_instance.board[end_sq[0]][end_sq[1]]:
            move_str = f"{piece_char}x{end_str}"
            if piece.name == 'P':
                # For pawn captures, include the start column
                move_str = f"{start_str[0]}x{end_str}"

        # Handle castling (O-O or O-O-O) - requires knowing the King/Rook start/end squares
        # For simplicity in this placeholder, we'll skip special castling SAN.

        # This part is the most accurate part of the simplification:
        if promotion:
            move_str += f"={promotion}"

        return move_str

    return f"Move Error ({start_str}{end_str})"  # Fallback


# --- Load assets ---
def load_assets(image_folder="images"):
    global BOARD_BACKGROUND, PIECE_IMAGES, BOT_IMAGE
    # Board
    board_path = os.path.join(image_folder, "200.png")
    if os.path.exists(board_path):
        img = pygame.image.load(board_path)
        BOARD_BACKGROUND = pygame.transform.scale(img, (BOARD_WIDTH, BOARD_HEIGHT))
    else:
        BOARD_BACKGROUND = None
    # Pieces
    for key, filename in FILENAME_MAP.items():
        path = os.path.join(image_folder, filename)
        if os.path.exists(path):
            img = pygame.image.load(path).convert_alpha()
            PIECE_IMAGES[key] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
    # Bot image
    bot_path = os.path.join(image_folder, "bot.png")
    if os.path.exists(bot_path):
        img = pygame.image.load(bot_path).convert_alpha()
        scale = min(SIDEBAR_WIDTH / img.get_width(), 150 / img.get_height())
        BOT_IMAGE = pygame.transform.smoothscale(img, (int(img.get_width() * scale), int(img.get_height() * scale)))


# --- Drawing Functions ---
def draw_board(screen):
    if BOARD_BACKGROUND:
        screen.blit(BOARD_BACKGROUND, (0, 0))
    else:
        colors = [pygame.Color("#EAF0CE"), pygame.Color("#7B955D")]
        for r in range(8):
            for c in range(8):
                color = colors[(r + c) % 2]
                pygame.draw.rect(screen, color, pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def draw_pieces(screen, board_state):
    for r in range(8):
        for c in range(8):
            piece = board_state[r][c]
            if piece:
                key = str(piece)
                if key in PIECE_IMAGES:
                    rect = PIECE_IMAGES[key].get_rect(
                        center=(c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2))
                    screen.blit(PIECE_IMAGES[key], rect)


def draw_highlights(screen, selected_square, legal_moves, last_move):
    s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    if last_move:
        start, end = last_move
        s.fill(COLOR_HIGHLIGHT_LAST_MOVE)
        screen.blit(s, (start[1] * SQUARE_SIZE, start[0] * SQUARE_SIZE))
        screen.blit(s, (end[1] * SQUARE_SIZE, end[0] * SQUARE_SIZE))
    if selected_square:
        s.fill(COLOR_HIGHLIGHT_SELECTED)
        screen.blit(s, (selected_square[1] * SQUARE_SIZE, selected_square[0] * SQUARE_SIZE))
    circle = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(circle, COLOR_HIGHLIGHT_LEGAL, (SQUARE_SIZE // 2, SQUARE_SIZE // 2), SQUARE_SIZE // 6)
    for move in legal_moves:
        screen.blit(circle, (move[1] * SQUARE_SIZE, move[0] * SQUARE_SIZE))


def draw_sidebar(screen):
    # Minimal dark sidebar
    pygame.draw.rect(screen, (28, 30, 33), (BOARD_WIDTH, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))

    offset_y = 25  # more top padding

    # Bot image at top
    if BOT_IMAGE:
        rect = BOT_IMAGE.get_rect()
        rect.centerx = BOARD_WIDTH + SIDEBAR_WIDTH // 2
        rect.top = offset_y
        screen.blit(BOT_IMAGE, rect)
        offset_y += rect.height + 25  # more spacing below bot image

    # Fonts - modern
    header_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
    info_font = pygame.font.SysFont("Segoe UI", 16)

    # AI info (minimal, inline, no box)
    screen.blit(header_font.render("StockFischer 1.0", True, (240, 240, 240)), (BOARD_WIDTH + 15, offset_y))
    offset_y += 35  # more spacing below title

    screen.blit(info_font.render(f"Have a good game!", True, (180, 180, 180)), (BOARD_WIDTH + 15, offset_y))
    offset_y += 35
    # Move history (minimal)
    screen.blit(info_font.render("Move History", True, (200, 200, 200)), (BOARD_WIDTH + 15, offset_y))
    offset_y += 25
    for move in MOVE_HISTORY[-12:]:
        screen.blit(info_font.render(move, True, (220, 220, 220)), (BOARD_WIDTH + 20, offset_y))
        offset_y += 22  # slightly taller spacing for readability


def draw_game_state(screen, game, selected_square, legal_moves, last_move):
    draw_board(screen)
    draw_highlights(screen, selected_square, legal_moves, last_move)
    draw_pieces(screen, game.board)
    draw_sidebar(screen)
    pygame.display.flip()


# --- AI Thread ---
def ai_search_worker(game_copy, max_depth, time_limit):
    global AI_RESULT
    AI_RESULT[2] = 'Search'
    book_move = None
    if game_copy.move_count < 20:
        try:
            book_move = get_polyglot_book_move(game_copy)
        except:
            pass
    if book_move:
        start, end, prom = book_move
        try:
            if game_copy.board[start[0]][start[1]].colour == game_copy.turn and end in game_copy.get_moves(start[0],
                                                                                                           start[1]):
                AI_RESULT[0], AI_RESULT[1] = book_move, "Book"
                AI_RESULT[2] = "Book"
                return
        except:
            pass
    start_time = time.time()
    best_move = None
    for depth in range(1, max_depth + 1):
        rem = time_limit - (time.time() - start_time)
        if rem < 0.2: break
        res = minimax_sse(game_copy, depth, float('-inf'), float('inf'), game_copy.turn == 'w', original_depth=depth,
                          start_time=start_time, time_limit=rem, principal_variation=best_move)
        if res is None: break
        if isinstance(res, tuple) and len(res) == 3: best_move = res
    AI_RESULT[0] = best_move
    AI_RESULT[1] = "Search"


# --- Main GUI ---
def main_gui():
    global AI_THREAD, AI_RESULT, AI_START_TIME, AI_APPLY_TIME, AI_STATUS, AI_TIME_USED, MOVE_HISTORY
    load_assets("images")
    game = Game()
    selected_square, legal_moves, last_move = None, [], None
    ai_thinking = False
    ai_move_ready = False
    running = True
    clock = pygame.time.Clock()
    AI_RESULT = [None, None, 'Search']

    while running:
        AI_STATUS = "Thinking..." if ai_thinking else "Idle"
        AI_TIME_USED = time.time() - AI_START_TIME if ai_thinking else 0.0

        if AI_THREAD and not AI_THREAD.is_alive():
            AI_THREAD = None
            delay = BOOK_MOVE_DELAY if AI_RESULT[2] == "Book" else 0.0
            AI_APPLY_TIME = time.time() + delay
            ai_thinking = False
            ai_move_ready = True

        if ai_move_ready and time.time() >= AI_APPLY_TIME:
            ai_move, source, _ = AI_RESULT
            ai_move_ready = False
            AI_RESULT = [None, None, 'Search']
            if ai_move:
                start, end, prom = ai_move

                # --- START SAN CONVERSION FOR AI MOVE ---
                san_move = generate_simple_san(game, start, end, prom)
                # --- END SAN CONVERSION FOR AI MOVE ---

                if game.make_move(start, end, prom):
                    last_move = (start, end)
                    MOVE_HISTORY.append(f"AI: {san_move}")  # Use SAN

        allow_input = not ai_move_ready and not ai_thinking
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            # ------------------------------------------------------------------
            # --- START: CRITICAL CHANGE TO PREVENT SIDEBAR/OUT-OF-BOUNDS CRASH ---
            # ------------------------------------------------------------------
            if allow_input and game.turn == HUMAN_COLOR and e.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = e.pos

                # Check if the click is WITHIN the 8x8 board area
                if mouse_x < BOARD_WIDTH and mouse_y < BOARD_HEIGHT:
                    col, row = mouse_x // SQUARE_SIZE, mouse_y // SQUARE_SIZE
                    click_sq = (row, col)

                    if selected_square:
                        start_r, start_c = selected_square
                        piece = game.board[start_r][start_c]
                        if click_sq == selected_square:
                            selected_square, legal_moves = None, []
                        elif click_sq in legal_moves:
                            end_r, end_c = click_sq
                            promotion = 'Q' if piece and piece.name == 'P' and (end_r == 0 or end_r == 7) else None

                            # --- START SAN CONVERSION FOR HUMAN MOVE ---
                            san_move = generate_simple_san(game, selected_square, click_sq, promotion)
                            # --- END SAN CONVERSION FOR HUMAN MOVE ---

                            if game.make_move((start_r, start_c), click_sq, promotion):
                                last_move = (selected_square, click_sq)
                                MOVE_HISTORY.append(f"You: {san_move}")  # Use SAN
                                selected_square, legal_moves = None, []
                                AI_RESULT = [None, None, 'Search']
                                AI_START_TIME = time.time()
                                AI_THREAD = threading.Thread(target=ai_search_worker,
                                                             args=(game.light_copy(), MAX_DEPTH, TIME_LIMIT))
                                AI_THREAD.start()
                                ai_thinking = True
                            else:
                                print("Illegal move")
                        else:
                            new_piece = game.board[row][col]
                            if new_piece and new_piece.colour == HUMAN_COLOR:
                                selected_square, legal_moves = click_sq, list(game.get_moves(row, col))
                            else:
                                selected_square, legal_moves = None, []
                    else:
                        piece = game.board[row][col]
                        if piece and piece.colour == HUMAN_COLOR:
                            selected_square = click_sq;
                            legal_moves = list(game.get_moves(row, col))
                        else:
                            selected_square, legal_moves = None, []

        draw_game_state(SCREEN, game, selected_square, legal_moves, last_move)
        clock.tick(30)
    if AI_THREAD and AI_THREAD.is_alive(): print("Stopping AI thread...")
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main_gui()