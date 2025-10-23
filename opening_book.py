# opening_book.py (Modified for Polyglot .bin files)

import chess
import chess.polyglot
import os
import random

WHITE_BOOK_FILE = 'white.bin'
BLACK_BOOK_FILE = 'black.bin'

white_reader = None
black_reader = None

#converts board reprsentation to python chess board
piece_map_to_chess = {
    'P': chess.PAWN, 'N': chess.KNIGHT, 'B': chess.BISHOP,
    'R': chess.ROOK, 'Q': chess.QUEEN, 'K': chess.KING
}

def your_board_to_chess_board(game_instance):
    board = chess.Board(fen=None) #empty board
    board.clear_board()
    for r in range(8):
        for c in range(8):
            piece_obj = game_instance.board[r][c]
            if piece_obj:
                square_index = chess.square(c, 7 - r)
                color = chess.WHITE if piece_obj.colour == 'w' else chess.BLACK
                piece_type = piece_map_to_chess.get(piece_obj.name)
                if piece_type:
                    board.set_piece_at(square_index, chess.Piece(piece_type, color))

    board.turn = chess.WHITE if game_instance.turn == 'w' else chess.BLACK

    board.castling_rights = 0
    if game_instance.castling.get('wKR'): board.castling_rights |= chess.BB_H1
    if game_instance.castling.get('wQR'): board.castling_rights |= chess.BB_A1
    if game_instance.castling.get('bKR'): board.castling_rights |= chess.BB_H8
    if game_instance.castling.get('bQR'): board.castling_rights |= chess.BB_A8


    if game_instance.enpassant:
        ep_row, ep_col = game_instance.enpassant
        board.ep_square = chess.square(ep_col, 7 - ep_row)
    else:
        board.ep_square = None


    board.halfmove_clock = game_instance.move_clock
    board.fullmove_number = (game_instance.move_count // 2) + 1


    return board



def load_polyglot_books():
    global white_reader, black_reader
    loaded_any = False
    print("Loading Polyglot opening books...")
    try:
        if os.path.exists(WHITE_BOOK_FILE):
            white_reader = chess.polyglot.open_reader(WHITE_BOOK_FILE)
            print(f"  OK: Loaded white opening book: {WHITE_BOOK_FILE}")
            loaded_any = True
        else:
            print(f"  WARNING: White book file not found at: {os.path.abspath(WHITE_BOOK_FILE)}")

        if os.path.exists(BLACK_BOOK_FILE):
            black_reader = chess.polyglot.open_reader(BLACK_BOOK_FILE)
            print(f"  OK: Loaded black opening book: {BLACK_BOOK_FILE}")
            loaded_any = True
        else:
            print(f"  WARNING: Black book file not found at: {os.path.abspath(BLACK_BOOK_FILE)}")

        if not loaded_any:
             print("WARNING: No Polyglot book files (.bin) found. Opening book disabled.")

    except Exception as e:
        print(f"ERROR loading Polyglot book(s): {e}. Opening book disabled.")
        white_reader = None
        black_reader = None


def get_polyglot_book_move(game_instance):
    reader = white_reader if game_instance.turn == 'w' else black_reader
    if not reader:
        return None

    try:
        board = your_board_to_chess_board(game_instance)
    except Exception as e:
        print(f"Error converting board state for book lookup: {e}")
        return None

    try:
        entry = reader.weighted_choice(board)
        if not entry:
            return None # Position not in book
        move = entry.move

        start_sq_index = move.from_square
        end_sq_index = move.to_square

        start_row, start_col = 7 - chess.square_rank(start_sq_index), chess.square_file(start_sq_index)
        end_row, end_col = 7 - chess.square_rank(end_sq_index), chess.square_file(end_sq_index)


        promotion_char = None
        if move.promotion:
            promotion_char = chess.piece_symbol(move.promotion).upper()

        return ((start_row, start_col), (end_row, end_col), promotion_char)

    except KeyError:
        return None
    except Exception as e:
        # Catch potential errors during lookup or conversion
        print(f"Error during Polyglot lookup/conversion: {e}")
        return None

load_polyglot_books()