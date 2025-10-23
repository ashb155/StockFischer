import chess
import chess.polyglot
import os

try:
    from game import index_to_notation
except ImportError:
    # Basic fallback
    def index_to_notation(pos):
        col_map = 'abcdefgh'
        row, col = pos
        if pos is None or not (0 <= row < 8 and 0 <= col < 8):
            return "-"
        return col_map[col] + str(8 - row)

WHITE_BOOK_FILE = 'white.bin'
BLACK_BOOK_FILE = 'black.bin'

white_reader = None
black_reader = None

def your_board_to_chess_board(game_instance):
    """Converts the custom game state to a python-chess Board object via FEN."""
    try:
        fen_rows = []
        for r in range(8):
            empty_count = 0
            fen_row = ""
            for c in range(8):
                piece = game_instance.board[r][c]
                if piece:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    symbol = piece.name
                    fen_row += symbol.upper() if piece.colour == 'w' else symbol.lower()
                else:
                    empty_count += 1
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        piece_placement = "/".join(fen_rows)

        active_color = game_instance.turn

        castling = ""
        if game_instance.castling.get('wKR'): castling += "K"
        if game_instance.castling.get('wQR'): castling += "Q"
        if game_instance.castling.get('bKR'): castling += "k"
        if game_instance.castling.get('bQR'): castling += "q"
        if not castling: castling = "-"

        en_passant_square = "-"
        if game_instance.enpassant:
            ep_row, ep_col = game_instance.enpassant
            potential_ep_target_alg = index_to_notation((ep_row, ep_col))
            can_capture_ep = False
            pawn_rank = 4 if active_color == 'b' else 3
            capture_rank = 2 if active_color == 'b' else 5


            if ep_row == capture_rank and potential_ep_target_alg != "-":
                for dc in [-1, 1]:
                    check_col = ep_col + dc
                    if 0 <= check_col < 8:
                        pawn_to_check = game_instance.board[pawn_rank][check_col]
                        if (pawn_to_check and
                            pawn_to_check.name == 'P' and
                            pawn_to_check.colour == active_color):
                            can_capture_ep = True
                            break # Found a pawn that can capture


            if can_capture_ep:
                en_passant_square = potential_ep_target_alg


        halfmove_clock = str(game_instance.move_clock)

        fullmove_number = str(game_instance.move_count + 1)

        fen_string = f"{piece_placement} {active_color} {castling} {en_passant_square} {halfmove_clock} {fullmove_number}"


        board = chess.Board(fen=fen_string)
        return board

    except Exception as e:
        print(f"!!! ERROR generating FEN or creating board: {e}")
        print(f"!!!   Game state during error:")
        print(f"!!!     Turn: {game_instance.turn}")
        print(f"!!!     Castling: {game_instance.castling}")
        print(f"!!!     En Passant: {game_instance.enpassant}")
        print(f"!!!     Move Clock: {game_instance.move_clock}")
        print(f"!!!     Move Count: {game_instance.move_count}")
        if 'fen_string' in locals():
             print(f"!!!   Attempted FEN: {fen_string}")
        return None


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

    board = your_board_to_chess_board(game_instance)
    if board is None:
        print("Skipping book lookup")
        return None


    try:
        entry = reader.weighted_choice(board)
        if not entry:
            return None
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
    except ValueError as e:
        print(f"Error: Invalid board state detected during book lookup: {e}")
        fen_for_error = board.fen() if board else "N/A (Board object was None)"
        print(f"Board FEN during error: {fen_for_error}")
        return None
    except Exception as e:
        print(f"Unexpected error during Polyglot lookup: {e}")
        fen_for_error = board.fen() if board else "N/A (Board object was None)"
        print(f"Board FEN during error: {fen_for_error}")
        return None

load_polyglot_books()