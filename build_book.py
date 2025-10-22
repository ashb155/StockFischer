import io
import os
import glob
import pickle
import chess
import chess.pgn
from game import Game, notation_to_index
from ai import calculate_zobrist_hash


BOOK_DATA_FILES = glob.glob('chess-openings-master/chess-openings-master/[a-e].tsv')
OUTPUT_BOOK_FILE = 'book.pkl'

def chess_move_to_engine_move(chess_move):
    from_sq = chess_move.from_square
    to_sq = chess_move.to_square

    start_row, start_col = 7 - (from_sq // 8), from_sq % 8
    end_row, end_col = 7 - (to_sq // 8), to_sq % 8

    promotion = None
    if chess_move.promotion:
        promotion = chess.piece_symbol(chess_move.promotion).upper()  # Q, R, B, N

    return ((start_row, start_col), (end_row, end_col), promotion)


def build_book():
    opening_book = {}
    processed_lines = 0
    skipped_lines = 0

    if not BOOK_DATA_FILES:
        print(f"Error: No .tsv files found.")
        return

    print(f"Found {len(BOOK_DATA_FILES)} TSV files. Starting build...")

    for tsv_file in BOOK_DATA_FILES:
        print(f"  Processing {os.path.basename(tsv_file)}...")
        try:
            with open(tsv_file, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    try:

                        eco, name, pgn_string = line.strip().split('\t')


                        pgn_io = io.StringIO(pgn_string)
                        game_node = chess.pgn.read_game(pgn_io)
                        if game_node is None:
                            skipped_lines += 1
                            continue

                        engine_game = Game()

                        current_node = game_node
                        while not current_node.is_end():
                            current_hash = calculate_zobrist_hash(engine_game)

                            next_node = current_node.variations[0]
                            next_chess_move = next_node.move
                            next_engine_move = chess_move_to_engine_move(next_chess_move)

                            if current_hash not in opening_book:
                                opening_book[current_hash] = []
                            if next_engine_move not in opening_book[current_hash]:
                                opening_book[current_hash].append(next_engine_move)


                            start, end, promo = next_engine_move
                            if not engine_game.make_move(start, end, promo):
                                skipped_lines += 1
                                break

                            current_node = next_node
                        else:
                            processed_lines += 1

                    except Exception as e:
                        skipped_lines += 1
                        continue

        except Exception as e:
            print(f"    Error reading file {tsv_file}: {e}")

    print("\n--- Build Complete ---")
    print(f"Processed {processed_lines} opening lines.")
    print(f"Skipped {skipped_lines} lines (malformed or invalid).")
    print(f"Generated book with {len(opening_book)} unique positions.")

    # Saves to a fast-loading file
    print(f"Saving book to {OUTPUT_BOOK_FILE}...")
    try:
        with open(OUTPUT_BOOK_FILE, 'wb') as f_out:
            pickle.dump(opening_book, f_out)
        print("Book saved successfully")
    except Exception as e:
        print(f"Error saving book: {e}")


if __name__ == "__main__":
    build_book()