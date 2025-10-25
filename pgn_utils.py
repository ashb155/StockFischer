import chess
import chess.pgn
import random
import time


def prepare_texel_dataset(pgn_file_path, output_file_path,
                          max_positions=5000000, min_elo=2000,
                          min_move=10, max_move=40, sample_rate=0.2):
    positions_extracted = 0
    games_processed = 0
    skipped_low_elo = 0
    skipped_error = 0

    print(f"Starting PGN processing: {pgn_file_path}")
    print(f"Target positions: {max_positions}, Min Elo: {min_elo}, Move range: [{min_move}-{max_move}], Sample rate: {sample_rate:.2f}")


    try:
        with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file, \
             open(output_file_path, "w", encoding='utf-8') as output_file:

            while positions_extracted < max_positions:
                try:
                    headers = chess.pgn.read_headers(pgn_file)
                    if headers is None:
                        break # End of file
                except Exception as e:
                    skipped_error += 1
                    while True:
                        pos = pgn_file.tell()
                        line = pgn_file.readline()
                        if not line or line.startswith('[Event "'):
                             if line:
                                pgn_file.seek(pos)
                             break
                    continue

                games_processed += 1

                try:
                    white_elo = int(headers.get("WhiteElo", 0))
                    black_elo = int(headers.get("BlackElo", 0))
                    result = headers.get("Result", "*") # Need result later

                    if white_elo < min_elo or black_elo < min_elo or result not in ['1-0', '0-1', '1/2-1/2']:
                        skipped_low_elo += 1
                        continue # Skip game if low ELO or invalid result

                except ValueError:
                    skipped_low_elo += 1 # Treat non-numeric ELO as low ELO
                    continue
                except Exception as e:
                    # print(f"Warning: Error processing headers for game {games_processed}. Skipping. Error: {e}") # Uncomment for debugging
                    skipped_error += 1
                    continue

                # 2. Parse the full game now that headers are okay
                try:
                    # read_game expects the file pointer to be at the start of the game's moves
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                         # print(f"Warning: Could not read game body for game {games_processed}. Headers were: {headers}") # Uncomment for debugging
                         skipped_error += 1
                         continue

                except Exception as e:
                    # print(f"Warning: Error reading game body for game {games_processed}. Skipping. Error: {e}") # Uncomment for debugging
                    skipped_error += 1
                    continue

                # Double-check result consistency
                final_result = game.headers.get("Result", "*")
                if final_result not in ['1-0', '0-1', '1/2-1/2']:
                    skipped_error +=1
                    continue


                # 3. Extract Positions (Midgame / Endgame based on move count)
                board = game.board()
                try:
                    for i, move in enumerate(game.mainline_moves()):
                        board.push(move)
                        # Use fullmove_number from the board FEN for accuracy
                        current_move_num = board.fullmove_number

                        # Apply move range filter and sampling
                        if min_move <= current_move_num <= max_move and random.random() < sample_rate:
                            # Check for checkmate/stalemate (often excluded from tuning)
                            if not board.is_checkmate() and not board.is_stalemate():
                                fen = board.fen()
                                # Write FEN and the final game result
                                output_file.write(f"{fen} | {final_result}\n")
                                positions_extracted += 1

                        if current_move_num > max_move or positions_extracted >= max_positions:
                            break

                except Exception as e: # Catch errors during move processing
                    # print(f"Warning: Error processing moves in game {games_processed}. Skipping rest of game. Error: {e}") # Uncomment for debugging
                    skipped_error +=1
                    continue # Move to next game

                # Progress reporting
                if games_processed % 5000 == 0:
                     print(f"Progress: Games={games_processed}, Positions={positions_extracted}, Skipped (Elo/Err)={skipped_low_elo}/{skipped_error}...")

                if positions_extracted >= max_positions:
                    print(f"Target number of positions ({max_positions}) reached.")
                    break

    except FileNotFoundError:
        print(f"Error: PGN file not found at '{pgn_file_path}'")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")
        return False

    print(f"\n--- PGN Processing Summary ---")
    print(f"Total Games Processed: {games_processed}")
    print(f"Positions Extracted:   {positions_extracted}")
    print(f"Games Skipped (Low Elo/Invalid Result): {skipped_low_elo}")
    print(f"Games Skipped (Parsing Errors):       {skipped_error}")
    print(f"Output data saved to: {output_file_path}")
    print("----------------------------")
    return True

# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Configuration ---
    PGN_INPUT_FILE = 'lichesspgn.pgn'
    TUNING_OUTPUT_FILE = 'tuning_data.txt'

    TARGET_POSITIONS = 1000000  # Number of positions to aim for
    MINIMUM_ELO = 2000         # Filter games by player strength
    MIN_MOVE_NUM = 12          # Start extracting after opening phase
    MAX_MOVE_NUM = 50          # Stop extracting in deep endgame
    POSITION_SAMPLE_RATE = 0.25 # Extract ~25% of eligible positions per game

    print("Starting PGN to Texel Data Conversion...")
    start_time = time.time()

    success = prepare_texel_dataset(
        pgn_file_path=PGN_INPUT_FILE,
        output_file_path=TUNING_OUTPUT_FILE,
        max_positions=TARGET_POSITIONS,
        min_elo=MINIMUM_ELO,
        min_move=MIN_MOVE_NUM,
        max_move=MAX_MOVE_NUM,
        sample_rate=POSITION_SAMPLE_RATE
    )

    end_time = time.time()

    if success:
        print(f"Data extraction completed successfully in {end_time - start_time:.2f} seconds.")
    else:
        print("Data extraction failed.")