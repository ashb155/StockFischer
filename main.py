from game import Game, move_to_algebraic, index_to_notation # Added index_to_notation
from board import print_board
from parser import parser
import os
from ai import minimax_sse
import time
import random
from opening_book import get_polyglot_book_move

# --- Search Depth Mapping ---
difficulty = {'easy': 2, 'medium': 3, "hard": 4}

# --- time limit mapping by Difficulty ---
time_limits = {'easy': 2.0, 'medium': 5.0, "hard": 10.0}

level = input('Choose level - Easy / Medium / Hard: ').strip().lower()

max_depth = difficulty.get(level, 4)
TIME_LIMIT = time_limits.get(level, 10.0)

colour = input('Choose colour - W / B: ').strip().lower()
if colour not in ['w', 'b']:
    print("Invalid colour, defaulting to white.")
    colour = 'w'


def refresh():
    os.system('cls' if os.name == 'nt' else 'clear')

game = Game()
while True:
    refresh()
    print("\nStockFischer 2.0")
    print_board(game.board)
    print(f"Turn: {'White' if game.turn == 'w' else 'Black'}")

    if game.state:
        print(f"Game State: {game.state}")
        if game.state in ["Checkmate", "Stalemate", "Draw (Threefold repetition)", "Draw (50-move rule)"]:
            print("--- Game Over ---")
            break

    if game.turn == colour:
        move_str = input("Enter your move")
        if move_str.lower() in ['quit', 'exit']:
            break
        try:
            start, end, promotion = parser(move_str, game)
            if start is None or end is None:
                 input("Invalid or ambiguous move format. Press Enter...")
                 continue
            if not game.make_move(start, end, promotion):
                input("Illegal move. Press Enter...")
        except Exception as e:
            input(f"Error parsing move '{move_str}': {e}. Press Enter...")
    else:
        print("AI thinking...")
        book_move = None
        if game.move_count < 20:
             book_move = get_polyglot_book_move(game)
        else:
             print("Book depth exceeded (move > 10), proceeding to search.")

        valid_book_move_played = False
        if book_move:
            (start, end, promotion) = book_move
            piece = game.board[start[0]][start[1]]
            if piece and piece.colour == game.turn:
                try:
                    legal_moves_for_piece = game.get_moves(start[0], start[1])
                except Exception as e:
                    print(f"Error getting legal moves for book validation: {e}")
                    legal_moves_for_piece = [] # Assume invalid if error occurs

                if end in legal_moves_for_piece:
                    algebraic = move_to_algebraic(game, start, end, promotion)
                    print(f"AI plays {algebraic}")
                    if not game.make_move(start, end, promotion):
                         print(f"ERROR: Failed to make validated book move {algebraic}. Searching instead.")
                         valid_book_move_played = False # Force search
                    else:
                         valid_book_move_played = True
                         time.sleep(0.5)
                else:
                    start_notation = index_to_notation(start)
                    end_notation = index_to_notation(end)
                    print(f"Warning: Book move {start_notation}{end_notation} invalid according to engine (target square not legal). Searching instead.")
            else:
                 start_notation = index_to_notation(start)
                 print(f"Warning: Book move from {start_notation} invalid according to engine (piece mismatch or empty start). Searching instead.")


        if not valid_book_move_played:
            print(f'AI searching... (Max Depth: {max_depth}, Time Limit: {TIME_LIMIT:.1f}s)')
            start_time = time.time()
            best_move_so_far = None
            best_score_so_far = None

            # Iterative Deepening Loop
            for depth in range(1, max_depth + 1):
                time_remaining = TIME_LIMIT - (time.time() - start_time)

                min_time_needed = 0.2
                if time_remaining < min_time_needed:
                    print(f"Time limit ({time_remaining:.2f}s) too low before starting depth {depth}.")
                    break

                print(f"Searching depth {depth}...")
                pv_move_tuple = best_move_so_far

                search_result = minimax_sse(game, depth, float('-inf'), float('inf'), game.turn == 'w',
                                            original_depth=depth, # Pass original_depth correctly
                                            start_time=start_time,
                                            time_limit=time_remaining,
                                            principal_variation=pv_move_tuple) # Pass PV move

                if search_result is None:
                    print(f"Search at depth {depth} interrupted or failed. Using result from previous depth (if any).")
                    break


                if isinstance(search_result, tuple) and len(search_result) == 3:
                     current_best_move = search_result
                     best_move_so_far = current_best_move # Update the best move found so far
                     algebraic_move = move_to_algebraic(game, *current_best_move)
                     print(f"Depth {depth} completed. Best move found: {algebraic_move} (Time: {time.time() - start_time:.2f}s)")

                     if TIME_LIMIT - (time.time() - start_time) < min_time_needed:
                          print("Time limit reached after completing depth.")
                          break
                else:

                    print(f"Warning: Search at depth {depth} did not return a valid move tuple.")
                    if not best_move_so_far:
                        break


            if best_move_so_far:
                (start, end, promotion) = best_move_so_far
                algebraic = move_to_algebraic(game, start, end, promotion)
                print(f"AI plays: {algebraic}")
                if not game.make_move(start, end, promotion):
                     print(f"ERROR: AI failed to make its calculated move {algebraic}!")
                     break
            else:

                print("AI failed to find a move via search. Making a random legal move.")
                fallback_move = None
                possible_starts = []

                for r in range(8):
                    for c in range(8):
                        piece = game.board[r][c]
                        if piece and piece.colour == game.turn:
                           possible_starts.append((r,c))

                if not possible_starts:
                     print("AI cannot find any pieces to move (Error or Game End?).")
                     break

                random.shuffle(possible_starts)

                for r_start, c_start in possible_starts:
                   try:
                       moves = game.get_moves(r_start, c_start)
                       if moves:
                           end_move = random.choice(moves)
                           promo = 'Q' if game.board[r_start][c_start].name == 'P' and (end_move[0] == 0 or end_move[0] == 7) else None
                           fallback_move = ((r_start, c_start), end_move, promo)
                           break
                   except Exception as e:
                       print(f"Error getting moves for fallback from {(r_start, c_start)}: {e}")
                       continue # Try next piece

                if fallback_move:
                    (start, end, promotion) = fallback_move
                    algebraic = move_to_algebraic(game, start, end, promotion)
                    print(f"AI plays: {algebraic}")
                    if not game.make_move(start, end, promotion):
                         print(f"ERROR: AI failed to make fallback move {algebraic}!")
                         break
                else:
                    print("AI cannot find any legal moves (Checkmate/Stalemate?).")
                    if game.is_checkmate(): game.state = "Checkmate"
                    elif game.is_stalemate(): game.state = "Stalemate"
                    print(f"Final State: {game.state}")
                    break


