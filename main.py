
from game import Game, move_to_algebraic
from board import print_board
from parser import parser
import os
from ai import minimax_sse, calculate_zobrist_hash
import time

from opening_book import get_book_move


def refresh():
    os.system('cls' if os.name == 'nt' else 'clear')


# --- Search Depth Mapping ---
difficulty = {'easy': 2, 'medium': 3, "hard": 4}

# --- time limit mapping by Difficulty ---
time_limits = {'easy': 2.0, 'medium': 5.0, "hard": 10.0}

level = input('Choose level - Easy / Medium / Hard').strip().lower()

# Get max depth and corresponding time limit based on the chosen level
max_depth = difficulty.get(level, 4)
TIME_LIMIT = time_limits.get(level, 10.0)

colour = input('Choose colour - W / B').strip().lower()

game = Game()
while True:
    refresh()
    print("\nStockFischer 2.0 (PVS/LMR + Ext. Book)")
    print_board(game.board)

    if game.state:
        print(game.state)
        if game.state in ["Checkmate", "Stalemate", "Draw (Threefold repetition)", "Draw (50-move rule)"]:
            break

    if game.turn == colour:
        move = input("Enter move: ")
        try:
            start, end, promotion = parser(move, game)
            if not game.make_move(start, end, promotion):
                input("Invalid move. Press Enter...")
        except Exception as e:
            input(f"Error: {e}. Press Enter...")
    else:
        current_hash = calculate_zobrist_hash(game)
        book_move = get_book_move(current_hash)

        if book_move:
            (start, end, promotion) = book_move
            algebraic = move_to_algebraic(game, start, end, promotion)
            game.make_move(start, end, promotion)
            print(f"AI plays (from book): {algebraic}")
            time.sleep(0.5)

        else:
            print(f'Thinking (Max Depth: {max_depth}, Time Limit: {TIME_LIMIT}s)')

            start_time = time.time()
            best_move_so_far = None

            # Search from depth 1 up to the chosen max_depth
            for depth in range(1, max_depth + 1):
                time_remaining = TIME_LIMIT - (time.time() - start_time)

                if time_remaining <= 0.1:  # Stop if less than 0.1s remains
                    break

                current_best_move = minimax_sse(game, depth, float('-inf'), float('inf'), game.turn == 'w',
                                                start_time=start_time,
                                                time_limit=time_remaining,
                                                principal_variation=best_move_so_far)

                if current_best_move is not None:
                    best_move_so_far = current_best_move
                    algebraic_move = move_to_algebraic(game, *current_best_move)
                    print(
                        f"Depth {depth} completed. Best move: {algebraic_move} (Time: {time.time() - start_time:.2f}s)")
                else:
                    print(f"Search at depth {depth} interrupted due to time limit.")
                    break

            if best_move_so_far:
                (start, end, promotion) = best_move_so_far
                algebraic = move_to_algebraic(game, start, end, promotion)
                game.make_move(start, end, promotion)
                print(f"AI plays: {algebraic}")
            else:
                print("AI failed to find any move .")