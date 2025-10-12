# File: main.py

from game import Game, move_to_algebraic
from board import print_board
from parser import parser
import os
from ai import minimax_sse
import time


def refresh():
    os.system('cls' if os.name == 'nt' else 'clear')  # for terminal


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
    print("\nStockFischer 2.0 (TT + ID Enabled)")
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
        print(f'Thinking (Max Depth: {max_depth}, Time Limit: {TIME_LIMIT}s)')

        # --- MODIFIED: Iterative Deepening Loop ---
        start_time = time.time()
        best_move_so_far = None

        # Search from depth 1 up to the chosen max_depth
        for depth in range(1, max_depth + 1):
            time_remaining = TIME_LIMIT - (time.time() - start_time)

            if time_remaining <= 0.1:  # Stop if less than 0.1s remains
                break

            # The search passes time tracking and the PV from the previous iteration
            current_best_move = minimax_sse(game, depth, float('-inf'), float('inf'), game.turn == 'w',
                                            start_time=start_time,
                                            time_limit=time_remaining,
                                            principal_variation=best_move_so_far)

            # Check if the search completed successfully (did not return None/timeout)
            if current_best_move is not None:
                best_move_so_far = current_best_move
                algebraic_move = move_to_algebraic(game, *current_best_move)
                print(f"Depth {depth} completed. Best move: {algebraic_move} (Time: {time.time() - start_time:.2f}s)")
            else:
                # If search failed (e.g., timed out mid-depth), rely on the best move from the previous depth
                print(f"Search at depth {depth} interrupted due to time limit.")
                break

        if best_move_so_far:
            (start, end, promotion) = best_move_so_far
            algebraic = move_to_algebraic(game, start, end, promotion)
            game.make_move(start, end, promotion)
            print(f"AI plays: {algebraic}")
        else:
            print("AI failed to find any move (Possible issue with first move generation).")