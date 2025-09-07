from game import Game, move_to_algebraic
from board import print_board
from parser import parser
import os
from ai import minimax_sse
import time


def refresh():
    os.system('cls' if os.name == 'nt' else 'clear') #for terminal

difficulty={'easy':2,'medium':3,"hard":5}

level=input('Choose level - Easy / Medium / Hard').strip().lower()
depth=difficulty.get(level,3)
colour=input('Choose colour - W / B').strip().lower()

game = Game()
while True:
    refresh()
    print("\nStockFischer 2.0")
    print_board(game.board)

    if game.state:
        print(game.state)
        if game.state in ["Checkmate", "Stalemate","Draw (Threefold repetition)","Draw (50-move rule)"]:
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
        print('Thinking')
        time.sleep(3)
        best_move=minimax_sse(game, depth,  float('-inf'), float('inf'), game.turn == 'w')
        if best_move:
            (start,end,promotion)=best_move
            algebraic = move_to_algebraic(game, start, end, promotion)
            game.make_move(start,end,promotion)
            print(f"AI plays: {algebraic}")

