from game import Game
from board import print_board
from parser import parser
import os

def refresh():
    os.system('cls' if os.name == 'nt' else 'clear') #for terminal
game = Game()
while True:
    refresh()
    print("\nStockFischer 1.0")
    print_board(game.board)

    if game.state:
        print(game.state)
        if game.state in ["Checkmate", "Stalemate","Draw (Threefold repetition)","Draw (50-move rule)"]:
            break

    move = input("Enter move: ")
    try:
        start, end, promotion = parser(move, game)
        if not game.make_move(start, end, promotion):
            input("Invalid move. Press Enter...")
    except Exception as e:
        input(f"Error: {e}. Press Enter...")
