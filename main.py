from game import Game
from board import print_board
from parser import parser
import os


def refresh():
        os.system('cls' if os.name == 'nt' else 'clear')

game = Game()

while True:
    refresh()
    print('''
     StockFischer 1.0''')
    print_board(game.board)

    move = input("Enter move: ")
    try:
        start, end = parser(move, game)

        if game.make_move(start, end):
            pass  # board will redraw on next loop
        else:
            print("Illegal move")
            input("Press Enter...")  # pause before redraw

    except Exception as e:
        print("Error:", e)
        break
