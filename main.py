from game import Game
from board import print_board
from parser import parser

game = Game()
print_board(game.board)

while True:
    move = input("Enter move: ")
    try:
        start, end = parser(move, game)

        if game.make_move(start, end):
            print_board(game.board)
            print("Move made")
            print("Move Count - ", game.move_count)
        else:
            print("Illegal move")

    except Exception as e:
        print("Error:", e)
        break