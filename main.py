from game import Game, notation_to_index
from board import print_board

game = Game()
print_board(game.board)

while True:
    start = notation_to_index(input("Enter start square "))
    end = notation_to_index(input("Enter end square:"))

    if game.make_move(start, end):
        print_board(game.board)
        print('Move made')
    else:
        print("Invalid")
        break