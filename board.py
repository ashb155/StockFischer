from pieces import Piece
from moves import pawn_moves

def create_board():
    empty=None
    board = [
        [Piece('b', 'R'), Piece('b', 'N'), Piece('b', 'B'), Piece('b', 'Q'), Piece('b', 'K'), Piece('b', 'B'),
         Piece('b', 'N'), Piece('b', 'R')],
        [Piece('b', 'P')] * 8,
        [empty] * 8,
        [empty] * 8,
        [empty] * 8,
        [empty] * 8,
        [Piece('w', 'P')] * 8,
        [Piece('w', 'R'), Piece('w', 'N'), Piece('w', 'B'), Piece('w', 'Q'), Piece('w', 'K'), Piece('w', 'B'),
         Piece('w', 'N'), Piece('w', 'R')]
    ]
    return board

def print_board(board):
    for row in board:
        print(" ".join(str(piece) if piece else '--' for piece in row))

board=create_board()
print_board(board)

pawn_position=[(6,i) for i in range(8)]
for row,col in pawn_position:
    moves=pawn_moves(board,row,col)
    print(f"Pawn at ({row},{col}) can move to : {moves}")