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
    for i,row in enumerate(board):
        print(8-i," ".join(str(piece) if piece else '--' for piece in row))
    print("  a  b  c  d  e  f  g  h")

