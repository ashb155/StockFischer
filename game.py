from board import create_board
from moves import pawn_moves, rook_moves, bishop_moves, knight_moves,queen_moves,king_moves
class Game:
    def __init__(self):
        self.board = create_board()
        self.turn='w'
        self.move_count=0
        self.history = []
        self.enpassant=None

    def get_moves(self,row,col):
        piece = self.board[row][col]
        if not piece:
            return []
        if piece.name== 'P':
            return pawn_moves(self.board,row,col)
        elif piece.name== 'N':
            return knight_moves(self.board,row,col)
        elif piece.name== 'B':
            return bishop_moves(self.board,row,col)
        elif piece.name== 'R':
            return rook_moves(self.board,row,col)
        elif piece.name== 'Q':
            return queen_moves(self.board,row,col)
        elif piece.name== 'K':
            return king_moves(self.board,row,col)

    def make_move(self,start,end):
        r1,c1 = start
        r2,c2 = end
        piece= self.board[r1][c1]


        if not piece or piece.colour != self.turn:
            return False

        if piece.name == 'P' and self.enpassant and end == self.enpassant: #enables en passant captures
            if piece.colour=='w':
                self.board[r2+1][c2] = None
            else:
                self.board[r2-1][c2] = None

        moves=self.get_moves(r1,c1)
        if (r2,c2) not in moves:
            print('Illegal')
            return False

        self.board[r2][c2]=piece
        self.board[r1][c1]=None

        self.enpassant = None
        if piece.name == 'P' and abs(r2 - r1) == 2: #adds possible en passant targets
            mid_row = (r1 + r2) // 2
            self.enpassant = (mid_row, c1)

        self.history.append(((r1,c1),(r2,c2),piece))

        self.turn='b' if self.turn== 'w' else 'w'
        self.move_count+=1
        return True

def notation_to_index(move):
    col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3,
               'e': 4, 'f': 5, 'g': 6, 'h': 7}
    col = col_map[move[0].lower()]
    row = 8 - int(move[1])
    return (row, col)

def index_to_notation(pos):
    col_map = 'abcdefgh'
    row, col = pos
    return col_map[col] + str(8 - row)