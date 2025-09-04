import copy
from board import create_board
from moves import pawn_moves, rook_moves, bishop_moves, knight_moves, queen_moves, king_moves

class Game:
    def __init__(self):

        self.board = create_board()
        self.turn = 'w'
        self.move_count = 0
        self.history = []
        self.enpassant = None
        self.state = None
        self.castling = {'wKR': True, 'wQR': True, 'bKR': True, 'bQR': True}


    def pseudo_moves(self, row, col):
        piece = self.board[row][col]
        if not piece:
            return []

        if piece.name == 'P':
            return pawn_moves(self.board, row, col, self.enpassant)
        elif piece.name == 'N':
            return knight_moves(self.board, row, col)
        elif piece.name == 'B':
            return bishop_moves(self.board, row, col)
        elif piece.name == 'R':
            return rook_moves(self.board, row, col)
        elif piece.name == 'Q':
            return queen_moves(self.board, row, col)
        elif piece.name == 'K':
            moves = king_moves(self.board, row, col)
            moves.extend(self._castling_moves(piece, row, col))
            return moves
        return []

    #castling
    def _castling_moves(self, king, row, col):
        moves = []
        if king.colour == 'w' and row == 7 and col == 4:
            if self.castling['wKR'] and self.board[7][5] is None and self.board[7][6] is None:
                if not self.square_attacked((7, 4), 'b') and not self.square_attacked((7, 5), 'b') and not self.square_attacked((7, 6), 'b'):
                    moves.append((7, 6))
            if self.castling['wQR'] and self.board[7][3] is None and self.board[7][2] is None and self.board[7][1] is None:
                if not self.square_attacked((7, 4), 'b') and not self.square_attacked((7, 3), 'b') and not self.square_attacked((7, 2), 'b'):
                    moves.append((7, 2))
        elif king.colour == 'b' and row == 0 and col == 4:
            if self.castling['bKR'] and self.board[0][5] is None and self.board[0][6] is None:
                if not self.square_attacked((0, 4), 'w') and not self.square_attacked((0, 5), 'w') and not self.square_attacked((0, 6), 'w'):
                    moves.append((0, 6))
            if self.castling['bQR'] and self.board[0][3] is None and self.board[0][2] is None and self.board[0][1] is None:
                if not self.square_attacked((0, 4), 'w') and not self.square_attacked((0, 3), 'w') and not self.square_attacked((0, 2), 'w'):
                    moves.append((0, 2))
        return moves

    #attack moves
    def attack_moves(self, row, col):
        piece = self.board[row][col]
        if not piece:
            return []
        if piece.name == 'P':
            direction = -1 if piece.colour == 'w' else 1
            moves = []
            nrow = row + direction
            if 0 <= nrow < 8:
                if col - 1 >= 0:
                    moves.append((nrow, col - 1))
                if col + 1 < 8:
                    moves.append((nrow, col + 1))
            return moves
        elif piece.name == 'N':
            return knight_moves(self.board, row, col)
        elif piece.name == 'B':
            return bishop_moves(self.board, row, col)
        elif piece.name == 'R':
            return rook_moves(self.board, row, col)
        elif piece.name == 'Q':
            return queen_moves(self.board, row, col)
        elif piece.name == 'K':
            return king_moves(self.board, row, col)
        return []

    #legal moves
    def get_moves(self, row, col):
        piece = self.board[row][col]
        moves = self.pseudo_moves(row, col)
        legal_moves = []
        for r, c in moves:
            copy_game = copy.deepcopy(self)
            if copy_game._force_move((row, col), (r, c)):
                if not copy_game.is_check(piece.colour):
                    legal_moves.append((r, c))
        return legal_moves


    def _force_move(self, start, end):
        r1, c1 = start
        r2, c2 = end
        piece = self.board[r1][c1]
        if not piece:
            return False

        if piece.name == 'P' and self.enpassant and end == self.enpassant:
            if piece.colour == 'w':
                self.board[r2 + 1][c2] = None
            else:
                self.board[r2 - 1][c2] = None

        self.board[r2][c2] = piece
        self.board[r1][c1] = None

        self.enpassant = None
        if piece.name == 'P' and abs(r2 - r1) == 2:
            self.enpassant = ((r1 + r2) // 2, c1)

        return True

    #makes moves and updates game state
    def make_move(self, start, end):
        if start is None or end is None:
            return False

        r1, c1 = start
        r2, c2 = end
        piece = self.board[r1][c1]

        if not piece or piece.colour != self.turn:
            return False

        moves = self.get_moves(r1, c1)
        if (r2, c2) not in moves:
            return False

        if piece.name == 'K' and abs(c2 - c1) == 2:
            if c2 == 6:
                self.board[r2][5] = self.board[r2][7]
                self.board[r2][7] = None
            elif c2 == 2:
                self.board[r2][3] = self.board[r2][0]
                self.board[r2][0] = None

        self.board[r2][c2] = piece
        self.board[r1][c1] = None

        if piece.name == 'K':
            if piece.colour == 'w':
                self.castling['wKR'] = False
                self.castling['wQR'] = False
            else:
                self.castling['bKR'] = False
                self.castling['bQR'] = False
        elif piece.name == 'R':
            if piece.colour == 'w':
                if (r1, c1) == (7, 0):
                    self.castling['wQR'] = False
                elif (r1, c1) == (7, 7):
                    self.castling['wKR'] = False
            else:
                if (r1, c1) == (0, 0):
                    self.castling['bQR'] = False
                elif (r1, c1) == (0, 7):
                    self.castling['bKR'] = False

        self.enpassant = None
        if piece.name == 'P' and abs(r2 - r1) == 2:
            self.enpassant = ((r1 + r2) // 2, c1)

        self.history.append(((r1, c1), (r2, c2), piece))
        self.turn = 'b' if self.turn == 'w' else 'w'
        self.move_count += 0.5

        if self.is_checkmate(self.turn):
            self.state = "Checkmate"
        elif self.is_stalemate(self.turn):
            self.state = "Stalemate"
        elif self.is_check(self.turn):
            self.state = "Check"
        else:
            self.state = None
        return True

    # Check is square attacked
    def square_attacked(self, square, colour):
        r, c = square
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece.colour == colour:
                    if square in self.attack_moves(row, col):
                        return True
        return False

    #Checks check
    def is_check(self, colour):
        king_pos = None
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.name == 'K' and piece.colour == colour:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        if not king_pos:
            return False
        return self.square_attacked(king_pos, 'b' if colour == 'w' else 'w')

    # Checks checkmate
    def is_checkmate(self, colour):
        if not self.is_check(colour):
            return False
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.colour == colour:
                    if self.get_moves(r, c):
                        return False
        return True

    # Checks stalemate
    def is_stalemate(self, colour):
        if self.is_check(colour):
            return False
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.colour == colour:
                    if self.get_moves(r, c):
                        return False
        return True


    def copy(self):
        return copy.deepcopy(self)



def notation_to_index(move):
    col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3,
               'e': 4, 'f': 5, 'g': 6, 'h': 7}
    col = col_map.get(move[0].lower())
    if col is None:
        return None
    try:
        row = 8 - int(move[1])
    except ValueError:
        return None
    return (row, col)


def index_to_notation(pos):
    col_map = 'abcdefgh'
    row, col = pos
    return col_map[col] + str(8 - row)
