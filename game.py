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
        self.state=None

    def pseudo_moves(self, row, col):  # <- renamed for clarity
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
            return king_moves(self.board, row, col)

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

    def _force_move(self, start, end):  # simulation
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
            mid_row = (r1 + r2) // 2
            self.enpassant = (mid_row, c1)

        return True

    def make_move(self, start, end):
        if start is None or end is None:
            print("Illegal Move")
            return False

        r1, c1 = start
        r2, c2 = end
        piece = self.board[r1][c1]

        if not piece or piece.colour != self.turn:
            print("No valid piece to move")
            return False

        # En passant capture
        if piece.name == 'P' and self.enpassant and end == self.enpassant:
            if piece.colour == 'w':
                self.board[r2 + 1][c2] = None
            else:
                self.board[r2 - 1][c2] = None

        moves = self.get_moves(r1, c1)
        if (r2, c2) not in moves:
            print("Illegal move")
            return False

        # Normal move
        self.board[r2][c2] = piece
        self.board[r1][c1] = None

        # En passant target square
        self.enpassant = None
        if piece.name == 'P' and abs(r2 - r1) == 2:
            mid_row = (r1 + r2) // 2
            self.enpassant = (mid_row, c1)

        self.history.append(((r1, c1), (r2, c2), piece))
        self.turn = 'b' if self.turn == 'w' else 'w'
        self.move_count += 0.5

        if self.is_checkmate(self.turn):
            self.state="Checkmate"
        elif self.is_stalemate(self.turn):
            self.state="Stalemate"
        elif self.is_check(self.turn):
            self.state="Check"
        else:
            self.state=None
        return True

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


        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.colour != colour:
                    moves = self.pseudo_moves(r, c)
                    if king_pos in moves:
                        return True
        return False

    def is_checkmate(self, colour):
        if not self.is_check(colour):
            return False

        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.colour == colour:
                    moves = self.get_moves(r, c)
                    if moves:  # if any legal move exists, not checkmate
                        return False
        return True

    def is_stalemate(self, colour):
        if self.is_check(colour):
            return False

        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.colour == colour:
                    moves = self.get_moves(r, c)
                    if moves:  # if any legal move exists, not stalemate
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
