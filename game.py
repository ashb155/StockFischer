import copy
from board import create_board
from moves import pawn_moves, rook_moves, bishop_moves, knight_moves, queen_moves, king_moves
from pieces import Piece


# game class
class Game:
    def __init__(self):

        self.board = create_board()
        self.turn = 'w'
        self.move_count = 0
        self.history = []
        self.enpassant = None
        self.state = None
        self.castling = {'wKR': True, 'wQR': True, 'bKR': True, 'bQR': True}

        self.move_clock = 0
        self.position_count = {}
        self.position_count[self.board_key()] = 1

    def pseudo_moves(self, row, col):
        piece = self.board[row][col]
        if not piece: return []
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
            moves.extend(self.castling_moves(piece, row, col))
            return moves
        return []

    def castling_moves(self, king, row, col):
        moves = []
        if king.colour == 'w' and row == 7 and col == 4:
            if self.castling['wKR'] and self.board[7][5] is None and self.board[7][6] is None:
                if not self.square_attacked((7, 4), 'b') and not self.square_attacked((7, 5),
                                                                                      'b') and not self.square_attacked(
                        (7, 6), 'b'):
                    moves.append((7, 6))
            if self.castling['wQR'] and self.board[7][3] is None and self.board[7][2] is None and self.board[7][
                1] is None:
                if not self.square_attacked((7, 4), 'b') and not self.square_attacked((7, 3),
                                                                                      'b') and not self.square_attacked(
                        (7, 2), 'b'):
                    moves.append((7, 2))
        elif king.colour == 'b' and row == 0 and col == 4:
            if self.castling['bKR'] and self.board[0][5] is None and self.board[0][6] is None:
                if not self.square_attacked((0, 4), 'w') and not self.square_attacked((0, 5),
                                                                                      'w') and not self.square_attacked(
                        (0, 6), 'w'):
                    moves.append((0, 6))
            if self.castling['bQR'] and self.board[0][3] is None and self.board[0][2] is None and self.board[0][
                1] is None:
                if not self.square_attacked((0, 4), 'w') and not self.square_attacked((0, 3),
                                                                                      'w') and not self.square_attacked(
                        (0, 2), 'w'):
                    moves.append((0, 2))
        return moves

    def attack_moves(self, row, col):
        piece = self.board[row][col]
        if not piece: return []
        if piece.name == 'P':
            direction = -1 if piece.colour == 'w' else 1
            moves = []
            nrow = row + direction
            if 0 <= nrow < 8:
                if col - 1 >= 0: moves.append((nrow, col - 1))
                if col + 1 < 8: moves.append((nrow, col + 1))
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

    # --- MODIFIED: Uses fast move/unmake system ---
    def get_moves(self, row, col):
        piece = self.board[row][col]
        if not piece or piece.colour != self.turn: return []

        moves = self.pseudo_moves(row, col)
        legal_moves = []

        for r, c in moves:
            # Store move details *before* forcing the move (for unmake)
            move_details = self.force_move_and_save((row, col), (r, c))

            # Check for legality: A move is legal if the king is NOT in check after the move
            if not self.is_check(piece.colour):
                legal_moves.append((r, c))

            # Unmake the move (restore the board state)
            self.unmake_move((row, col), (r, c), move_details)

        return legal_moves

    # --- NEW: force_move_and_save (For internal use in get_moves) ---
    def force_move_and_save(self, start, end, promotion='Q'):
        r1, c1 = start;
        r2, c2 = end
        piece = self.board[r1][c1]

        old_castling = self.castling.copy()
        old_enpassant = self.enpassant
        captured = self.board[r2][c2]
        captured_pos = None

        if piece.name == 'P' and self.enpassant and end == self.enpassant:
            captured_pos = (r2 + 1, c2) if piece.colour == 'w' else (r2 - 1, c2)
            captured = self.board[captured_pos[0]][captured_pos[1]]
            self.board[captured_pos[0]][captured_pos[1]] = None

        self.board[r2][c2] = piece
        self.board[r1][c1] = None

        if piece.name == 'K' and abs(c2 - c1) == 2:
            if c2 == 6:  # Kingside
                rook = self.board[r2][7];
                self.board[r2][5] = rook;
                self.board[r2][7] = None
                rook_move = ((r2, 7), (r2, 5), rook)
            elif c2 == 2:  # Queenside
                rook = self.board[r2][0];
                self.board[r2][3] = rook;
                self.board[r2][0] = None
                rook_move = ((r2, 0), (r2, 3), rook)
            else:
                rook_move = None
        else:
            rook_move = None

        promoted_to = None
        if piece.name == 'P':
            if (piece.colour == 'w' and r2 == 0) or (piece.colour == 'b' and r2 == 7):
                self.board[r2][c2] = Piece(piece.colour, promotion)
                promoted_to = self.board[r2][c2]

        if piece.name == 'K':
            if piece.colour == 'w':
                self.castling['wKR'] = False; self.castling['wQR'] = False
            else:
                self.castling['bKR'] = False; self.castling['bQR'] = False
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

        new_enpassant = None
        if piece.name == 'P' and abs(r2 - r1) == 2:
            new_enpassant = ((r1 + r2) // 2, c1)
        self.enpassant = new_enpassant

        return (piece, captured, captured_pos, rook_move, promoted_to, old_enpassant, old_castling)

    # --- NEW: unmake_move (Restores state to pre-move quickly) ---
    def unmake_move(self, start, end, move_details):
        (piece, captured, captured_pos, rook_move, promoted_to, old_enpassant, old_castling) = move_details
        r1, c1 = start;
        r2, c2 = end

        self.board[r1][c1] = piece
        self.board[r2][c2] = None

        if captured:
            restore_pos = captured_pos if captured_pos else end
            self.board[restore_pos[0]][restore_pos[1]] = captured

        if rook_move:
            r_start, r_end, rook = rook_move
            self.board[r_start[0]][r_start[1]] = rook
            self.board[r_end[0]][r_end[1]] = None

        self.enpassant = old_enpassant
        self.castling = old_castling

    # --- MODIFIED: Master move executor (updated to be robust) ---
    def _force_move(self, start, end, promotion='Q'):
        r1, c1 = start;
        r2, c2 = end;
        piece = self.board[r1][c1]

        if not piece: return False

        old_castling = self.castling.copy()
        old_enpassant = self.enpassant
        old_move_clock = self.move_clock
        old_turn = self.turn

        captured = self.board[r2][c2];
        captured_pos = None
        if piece.name == 'P' and self.enpassant and end == self.enpassant:
            captured_pos = (r2 + 1, c2) if piece.colour == 'w' else (r2 - 1, c2)
            captured = self.board[captured_pos[0]][captured_pos[1]]
            self.board[captured_pos[0]][captured_pos[1]] = None

        self.board[r2][c2] = piece;
        self.board[r1][c1] = None

        rook_move = None
        if piece.name == 'K' and abs(c2 - c1) == 2:
            if c2 == 6:
                rook = self.board[r2][7];
                self.board[r2][5] = rook;
                self.board[r2][7] = None
                rook_move = ((r2, 7), (r2, 5), rook)
            elif c2 == 2:
                rook = self.board[r2][0];
                self.board[r2][3] = rook;
                self.board[r2][0] = None
                rook_move = ((r2, 0), (r2, 3), rook)

        promoted_to = None
        if piece.name == 'P':
            if (piece.colour == 'w' and r2 == 0) or (piece.colour == 'b' and r2 == 7):
                self.board[r2][c2] = Piece(piece.colour, promotion)
                promoted_to = self.board[r2][c2]

        if piece.name == 'K':
            if piece.colour == 'w':
                self.castling['wKR'] = False; self.castling['wQR'] = False
            else:
                self.castling['bKR'] = False; self.castling['bQR'] = False
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

        new_enpassant = None
        if piece.name == 'P' and abs(r2 - r1) == 2: new_enpassant = ((r1 + r2) // 2, c1)
        self.enpassant = new_enpassant

        return (piece, captured, captured_pos, rook_move, promoted_to, old_enpassant, old_castling, old_move_clock,
                old_turn)

    def make_move(self, start, end, promotion='Q'):
        if start is None or end is None: return False
        r1, c1 = start;
        piece = self.board[r1][c1]
        if not piece or piece.colour != self.turn: return False

        moves = self.get_moves(r1, c1)
        if end not in moves: return False

        move_details = self._force_move(start, end, promotion)
        (moved_piece, captured_piece, captured_pos, rook_move, promoted_to, old_enpassant, old_castling, old_move_clock,
         old_turn) = move_details

        history_entry = (start, end, moved_piece, captured_piece, captured_pos, rook_move, promoted_to,
                         old_enpassant, old_castling, old_move_clock, old_turn)
        self.history.append(history_entry)

        self.turn = 'b' if self.turn == 'w' else 'w'
        if self.turn == 'w': self.move_count += 1

        if moved_piece.name == 'P' or captured_piece:
            self.move_clock = 0
        else:
            self.move_clock += 1

        key = self.board_key()
        self.position_count[key] = self.position_count.get(key, 0) + 1

        if self.is_checkmate(self.turn):
            self.state = "Checkmate"
        elif self.is_stalemate(self.turn):
            self.state = "Stalemate"
        elif self.move_clock >= 100:
            self.state = "Draw (50-move rule)"
        elif self.position_count.get(key, 0) >= 3:
            self.state = "Draw (Threefold repetition)"
        elif self.is_check(self.turn):
            self.state = "Check"
        else:
            self.state = None
        return True

    def full_unmake_move(self):
        if not self.history: return False
        (start, end, piece, captured, captured_pos, rook_move, promoted_to,
         old_enpassant, old_castling, old_move_clock, old_turn) = self.history.pop()

        r1, c1 = start;
        r2, c2 = end
        self.board[r1][c1] = piece
        if promoted_to:
            self.board[r2][c2] = None
        else:
            self.board[r2][c2] = None

        if captured:
            restore_pos = captured_pos if captured_pos else end
            self.board[restore_pos[0]][restore_pos[1]] = captured

        if rook_move:
            r_start, r_end, rook = rook_move
            self.board[r_start[0]][r_start[1]] = rook
            self.board[r_end[0]][r_end[1]] = None

        self.enpassant = old_enpassant
        self.castling = old_castling
        self.move_clock = old_move_clock
        self.turn = old_turn
        if self.turn == 'b': self.move_count -= 1

        key = self.board_key()
        self.position_count[key] -= 1
        if self.position_count[key] == 0: del self.position_count[key]

        self.state = None
        if self.is_check(self.turn): self.state = "Check"
        return True

    def square_attacked(self, square, colour):
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece.colour == colour:
                    if square in self.attack_moves(row, col): return True
        return False

    def is_check(self, colour):
        king_pos = None
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.name == 'K' and piece.colour == colour:
                    king_pos = (r, c);
                    break
            if king_pos: break
        if not king_pos: return False
        return self.square_attacked(king_pos, 'b' if colour == 'w' else 'w')

    def is_checkmate(self, colour):
        if not self.is_check(colour): return False
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.colour == colour:
                    if self.get_moves(r, c): return False
        return True

    def is_stalemate(self, colour):
        if self.is_check(colour): return False
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.colour == colour:
                    if self.get_moves(r, c): return False
        return True

    def copy(self):
        return copy.deepcopy(self)

    def light_copy(self):
        copy = Game()
        copy.board = [[piece for piece in row] for row in self.board]
        copy.turn = self.turn
        copy.enpassant = self.enpassant
        copy.castling = self.castling.copy()
        copy.move_count = self.move_count
        copy.move_clock = self.move_clock
        copy.state = self.state
        return copy

    def board_key(self):
        rows = []
        for r in range(8):
            row = []
            for c in range(8):
                p = self.board[r][c]
                row.append(p.colour + p.name if p else ".")
            rows.append("".join(row))
        return "/".join(rows) + " " + self.turn


def notation_to_index(move):
    col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3,
               'e': 4, 'f': 5, 'g': 6, 'h': 7}
    col = col_map.get(move[0].lower())
    if col is None: return None
    try:
        row = 8 - int(move[1])
    except ValueError:
        return None
    return (row, col)


def index_to_notation(pos):
    col_map = 'abcdefgh'
    row, col = pos
    return col_map[col] + str(8 - row)


def move_to_algebraic(game, start, end, promotion=None):
    piece = game.board[start[0]][start[1]]
    target_piece = game.board[end[0]][end[1]]

    if piece.name == 'K' and abs(start[1] - end[1]) == 2:
        return 'O-O' if end[1] == 6 else 'O-O-O'

    move_str = ''
    if piece.name != 'P': move_str += piece.name

    if target_piece or (piece.name == 'P' and start[1] != end[1]):
        if piece.name == 'P': move_str += index_to_notation(start)[0]
        move_str += 'x'

    move_str += index_to_notation(end)

    if promotion: move_str += f"={promotion}"

    return move_str