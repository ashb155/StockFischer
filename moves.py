from pieces import Piece

def pawn_moves(board,row,col,enpassant=None):
    piece=board[row][col]
    moves=[]

    if not piece or piece.name != 'P':
        return moves

    direction = -1 if piece.colour=='w' else 1  #white moves to lower index while vice versa for black
    if 0 <= row+direction < 8 and board[row+direction][col] is None:
        moves.append((row+direction,col))

        start_row = 6 if piece.colour == 'w' else 1  #pawn 2 places for 1st move
        if row == start_row and board[row + 2 * direction][col] is None:
            moves.append((row+2*direction,col))


    for diagonal in [-1,1]:     #capturing
        ncol= col + diagonal
        nrow= row + direction
        if 0 <= nrow < 8 and 0 <= ncol < 8:
            target = board[nrow][ncol]
            if target and target.colour != piece.colour:
                moves.append((nrow,ncol))
            if enpassant and (nrow, ncol) == enpassant:
                moves.append((nrow, ncol))

    return moves

def rook_moves(board,row,col):
    piece=board[row][col]
    moves=[]
    if not piece or piece.name!='R':
        return moves

    directions=[(1,0),(-1,0),(0,1),(0,-1)]
    for dr,dc in directions:
        r,c = row+dr, col+dc
        while 0 <= r < 8 and 0 <= c < 8:
            target = board[r][c]
            if target is None:  #empty square
                moves.append((r,c))
            elif target.colour != piece.colour: #if same colour then occupied square
                moves.append((r,c))
                break
            else:
                break
            r+=dr
            c+=dc
    return moves

def knight_moves(board,row,col):
    piece=board[row][col]
    moves=[]
    if not piece or piece.name !='N':
        return moves

    directions=[(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    for dr,dc in directions:
        r,c = row+dr, col+dc
        if 0 <= r < 8 and 0 <= c < 8:
            target = board[r][c]
            if target is None or target.colour != piece.colour:
                moves.append((r, c))
    return moves

def bishop_moves(board,row,col):
    piece=board[row][col]
    moves=[]
    if not piece or piece.name != 'B':
        return moves

    directions = [(1,1),(1,-1),(-1,1),(-1,-1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            target = board[r][c]
            if target is None:
                moves.append((r, c))
            elif target.colour != piece.colour:
                moves.append((r, c))
                break
            else:
                break
            r += dr
            c += dc
    return moves

def queen_moves(board, row, col):
    return rook_moves(board, row, col) + bishop_moves(board, row, col)


def king_moves(board, row, col):
    piece = board[row][col]
    moves = []
    if not piece or piece.name != 'K':
        return moves

    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            target = board[r][c]
            if target is None or target.colour != piece.colour:
                moves.append((r,c))
    return moves