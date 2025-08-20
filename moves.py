from pieces import Piece

def pawn_moves(board,row,col):
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


    for diagonal in range(-1,2):
        ncol= col + diagonal
        nrow= row + direction
        if 0 <= nrow < 8 and 0 <= ncol < 8:
            target = board[nrow][ncol]
            if target and target.colour != piece.colour:
                moves.append((nrow,ncol))

    return moves

