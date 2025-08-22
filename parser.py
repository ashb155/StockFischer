from game import notation_to_index,index_to_notation

def parser(move,game):
    move=move.strip()
    pieces = {'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K'}
    piece_type = 'P'
    if move[0] in pieces:
        piece_type = pieces[move[0]]
        move=move[1:]

    destsq=move[-2:]
    dest=notation_to_index(destsq)

    candidates = []
    for r in range(8):
        for c in range(8):
            piece = game.board[r][c]
            if piece and piece.colour == game.turn and piece.name == piece_type:
                if dest in game.get_moves(r,c):
                    candidates.append((r,c))

    if len(candidates) == 1:
        return candidates[0], dest
    else:
        return False