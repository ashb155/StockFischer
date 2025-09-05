from game import notation_to_index,index_to_notation

def parser(move,game):
    move=move.strip()

    #castling notation
    if move =='0-0':
        if game.turn == 'w':
            return (7,4),(7,6),None
        else:
            return (0,4),(0,6),None

    if move == '0-0-0':
        if game.turn == 'w':
            return (7,4),(7,2),None
        else:
            return (0,4),(0,2),None

    #promotion
    promotion = None
    if len(move) > 2 and move[-1] in "QRBN":
        promotion = move[-1]
        move = move[:-1]

    pieces = {'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K'}
    piece_type = 'P'
    if move[0] in pieces:
        piece_type = pieces[move[0]]
        move=move[1:]

    #capture marker
    capture= "x" in move
    move=move.replace("x","")

    if len(move)<2:
        return None, None, None

    #destination square
    destsq=move[-2:]
    dest=notation_to_index(destsq)

    disambig = move[:-2]
    file_hint, rank_hint = None, None
    for ch in disambig:
        if ch in "abcdefgh":
            file_hint = ch
        elif ch in "12345678":
            rank_hint = ch

    candidates = []
    for r in range(8):
        for c in range(8):
            piece = game.board[r][c]
            if piece and piece.colour == game.turn and piece.name == piece_type:
                if dest in game.get_moves(r, c):
                    notation = index_to_notation((r, c))
                    if file_hint and notation[0] != file_hint:
                        continue
                    if rank_hint and notation[1] != rank_hint:
                        continue
                    candidates.append((r, c))

    if len(candidates) == 1:
        return candidates[0], dest, promotion
    else:
        return None, None, None