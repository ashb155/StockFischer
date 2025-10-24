# piece class
class Piece:
    def __init__(self, colour, name):
        self.colour = colour
        self.name = name

    def __str__(self):
        return self.colour + self.name

    def __repr__(self):
        return f"Piece('{self.colour}', '{self.name}')"

    def __eq__(self, other):
        if not isinstance(other, Piece):
            return False
        return self.colour == other.colour and self.name == other.name

    def __hash__(self):
        return hash((self.colour, self.name))
