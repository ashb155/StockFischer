#piece class
class Piece:
    def __init__(self,colour,name):
        self.colour=colour
        self.name=name

    def __str__(self):
        return self.colour + self.name
