import random
import pickle
import os

BOOK_FILE = 'book.pkl'
BOOK = {}

def load_book():
    global BOOK
    if os.path.exists(BOOK_FILE):
        try:
            with open(BOOK_FILE, 'rb') as f:
                BOOK = pickle.load(f)
            print(f"Extensive opening book loaded ({len(BOOK)} positions).")
        except Exception as e:
            print(f"Error loading opening book file {BOOK_FILE}: {e}")
            BOOK = {}
    else:
        print(f"Warning: '{BOOK_FILE}' not found. Run 'build_book.py' first.")
        BOOK = {}

load_book()

def get_book_move(game_hash):
    if game_hash in BOOK:
        moves = BOOK[game_hash]
        if moves:
            return random.choice(moves)

    return None