import random
from collections import defaultdict

class MENACE:
    def __init__(self):
        self.matchboxes = defaultdict(lambda: [1] * 9)  # beads per move
        self.history = []  # store (state, move)

    def get_state(self, board):
        return ''.join(map(str, board))

    def choose_move(self, board):
        state = self.get_state(board)
        beads = self.matchboxes[state]
        valid_moves = [i for i in range(9) if board[i] == 0]

        if not valid_moves:
            return None

        # weighted choice
        weights = [beads[i] if i in valid_moves else 0 for i in range(9)]
        total = sum(weights)

        if total == 0:
            move = random.choice(valid_moves)
        else:
            move = random.choices(range(9), weights=weights)[0]

        self.history.append((state, move))
        return move

    def update(self, reward):
        for state, move in self.history:
            self.matchboxes[state][move] = max(1, self.matchboxes[state][move] + reward)
        self.history = []
