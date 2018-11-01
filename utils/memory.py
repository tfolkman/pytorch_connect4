"""
A class to store game states for learning
"""

from collections import deque


class Memory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.long_term_memory = deque(maxlen=self.memory_size)
        self.short_term_memory = deque(maxlen=self.memory_size)

    def commit_short_term_memory(self, identities, state, action_values):
        for game_state, action_value in identities(state, action_values):
            self.short_term_memory.append({
                'board': game_state.board,
                'game_state': game_state,
                'id': game_state.id,
                'action_value': action_value,
                'player_turn': game_state.playerTurn
            })

    def commit_long_term_memory(self):
            for i in self.short_term_memory:
                self.long_term_memory.append(i)
            self.short_term_memory.clear()