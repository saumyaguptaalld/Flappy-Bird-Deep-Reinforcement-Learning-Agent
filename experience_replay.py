from collections import deque
import random 


class ReplayMemory():
    # create a FIFO memory - Experience relay
    def __init__(self, maxlen, seed = None): # max_len is no of experiences, seed is the initial value for random sampling of the experiences
        self.memory = deque([],maxlen=maxlen)

    def append(self, new_memory):
        self.memory.append(new_memory)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    # current buffer size 
    def __len__(self):
        return len(self.memory)


# Creating a FIFO memory , deque , which lets in new experiences from rear and moves out old experiences from front