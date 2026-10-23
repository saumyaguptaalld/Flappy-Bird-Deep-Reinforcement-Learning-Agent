# DQN Flappy Bird — Project Documentation

This project trains a Deep Q-Learning (DQN) agent to play `FlappyBird-v0`
using Gymnasium. 

## The big picture

DQN teaches an agent to play a game by learning a function `Q(state, action)`
that predicts how good each action is in a given state. Since Flappy Bird's
state space is continuous (bird height, velocity, pipe distances, etc.), we
can't store this in a lookup table — so a neural network approximates it
instead.

The project splits into five files:

| File | Role |
|---|---|
| `dqn.py` | The neural network: the "brain" that predicts Q values |
| `experience_replay.py` | The memory that stores and re-samples past gameplay |
| `parameters.yaml` | All hyperparameters |
| `agent.py` | The orchestrator: trains or tests the agent |
| `game_flappy_bird.py` | Optional: play the game yourself, no AI involved |

## `dqn.py` — the brain

Defines a small feed-forward network: `state_dim → hidden layer → action_dim`.
Feed it a state (a vector of numbers describing the game), and it outputs
one number per possible action — its predicted Q-value (expected future
reward). For Flappy Bird that's 2 outputs: "do nothing" (0) and "flap" (1).
The agent later just picks whichever output is highest (when exploiting).

a plain regression network. All the reinforcement-learning logic lives in `agent.py`.

## `experience_replay.py` — the memory

A DQN doesn't learn from the current moment the way you might expect. If
it trained on each frame right as it happened, consecutive frames would be
nearly identical (the bird barely moves frame to frame), making the
training data highly correlated and learning unstable.

Instead, every experience the agent has —
`(state, action, next_state, reward, terminated)` — gets dropped into a
buffer (a `deque` with a max length, so it's FIFO: once full, the oldest
experience is discarded automatically). Later, training pulls a **random
mini-batch** out of this buffer, which breaks the correlation and gives
more stable gradient updates.

- `append()` adds a new experience
- `sample()` randomly draws a mini-batch (without replacement)
- `__len__` lets the agent check whether there's enough memory yet to
  bother training
- `seed` (optional) makes `sample()`'s randomness reproducible across runs

## `parameters.yaml` — the settings

All hyperparameters live here instead of hardcoded in Python, under a
named block (`flappybirdv0`). This means you can define several different
configurations in the same file and run any of them from the command line
without touching code:

```
python agent.py flappybirdv0 --train
python agent.py flappybirdv1 --train
```

Key settings:
- **`epsilon_init` / `epsilon_decay` / `epsilon_min`** — control
  exploration vs. exploitation (see below)
- **`replay_memory_size` / `mini_batch_size`** — how big the memory
  buffer is, and how many experiences get sampled per training step
- **`network_sync_rate`** — how often the target network gets refreshed
  (see below)
- **`alpha`** — learning rate for the optimizer
- **`gamma`** — discount factor: how much future rewards matter relative
  to immediate ones
- **`reward_threshold`** — ends an episode early if reached (sor press Ctrl+C to end training)

## `agent.py` — where it all runs

### Device selection

Checks for an Apple Silicon GPU (`mps`),NVIDIA (`cuda`), then
falls back toif not then `cpu`. 

### `Agent.__init__`

Opens `parameters.yaml`, pulls out the named block, and stores every value
as an attribute (`self.gamma`, `self.alpha`, etc). Also sets up file paths
for saving the trained model weights and a training log, named after the
parameter set so different configs don't overwrite each other's results.

### `Agent.run(is_training, render)`

The actual game loop — behaves differently depending on mode:

- **Training mode**: creates a `ReplayMemory`, starts `epsilon` at its
  initial (high) value, and creates **two** copies of the network: a
  `policy_dqn` (updated every training step) and a `target_dqn` (a frozen
  copy used only to compute stable targets). Using one constantly changing
  network to both pick actions and grade itself makes training chase a
  moving target and destabilizes learning. The target network is freeze a second
  copy and only periodically sync it.
- **Test mode**: skips all of that, loads the saved weights from training
  (`map_location=device` makes this safe even if training happened on a
  different device than testing), and puts the network in eval mode to
  watch it play.

Then it loops over episodes forever (`itertools.count()` — stop manually
with Ctrl+C, or let `reward_threshold` end an episode early). Within each
episode, every step follows the same pattern:

1. **Epsilon-greedy action choice** — with probability `epsilon`, pick a
   random action (explore); otherwise ask the policy network for its
   best-guess action (exploit). Epsilon starts near 0.1 (10%
   exploring) and decays after every episode, so the agent gradually
   shifts from "trying things" to "using what it's learned" — but never
   drops below `epsilon_min`, always keeping a little randomness so it
   doesn't get stuck in a rut.
2. **Step the environment**, collect the reward and next state.
3. **If training**, store the transition in memory.
4. Once memory has more experiences than one mini-batch's worth, sample a
   batch and call `optimize()` to actually train the network.
5. Periodically (every `network_sync_rate` steps), copy the policy
   network's weights into the target network.
6. If this episode beat the best reward seen so far, save the model
   checkpoint and log it.
7. Print the episode's reward and current epsilon, so you can watch
   exploration decay in real time in the terminal.

### `Agent.optimize(mini_batch, policy_dqn, target_dqn)`

The actual learning step, run once per mini-batch:

1. Unpack the batch into separate stacked tensors (all states together,
   all actions together, etc).
2. Compute the **target** Q-value for each experience using the Bellman
   equation: `reward + gamma * best_Q(next_state)` — but only if the
   episode didn't terminate there; if it did, there's no next state to
   look ahead into, so the target is just the reward. Computed with
   `target_dqn` under `torch.no_grad()`, since it's meant to be a fixed
   reference point, not something we backpropagate through.
3. Compute the **predicted** Q-value from `policy_dqn` for the specific
   action that was actually taken (`gather()` picks that one value out of
   the network's per-action output).
4. Take the mean-squared-error loss between predicted and target,
   backpropagate, and step the optimizer — the one point where the
   network's weights actually change.

### Command-line entry point

```
python agent.py flappybirdv0 --train      # train, no rendering (faster)
python agent.py flappybirdv0              # test: loads saved weights, renders the game
```

## `game_flappy_bird.py` — play it yourself (optional)

Not part of the training pipeline at all — `agent.py` never imports or
calls it. It's a standalone script where *you* control the bird with the
spacebar, using the same Gymnasium environment. Useful for sanity-checking
the environment before training, and for getting an intuitive feel for
what "flap" actually does and how rewards behave.

It opens the environment with `render_mode="human"` (which already creates
a window via pygame internally), grabs a handle to that window, then loops:
poll for a spacebar press → step the environment with `action = 1` if
pressed else `0` → render. Close the window or hit Ctrl+C to quit.

## Running the project end-to-end

```bash
# 1. (Optional) Play the game yourself first
python game_flappy_bird.py

# 2. Train the agent — runs indefinitely, stop with Ctrl+C when satisfied
python agent.py flappybirdv0 --train

# 3. Watch the trained agent play
python agent.py flappybirdv0
```

Trained model weights and logs are saved automatically under `runs/`,
named after the parameter set used (e.g. `runs/flappybirdv0.pt`,
`runs/flappybirdv0.log`).

## Key concepts, summarized

- **Epsilon-greedy**: balances trying new things (explore) against using
  what's already learned (exploit); epsilon decays over training.
- **Experience replay**: breaks correlation between consecutive frames by
  training on random past experiences instead of the current one.
- **Target network**: a periodically-frozen copy of the policy network,
  used to keep training targets stable instead of chasing a
  constantly moving reference.
- **Bellman equation**: `target Q = reward + gamma * best future Q`,
  the mathematical backbone of how Q-learning propagates reward
  information backward through time.
- **Discount factor (gamma)**: how much the agent values future reward vs.
  immediate reward.
