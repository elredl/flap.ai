import tensorflow as tf
from flappy_ai import FlappyBirdEnv


agent = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3)
])
# Load the trained weights
agent.load_weights('flappy_ppo_final.weights.h5') 

def get_action(state):
    out = agent(state[None])[0]
    logits = out[:2]
    action = tf.random.categorical(logits[None], 1)[0, 0]
    return int(action)

env = FlappyBirdEnv(render_mode="human", frame_skip=1) 
s, i = env.reset()

while True:
    a = get_action(tf.constant(s, tf.float32))
    s, r, done, truncated, info = env.step(a)
    env.render()
    if done or truncated:
        print("Score:", info["score"])
        s, _ = env.reset()
