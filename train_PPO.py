import tensorflow as tf
import numpy as np
from flappy_ai import FlappyBirdEnv

env = FlappyBirdEnv(render_mode=None)
# the input shape is 5 for the player, velocity, distance to pipe, gap position, pipe speed
# the first 2 = policy(flap or don’t), last 1 = value(how good is this state?)
agent = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation = 'relu', input_shape = (5,)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(3)
])
# the hyperparamters opt = stable optimizer, gama = how much we care about future rewards, lam = GAE smoothing (makes learning stable), clip = prevents huge policy updates
opt = tf.keras.optimizers.Adam(3e-4)
gama = 0.99
lam = 0.95
clip = 0.2

# this is how the bird decides to flap or not
def get_action_value(state):
    out = agent(state[None])[0]
    logits = out[:2]
    value = out[2]
    probs = tf.nn.softmax(logits)
    action = tf.random.categorical(logits[None], 1)[0, 0]
    logp = tf.nn.log_softmax(logits)[action]

    return int(action), logp, value

def train(states, actions, old_logps, advantages, returns):
    with tf.GradientTape() as tape:
        out = agent(states)
        logits = out[:, :2]
        values = out[:, 2:]
        logps = tf.nn.log_softmax(logits)
        new_logps = tf.reduce_sum(tf.one_hot(actions,2) * logps, axis = 1)
        ratio = tf.exp(new_logps - old_logps)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1 - clip, 1 + clip) * advantages
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        value_loss = tf.reduce_mean((values[:, 0] - returns)**2)
        entropy = -tf.reduce_mean(tf.nn.softmax(logits) * logps)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    
    grads = tape.gradient(loss, agent.trainable_variables)
    opt.apply_gradients(zip(grads, agent.trainable_variables))

s, _ = env.reset()
s = tf.constant(s, tf.float32)
buf = {'state':[], 'action':[], 'reward':[], 'd':[], 'logp':[], 'value':[]}

try:
    for step in range(1000000):
        for _ in range(2048):
            a, logp, v = get_action_value(s)
            ns, r, done, _, info = env.step(a)
            buf['state'].append(s)
            buf['action'].append(a)
            buf['reward'].append(r)
            buf['d'].append(done)
            buf['logp'].append(logp)
            buf['value'].append(v.numpy())
            s = tf.constant(ns, tf.float32)
            if done:
                print("Score:", info["score"])
                s, _ = env.reset()
                s = tf.constant(s, tf.float32)

        _, _, last_v = get_action_value(s)
        values = [v for v in buf['value']] + [last_v.numpy()]
        adv = []
        gae = 0
        for t in reversed(range(len(buf['reward']))):
            delta = buf['reward'][t] + gama * values[t + 1] * (1 - buf['d'][t]) - values[t]
            gae = delta + gama * lam * (1 - buf['d'][t]) * gae
            adv.append(gae)
        adv = adv[::-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        ret = np.array(adv) + values[:-1]

        states = tf.stack(buf['state'])
        actions = tf.constant(buf['action'], tf.int32)
        old_logps = tf.stack(buf['logp'])
        advantages = tf.constant(adv, tf.float32)
        returns = tf.constant(ret, tf.float32)

        # Train for 4 epochs
        for _ in range(4):
            train(states, actions, old_logps, advantages, returns)

        for k in buf: buf[k].clear()

except KeyboardInterrupt:
    print("\nTraining stopped by user, saving final model.")
    agent.save_weights('flappy_ppo_final.weights.h5')
print("Done")
