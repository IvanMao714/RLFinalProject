import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np

class Agent:
    def __init__(self, num_states, num_actions, learning_rate=0.001, gamma=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.states = []
        self.actions = []
        self.rewards = []

        # Build models
        self._build_model()

    def _build_model(self):
        # Actor Model
        inputs = tf.keras.Input(shape=(self.num_states,))
        common = tf.keras.layers.Dense(24, activation='relu')(inputs)
        action = tf.keras.layers.Dense(self.num_actions, activation='softmax')(common)
        self.actor_model = tf.keras.Model(inputs=inputs, outputs=action)
        self.actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                                 loss=self._actor_loss)

        # Critic Model
        value = tf.keras.layers.Dense(1)(common)
        self.critic_model = tf.keras.Model(inputs=inputs, outputs=value)
        self.critic_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                                  loss='mse')

    # def _actor_loss(self, y_true, y_pred):
    #     # y_true contains advantages and selected actions
    #     advantages = y_true[:, 0]
    #     actions = y_true[:, 1:]
    #     advantages = tf.cast(advantages, tf.float32)
    #     actions = tf.cast(actions, tf.float32)  # Cast actions to float32
    #     action_probs = tf.reduce_sum(y_pred * actions, axis=1)
    #     log_probs = tf.math.log(action_probs + 1e-10)
    #     loss = -tf.reduce_mean(log_probs * advantages)
    #     return loss
    def _actor_loss(self, y_true, y_pred):
        # y_true contains advantages and selected actions
        advantages = y_true[:, 0]
        actions = y_true[:, 1:]
        # No need to cast if actions are already float32, but casting doesn't hurt
        advantages = tf.cast(advantages, tf.float32)
        actions = tf.cast(actions, tf.float32)
        action_probs = tf.reduce_sum(y_pred * actions, axis=1)
        log_probs = tf.math.log(action_probs + 1e-10)
        loss = -tf.reduce_mean(log_probs * advantages)
        return loss

    def policy(self, state):
        state = np.reshape(state, [1, self.num_states])
        action_probs = self.actor_model.predict(state, verbose=0)
        action = np.random.choice(self.num_actions, p=action_probs[0])
        return action

    def value(self, state):
        state = np.reshape(state, [1, self.num_states])
        value = self.critic_model.predict(state, verbose=0)
        return value[0][0]

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def train(self, next_state, done):
        if not self.states:
            return

        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)

        # Compute discounted rewards
        target_values = []
        cumulative = 0 if done else self.value(next_state)
        for reward in reversed(rewards):
            cumulative = reward + self.gamma * cumulative
            target_values.insert(0, cumulative)
        target_values = np.array(target_values)

        # # Compute advantages
        # values = self.critic_model.predict(states).flatten()
        # advantages = target_values - values

        # # One-hot encode actions
        # actions_one_hot = np.zeros((len(actions), self.num_actions))
        # actions_one_hot[np.arange(len(actions)), actions] = 1
        #
        # # Prepare actor targets (advantages and actions)
        # actor_targets = np.hstack([advantages.reshape(-1, 1), actions_one_hot])
        # Compute advantages
        values = self.critic_model.predict(states).flatten()
        advantages = target_values - values
        advantages = advantages.astype(np.float32)  # Ensure advantages are float32

        # One-hot encode actions
        actions_one_hot = np.zeros((len(actions), self.num_actions), dtype=np.float32)
        actions_one_hot[np.arange(len(actions)), actions] = 1.0  # Use 1.0 to ensure float32

        # Prepare actor targets (advantages and actions)
        actor_targets = np.hstack([advantages.reshape(-1, 1), actions_one_hot])

        # Train actor and critic
        self.actor_model.fit(states, actor_targets, verbose=0)
        self.critic_model.fit(states, target_values, verbose=0)

        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
