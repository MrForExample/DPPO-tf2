import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import gym

import matplotlib.pyplot as plt
import seaborn as sns

env_id = "Pendulum-v0"
# Max number of episodes for training
episodes_number = 1000
# Max number of steps within one episode, set to 0 for reset only when episode is done
episode_steps_length = 200
critic_update_step = 15
actor_update_step = 10
# Use for reward feed in agent during training, not for log or print
reward_scale = 0.1
# Agent update after number of batch step, set to 0 for update only when episode is done
batch = 32
# Training use TD-lambda-return of Monte Carlo return
use_td_lambda = True
lam = 0.95
gamma = 0.9
actor_hidden_layers = [100]
critic_hidden_layers = [100]
hidden_activation = 'relu'
actor_mean_activation = 'tanh'
actor_std_activation = 'softplus'
actor_lr = lambda f: f * 1e-4 + 5e-5
critic_lr = lambda f: f * 2e-4 + 1e-4
cliprange = 0.2
# parameters for save, load, log
max_checkpoints_number = 5
save_every_number_episodes = 100
log_every_number_episodes = 10
eval_every_number_episodes = 25
critic_model_path = "./models/critic"
actor_model_path = "./models/actor"
log_path = "./models/logs"

class PPO:
    def __init__(self, s_space, a_space):
        self.s_space = s_space
        self.a_space = a_space

        s_max = []
        for h, l in zip(self.s_space.high, self.s_space.low):
            m = abs(h) if abs(h) > abs(l) else abs(l)
            s_max.append(m)
        self.s_space_max = np.array(s_max)
        print("State space scale: {}".format(self.s_space_max))

        a_max = []
        for h, l in zip(self.a_space.high, self.a_space.low):
            m = abs(h) if abs(h) > abs(l) else abs(l)
            if m == float('Inf'):
                m = 1
            a_max.append(m)
        self.a_space_max = np.array(a_max)
        print("Action space scale: {}".format(self.a_space_max))

        c_init = keras.initializers.RandomNormal() 
        critic_latent, critic_input_layer = self.mlp(self.s_space.shape, critic_hidden_layers, c_init)
        self.critic = self.build_critic(critic_input_layer, 1, critic_latent, c_init)

        a_init = keras.initializers.RandomNormal() 
        self.action_num = self.a_space.shape[0]
        actor_latent, actor_input_layer = self.mlp(self.s_space.shape, actor_hidden_layers, a_init)
        self.actor = self.build_actor(actor_input_layer, self.action_num, actor_latent, a_init)
        self.c_optimizer = keras.optimizers.Adam()
        self.a_optimizer = keras.optimizers.Adam()

        self.load_model(critic_model_path, self.critic)
        self.load_model(actor_model_path, self.actor)       

    def load_model(self, path, model):
        if path is not None:
            print("Loading model...from: {}".format(path))
            ckpt = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=max_checkpoints_number)
            ckpt.restore(manager.latest_checkpoint)

    def save_model(self, path, model):
        if path is not None:
            print("Saveing model...from: {}".format(path))
            ckpt = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=max_checkpoints_number)
            manager.save()

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        #s /= self.s_space_max
        return self.critic(s)

    def get_a(self, s, is_determinative = False):
        s = s[np.newaxis, :]
        #s /= self.s_space_max
        action_all = self.actor(s)
        mean, std = tf.split(action_all, [self.action_num, self.action_num], 1)
        mean *= self.a_space_max
        if is_determinative:
            action, prob = np.squeeze(mean, 0), np.array([1])
        else :
            action_pd = tfp.distributions.Normal(mean, std)
            action = np.clip(action_pd.sample(1), self.a_space.low, self.a_space.high)
            prob = action_pd.prob(action)
            action, prob = np.squeeze(action, (0, 1)), np.squeeze(prob, (0, 1))
        return action, prob

    def update(self, bs, ba, br, bv, oldpi_prob, actor_lrnow, critic_lrnow):
        #bs /= self.s_space_max
        advantage  = br - bv
        # Normalize the advantages
        #advantage  = (advantage  - tf.reduce_mean(advantage )) / (keras.backend.std(advantage) + 1e-8)
        all_vf_loss = 0
        for _ in range(critic_update_step):
            with tf.GradientTape() as v_tape:
                vf_loss = tf.reduce_mean(tf.square(br - self.critic(bs)))

            grads = v_tape.gradient(vf_loss, self.critic.trainable_variables)
            self.c_optimizer.learning_rate = critic_lrnow
            grads_and_vars = zip(grads, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(grads_and_vars)

            all_vf_loss += vf_loss

        all_pg_loss = 0
        for _ in range(actor_update_step):
            with tf.GradientTape() as a_tape:
                action_all = self.actor(bs)
                mean, std = tf.split(action_all, [self.action_num, self.action_num], 1)
                mean *= self.a_space_max
                action_pd = tfp.distributions.Normal(mean, std)

                #ratio = tf.exp(action_pd.log_prob(ba) - oldpi_log_prob)
                ratio = action_pd.prob(ba) / (oldpi_prob + 1e-5)

                pg_losses1 = -advantage * ratio
                pg_losses2 = -advantage * tf.clip_by_value(ratio, 1-cliprange, 1+cliprange)
                pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))

            grads = a_tape.gradient(pg_loss, self.actor.trainable_variables)
            self.a_optimizer.learning_rate = actor_lrnow
            grads_and_vars = zip(grads, self.actor.trainable_variables)
            self.a_optimizer.apply_gradients(grads_and_vars)

            all_pg_loss += pg_loss

        return all_vf_loss, all_pg_loss

    def mlp(self, input_shape, layers, weight_init):
        # Build input layer
        x_input = keras.Input(shape=input_shape)
        h = x_input
        # Build hidden layer
        for num_hidden in layers:
            h = keras.layers.Dense(units=num_hidden, activation=hidden_activation, kernel_initializer=weight_init)(h)          
        return h, x_input

    # Build actor's output layer
    def build_actor(self, x_input, output_num, actor_latent, weight_init):
        mu = keras.layers.Dense(output_num, actor_mean_activation, kernel_initializer=weight_init)(actor_latent)
        sigma = keras.layers.Dense(output_num, actor_std_activation, kernel_initializer=weight_init)(actor_latent)
        a_output = keras.layers.concatenate([mu, sigma])
        network = keras.Model(inputs=[x_input], outputs=[a_output])
        return network

    # Build critic's output layer
    def build_critic(self, x_input, output_num, critic_latent, weight_init):
        v_output = keras.layers.Dense(units=output_num, kernel_initializer=weight_init)(critic_latent)
        network = keras.Model(inputs=[x_input], outputs=[v_output])
        return network

def TD_lambda_return(agent, buffer_r, buffer_s, s_):
    td_lambda, discounted_return, v_s_list = [], [], []
    v_s_i = np.array(agent.get_v(s_)).item()
    # last_return_length = T - t - 1
    last_return_length = 0
    # Calculate each lambda-return for the time step within the batch 
    for i_b in range(len(buffer_r)-1, -1, -1):
        td_return = 0
        # Calculate TD-error at time step corresponding to i_b
        now_return = buffer_r[i_b] +  gamma * v_s_i
        discounted_return.append(now_return)
        
        v_s_i = np.array(agent.get_v(buffer_s[i_b])).item()
        v_s_list.append(v_s_i)
        # Calculate TD-lambda-return at time step corresponding to i_b
        for i_return in range(last_return_length):
            discounted_return[i_return] = buffer_r[i_b] + gamma * discounted_return[i_return]
            td_return = discounted_return[i_return + 1] + lam * td_return

        td_return = (1 - lam) * td_return + lam ** last_return_length * discounted_return[0]
        td_lambda.append(td_return)
        last_return_length += 1

    v_s_list.reverse()
    td_lambda.reverse()
    return td_lambda, v_s_list

def MC_return(agent, buffer_r, buffer_s, s_):
    discounted_return, v_s_list = [], []
    # Calculate each discounted return for the time step within the batch
    v_s_ = np.array(agent.get_v(s_)).item()
    for i_b in range(len(buffer_r)-1, -1, -1):
        v_s_ = buffer_r[i_b] + gamma * v_s_
        discounted_return.append(v_s_)

        v_s_i = np.array(agent.get_v(buffer_s[i_b])).item()
        v_s_list.append(v_s_i)

    v_s_list.reverse()
    discounted_return.reverse()
    return discounted_return, v_s_list 

def evaluate(eval_env, agent):
    # Show the agent's perform
    s = eval_env.reset()
    ep_r = 0
    t = 0
    while True:
        t += 1
        eval_env.render()
        a, _ = agent.get_a(s, True)
        s_, r, done, _ = eval_env.step(a)
        done = t == episode_steps_length or done
        s = s_
        ep_r += r

        if done:
            print("Average episode reward: {}".format(ep_r/t))
            break

def learn():
    env = gym.make(env_id)
    print("Observation space: {}".format(env.observation_space))
    if type(env.observation_space) == gym.spaces.Box:
        print("Observation bounds: high: {}, low: {}".format(env.observation_space.high, env.observation_space.low))
    print("Action space: {}".format(env.action_space))
    if type(env.action_space) == gym.spaces.Box:
        print("Action bounds: high: {}, low: {}".format(env.action_space.high, env.action_space.low))

    ppo = PPO(env.observation_space, env.action_space)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = log_path + '/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    train_reward = tf.keras.metrics.Mean('train_reward', dtype=tf.float32)
    train_reward_max = tf.keras.metrics.Mean('train_reward_max', dtype=tf.float32)
    train_reward_min = tf.keras.metrics.Mean('train_reward_min', dtype=tf.float32)
    train_reward_std = tf.keras.metrics.Mean('train_reward_std', dtype=tf.float32)
    train_vf_loss = tf.keras.metrics.Mean('train_vf_loss', dtype=tf.float32)
    train_pg_loss = tf.keras.metrics.Mean('train_pg_loss', dtype=tf.float32)

    # Training ppo for given number of episodes
    for i_episode in range(episodes_number):
        s = env.reset()
        buffer_s, buffer_a, buffer_r, buffer_p, ep_all_r = [], [], [], [], []
        ep_r = ep_vf_loss = ep_pg_loss = 0
        ep_max_r = float('-inf')
        ep_min_r = float('+inf')
        t = 0
        # Training at each episode for given maximum number of steps
        while True:
            t += 1
            if (i_episode + 1) % 10 == 0:
                env.render()

            a, p = ppo.get_a(s)
            s_, r, done, info = env.step(a)
            done = t == episode_steps_length - 1 or done

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r * reward_scale)
            buffer_p.append(p)

            ep_all_r.append(r)
            ep_r += r
            if r > ep_max_r: ep_max_r = r
            if r < ep_min_r: ep_min_r = r

            #print('State: {}, Action: {}, Log_Probability: {}'.format(s, a, p))
            # update ppo
            if (batch != 0 and t % batch == 0) or done:
                if use_td_lambda:
                    final_return, v_s_list = TD_lambda_return(ppo, buffer_r, buffer_s, s_)
                else :
                    final_return, v_s_list = MC_return(ppo, buffer_r, buffer_s, s_)

                #print("\nfinal_return: {}, type: {}".format(final_return, type(final_return)))
                bs, ba, br, bv, bp = np.array(buffer_s), np.array(buffer_a), np.array(final_return)[:, np.newaxis], np.array(v_s_list)[:, np.newaxis], np.array(buffer_p)
                # from training begin: 1 -> end: 0 
                frac = 1 - (i_episode / episodes_number)
                # Calculate the learning rate and cliprange
                actor_lrnow = actor_lr(frac)
                critic_lrnow = critic_lr(frac)

                vf_loss, pg_loss = ppo.update(bs, ba, br, bv, bp, actor_lrnow, critic_lrnow)
                ep_vf_loss += vf_loss
                ep_pg_loss += pg_loss

                if done:
                    break
                buffer_s, buffer_a, buffer_r, buffer_p = [], [], [], []

            s = s_

        # Log for tensorboard to plot
        avg_ep_r = ep_r/t
        avg_ep_vf_loss = ep_vf_loss/t
        avg_ep_pg_loss = ep_pg_loss/t
        ep_r_std = np.std(ep_all_r)
        
        train_reward(avg_ep_r)
        train_reward_max(ep_max_r)
        train_reward_min(ep_min_r)
        train_reward_std(ep_r_std)
        train_vf_loss(avg_ep_vf_loss)
        train_pg_loss(avg_ep_pg_loss)
        if (i_episode + 1) % log_every_number_episodes == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('Average over {} episode reward'.format(log_every_number_episodes), train_reward.result(), step=i_episode)
                tf.summary.scalar('Average over {} episode max reward'.format(log_every_number_episodes), train_reward_max.result(), step=i_episode)
                tf.summary.scalar('Average over {} episode min reward'.format(log_every_number_episodes), train_reward_min.result(), step=i_episode)
                tf.summary.scalar('Average over {} episode standard deviation'.format(log_every_number_episodes), train_reward_std.result(), step=i_episode)
                tf.summary.scalar('Average over {} episode value function loss'.format(log_every_number_episodes), train_vf_loss.result(), step=i_episode)
                tf.summary.scalar('Average over {} episode policie function loss'.format(log_every_number_episodes), train_pg_loss.result(), step=i_episode)
            train_reward.reset_states()
            train_reward_max.reset_states()
            train_reward_min.reset_states()
            train_reward_std.reset_states()
            train_vf_loss.reset_states()
            train_pg_loss.reset_states()

        print("Ep_id: {}, Avg_ep_r: {:.5f}, Ep_r_max: {:.5f}, Ep_r_min: {:.5f}, Ep_r_std: {:.5f}, Avg_vf_loss: {:.5f}, Avg_pg_loss: {:.5f}".format(i_episode, avg_ep_r, ep_max_r, ep_min_r, ep_r_std, avg_ep_vf_loss, avg_ep_pg_loss))

        # Save model
        if (i_episode + 1) % save_every_number_episodes == 0:
            ppo.save_model(critic_model_path, ppo.critic)
            ppo.save_model(actor_model_path, ppo.actor)

        # Evaluate model
        if (i_episode + 1) % eval_every_number_episodes == 0:
            evaluate(env, ppo)

    # Show the result
    while True:
        evaluate(env, ppo)

    env.close()

if __name__ == "__main__":
    learn()
    