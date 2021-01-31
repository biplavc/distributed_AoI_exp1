from tf_environment import *
from create_graph_1 import *
# if comet:
#     from main_tf import experiment

random.seed(42)
np.random.seed(42)
# tf.random.set_seed(42)

tempdir = tempfile.gettempdir()

## source = https://github.com/tensorflow/agents/blob/master/docs/tutorials/7_SAC_minitaur_tutorial.ipynb


def tf_sac(I, drones_coverage, folder_name, deployment, packet_update_loss, packet_sample_loss, periodicity, adj_matrix, tx_rx_pairs, tx_users):
    
    
    all_actions = []    ## save all actions over all steps of all episodes  
    print(f"\n\nSAC started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss}, periodicity = {periodicity}, tx_rx_pairs = {tx_rx_pairs}, tx_users = {tx_users}, UL_capacity = {UL_capacity}, DL_capacity = {DL_capacity} and {deployment} deployment")
    print(f"\n\nSAC started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss}, periodicity = {periodicity},, tx_rx_pairs = {tx_rx_pairs}, tx_users = {tx_users}, UL_capacity = {UL_capacity}, DL_capacity = {DL_capacity} and {deployment} deployment", file = open(folder_name + "/results.txt", "a"), flush = True)

    train_py_env = UAV_network(I, drones_coverage, "train net", folder_name, packet_update_loss, packet_sample_loss, periodicity, adj_matrix, tx_rx_pairs, tx_users)
    eval_py_env = UAV_network(I, drones_coverage, "eval net", folder_name, packet_update_loss, packet_sample_loss, periodicity, adj_matrix, tx_rx_pairs, tx_users)
    
    collect_env = tf_py_environment.TFPyEnvironment(train_py_env) # doesn't print out
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    collect_env.reset()
    eval_env.reset()
    
    final_step_rewards = []
    
    sac_returns = []
    
    initial_collect_steps = 1000  # @param {type:"integer"}

    batch_size = 16 # @param {type:"integer"}

    critic_learning_rate = 3e-4 # @param {type:"number"}
    actor_learning_rate = 3e-4 # @param {type:"number"}
    alpha_learning_rate = 3e-4 # @param {type:"number"}
    target_update_tau = 0.005 # @param {type:"number"}
    target_update_period = 1 # @param {type:"number"}
    gamma = 0.99 # @param {type:"number"}
    reward_scale_factor = 1.0 # @param {type:"number"}

    actor_fc_layer_params = fc_layer_params
    critic_joint_fc_layer_params = fc_layer_params
    
    
    # GPU AND STRATEGY
    use_gpu = True #@param {type:"boolean"}

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)
    
    ## AGENTS
    
    observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(collect_env))


    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
                (observation_spec, action_spec),
                observation_fc_layer_params=None,
                action_fc_layer_params=None,
                joint_fc_layer_params=critic_joint_fc_layer_params,
                kernel_initializer='glorot_uniform',
                last_kernel_initializer='glorot_uniform')
        
    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(observation_spec, action_spec,       fc_layer_params=actor_fc_layer_params,      continuous_projection_net=(tanh_normal_projection_network.TanhNormalProjectionNetwork))
        
    
    with strategy.scope():
        train_step = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
                time_step_spec,
                action_spec,
                actor_network           = actor_net,
                critic_network          = critic_net,
                actor_optimizer         = tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
                critic_optimizer        = tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
                alpha_optimizer         = tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_learning_rate),
                target_update_tau       = target_update_tau,
                target_update_period    = target_update_period,
                td_errors_loss_fn       = tf.math.squared_difference,
                gamma                   = gamma,
                reward_scale_factor     = reward_scale_factor,
                train_step_counter      = train_step)

        tf_agent.initialize()
        
    ## REPLAY BUFFER
    
    table_name  = 'uniform_table'
    table       = reverb.Table(
                table_name,
                max_size     = replay_buffer_capacity,
                sampler      = reverb.selectors.Uniform(),
                remover      = reverb.selectors.Fifo(),
                rate_limiter = reverb.rate_limiters.MinSize(1))

    reverb_server = reverb.Server([table])
    
    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
                    tf_agent.collect_data_spec,
                    sequence_length = 2,
                    table_name      = table_name,
                    local_server    = reverb_server)
    
    dataset       = reverb_replay.as_dataset(sample_batch_size=batch_size, num_steps=2).prefetch(50)
    
    experience_dataset_fn = lambda: dataset
    
    
    # POLICIES
    
    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_eval_policy, use_tf_function=True)
    
    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy, use_tf_function=True)
    
    random_policy = random_py_policy.RandomPyPolicy(collect_env.time_step_spec(), collect_env.action_spec())
    
    ## ACTORS
    
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(reverb_replay.py_client,table_name,sequence_length=2,stride_length=1)
    
    initial_collect_actor = actor.Actor(collect_env, random_policy,  train_step, steps_per_run=initial_collect_steps, observers=[rb_observer])
    
    initial_collect_actor.run()
    
    env_step_metric = py_metrics.EnvironmentSteps()
    
    collect_actor = actor.Actor(collect_env, collect_policy, train_step, steps_per_run=1, metrics=actor.collect_metrics(10), summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),    observers=[rb_observer, env_step_metric])
    
    eval_actor = actor.Actor(eval_env, eval_policy, train_step, episodes_per_run=num_eval_episodes, metrics=actor.eval_metrics(num_eval_episodes), summary_dir=os.path.join(tempdir, 'eval'),)
    
    ## LEARNERS
    
    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)
    
    policy_save_interval = 5000 # @param {type:"integer"}

    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [triggers.PolicySavedModelTrigger(saved_model_dir, tf_agent, train_step, interval=policy_save_interval),   triggers.StepPerSecondLogTrigger(train_step, interval=1000),]

    agent_learner = learner.Learner( tempdir, train_step, tf_agent, experience_dataset_fn, triggers=learning_triggers)
    
    ##  METRICS AND EVALUATION
    
    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
        return results

    metrics = get_eval_metrics()
    
    def log_eval_metrics(step, metrics):
        eval_results = (', ').join('{} = {:.2f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))

    log_eval_metrics(0, metrics)
    
    ## TRAINING THE AGENT
    
    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]
    UAV_returns = [sum(eval_py_env.UAV_age.values())]

    for _ in range(num_iterations):
    # Training.
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy

        if step % eval_interval == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])
            UAV_returns.append(sum(eval_py_env.UAV_age.values()))

        if log_interval and step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

    rb_observer.close()
    reverb_server.stop()
    
    sac_returns = returns
    
     
    pickle.dump(sac_returns, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_returns.pickle", "wb"))
    pickle.dump(UAV_returns, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_UAV_returns.pickle", "wb"))
    pickle.dump(final_step_rewards, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_final_step_rewards.pickle", "wb"))
    pickle.dump(eval_py_env.tx_attempt_dest, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_tx_attempt_dest.pickle", "wb"))
    pickle.dump(eval_py_env.tx_attempt_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_tx_attempt_UAV.pickle", "wb"))
    pickle.dump(eval_py_env.preference, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_preference.pickle", "wb")) 

    pickle.dump(eval_py_env.sample_time, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_sample_time.pickle", "wb"))

    pickle.dump(eval_py_env.dest_age, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_dest_age.pickle", "wb")) 
    pickle.dump(eval_py_env.UAV_age, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_UAV_age.pickle", "wb"))
    pickle.dump(eval_py_env.age_dist_dest, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_age_dist_dest.pickle", "wb"))
    pickle.dump(eval_py_env.age_dist_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_age_dist_UAV.pickle", "wb"))
    
    pickle.dump(eval_py_env.attempt_sample, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_attempt_sample.pickle", "wb"))
    pickle.dump(eval_py_env.success_sample, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_success_sample.pickle", "wb"))
    pickle.dump(eval_py_env.attempt_update, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_attempt_update.pickle", "wb"))
    pickle.dump(eval_py_env.success_update, open(folder_name + "/" + deployment + "/" + str(I) + "U_sac_success_update.pickle", "wb"))
    
    
    print("\nSAC scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards[-5:]), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", np.mean(sac_returns[-5:]), " : end with final state of ", eval_py_env._state, " with shape ", np.shape(eval_py_env._state))
    
    print("\nSAC scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards[-5:]), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", np.mean(sac_returns[-5:]), " : end with final state of ", eval_py_env._state, " with shape ", np.shape(eval_py_env._state), file = open(folder_name + "/results.txt", "a"), flush = True)

    
    print(f"SAC ended for {I} users and {deployment} deployment")
    return sac_returns, final_step_rewards, all_actions
    