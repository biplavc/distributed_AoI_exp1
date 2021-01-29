from tf_environment import *
from create_graph_1 import *
import collections
from collections import defaultdict
import itertools
import operator
from datetime import datetime
import sys

# random.seed(42)

def find_mad_action(eval_env): ## maximal age difference
    ## https://pynative.com/python-random-seed/
    seedValue = random.randrange(sys.maxsize)
    random.seed(seedValue) ## so that whenever randomness arises, different actions will be taken

    # print(f"eval_env.tx_rx_pairs = {eval_env.tx_rx_pairs}")
    ## the age is the difference of a user's age at the destination node and its age at the BS    
    mad_age_dest = {tuple(x):(eval_env.dest_age[tuple(x)] - eval_env.UAV_age[x[0]]) for x in eval_env.tx_rx_pairs}

    if verbose:
        print(f"age difference of users at dest is {mad_age_dest} where age at dest is {eval_env.dest_age}, age at UAV is {eval_env.UAV_age}")  
        print(f"mad_age_dest = {mad_age_dest}")
   
    ## pair selection for updating begins

    sampleDict_copy = copy.deepcopy(mad_age_dest)

    updated_users = []
    while len(updated_users) < eval_env.DL_capacity:

        itemMaxValue = max(sampleDict_copy.items(), key=lambda x: x[1])
        # print('Max Age Difference : ', itemMaxValue[1])
        listOfKeys = list() ## covered UAVs that will be removed once added to the selected_UAVs
        # Iterate over all the items in dictionary to find keys with max value
        for key, value in sampleDict_copy.items():
            if value == itemMaxValue[1]:
                listOfKeys.append(key)



        remaining_capacity = eval_env.DL_capacity - len(updated_users)
        if remaining_capacity > 0:
            # print(f"remaining_capacity = {remaining_capacity}")
            if len(listOfKeys) > remaining_capacity: ## listOfKeys can be filled with some combination
                updated_users.extend(random.sample(listOfKeys, remaining_capacity))
            else: ## entire listOfKeys can be entered
                updated_users.extend(listOfKeys)
        
        
        for items in listOfKeys:
            del sampleDict_copy[items] ## covered users will be deleted

        # print(f"sampleDict_copy for next loop is {sampleDict_copy} with dest_age = {eval_env.dest_age} and selected_UAVs = {selected_UAVs}")
    
    updated_users = [list(x) for x in updated_users]
    
    if verbose: 
        print(f"updated_users = {updated_users}")
        
    ## users for updating completed, then select user for sampling
    
    sampling_possibilities = [] ## for each UAV, combination of the capacity number of users
    for m in eval_env.UAV_list:
        users_UAV = eval_env.act_coverage[m] # users under the UAV m
        if len(users_UAV) < eval_env.UL_capacity:
            sample = list(combinations(users_UAV, len(users_UAV)))
            sampling_possibilities.extend(sample)
        else:
            sample = list(combinations(users_UAV, eval_env.UL_capacity))
            sampling_possibilities.extend(sample)


    ## all sampling actions collected
    
    ## select users to sample
    
    selected_users_overall = []
    
    for UAV in eval_env.UAV_list: # make a dict for each UAV storing the age of the users under it
        
        selected_users_UAV = [] ## users selected under this UAV
        
        users_covered = eval_env.act_coverage[UAV]
        users_age = {user:eval_env.UAV_age[user] for user in users_covered}  ## dict
        
        if verbose:
        
            print(f"under UAV {UAV}, users covered are {users_covered} and their ages are {users_age} when the general UAV age was {eval_env.UAV_age}")
              
        
        while len(selected_users_UAV) < eval_env.UL_capacity and len(users_age) > 0: ## users selected under this UAV should be less than capacity but also in cases when UAV has less users, second condition here invoked
        
            itemMaxValue = max(users_age.items(), key=lambda x: x[1])
            # print('Max Age  : ', itemMaxValue[1])
            listOfKeys = list() ## covered UAVs that will be removed once added to the selected_users_UAV
            # Iterate over all the items in dictionary to find keys with max value
            for key, value in users_age.items():
                if value == itemMaxValue[1]:
                    listOfKeys.append(key)


            remaining_capacity = eval_env.UL_capacity - len(selected_users_UAV)
            if remaining_capacity > 0:
                # print(f"remaining_capacity = {remaining_capacity}")
                if len(listOfKeys) > remaining_capacity: ## listOfKeys can be filled with some combination
                    selected_users_UAV.extend(random.sample(listOfKeys, remaining_capacity))
                else: ## entire listOfKeys can be entered
                    selected_users_UAV.extend(listOfKeys)
            
            
            for items in listOfKeys:
                del users_age[items] ## covered users will be deleted

        if verbose:
            print(f"users sampled under UAV {UAV} are {selected_users_UAV}")
        
        selected_users_overall.extend(selected_users_UAV)
        
    sampled_users = selected_users_overall

    if verbose:
        print(f"sampled_users = {sampled_users}")
    
############################################

    mad_action = list(itertools.product([sampled_users], [updated_users]))
        
    ## finally collect the result and compare with the action space of the environment
    updated_users = sorted(mad_action[0][1]) ## doesn't affect so keep it. but if due to this 11-10 and 10-11 are same, wrong
    sampled_users = sorted(mad_action[0][0])
       
    actual_mad_action = None # this has to change
    
    for action in range(eval_env.action_size):
        eval_action = eval_env.map_actions(action)
        # if verbose:
        #     print(f"\nm = {action}, eval_action={eval_action}, eval_action[0]={list(eval_action[0])}, eval_action[1]={list(eval_action[1])}, sampled_users = {sampled_users}, updated_UAVs = {updated_UAVs}")
        
        if sorted(list(eval_action[0]))==sorted(sampled_users) and sorted(list(eval_action[1]))==sorted(updated_users):
            actual_mad_action = action
            break

    if verbose:
        print(f"updated_users are {updated_users}, sampled_users = {sampled_users}, mad_action = {mad_action}, actual_mad_action = {actual_mad_action}")
        
    assert actual_mad_action!=None
        
    return actual_mad_action    


def mad_scheduling(I, drones_coverage, folder_name, deployment, packet_update_loss, packet_sample_loss, periodicity, adj_matrix, tx_rx_pairs, tx_users):  ## maximal age difference
    print(f"\n\nmad started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss}, periodicity = {periodicity}, tx_rx_pairs = {tx_rx_pairs}, tx_users = {tx_users}, UL_capacity = {UL_capacity}, DL_capacity = {DL_capacity} and {deployment} deployment")
    print(f"\n\nmad started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss}, periodicity = {periodicity}, tx_rx_pairs = {tx_rx_pairs}, tx_users = {tx_users}, UL_capacity = {UL_capacity}, DL_capacity = {DL_capacity} and {deployment} deployment", file = open(folder_name + "/results.txt", "a"), flush = True)
    # do scheduling for MAX_STEPS random_episodes times and take the average
    final_step_rewards = []
    overall_ep_reward  = []
    UAV_returns        = []
    
    all_actions = [] # just saving all actions to see the distribution of the actions
    
    age_dist_UAV =  {} ## stores the episode ending (after MAX_STEPS) age for each user at UAV
    age_dist_dest  =  {} ## stores the episode ending (after MAX_STEPS) age for each pair at destination
    
    sample_time = {}
    for ii in periodicity.keys():
        sample_time[ii] = []
    
    attempt_sample = []
    success_sample = []
    attempt_update = []
    success_update = []
    
    
    for ep in range(random_episodes): # how many times this policy will be run, similar to episode

        ep_reward = 0
        eval_env = UAV_network(I, drones_coverage, "eval_net", folder_name, packet_update_loss, packet_sample_loss, periodicity, adj_matrix, tx_rx_pairs, tx_users) ## will have just the eval env here

        eval_env.reset() # initializes age

        episode_wise_attempt_sample = 0
        episode_wise_success_sample = 0
        episode_wise_attempt_update = 0
        episode_wise_success_update = 0    

        
        if ep==0:
            print(f"\nmad scheduling and {deployment} placement with {I} users, coverage is {eval_env.act_coverage}, DL_capacity is {eval_env.DL_capacity}, UL_capacity = {eval_env.UL_capacity}, action space size is {eval_env.action_size} and they are {eval_env.actions_space} \n\n", file = open(folder_name + "/action_space.txt", "a"), flush = True)
            print(f"\nmad scheduling and {deployment} placement with {I} users, coverage is {eval_env.act_coverage}, DL_capacity is {eval_env.DL_capacity}, UL_capacity = {eval_env.UL_capacity}, action space size is {eval_env.action_size} ", file = open(folder_name + "/results.txt", "a"), flush = True)
            
            age_dist_UAV.update(eval_env.age_dist_UAV) # age_dist_UAV will have the appropriate keys and values
            age_dist_dest.update(eval_env.age_dist_dest) # age_dist_dest will have the appropriate keys and values ## working 
            
        eval_env.current_step  = 1
            
        for i in eval_env.user_list:
            eval_env.tx_attempt_UAV[i].append(0) # 0 will be changed in _step for every attempt
            # age_dist_UAV[i].append(eval_env.UAV_age[i]) # eval_env.UAV_age has been made to 1 by now

        for i in eval_env.tx_rx_pairs:
            eval_env.tx_attempt_dest[tuple(i)].append(0) # 0 will be changed in _step for every attempt
            # age_dist_dest[tuple(i)].append(eval_env.dest_age[tuple(i)]) # eval_env.dest_age has been made to 1 by now
                
        for i in range(eval_env.action_size):
            eval_env.preference[i].append(0) 
            
        for x in range(MAX_STEPS):
            # print(x)
            # print("inside MAD - ", x, " step started")      ## runs MAX_STEPS times       
            selected_action = find_mad_action(eval_env)
            all_actions.append(selected_action)
            # print("all_actions", type(all_actions))
            eval_env.preference[selected_action][-1] = eval_env.preference[selected_action][-1] + 1
            action = eval_env.map_actions(selected_action)
            updated_users = list(action[1])
            sampled_users = list(action[0])

                
            if verbose:
                print(f"\n\ncurrent episode = {ep}\n\n")
                print(f" step = {eval_env.current_step}, mad selection is {selected_action}, actual action is {action}, updated_users = {updated_users}, sampled_users = {sampled_users}")
                
            if eval_env.current_step==1: ## updating
            # step 1 so dest has nothing to get from UAV
                for k in eval_env.tx_rx_pairs:
                    eval_env.dest_age[tuple(k)] = eval_env.dest_age[tuple(k)]+1
                    
            else: # not time step = 1
                for i in eval_env.tx_rx_pairs: ## updating
                    if i in updated_users:
                        # ## find associated UAV
                        # for kk in eval_env.act_coverage:
                        #     if i in eval_env.act_coverage[kk]:
                        #         associated_UAV = kk
                        ##
                        episode_wise_attempt_update = episode_wise_attempt_update + 1
                        eval_env.tx_attempt_dest[tuple(i)][-1] = eval_env.tx_attempt_dest[tuple(i)][-1] + 1
                        chance_update_loss = round(random.random(), 2)
                        if verbose:
                            # print(f"user {i}'s associated UAV is {associated_UAV}")
                            print(f"for pair {i}, chance_update_loss = {chance_update_loss} and eval_env.update_loss = {eval_env.update_loss[tuple(i)]} ")
                        if chance_update_loss > eval_env.update_loss[tuple(i)]:
                            if verbose:
                                print(f"pair {i} was selected to be updated. so pair {i}'s age at dest will become {i[0]}'s age at the UAV +1, i.e. eval_env.UAV_age[i[0]]+1={eval_env.UAV_age[i[0]] + 1}")
                            eval_env.dest_age[tuple(i)] = eval_env.UAV_age[i[0]] + 1 # age for the next slot, like how I update current_sample in my SWIFT work
                            episode_wise_success_update = episode_wise_success_update + 1
                        
                        else:
                            eval_env.dest_age[tuple(i)] = eval_env.dest_age[tuple(i)] + 1
                            if verbose:
                                print(f'pair {i} was updated but had update failure')
                                
                    else:
                        if verbose:
                            print("pair ", i, " was not updated")
                        eval_env.dest_age[tuple(i)] = eval_env.dest_age[tuple(i)] + 1
                            
            ## updating process done
            
            ## sampling process start
            
            for i in eval_env.user_list: 
                if i in sampled_users:
                    sample_time[i].append(eval_env.current_step)
                    chance_sample_loss = round(random.random(),2)
                    eval_env.tx_attempt_UAV[i][-1] = eval_env.tx_attempt_UAV[i][-1] + 1
                    episode_wise_attempt_sample = episode_wise_attempt_sample + 1
                    if verbose:
                        print(f" for user {i}, chance_sample_loss = {chance_sample_loss} and eval_env.sample_loss = {eval_env.sample_loss[i]} ")
                    if chance_sample_loss > eval_env.sample_loss[i]:
                        if x%periodicity[i]==0:
                            if verbose:
                                print("slot = ", x, " - user ", i, " period = ", periodicity[i], " was selected to sample and sampled")
                            eval_env.UAV_age[i] = 1 # age for the next slot, like how I update current_sample in my SWIFT work
                            episode_wise_success_sample = episode_wise_success_sample + 1 
                        else:
                            if verbose:
                                print("slot = ", x, " - user ", i, " period = ", periodicity[i], " was selected to sample but not sampled")                    
                    else:
                        eval_env.UAV_age[i] = eval_env.UAV_age[i] + 1
                        if verbose:
                            print(f'user {i} was sampled but had sample failure')
    
                else:
                    if verbose:
                        print("user ", i, " was not sampled")
                    eval_env.UAV_age[i] = eval_env.UAV_age[i] + 1
                    
                    
            ## sampling process done
            
            if verbose:
                print(f"time = {eval_env.current_step}, sample_time = {sample_time}")
                print(f"tx_attempt_UAV has become {eval_env.tx_attempt_UAV} and tx_attempt_dest has become {eval_env.tx_attempt_dest}")
                         
                print(f"\n step = {eval_env.current_step} of episode {ep} ended, UAV_age = {eval_env.UAV_age} with sum = {np.sum(list(eval_env.UAV_age.values()))}, dest_age = {eval_env.dest_age} with sum = {np.sum(list(eval_env.dest_age.values()))}, tx_attempt_UAV = {eval_env.tx_attempt_UAV}, tx_attempt_dest = {eval_env.tx_attempt_dest}") ##, preference = {eval_env.preference}")
                
                print(f"episode_wise_attempt_sample = {episode_wise_attempt_sample}")
                print(f"episode_wise_success_sample = {episode_wise_success_sample}")
                print(f"episode_wise_attempt_update = {episode_wise_attempt_update}")
                print(f"episode_wise_success_update = {episode_wise_success_update}")
                # time.sleep(2)  
                
            if eval_env.current_step==MAX_STEPS:
                final_reward = np.sum(list(eval_env.dest_age.values()))
                
            eval_env.current_step += 1
            ep_reward = ep_reward + np.sum(list(eval_env.dest_age.values()))
          
        
        attempt_sample.append(episode_wise_attempt_sample)
        success_sample.append(episode_wise_success_sample)
        attempt_update.append(episode_wise_attempt_update)
        success_update.append(episode_wise_success_update)
        
        final_step_rewards.append(final_reward)
        overall_ep_reward.append(ep_reward)
        UAV_returns.append(sum(eval_env.UAV_age.values()))
        

        for i in eval_env.user_list:
            age_dist_UAV[i].append(eval_env.UAV_age[i]) 

        for i in eval_env.tx_rx_pairs:
            age_dist_dest[tuple(i)].append(eval_env.dest_age[tuple(i)]) 
            
       
        if verbose:
            print(f"age_dist_UAV = {age_dist_UAV}, age_dist_dest = {age_dist_dest}")
            print(f"results for step {eval_env.current_step-1} of episode {ep}")
            print(f"attempt_sample = {attempt_sample}")
            print(f"success_sample = {success_sample}")
            print(f"attempt_update = {attempt_update}")
            print(f"success_update = {success_update}")
            # time.sleep(10)
            print(f"\n*****************************************************\n")

        
    pickle.dump(UAV_returns, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_UAV_returns.pickle", "wb"))
    pickle.dump(age_dist_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_age_dist_UAV.pickle", "wb"))
    pickle.dump(age_dist_dest, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_age_dist_dest.pickle", "wb"))
    
    pickle.dump(eval_env.sample_time, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_sample_time.pickle", "wb"))
    
    pickle.dump(attempt_sample, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_attempt_sample.pickle", "wb"))
    pickle.dump(success_sample, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_success_sample.pickle", "wb"))
    pickle.dump(attempt_update, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_attempt_update.pickle", "wb"))
    pickle.dump(success_update, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_success_update.pickle", "wb"))
    
    
    print("\nMAD scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", np.mean(overall_ep_reward), " : end with final state of ", eval_env._state, " with shape ", np.shape(eval_env._state))
    
    print("\nMAD scheduling ", deployment, " placement, ", I, " users - avg of final_step_rewards = ", np.mean(final_step_rewards), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", np.mean(overall_ep_reward), " : end with final state of ", eval_env._state, " with shape ", np.shape(eval_env._state), file = open(folder_name + "/results.txt", "a"), flush = True)
   
    assert(len(final_step_rewards)==len(final_step_rewards))
    return overall_ep_reward, final_step_rewards, all_actions
    