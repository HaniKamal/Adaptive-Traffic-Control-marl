import os
import sys
import argparse
import matplotlib.pyplot as plt

from src.env import TrafficEnv
from src.maddpg import MADDPG
from src.utils import get_average_travel_time
from src.utils import get_average_CO2
from src.utils import get_average_fuel
from src.utils import get_average_length



parser = argparse.ArgumentParser()
parser.add_argument("-R", "--render", action="store_true",
                    help= "whether render while training or not")
args = parser.parse_args()

if __name__ == "__main__":

    # Before the start, should check SUMO_HOME is in your environment variables
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # configuration
    state_dim = 10
    action_dim = 2
    n_agents = 2
    n_episode = 10

    # Create an Environment and RL Agent
    env = TrafficEnv("gui") if args.render else TrafficEnv()
    agent = MADDPG(n_agents, state_dim, action_dim)

    # Train your RL agent
    performance_list = []
    co2_emission = []
    fuel_cons = []
    route_length = []
    for episode in range(n_episode):

        state = env.reset()
        reward_epi = []
        actions = [None for _ in range(n_agents)]
        action_probs = [None for _ in range(n_agents)]
        done = False

        while not done:
            # select action according to a given state
            for i in range(n_agents):
                action, action_prob = agent.select_action(state[i, :], i)
                actions[i] = action
                action_probs[i] = action_prob

            # apply action and get next state and reward
            before_state = state
            state, reward, done = env.step(actions)

            # make a transition and save to replay memory
            transition = [before_state, action_probs, state, reward, done]
            agent.push(transition)

            # train an agent
            if agent.train_start():
                for i in range(n_agents):
                    agent.train_model(i)
                agent.update_eps()

            if done:
                break

        env.close()
        average_traveling_time = round( get_average_travel_time(), 2)
        performance_list.append(average_traveling_time)

        average_length = get_average_length()
        route_length.append(average_length)

        average_CO2 = round(get_average_CO2() / average_length , 2) 
        co2_emission.append(average_CO2)
        
        average_fuel = round((get_average_fuel() / average_length) + 3 , 2)
        fuel_cons.append(average_fuel)

        

        print(f"Episode: {episode+1}\t Average Traveling Time:{average_traveling_time}\t Average CO2(g/km):{ average_CO2}\t Average Fuel Consumption(L/100km):{average_fuel}\t Eps:{round(agent.eps,2)}")

    # Save the model
    agent.save_model("results/trained_model.th")

    # Plot the performance
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10.8, 7.2), dpi=120)
    plt.plot(performance_list)
    plt.xlabel('# of Episodes')
    plt.ylabel('Average Traveling Time - sec')
    plt.title('Performance of MADDPG for raveling Time')
    plt.savefig('./results/performance.png')

    # Plot the Co2 Emissions
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10.8, 7.2), dpi=120)
    plt.plot(co2_emission)
    plt.xlabel('# of Episodes')
    plt.ylabel('Average CO2 Emissions - g/km')
    plt.title('Performance of MADDPG for CO2 Emissions')
    plt.savefig('./results/Co2_emissions.png')

    # Plot the Fuel Consumption
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10.8, 7.2), dpi=120)
    plt.plot(fuel_cons)
    plt.xlabel('# of Episodes')
    plt.ylabel('Average Fuel Consumption L/100km')
    plt.title('Performance of MADDPG for Fuel Consumption')
    plt.savefig('./results/fuel_consumption.png')

    