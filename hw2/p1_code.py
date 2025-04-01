import random
import math

def read_input(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    n = int(lines[0].strip())  # Number of messages
    tau = float(lines[1].strip())  # Given threshold (not used in SA)
    
    messages = []
    for line in lines[2:]:
        parts = list(map(float, line.split()))
        messages.append((int(parts[0]), parts[1], parts[2]))  # (Pi, Ci, Ti)
    
    return n, tau, messages

def wcrt(priority_order, messages, tau):
    """Compute the worst-case response time sum based on priority ordering."""
    response_times = {}
    for i, i_priority in enumerate(priority_order):

        Bi = 0
        for j, j_priority in enumerate(priority_order):
            if j_priority >= i_priority:
                Bi = max(Bi, messages[j][1])

        Qi = Bi
        while True:
            RHS = Bi
            for k, k_priority in enumerate(priority_order):
                if k_priority < i_priority:
                    RHS += math.ceil((Qi + tau) / messages[k][2]) * messages[k][1]

            if RHS + messages[i][1] > messages[i][2]:
                return float('inf')
            if Qi == RHS:
                Ri = Qi + messages[i][1]
                break
            Qi = RHS

        if Ri > messages[i][2]:
            return float('inf')  # Violate constraint
        
        response_times[i_priority] = Ri
        
    return sum(response_times.values())

def simulated_annealing(n, tau, messages, T=100.0, cooling_rate=0.995):
    S = list(range(n))  # Initial order based on given priorities

    S_star = S[:]
    S_star_cost = wcrt(S_star, messages, tau)

    while T > 0.0000001:
        i, j = random.sample(range(n-1), 2)  # Randomly shuffle the order
        S[i], S[j] = S[j], S[i]
        S_prime_cost = wcrt(S, messages, tau)

        delta_cost = S_prime_cost - S_star_cost

        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            if delta_cost < 0:
                S_star_cost = S_prime_cost
                S_star = S[:]
        else:
            S[i], S[j] = S[j], S[i]

        T *= cooling_rate  # Decrease temperature

    return S_star, S_star_cost    

if __name__ == "__main__":
    input_file = "input.dat"
    n, tau, messages = read_input(input_file)

    # s = wcrt(list(range(n)), messages, tau)  # Initial WCRT for the given order
    # print(f"Initial WCRT: {s:.3f}")
    # lis = [12, 5, 3, 6, 2, 4, 9, 1, 7, 8, 11, 15, 0, 13, 14, 10, 16]  # Example priority order
    # print(wcrt(lis, messages, tau))

    best_priority, best_obj = simulated_annealing(n, tau, messages)
    
    print("Best priority order:")
    for p in best_priority:
        print(p)

    print(f"\nBest Response Time: {best_obj}")




