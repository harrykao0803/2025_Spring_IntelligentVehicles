import math 


def calculate_Bi(msg_num, i, Ci):
    Bi = 0
    for j in range(i, msg_num):
        Bi = max(Bi, Ci[j])
    return Bi


def calculate_Ri(i, Bi, right_Qi, Ci, Ti, tau):
    Qi = right_Qi

    RHS = Bi
    for j in range(i):
        RHS += math.ceil((right_Qi + tau) / Ti[j]) * Ci[j]

    if RHS + Ci[i] > Ti[i]:
        print("This system is not schedulable.")
        return -1
    elif Qi == RHS:
        return Qi + Ci[i]
    else:
        return calculate_Ri(i, Bi, RHS, Ci, Ti, tau)


if __name__ == '__main__':

    with open('input.dat', 'r') as file:
        lines = file.readlines()

    msg_num = int(lines[0].strip())  # 17
    tau = float(lines[1].strip())    # 0.002

    Ci = []  # transmission time
    Ti = []  # period

    for line in lines[2:]:
        _, c, t = line.strip().split()
        Ci.append(float(c))
        Ti.append(int(t))

    for i in range(msg_num):
        Bi = calculate_Bi(msg_num, i, Ci)
        Ri = calculate_Ri(i, Bi, Bi, Ci, Ti, tau)
        print(f"message {i:<2}: {Ri:.3f}")

