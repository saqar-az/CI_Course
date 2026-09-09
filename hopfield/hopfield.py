import numpy as np

class Hopfiled:
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.W = np.zeros((n_neurons, n_neurons)) # weight matrix the size of n*n filled with zeros

    # training to store the patterns, use hebbian learning: wij = (1/n) * sigma (xi xj) for i!=j
    def train(self, patterns):
        self.W = np.zeros((self.n, self.n)) # makes the weight matrix filled with zeros size of n*n
        for p in patterns:
            self.W += np.outer(p, p) # hebbian learning: sigma (xi xj) for i!=j
        np.fill_diagonal(self.W, 0) # every diaginal elements is set to zero (no loop with itself)
        self.W /= self.n # 1/n

    # calculate the energy function: -1/2 * sigam i sigma j wij *sisj
    def energy(self, state):
        """ E = -0.5 * s^T W s (biases set to zero). """
        # np.outer (state,state) -> makes sigma sisj
        return -0.5 * np.sum(self.W * np.outer(state, state))

    # each neuron is updated one step at a time, state is the starting state
    def update(self, state, max_iter=1000, return_energies=False):
        s = state.copy() # copy of the state
        energies = [self.energy(s)] if return_energies else None

        for _ in range(max_iter):
            changed = False # at start nothing has changed so flag is false if change happens it becomes true
            for i in np.random.permutation(self.n): # randomly updadting neurons based on index
                h = self.W[i] @ s # dot of self.w[i] and state -> hi = sigma wijsj
                new_val = np.sign(h) # if h>0 then new_val is +1, if h<0 then new_val is -1 and if h=0 then new_val is 0
                if new_val == 0:                # tie – keep old state
                    new_val = s[i]
                if new_val != s[i]:
                    s[i] = new_val # update
                    changed = True
                    if return_energies:
                        energies.append(self.energy(s))
            if not changed:
                break

        return (s, energies) if return_energies else s

    # recall phase
    def recall(self, state):
            return self.update(state, return_energies=True)

if __name__ == "__main__":

    # 2 patterns
    pattern1 = np.array([ 1,  1, -1, -1])
    pattern2 = np.array([-1, -1,  1,  1])

    # hopfiled with 4 neurons
    n_neurons = 4
    net = Hopfiled(n_neurons)
    net.train([pattern1, pattern2])
     
    # pattern 1 noisy
    noisy1 = np.array([-1,  1, -1, -1])

    # recall that runs asynchronous updates
    final_state, energies = net.recall(noisy1)

    print("final state: ", final_state)
    print("matches? ", np.array_equal(final_state, pattern1))
    print("\nenergy history: ")
    for i, e in enumerate(energies):
        print(f"step {i:2d}: {e:.4f}")