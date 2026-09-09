import numpy as np
import matplotlib.pyplot as plt

# function for drawing the membership function
def zoozanaqe(x, a, b, c, d):
    if x < a or x > d:
        return 0.0
    elif a <= x <= b:
        if b == a:
            return 1.0
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    elif c <= x <= d:
        if d == c:
            return 0.0
        return (d - x) / (d - c)
    else:
        return 0.0

# points to be send to zoozanaqe function to make the membership functions
error_sets = {
    'Cold':   [-5, -4, -1, 0],
    'Normal': [-2, -1, 2, 3],
    'Hot':    [ 2, 3, 4, 5]
}
power_sets = {
    'Low':    [0, 0, 20, 40],
    'Medium': [30, 40, 60, 70],
    'High':   [60, 70, 90, 100]
}

# input and output domain
error_domian = np.arange(-5,5.01,0.01)
power_domian = np.arange(0, 101, 1) 

# fuzzification function to compute membership degrees, returns true memberships
def fuzzify(inp):
    membership = {}
    for name, params in error_sets.items():  
        membership[name] = zoozanaqe(inp, *params)  
    return membership

# rules to get the output membership function based on fuzzy_input that has active memberships
def rules(fuzzy_input):
    # 1st rule
    y_cold = fuzzy_input['Cold'] 
    # 2nd rule  
    y_medium = fuzzy_input['Normal']  
    # 3rd rule
    y_hot = fuzzy_input['Hot']     

    def clip(activ_name, activ_deg):
        params = power_sets[activ_name]
        clipped = []
        for p in power_domian:
            original_mf = zoozanaqe(p, *params)
            clipped.append(min(original_mf, activ_deg))
        return np.array(clipped)

    clipped_low = clip('Low', y_hot)   
    clipped_med = clip('Medium', y_medium) 
    clipped_high = clip('High', y_cold)  
    aggregated = np.maximum(clipped_low, np.maximum(clipped_med, clipped_high))

    return aggregated, (clipped_low, clipped_med, clipped_high)

# getting the centroid based on the formula
def defuzzify_centroid(aggregated_mf, universe):
    numerator = np.sum(universe * aggregated_mf)
    denominator = np.sum(aggregated_mf)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def fuzzy_controller(error_input):
    fuzzy_in = fuzzify(error_input)
    aggregated, _ = rules(fuzzy_in)
    crisp_power = defuzzify_centroid(aggregated, power_domian)
    return crisp_power

def plots():
    plt.figure()
    for name, params in error_sets.items():
        y = [zoozanaqe(x, *params) for x in error_domian]
        plt.plot(error_domian, y, label=name, lw=2)
    plt.xlabel('error')
    plt.ylabel('membership degree')
    plt.grid(True)
    plt.show()

    plt.figure()
    for name, params in power_sets.items():
        y = [zoozanaqe(p, *params) for p in power_domian]
        plt.plot(power_domian, y, label=name, lw=2)
    plt.xlabel('power')
    plt.ylabel('membership degree')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plots()
    test = [-3.0, -1.0]
    for e in test:
        power_out = fuzzy_controller(e)
        print(f"error = {e}°C -> power = {power_out}%")