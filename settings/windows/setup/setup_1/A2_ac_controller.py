# Assignment 2: Air Conditioner controller using fuzzy logic (Mamdani)
# pip install scikit-fuzzy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. Inputs and output with their ranges (universe of discourse)
temp     = ctrl.Antecedent(np.arange(10, 41, 1), 'temperature')   # degrees C
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')      # percent
power    = ctrl.Consequent(np.arange(0, 101, 1), 'power')         # AC power %

# 2. Membership functions (triangular)
temp['cold']        = fuzz.trimf(temp.universe, [10, 10, 22])
temp['comfortable'] = fuzz.trimf(temp.universe, [18, 24, 30])
temp['hot']         = fuzz.trimf(temp.universe, [26, 40, 40])

humidity['low']    = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['medium'] = fuzz.trimf(humidity.universe, [20, 50, 80])
humidity['high']   = fuzz.trimf(humidity.universe, [50, 100, 100])

power['low']    = fuzz.trimf(power.universe, [0, 0, 40])
power['medium'] = fuzz.trimf(power.universe, [30, 50, 70])
power['high']   = fuzz.trimf(power.universe, [60, 100, 100])

# 3. Rule base  (& = AND = min, | = OR = max)
rules = [
    ctrl.Rule(temp['cold'], power['low']),
    ctrl.Rule(temp['comfortable'] & humidity['low'], power['low']),
    ctrl.Rule(temp['comfortable'] & (humidity['medium'] | humidity['high']), power['medium']),
    ctrl.Rule(temp['hot'] & humidity['low'], power['medium']),
    ctrl.Rule(temp['hot'] & (humidity['medium'] | humidity['high']), power['high']),
]

# 4. Build system and simulate
ac = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

tests = [(15, 30), (24, 60), (32, 70), (38, 90)]
for t, h in tests:
    ac.input['temperature'] = t
    ac.input['humidity'] = h
    ac.compute()   # fuzzification -> rules -> aggregation -> centroid defuzzification
    print(f"Temp = {t} C, Humidity = {h} %  ->  AC Power = {ac.output['power']:.2f} %")

# Optional graphs:
# import matplotlib.pyplot as plt
# temp.view(); humidity.view(); power.view(sim=ac); plt.show()
