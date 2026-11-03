# Assignment 3: Fuzzy logic control of a simulated robotic arm
# Input: error = target angle - current angle, Output: angle change per step
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Variables
error = ctrl.Antecedent(np.arange(-180, 181, 1), 'error')
move  = ctrl.Consequent(np.arange(-20, 21, 1), 'move')

# 2. Membership functions
error['neg_big']   = fuzz.trimf(error.universe, [-180, -180, -45])
error['neg_small'] = fuzz.trimf(error.universe, [-90, -30, 0])
error['zero']      = fuzz.trimf(error.universe, [-10, 0, 10])
error['pos_small'] = fuzz.trimf(error.universe, [0, 30, 90])
error['pos_big']   = fuzz.trimf(error.universe, [45, 180, 180])

move['neg_big']   = fuzz.trimf(move.universe, [-20, -20, -8])
move['neg_small'] = fuzz.trimf(move.universe, [-12, -5, 0])
move['zero']      = fuzz.trimf(move.universe, [-2, 0, 2])
move['pos_small'] = fuzz.trimf(move.universe, [0, 5, 12])
move['pos_big']   = fuzz.trimf(move.universe, [8, 20, 20])

# 3. Rules: far from target -> move fast, close -> move slowly
rules = [
    ctrl.Rule(error['neg_big'],   move['neg_big']),
    ctrl.Rule(error['neg_small'], move['neg_small']),
    ctrl.Rule(error['zero'],      move['zero']),
    ctrl.Rule(error['pos_small'], move['pos_small']),
    ctrl.Rule(error['pos_big'],   move['pos_big']),
]
arm = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

# 4. Simulate arm moving from 0 to 120 degrees
target, angle = 120.0, 0.0
history = [angle]
for step in range(1, 101):
    e = target - angle
    arm.input['error'] = e
    arm.compute()
    angle += arm.output['move']
    history.append(angle)
    print(f"Step {step:3d} | Angle = {angle:7.2f} | Error = {target - angle:7.2f}")
    if abs(target - angle) < 0.5:
        print(f"Target {target} reached in {step} steps")
        break

plt.plot(history, marker='o')
plt.axhline(target, color='r', linestyle='--', label='Target')
plt.xlabel('Step'); plt.ylabel('Arm angle (deg)')
plt.title('Fuzzy Controlled Robotic Arm'); plt.legend(); plt.show()
