import pytest
import numpy as np

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Reinforcement_Learning_System import Reinforcement_Learning_System

def test_select_state():
    final_policy_for_step = [0.1, 0.2, 0.3, 0.4]
    RLS = Reinforcement_Learning_System()
    action = RLS.select_state(final_policy_for_step)
    assert action in [0, 1, 2, 3]
    final_policy_for_step = [0,0,0,1]
    assert action==3
    final_policy_for_step = [0,0,1,0]
    assert action==2
    final_policy_for_step = [0,1,0,0]
    assert action==1
    final_policy_for_step = [1,0,0,0]
    assert action==0


