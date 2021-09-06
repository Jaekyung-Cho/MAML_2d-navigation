## MAML-RL 2D navigation
<br>

Source paper
+ Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." International Conference on Machine Learning. PMLR, 2017.
 
<br>

Implimented by 
+ Jaekyung Cho, Autonomous Robot Intelligence Lab, Seoul National Univ.  

<br>

Source codes  
+ 2D navigation environment (modified) : https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/maml_examples/point_env_randgoal.py  
+ PPO :  
https://github.com/nikhilbarhate99/PPO-PyTorch

<br>
<br>

---
### Training Each


> python maml.py --config=/configs/2d-navigation.yaml  
> python baseline1.py --config=/configs/2d-navigation.yaml  
> python baseline2.py --config=/configs/2d-navigation.yaml  
> python baseline3.py --config=/configs/2d-navigation.yaml  

<br>

### Training All

> python train_all.py --config=/configs/2d-navigation.yaml --maml --bs1 --bs2 --bs3  

result will be saved in **results** directory

<br>

### Test

> python test_all.py --config=/configs/2d-navigation.yaml

<br>

---

### Result 