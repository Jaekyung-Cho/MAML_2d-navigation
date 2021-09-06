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

<br>
we trained three algorithms with tasks which the goal distribution is [-0.5, 0.5] and we evaluate with tasks which the goal distribution is [-1, 1]. We used 100 evaluation tasks and averaging them.
<br>
<br>
A blue dot is the goal, and a green dot is initial point(origin).  
A red dot is the moving robot.
  
Only 3 update is enough to solve 2d-navigation task for MAML.
<br>  
<img src="results/maml_0_update.png" style="height: 150px;">
<img src="results/maml_1_update.png" style="height: 150px;">
<img src="results/maml_2_update.png" style="height: 150px;">
<img src="results/maml_3_update.png" style="height: 150px;">
<img src="results/maml_4_update.png" style="height: 150px;">
<img src="results/maml_5_update.png" style="height: 150px;">
<br><br>
However baseline1 (pretrained) cannot solve 2d-navigation task. It looks like overfitted to training tasks.
<br><br> 
<img src="results/baseline1_0_update.png" style="height: 150px;">
<img src="results/baseline1_1_update.png" style="height: 150px;">
<img src="results/baseline1_2_update.png" style="height: 150px;">
<img src="results/baseline1_3_update.png" style="height: 150px;">
<img src="results/baseline1_4_update.png" style="height: 150px;">
<img src="results/baseline1_5_update.png" style="height: 150px;">
<br><br>
Baseline2 (random initialized) tend to be adapted to new tasks but much slower than MAML.
<br><br>
<img src="results/baseline2_0_update.png" style="height: 150px;">
<img src="results/baseline2_1_update.png" style="height: 150px;">
<img src="results/baseline2_2_update.png" style="height: 150px;">
<img src="results/baseline2_3_update.png" style="height: 150px;">
<img src="results/baseline2_4_update.png" style="height: 150px;">
<img src="results/baseline2_5_update.png" style="height: 150px;">
<br><br>
The log-scale reward summation graph is following.  
<br><br>
<img src="results/MAML_result.png" style="height: 200px;">
