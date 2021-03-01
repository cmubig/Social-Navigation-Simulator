# Social Navigation Simulator

<img src="docs/_static/combo.gif" alt="Agents spelling ``CADRL''">

A social navigation simulator for simulating a more realistic pedestrian crowd, with
various settings and algorithms available to generate dense and rare crowd scenarios.

This simulator is developed based on the original work from [here](https://github.com/mit-acl/gym-collision-avoidance):

---
**References**  


**Journal Version:** M. Everett, Y. Chen, and J. P. How, "Collision Avoidance in Pedestrian-Rich Environments with Deep Reinforcement Learning", in review, [Link to Paper](https://arxiv.org/abs/1910.11689)

**Conference Version:** M. Everett, Y. Chen, and J. P. How, "Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018. [Link to Paper](https://arxiv.org/abs/1805.01956), [Link to Video](https://www.youtube.com/watch?v=XHoXkWLhwYQ)

This repo also contains the trained policy for the SA-CADRL paper (referred to as CADRL here) from the proceeding paper: Y. Chen, M. Everett, M. Liu, and J. P. How. “Socially Aware Motion Planning with Deep Reinforcement Learning.” IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Vancouver, BC, Canada, Sept. 2017. [Link to Paper](https://arxiv.org/abs/1703.08862)  

Social Force Model  
Helbing, D., & Molnar, P. (1995). Social force model for pedestrian dynamics. Physical review E, 51(5), 4282.
[link to paper](https://arxiv.org/abs/cond-mat/9805244)

**Implementation References**  

Social Force Implementation modified from [here](https://github.com/svenkreiss/socialforce)

---

### Download and install  
```
git clone https://github.com/cmubig/Social-Navigation-Simulator.git --recurse-submodules
cd Social-Navigation-Simulator
sudo ./install.sh
```  
---

### Example Usage  
```
cd experiments
./run_test_experiments
```

---  
### Different mode usage
exp1 is for ( algorithm-generated dataset using settings from dataset (e.g. UNIV ) )  
exp2 is for ( algorithm-generated dataset using population density from 0.1 to 1.0 )  
exp3 is for ( 1 vs n-1 ) mixed algorithms case, where 1 agents running algorithm A vs n-1 agents running algorithm B  
exp4 is for ( 50% vs 50% ) mixed algorithms case, where 50% agents running algorithm A vs 50% agents running algorithm B  

corresponding command format:  
  
```bash
# For exp1
python3 src/run_experiment_1.py --output_name "exp1_ETH_CADRL" --experiment_num 1 --algorithm_name "CADRL"\ 
--experiment_iteration 3 --timeout 60 --dataset_name "ETH"
```
  
```bash
# For exp2
python3 src/run_experiment_1.py --output_name "exp2_0.1_CADRL" --experiment_num 2 --algorithm_name "CADRL"\
--experiment_iteration 3 --timeout 60 --population_density 0.1
```

```bash
# For exp3  ( --algorithm_name [ "algorithm_A", "algorithm_B" ] )
python3 src/run_experiment_1.py --output_name "exp3_0.3_CADRL_RVO" --experiment_num 3 --algorithm_name ["CADRL","RVO"]\
--experiment_iteration 3 --timeout 60 --population_density 0.3
```
  
```bash
# For exp4  ( --algorithm_name [ "algorithm_A", "algorithm_B" ] )
python3 src/run_experiment_1.py --output_name "exp4_0.3_CADRL_RVO" --experiment_num 4 --algorithm_name ["CADRL","RVO"]\
--experiment_iteration 3 --timeout 60 --population_density 0.3
```

By following the format above, create your simulation command,   
comment out the default simulation command inside run_test_experiment.sh,  
put your command in the end of run_test_experiment.sh


---

### Miscellaneous
To change the __plot_while_simulation__ or __output_animation__ setting, change experiment/src/master_config_deploy.py
```python
self.SHOW_EPISODE_PLOTS  = False    # plot and show while simulation? 
self.SAVE_EPISODE_PLOTS  = self.ANIMATE_EPISODES = True  #save simulation result as animation as well?
```
---

### About the original simulator code

Please see [the documentation](https://gym-collision-avoidance.readthedocs.io/en/latest/)!

### Original code citation:

```
@inproceedings{Everett18_IROS,
  address = {Madrid, Spain},
  author = {Everett, Michael and Chen, Yu Fan and How, Jonathan P.},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  date-modified = {2018-10-03 06:18:08 -0400},
  month = sep,
  title = {Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning},
  year = {2018},
  url = {https://arxiv.org/pdf/1805.01956.pdf},
  bdsk-url-1 = {https://arxiv.org/pdf/1805.01956.pdf}
}
```
