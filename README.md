
<img src="docs/_static/combo.gif" alt="Agents spelling ``CADRL''" width=300>

# Social Navigation Simulator (SNS) 

**Social Navigation Simulator (SNS) is developed based on [Gym-Collision-Avoidance (GCA)](https://github.com/mit-acl/gym-collision-avoidance) to simulate diverse types of realistic pedestrian crowds, with
various settings and algorithms available to generate dense and rare crowd scenarios.**

The SNS simulator addtionally provides the following features, in addition to what is available in the GCA simulator:  

(Need to elaborate and clearly describe each advantage)

- Addition of more social navigation policies (list specific algorithms and citations here)
- Support different scene experiment setups to generate dense and rare crowd scenarios.
- Dynamic number of agents in a scene (There can be a different number of agents during the same simulation)
- Asynchronous entrance and exit of agents in a scene ( To simulate the dataset more accurately )
- Collision timeout and resume ( Collided agents will halt their action for N seconds )
- Output agents trajectories during simulation, in the format of datasets such as ETH,ZARA1,ZARA2.
- Agents can be easily configured to use different social navigation algorithms as their policies. 
  (e.g. CADRL vs Socialforce )

## Project Contributors  
Sam Shum (cshum@andrew.cmu.edu)

Advaith Sethuraman

Dapeng Zhao

PI: [Jean Oh](http://www.cs.cmu.edu/~jeanoh/)

---
<!--
This simulator is developed based on the original work from [here](https://github.com/mit-acl/gym-collision-avoidance):  

**Journal Version:** M. Everett, Y. Chen, and J. P. How, "Collision Avoidance in Pedestrian-Rich Environments with Deep Reinforcement Learning", in review, [Link to Paper](https://arxiv.org/abs/1910.11689)

**Conference Version:** M. Everett, Y. Chen, and J. P. How, "Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018. [Link to Paper](https://arxiv.org/abs/1805.01956), [Link to Video](https://www.youtube.com/watch?v=XHoXkWLhwYQ)

This repo also contains the trained policy for the SA-CADRL paper (referred to as CADRL here) from the proceeding paper: Y. Chen, M. Everett, M. Liu, and J. P. How. “Socially Aware Motion Planning with Deep Reinforcement Learning.” IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Vancouver, BC, Canada, Sept. 2017. [Link to Paper](https://arxiv.org/abs/1703.08862)  
-->

## Download and install  
```
git clone https://github.com/cmubig/Social-Navigation-Simulator.git --recurse-submodules
cd Social-Navigation-Simulator
sudo ./install.sh
```  
## Library Requirement and Suggested Version
The following libraries will be automatically installed in the virtual environment(venv) created by the installation script.  
Your original python workspace, library versions will not be affected.  

The following libraries should already exist in the venv, when you run the install script.

* tensorflow==1.15.2  
* Pillow  
* PyOpenGL   
* pyyaml  
* matplotlib>=3.0.0  
* shapely  
* pytz  
* imageio==2.4.1  
* gym  
* moviepy  
* pandas  
* networkx  
* torch===1.2.0  
* attrdict  

## Hardware, Software System Requirements  
These are the specifications of the hardware used during development.  
Different combinations of hardware could work and will yield different speeds.  

* Intel(R) Core(TM) i7-4720HQ CPU @ 2.60GHz
* 8GB ram
* GeForce GTX 960M (4GB)  

These are the specifications of the software used during development.
Different combinations of software might work.  

* Ubuntu 18.04
* CUDA 10.0  (CUDA Version 10.0.326 / Cuda compilation tools, release 10.0, V10.0.326)
* cuDNN 7.6.5
* Python 3.6.9


---
## Example Usage  
```
cd experiments
./run_test_experiments
```

## Different mode usage
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


## Miscellaneous
To change the __plot_while_simulation__ or __output_animation__ setting, change experiment/src/master_config_deploy.py
```python
self.SHOW_EPISODE_PLOTS  = False    # plot and show while simulation? 
self.SAVE_EPISODE_PLOTS  = self.ANIMATE_EPISODES = True  #save simulation result as animation as well?
```
---

## References

**Social LSTM**  
Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L., & Savarese, S. (2016). Social lstm: Human trajectory prediction in crowded spaces. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 961-971).  
[link to paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html)

**Social GAN**  
Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., & Alahi, A. (2018). Social gan: Socially acceptable trajectories with generative adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2255-2264).  
[link to paper](https://arxiv.org/abs/1803.10892)

**Social Force Model**  
Helbing, D., & Molnar, P. (1995). Social force model for pedestrian dynamics. Physical review E, 51(5), 4282.  
[link to paper](https://arxiv.org/abs/cond-mat/9805244)

**ORCA (RVO2)**  
Snape, J., Guy, S. J., Van Den Berg, J., & Manocha, D. (2014). Smooth coordination and navigation for multiple differential-drive robots. In Experimental Robotics (pp. 601-613). Springer, Berlin, Heidelberg.  
[link to paper](https://link.springer.com/chapter/10.1007/978-3-642-28572-1_41)

**Social-STGCNN**
Mohamed, A., Qian, K., Elhoseiny, M., & Claudel, C. (2020). Social-stgcnn: A social spatio-temporal graph convolutional neural network for human trajectory prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14424-14432).  
[link to paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.html)

**Constant Velocity Model (CVM)**  
Schöller, C., Aravantinos, V., Lay, F., & Knoll, A. (2020). What the constant velocity model can teach us about pedestrian motion prediction. IEEE Robotics and Automation Letters, 5(2), 1696-1703.  
[link to paper](https://ieeexplore.ieee.org/abstract/document/8972605?casa_token=5Eby3flWY1IAAAAA:sjTaJbAjP_dSKMA6kDT21HA6fTdyF1ucqWC9LeW-eYk45bDPeYR9BobApeI74UPL8W8VgwuYfg)

**Social-PEC (SPEC)**  
Zhao, D., & Oh, J. (2020). Noticing Motion Patterns: Temporal CNN with a Novel Convolution Operator for Human Trajectory Prediction. IEEE Robotics and Automation Letters.  
[link to paper](https://ieeexplore.ieee.org/abstract/document/9309403/?casa_token=pV4aFJU4-0UAAAAA:NRC5vkADgA7Jd4cmX9HcV4pXuqBxQxAx8-GugQIVSSiTqpOiehJZW1TYo4dBlLUDNWwxHDCCZg)




## Implementation References

Social Force Implementation modified from [here](https://github.com/svenkreiss/socialforce)  
Social LSTM Implementation modified from [here](https://github.com/quancore/social-lstm)  
Social GAN Implementation modified from [here](https://github.com/agrimgupta92/sgan)  
Social STGCNN Implementation modified from [here](https://github.com/abduallahmohamed/Social-STGCNN)  
Constant Velocity Model modified from [here](https://github.com/cschoeller/constant_velocity_pedestrian_motion)  
CADRL Implementation from [here](https://github.com/mit-acl/gym-collision-avoidance)  
RVO2 Implementation from [here](https://github.com/mit-acl/gym-collision-avoidance), [here](https://github.com/mit-acl/Python-RVO2/tree/56b245132ea104ee8a621ddf65b8a3dd85028ed2)  


---

## About the original simulator code

Please see [the documentation](https://gym-collision-avoidance.readthedocs.io/en/latest/)!


## Code citation for the original simulator version:

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



