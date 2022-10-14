# RCM-AIRL #

This repository contains code and data used in Zhao, Z., & Liang, Y. (2022). Deep Inverse Reinforcement Learning 
for Route Choice Modeling. arXiv preprint arXiv:2206.10598.

## Introduction ##
  Route choice modeling is a fundamental task in transportation planning and demand forecasting. Classical methods generally adopt the discrete choice model (DCM) framework with linear utility functions and high-level route characteristics. While several recent studies have started to explore the applicability of deep learning for travel choice modeling, they are all path-based with relatively simple model architectures and cannot take advantage of detailed link-level features. Existing link-based models can capture the link-level features and dynamic nature of link choices within the trip, but still assume linear relationships and are generally computational costly. To address these issues, this study proposes a general deep inverse reinforcement learning (IRL) framework for link-based route choice modeling, which is capable of incorporating high-dimensional features and capturing complex relationships. 

## Data ##
  The trajectory data used in this research covers a specific region of Shanghai, China. All the trajectory and network data used in this study can be found in the data repository.

## Requirements ##
  * python (We use python3.8)
  * torch
  * pandas
  * numpy
  * collections  

## How to run ##
  * RCM-BC: run train_bc.py
  * RCM-GAIL: run train_gail.py
  * RCM-AIRL: run train_airl.py


  
