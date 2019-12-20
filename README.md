# DeltaDOGS
This is a repo that implements original Delta-DOGS algorithm to optimize black-box function without derivative information. 
The objective function is expensive-to-evaluate and no analytical expression is available. The 

## Prerequisites
```
Python 3
scipy==1.1.0
numpy==1.14.0
```

## How to use
Fork the repo onto local machine, enter the root folder. In terminal, run the command
```
python3 1Dexample.py
```
or
```
python3 2Dexample.py
```

## Options for ```DeltaDOGS```

## Result
The subroutine of the class object ```DeltaDOGS``` ```deltadogs_optimizer()``` returns the minimizer found by running the optimization solver. Regarding the 1D and 2D test problem, the optimization results are demonstrated in figures for each iteration. The final result containing all the evaluated data points and their corresponding function values are saved in ```data.mat```. 

All the plots are generated in the folder: ```root/plot```.

1. The optimization code will plot the values of objective function, discrete search function, and continuous search function in the folder `plot` under the root directory. E.g.:
![1](/figures/plot1D12.png)

2. After the optimization completes, the information of candidate point and distance will be plotted as follow:
![2](/figures/Candidate_point.png)
![3](/figures/Distance.png)
