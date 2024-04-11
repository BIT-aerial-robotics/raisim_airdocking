# raisim_airdocking

## Dependencies
- [RaiSim](https://raisim.com/sections/Installation.html)

- [Aquaml](https://github.com/BIT-aerial-robotics/AquaML)


### Installation



#### Step 0

##### Create directory for all projects:

```
mkdir ~/raisim_worspace 
cd ~/raisim_workspace
```

Instead of `~/raisim_worspace` you could use any directory you like. It is given just as an example

#### Step 1

Clone this repository:

```
cd ~/raisim_worspace
git clone https://github.com/BIT-aerial-robotics/raisim_airdocking.git
cd raisim_airdocking
```

## Training



The repository provides a training example connected to the environment based on IQL, which can be referenced for training purposes.

```
cd iql
python IQLAirDocking.py
```

## Test



```
cd iql
python test_iql_airdocking.py
```

Regarding relevant path information, adjustments can be made according to one's own directory structure.