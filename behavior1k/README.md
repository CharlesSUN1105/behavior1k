# behavior1k

Code for behavior1k.

Tested with Python 3.10, but other versions should be ok.

# Install

1. Create and activate a conda environment.
2. Clone the repo (**note**, make sure to include the `--recursive` flag):
  - (ssh) `git clone --recursive git@github.com:cmower/behavior1k.git`
  - (https) `git clone --recursive https://github.com/cmower/behavior1k.git`
3. Change directory: `cd behavior1k`
4. Install [pinocchio](https://stack-of-tasks.github.io/pinocchio/): `conda install pinocchio -c conda-forge` (**note**, do __not__ use `pip`)
5. Install [spatial casadi](https://github.com/cmower/spatial-casadi): `pip install extern/spatial-casadi`
6. Install [figure_eight](https://github.com/cmower/figure_eight): `pip install extern/figure_eight`
7. Install `b1k`: `pip install -e .`

See `scripts` for examples.

# Programs

All planners/IKs are implementations of an abstract `Program` class (for mathematical *program*, since they are all methods based on numerical optimization schemes).
Each child class of `Program` contains the following abstract methods.
* `Program.reset(config)`: resets the problem
* `success = Program.solve()`: solves the problem
* `sol = Program.get_solution()`: retrieves the solution from the last call to `Program.solve`

**Note**
* For the `config`, for *all* joint-state related parameters, the ordering of the joints should be the same as in the `joint_names` attribute of the `Program` sub-class.
* *No default parameters* are given, please see the examples in `scripts/` for suggested values.

Check the code and examples (in `scripts/`) to see each planner/IK in action.

## `SingleArmAndTorsoPlanner`

```
from b1k.planner.single_arm import SingleArmAndTorsoPlanner
```

Plans in the joint space a trajectory only for a single arm (including the torso), given a joint state goal.

Cost function:
* Minimize joint velocity
* Minimize joint acceleration

Constraints
* dynamics (euler integration)
* Boundary conditions (initial/final joint position/velocity/acceleration)
* Joint position/velocity limits

Constructor
* `side` [`str`]: give `"left"` or `"right"` to specify which arm will move.
* `T` [`int`]: number of knot points in the trajectory. *Note*, higher can lead to longer compute times.
* `joint_limit_safety` [`float`]: number in `[0.0, 1.0]` that scales the joint position limits (default is `0.99`).

Configuration
```python
config = {
	"Q0": .., # np.ndarray (dof, T), warm start for joint position trajectory
	"dQ0": .., # np.ndarray (dof, T), warm start for joint velocity trajectory
	"ddQ0": .., # np.ndarray (dof, T), warm start for joint acceleration trajectory
	"duration": .., # float, duration for the trajectory [seconds]
	"q0": .., # np.ndarray (dof,) or list[float], robot current/initial joint position
	"qF": .., # np.ndarray (dof,) or list[float], robot final joint position
	"w_dQ": .., # float, weight on joint velocity cost term
	"w_ddQ": .., # float, weight on joint acceleration cost term
}
```

Output
```python
Q = planner.get_solution()
```
* `Q` [`dict`]: interpolated trajectory solution (absolute joint positions, ie in `[lower, upper]` joint limits). Keys are the joint names (from the URDF), and the values are functions of time (`t`).

## `SingleArmAndTorsoConstraintPlanner`

```
from b1k.planner.single_arm import SingleArmAndTorsoConstraintPlanner
```

Plans in the joint space a trajectory only for a single arm (including the torso), given a direction and distance for the hand to move.
The hand will move along the line given by the provided direction vector from the robot start state for a given distance.

**Note**, the planner seems to work reasonable well, however the hand does not seem to get to the end. I suspect a bug in how I define the points on the line.

Cost function
* minimize joint velocity/acceleration
* maintain gaze to end-effector position
* maintain zero end-effector rotational velocity

Constraints
* dynamics (euler integration)
* Boundary conditions (initial joint position, initial/final joint velocity/acceleration)
* Joint position/velocity limits
* Track defined line by the end-effector position

Constructor
* `side` [`str`]: give `"left"` or `"right"` to specify which arm will move.
* `T` [`int`]: number of knot points in the trajectory. *Note*, higher can lead to longer compute times.
* `joint_limit_safety` [`float`]: number in `[0.0, 1.0]` that scales the joint position limits (default is `0.99`).

Configuration
```python
config = {
	"Q0": .., # np.ndarray (dof, T), warm start for joint position trajectory
	"dQ0": .., # np.ndarray (dof, T), warm start for joint velocity trajectory
	"ddQ0": .., # np.ndarray (dof, T), warm start for joint acceleration trajectory
	"duration": .., # float, duration for the trajectory [seconds]
	"q0": .., # np.ndarray (dof,) or list[float], robot current/initial joint position
	"dr": .., # np.ndarray (3,) or list[float], direction vector for end-effector (no need to normalize, will be done automatically by planner)
	"d": .., # float, distance to move the end-effector in the given direction
	"w_dQ": .., # float, weight on joint velocity cost term
	"w_ddQ": .., # float, weight on joint acceleration cost term
	"w_r": .., # float, weight on the rotational velocity cost term
	"p_err_max": .., # float, maximum error for the end-effector position goal away from the line
	"maintain_gaze": .., # float, weight for maintain gaze cost term (set to `0.0` to turn off)
}
```

Output
```python
Q = planner.get_solution()
```
* `Q` [`dict`]: interpolated trajectory solution (absolute joint positions, ie in `[lower, upper]` joint limits). Keys are the joint names (from the URDF), and the values are functions of time (`t`).


## `WholeBodyPlanner`

```
from b1k.planner.whole_body import WholeBodyPlanner
```

Plans in the joint space a trajectory, given a joint state goal (including torso, and both arms).

Note, `WholeBodyPlanner` supports collision avoidance based on several spheres defined in the robot base frame.
In some cases, the planning problem can be tricky to solve.
A key reason often due to the initial guess: if the solver is not warm-started properly, then the solver can lead to mathematically feasible trajectories but not physically viable solutions (due to modelling errors).

One method you can use to resolve the issue is to
(1) solve the planning problem without collisions, and
(2) use the solution to the previous problem as the initial guess to the solver with collisions.
See `scripts/move_whole_body.py` for an example.

Cost function
* minimize joint velocity/acceleration

Constraints
* dynamics (euler integration)
* Boundary conditions (initial/final joint position/velocity/acceleration)
* Joint position/velocity limits
* Collision avoidance: given a number of spheres representing the constraint, collision spheres are defined on the robot body (run the script `scripts/visualize_collision_spheres.py`)

Constructor
* `T` [`int`]: number of knot points in the trajectory. *Note*, higher can lead to longer compute times.
* `collision_sphere_names` [`list[str]`]: a list of unique names for the collision spheres, default is empty list.
* `joint_limit_safety` [`float`]: number in `[0.0, 1.0]` that scales the joint position limits (default is `0.99`).

Configuration
```python
config = {
	"Q0": .., # np.ndarray (dof, T), warm start for joint position trajectory
	"dQ0": .., # np.ndarray (dof, T), warm start for joint velocity trajectory
	"ddQ0": .., # np.ndarray (dof, T), warm start for joint acceleration trajectory
	"duration": .., # float, duration for the trajectory [seconds]
	"q0": .., # np.ndarray (dof,) or list[float], robot current/initial joint position
	"qF": .., # np.ndarray (dof,) or list[float], robot final joint position
	"w_dQ": .., # float, weight on joint velocity cost term
	"w_ddQ": .., # float, weight on joint acceleration cost term
	"collision_spheres": .., # list[b1k.collisions.CollisionSphere], defines the positition and radii for each collision sphere
}
```

Output
```python
Q, dQ, ddQ = planner.get_solution()
```
* `Q` [`dict`]: interpolated trajectory solution (absolute joint positions, ie in `[lower, upper]` joint limits). Keys are the joint names (from the URDF), and the values are functions of time (`t`).
* `dQ` [`dict`]: interpolated trajectory solution (absolute joint velocities). Keys are the joint names (from the URDF), and the values are functions of time (`t`).
* `ddQ` [`dict`]: interpolated trajectory solution (absolute joint accelerations). Keys are the joint names (from the URDF), and the values are functions of time (`t`).

## `GlobalSingleArmIK`

```
from b1k.ik import GlobalSingleArmIK
```

Given a position goal (3d translation) and a rotation goal (quaternion, `xyzw`) for a given arm, the IK solution is found.
The solver includes the torso and arm in kinematic chain.

Cost function
* minimize the joint state distance to a given nominal configuration
* minimize the gaze distance to the target position (ie the solver will try to keep the position target at the center of the Zed camera frame, ie the camera in the head).

Constraints
* end-effector position/rotation
* Joint position limits

Constructor
* `side` [`str`]: give `"left"` or `"right"` to specify which arm will move.
* `joint_limit_safety` [`float`]: number in `[0.0, 1.0]` that scales the joint position limits (default is `0.99`).

Configuration
```python
config = {
	"q0": .., # np.ndarray (dof,) or list[float], robot current/initial joint position (used as initial guess too)
	"pG": .., # np.ndarray (3,) or list[float], goal position for end-effector (xyz)
	"rG": .., # np.ndarray (4,) or list[float], goal rotation (quaternion) for end-effector (xyzw)
	"p_err_max": .., # float, maximum error for the end-effector position goal
	"r_err_max": .., # float, maximum rotation error for the end-effector rotation goal
	"maintain_gaze": .., # float, weight for maintain gaze cost term (set to `0.0` to turn off)
}
```

Output
```python
q = planner.get_solution()
```
* `q` [`dict`]: joint positions IK solution. Keys are joint names (from the URDF) and the values are the joint positions as a `float`.

## `GlobalDualArmIK`

```
from b1k.ik import GlobalDualArmIK
```

Given a position goal (3d translation) and a rotation goal (quaternion, `xyzw`) for each arm, the IK solution is found.
The solver includes the torso and arm in kinematic chain and includes collision avoidance for a number of collision spheres.

Cost function
* minimize the joint state distance to a given nominal configuration
* minimize the gaze distance to the target position (ie the solver will try to keep the mid-point of the two position targets at the center of the Zed camera frame, ie the camera in the head).

Constraints
* end-effector position/rotation for both arms
* Joint position limits
* Collision avoidance: given a number of spheres representing the constraint, collision spheres are defined on the robot body (run the script `scripts/visualize_collision_spheres.py`)

Constructor
* `joint_limit_safety` [`float`]: number in `[0.0, 1.0]` that scales the joint position limits (default is `0.99`).
* `collision_sphere_names` [`list[str]`]: a list of unique names for the collision spheres, default is empty list.

Configuration
```python
config = {
	"q0": .., # np.ndarray (dof,) or list[float], robot current/initial joint position (used as initial guess too)
	"pG_left": .., # np.ndarray (3,) or list[float], goal position for left end-effector (xyz)
	"rG_left": .., # np.ndarray (4,) or list[float], goal rotation (quaternion) for left end-effector (xyzw)
	"pG_right": .., # np.ndarray (3,) or list[float], goal position for right end-effector (xyz)
	"rG_right": .., # np.ndarray (4,) or list[float], goal rotation (quaternion) for right end-effector (xyzw)
	"p_err_max": .., # float, maximum error for the end-effector position goal (for both hands)
	"r_err_max": .., # float, maximum rotation error for the end-effector rotation goal (for both hands)
	"maintain_gaze": .., # float, weight for maintain gaze cost term (set to `0.0` to turn off)
	"collision_spheres": .., # list[b1k.collisions.CollisionSphere], defines the positition and radii for each collision sphere
}
```

Output
```python
q = planner.get_solution()
```
* `q` [`dict`]: joint positions IK solution. Keys are joint names (from the URDF) and the values are the joint positions as a `float`.


## `DualArmIKController`

```
from b1k.ik import DualArmIKController
```

Given a position goal (3d translation) and a rotation goal (quaternion, `xyzw`) for each arm, the IK solution is found.
The solver includes the torso and arm in kinematic chain.

Cost function
* minimize end-effector distance to goal position/rotation
* minimize the joint state velocity
* minimize the gaze distance to the target position (ie the solver will try to keep the mid-point of the two end-effector positions at the center of the Zed camera frame, ie the camera in the head).
* minimize the joint state distance to a given nominal configuration

Constraints
* Joint position/velocity limits
* Linearized robot stability constraint (avoid tipping)

Constructor
* `dt` [`float`]: controller time step
* `joint_limit_safety` [`float`]: number in `[0.0, 1.0]` that scales the joint position limits (default is `0.99`).

Configuration
```python
config = {
	"q": .., # np.ndarray (dof,) or list[float], robot current/initial joint position
	"pG_left": .., # np.ndarray (3,) or list[float], goal position for left end-effector (xyz)
	"rG_left": .., # np.ndarray (4,) or list[float], goal rotation (quaternion) for left end-effector (xyzw)
	"pG_right": .., # np.ndarray (3,) or list[float], goal position for right end-effector (xyz)
	"rG_right": .., # np.ndarray (4,) or list[float], goal rotation (quaternion) for right end-effector (xyzw)
	"w_dq": .., # float, weight on the joint velocity cost term
	"w_p": .., # float, weight on both the end-effector position cost term
	"w_r": .., # float, weight on both the end-effector rotation cost term
	"w_gaze": .., # float, weight on both the end-effector rotation cost term
	"qn": .., # np.ndarray (dof,) or list[float], robot nominal joint position
	"w_qn": .., # float, weight on the nominal configuration cost term
}
```

Output
```python
dq = planner.get_solution()
```
* `dq` [`np.ndarray`]: joint velocities IK solution. Ordering of joints is the same as in ik.joint_names
