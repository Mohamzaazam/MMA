# MASS (Muscle-Actuated Skeletal System)

![Teaser](png/Teaser.png)

## Abstract

This code implements a basic simulation and control for a full-body **Musculoskeletal** system. Skeletal movements are driven by the actuation of the muscles, coordinated by activation levels. Interfacing with Python and PyTorch, it enables the use of Deep Reinforcement Learning (DRL) algorithms such as Proximal Policy Optimization (PPO).

## Publications

**Scalable Muscle-actuated Human Simulation and Control**  
Seunghwan Lee, Kyoungmin Lee, Moonseok Park, and Jehee Lee  
*ACM Transactions on Graphics (SIGGRAPH 2019), Volume 37, Article 73.*

- **Project Page**: http://mrl.snu.ac.kr/research/ProjectScalable/Page.htm
- **YouTube**: https://youtu.be/a3jfyJ9JVeM
- **Paper**: http://mrl.snu.ac.kr/research/ProjectScalable/Paper.pdf

## Prerequisites

This project uses **[Pixi](https://pixi.sh/)** for dependency management and building. You do not need to manually install C++ libraries or Python packages.

- **OS**: Ubuntu 20.04+ or WSL2 (Ubuntu 24 recommended)
- **Pixi**: Install Pixi by running:
  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```

## How to Build

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/MASS.git
    cd MASS
    ```

2.  **Build the project**:
    Simply run the following command. Pixi will automatically install all dependencies (DART, Eigen, PyTorch, etc.) and compile the C++ code.
    ```bash
    pixi run build
    ```

## How to Run

### 1. Visualization (Render)
To see the simulation visualization:

```bash
./build/render data/metadata.txt
```
*Note: You need an X server (like VcXsrv) running if you are on WSL2.*

### 2. Training (Python)
To run the training script:

```bash
pixi run train
```
Or manually:
```bash
pixi shell
python python/main.py -d data/metadata.txt
```

### 3. TensorBoard (Visualize Training Metrics)
To visualize training metrics in real-time:

```bash
pixi run tensorboard
```

Then open your browser and navigate to `http://localhost:6006` to view the training metrics including:
- Loss (Actor, Critic, Muscle)
- Training metrics (Avg Return, Avg Reward, Noise, etc.)

### 4. Run Pre-trained Models
To view a pre-trained model (if you have one in `nn/` folder):

```bash
./build/render data/metadata.txt ./nn/your_model.pt ./nn/your_muscle_model.pt
```

## Project Structure

- **`core/`**: C++ core simulation code (Environment, Character, Muscle, etc.).
- **`render/`**: OpenGL/GLUT rendering code.
- **`python/`**: Python scripts for RL training and environment management.
- **`data/`**: Configuration files, skeletons, and motion data.
- **`build/`**: Compiled executables and libraries (created after build).

## Model Creation & Retargeting

*This module is an ongoing project.*

📊 Monitoring & Profiling
12. Profile Your Code
Add profiling to find bottlenecks:

bash
# For Python side
python -m cProfile -o profile.stats python/main.py -d data/metadata.txt

# For C++ side, use perf
perf record -g ./build/render data/metadata.txt
perf report
13. Monitor GPU Utilization
bash
watch -n 0.5 nvidia-smi
If GPU utilization < 80%, increase batch sizes or parallelize more.

### Requirements
- Maya
- MotionBuilder

There is a sample model in the `data/maya` folder. If you want to edit the model, you will need to write your own Maya-Python export code and XML writer to ensure the simulation code correctly reads the musculoskeletal structure. A rig model is also provided for retargeting new motions.