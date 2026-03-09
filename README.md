# Phy-Embedded-Pre
## 📦 Data Download

The data files required for this project have been uploaded to Baidu Netdisk. Please obtain them via the link below:

* **Data File**: `data_collect_01.bag`
* **Download Link**:[Click here to download from Baidu Netdisk](https://pan.baidu.com/s/1-21Q_zwMervypL9VfSFevg)
* **Extraction Code**: `0413`

> 💡 **Tip**: It is recommended to copy the extraction code before clicking the link, as the system will usually fill it in automatically.

---

### 📂 Usage Instructions

Based on our tests, this project works perfectly on both **Ubuntu 18.04** and **Ubuntu 20.04**.

#### 1. Create a Virtual Environment
*Note: Make sure to activate the environment before installing the dependencies.*
```bash
conda create -n your_env_name python=3.9
conda activate your_env_name
pip install numpy opencv-python matplotlib scipy tqdm ultralytics rospkg
pip install torch torchvision torchaudio

#### 2. Process Bag Files to Generate Dataset
# Note: Please modify the bag file path inside the script before running
python generate_data.py

# Verify the validity and data flow of the dataset
python debug_dataflow.py

#### 3. Train and Evaluate the Prediction Network

Note: The `--data_dirs` argument is required. Please replace the path with your actual dataset directory.

**To train the model:**
```bash
# Basic training
python train_eval.py --mode train --data_dirs ./Data_1.5s/dataset_processed_01

# You can also specify the network and save directory
python train_eval.py --mode train --data_dirs ./Data_1.5s/dataset_processed_01 --exp ours --save_dir ./checkpoints_ours

To evaluate the model:
# Evaluate the model (it will automatically load 'best_model.pth' from the save_dir)
python train_eval.py --mode test --data_dirs ./Data_1.5s/dataset_processed_01 --save_dir ./checkpoints_ours

# Evaluate and generate visualization results
python train_eval.py --mode test --data_dirs ./Data_1.5s/dataset_processed_01 --save_dir ./checkpoints_ours --visualize

#### 4. ROS Deployment and Visualization
Open a terminal to play the ROS bag file:
rosbag play new_collect.bag

python deploy_ros_vis.py
