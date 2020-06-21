# Note to Reviewer

## Team
[Chi-shing Tong](cstong@umich.edu)

_Note_: I am submitting this project as individual.

# Programming a Real Self-Driving Car

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

## Tensorflow Object Detection API Local Installation(Windows 10 64bit)

1. Create conda environment `tf-gpu`

``` shell
conda env create --name tf-gpu
conda activate tf-gpu
```

2. Install tensorflow-gpu version `1.14` because object detection API still haven't been updated to work seamless with tensorflow 2.x.

``` shell
conda install tensorflow-gpu=1.14
```

3. Clone [Tensorflow Models](https://github.com/tensorflow/models.git) and checkout tag `v1.13.0`. This is the version that works with tensorflow `1.14`.

4. Downgrade `numpy` to version `1.17` due to incompatability issue of `1.18`.

5. Follow the `installation.md` under `/object_detection/g3doc`. Note that in windows `pycocotools` need to be installed by running the below commands:

``` shell
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

6. Change current directory to `/models/research` and run `python setup.py install` to install the libraries.

7. Change current directory to `/models/research/slim`, delete file `BUILD` and run `python setup.py install`. You can restore `BUILD` afterwards by running `git checkout HEAD -- BUILD`.

8. `~/anaconda3/envs/tf-gpu/lib/sit-packages/tensorflow/python/lib/io/file_io.py` has to be patched to successfully run the evaluation step.

``` python
def recursive_create_dir_v2(path):
  """Creates a directory and all parent/intermediate directories.

  It succeeds if path already exists and is writable.

  Args:
    path: string, name of the directory to be created

  Raises:
    errors.OpError: If the operation fails.
  """
  # pywrap_tensorflow.RecursivelyCreateDir(compat.as_bytes(path))
  os.makedirs(dirname, exist_ok=True)
```

## Tips

1. Add following to `main()` in `models/research/object_detection/model_main.py`. This prevent tensorflow allocating all the memories at once and only allow GPU to use as much memory as needed.

``` python
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, session_config=session_config)
```

2. Add `tf.logging.set_verbosity(tf.logging.INFO)` to `models/research/object_detection/model_main.py` will allow you see the training progress every 100 steps in terminal.

3. To adjust the frequency of saving checkpoint, you can simply add `save_checkpoints_secs` or `save_checkpoints_steps` in this line.
`models/research/object_detection/model_main.py`.

``` python
config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, 
                                session_config=session_config,
                                save_checkpoints_steps=1000)
```

 config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir) 
4. To prevent checkpoint saving to trigger evaluation, simply pass `throttle_secs` to `EvalSpec` in `models/research/object_detection/model_lib.py`

``` python
eval_specs.append(
        tf.estimator.EvalSpec(
          name=eval_spec_name,
          input_fn=eval_input_fn,
          steps=None,
          exporters=exporter,
          throttle_secs=864000))
```
5. It's possible to train in tensorflow `1.14` and freeze the graph in `1.4` to be satisfy the Udacity requirement. All you need to do is obtain the `.config` and `model.ckpt`. Create a new conda environment with tensorflow `1.4` installed. Checkout commit `f7e99c08` in `models` (thanks to [this post](https://github.com/alex-lechner/Traffic-Light-Classification) pointing out the compatible version) and follow the instructions in `/model/research/object_detection/g3doc/exporting_models.md` to freeze the graph, which can then be deployed to the project.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
