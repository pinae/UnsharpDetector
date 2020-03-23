# UnsharpDetector
AI-application to automatically identify and delete blurry photos.

## Installation

Load the 64 bit version of Python from [python.org](https://python.org) and
install it. You can skip this step on Linux because Python is usually included 
in your distribution.

Get the code using the download link or clone this repository with git.
If you are using Windows or macOS get git from 
[git-scm.com](https://git-scm.com). Use your package manager on Linux.

The next step is to create a virtual environment. This makes sure that
the following steps can not interfere with other python programs. Create
the virtualenv on macOS and Linux with the following command (in the 
directory where you downloaded the code):

```plaintext
python3 -m venv env
```

Windows does not find `python.exe` by default. So you may have to specify
the full path:

```shell script
..\..\AppData\Local\Programs\Python\Python36\python.exe -m venv env
```

Start the virtualenv on Windows with `env\Scripts\activate.bat`. On 
Linux and macOS use `source env/bin/activate`.

Install all other dependencies with `pip`:

```shell script
pip install -r requirements.txt
```

If you are using a Nvidia GPU and CUDA you may use `requirements-gpu.txt`.
TensorFlow will use a version which uses your GPU to run the neural
network.

## Usage

To use the program activate the virtualenv first with 
`env\Scripts\activate.bat` (Windows) or `source env/bin/activate` 
(Linux and macOS).

Run the graphical Application with: 
```shell script
python inference_gui.py
```

The program starts after a couple of seconds (initialization of 
TensorFlow). Initially it displays an empty list. You fill the list by
clicking the Button in the upper left corner and selecting a path. The
software will load all the images in this folder which may take a couple 
of seconds depending on the number and the size of the images. The 
classification starts immediately in the background.

You may immediately mark images for keeping or removal using the mouse.
The neural network will analyze all the images you do not classify 
manually. The dashed line around these images indicates the decision of 
the network. Green means definitely sharp. Red means blurry. Brown is 
something in between and a good candidate to override the decision.

If the thumbnail is too small to decide if an image is sharp enough to 
keep you may click on the thumbnail. This opens the image in full 
resolution in the preview area on the right.

If you are happy with all the decisions for the images click on the red
button in the upper right corner. This deletes all the images which were
marked for removal (red border) without further questions.

## Training

This repository comes with weights and settings for a pretrained neural 
network. If you want to experiment with different network architectures
you can simply change the config and run `train.py`.

The code uses sacred to keep track of the experiments. To use this magic
create a file named `secret_settings.py` which defines two variables:
1. `mongo_url`: The url of your mongodb with credentials.
2. `db_name`: The database name you are using in the mongodb.

If you are training on a dedicated server you can create queued 
experiments with `-q` on your notebook and start `queue_manager.py` on 
the server. It will automatically fetch queued experiments from your 
database and run them.

Most network architectures will learn some specifics of the generated 
datasets after 2-5 epochs. Training for 50 epochs (my default setting)
leads to something which looks like overfitting. So if you want the best
accuracy on validation data you may want to train for only 2-5 Epochs. 
But this also depends on the size of your dataset.

Also make sure you have no blurry images in your training dataset. This
greatly reduces the accuracy. My intention was to use images from 
vacations where I had already manually deleted all blurry images. I 
trained with ca. 2000 images.  

## Instabilities

I stumbled upon some instabilities of this program. Sometimes it crashes
with a not very informative segmentation fault. This has something to do
with the C-code from Qt or TensorFlow. It happened randomly. If you run 
into this problem try doing the same thing again. It may just work at the 
second attempt. If you have any idea what triggers these crashes please
create an issue with your idea.