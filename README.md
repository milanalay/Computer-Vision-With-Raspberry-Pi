# Computer-Vision-With-Raspberry-Pi

This project uses [TensorFlow Lite](https://tensorflow.org/lite) with Python on a Raspberry Pi to perform real-time image segmentation using images streamed from the camera.

A Python file for image segmentation using the Macbook camera is also included in this project for testing purpose.

Assuming that a latest version of Python3 and a virtual environment is installed into the device, we can easily run the project. 

Virtual environment (venv) is a tool that helps you isolate project-specific dependencies. It creates a self-contained directory that includes a Python interpreter, libraries, and other packages needed for your project.
(https://virtualenv.pypa.io/en/latest/installation.html)

To run this project to your device, run the following commands in the terminal
1. Create virtualenv
    virtualenv venv -p python3
    (venv is the name of your virtual environment)

2. Activate the virtual environment
    source venv/bin/activate

3. Clone the repo inside the virtual environment
    cd venv
    git clone https://github.com/milanalay/Computer-Vision-With-Raspberry-Pi.git

4. Install the required dependencies
    cd Computer-Vision-With-Raspberry-Pi
    pip install -r requirements.txt

5. Finally, run the python file segment_raspberrypi_camera.py if your are using raspberry pi or run segment_macbook_camera.py if you are using macbook
    python segment_raspbarrypi_camera.py
    or
    python segment_macbook_camera.py

