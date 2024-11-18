Setup:

1) Create a conda environment and install ultralytics for detection and segmentation
2) Check for the torch version and install accordingly
3) Docker Support:
    
    Use the given docker image if needed support from docker
    ```Docker
    FROM ubuntu:22.04

    # Set environment variables
    ENV DEBIAN_FRONTEND=noninteractive
    ENV TZ=UTC
    
    # Install necessary dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        gnupg2 \
        && rm -rf /var/lib/apt/lists/*
    
    # Add NVIDIA repository
    RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
        && dpkg -i cuda-keyring_1.0-1_all.deb \
        && rm cuda-keyring_1.0-1_all.deb
    
    # Install CUDA 12.1
    RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-toolkit-12-1 \
        && rm -rf /var/lib/apt/lists/*
    
    # Set CUDA environment variables
    ENV PATH=/usr/local/cuda-12.1/bin:${PATH}
    ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}
    
    # Create and set the working directory
    WORKDIR /app

    # Set the default command
    CMD ["/bin/bash"]
    
    ```

5)  Connect to ROS2 and pass images to the module


Process:

Object detection:
1) For Object detection I used Roboflow for data annotation using inbuilt DINO tool.
2) Then seperated the whole data set into test, train and validation set with augmentaion creating 3x times of images.
3) Used the train set to train the base yolov11 model to fit perfectly for the given dataset.

For training:
```python
    from ultralytics import YOLO
    model = YOLO("yolo11m.pt")
    model.train(data = "data.yaml" , imgsz=640, batch = 8, epochs = 100, workers=1,device=0)
```

Image Segmentation:
1) For Image Segmentation also I used Roboflow for data annotation but did every thing manually as there is no predefined model.
2) So for this I selected a random of 40 images and did annotation for these.
3) Then used these annotated images(approx 100 after augmentation) for further training the base yolov11 segmentation model.
4) After training, I used the final segmentation model to segment the remaining images

For training:
```python
    from ultralytics import YOLO
    model = YOLO("yolo11m_seg.pt")
    model.train(data = "data.yaml" , imgsz=640, batch = 8, epochs = 100, workers=1,device=0)
```

Download the model weights from [here](https://drive.google.com/drive/folders/197itXF1hBAPCWeMIsXq8LiZ5_OAOyxpK?usp=sharing)

After downloading keep under object_detection_segmentation/segment and object_detection_segemntation/detection

pallet segmentation.v1i.yolov11.zip - dataset used for training image segmentation model

pallet finder.v1i.yolov11.zip - dataset used for training object detection model


