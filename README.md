# DAIN_ESRGAN_DEOLDIFY

The purpose of this project is to integrate [DAIN](https://github.com/baowenbo/DAIN), [ESRGAN](https://github.com/xinntao/ESRGAN) and [DEOLDIFY](https://github.com/jantic/DeOldify) into an assembly line which can support video frame interplation/super resolution/colorization in a single runtime environment. Thanks for the great work of those open source repos. 

# How to use 
Clone the repo and it's dependecies. 
```
git clone
```

Build the docker image. It need take some time to pull all its dependencies.  
```
docker build -t dain_esrgan_deoldify .
```

Start the container (Notes: I only tested on GPU environment)  
```
docker run -itd --runtime nvidia --name dain_esrgan_deoldify --hostname ubuntu dain_esrgan_deoldify
```

Login the container  
```
docker exec -it dain_esrgan_deoldify /bin/bash
```

Run the example, you can check the result in folder data/output  
```
python run.py -i shanghai1937.mp4
```

If you want to test your own video, please put it into data/input/your_video.mp4 and execute:  
```
python run.py -i your_video.mp4  
```

By default, DAIN, ESRGAN and DEOLDIFY will be processed in sequence. If you want to customize the steps, you can use '-p' parameter:
```
python run.py -i your_video.mp4 -p dain,deoldify,build
python run.py -i your_video.mp4 -p esrgan,build
```
The 'build' step here is used to combine the final frames into a video.
