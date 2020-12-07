FROM tensorflow/tensorflow:1.14.0-py3

USER root

RUN \
	apt-get update && apt-get install -q -y \
		libsm6 libxext6 libxrender-dev cmake\
		python3-pip &&\
	pip3 install opencv-python==4.2.0.34 scipy dlib imutils opencv-contrib-python==3.4.5.20

RUN apt-get install -qqy x11-apps
ENV DISPLAY :0
ENV QT_X11_NO_MITSHM=1
CMD xeyes

WORKDIR /home/env
