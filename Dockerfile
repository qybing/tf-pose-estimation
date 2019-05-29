FROM docker.io/python:3.6.5-slim
RUN  apt-get update && apt-get install -yq --no-install-recommends --no-install-suggests libgtk2.0-dev swig && \
mkdir /root/.pip/ && cd /root/.pip/ && touch pip.conf && echo '[global]' >> pip.conf && \
echo 'index-url = https://pypi.tuna.tsinghua.edu.cn/simple' >> pip.conf && echo 'trusted-host = pypi.tuna.tsinghua.edu.cn' >> pip.conf
COPY . /opt/code/tf-pose-estimation
WORKDIR /opt/code/tf-pose-estimation
RUN cd /opt/code/tf-pose-estimation/tf_pose/pafprocess && swig -python -c++ pafprocess.i &&  \
pip install --upgrade pip && pip install numpy && python3 setup.py build_ext --inplace && \
cd /opt/code/tf-pose-estimation && rm -rf /var/lib/apt/lists/* && \
pip install -r requirements.txt
CMD python3 video_read.py
