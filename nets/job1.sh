#block(name=kitti_train, threads=1, memory=6000, subtasks=1, hours=24, gpus=1)
source /media/remote_home/mtriet/.bashrc 
source /media/remote_home/mtriet/virtualenv/myVE/bin/activate
echo "Kitti"
caffe train -solver kitti_solver.prototxt -weights fcn8s-heavy-pascal.caffemodel
