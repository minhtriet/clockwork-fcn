#block(name=youtube_mask_train, threads=1, memory=6000, subtasks=1, hours=24, gpus=1)
echo "Youtube"
caffe train -solver youtube_solver.prototxt -weights fcn8s-heavy-pascal.caffemodel
