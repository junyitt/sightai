import cv2
import time
from tool.darknet2pytorch import Darknet
from tool.utils import plot_boxes_cv2, load_class_names
from tool.torch_utils import do_detect
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    cfgfile = "./cfg/yolov4.cfg"
    weightfile = "./pretrained/yolov4.weights"
    namesfile = 'data/coco.names'
    imgfile = "./image/image/init.png"
    imgfile2 = "./media/059_L.png"
    use_cuda = True
    
    # Initiate model 
    t0 = time.time()

    darknet_model = Darknet(cfgfile)
    width, height = (darknet_model.width, darknet_model.height)
    darknet_model.load_weights(weightfile)
    if use_cuda:
        darknet_model.cuda()
    class_names = load_class_names(namesfile)

    t1 = time.time()
    total_time = round(t1-t0, 2)
    print("1 - Initiated DepthModel. -- {} minutes {} seconds".format(total_time//60, total_time % 60))

    print("====================================")
    print("====================================")
    print("====================================")
    # Inference 
    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    start = time.time()
    boxes = do_detect(darknet_model, sized, 0.4, 0.6, use_cuda)
    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
    print(boxes)
    plot_boxes_cv2(img, boxes[0], savename='demo_yolo.jpg', class_names=class_names)
    print("====================================")
    print("====================================")
    print("====================================")
