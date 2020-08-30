import time
import skimage.transform
from monodepth.main_monodepth_pytorch import Model as DepthModel
from easydict import EasyDict as edict
from logzero import setup_logger, logging 


class SightAI:
    def __init__(self):
        self.init_log()
        self.init_depth_model()
        pass
    
    def init_log(self):
        self.log = setup_logger(name="sightailog", logfile=os.path.join("./logs/sightai.txt"), level=logging.INFO)
        self.log.info("1 - Initiated log.")

    def init_depth_model(self):
        t0 = time.time()
        dict_parameters_test = edict({'data_dir':'./image',
                                    'model_path':'./pretrained/monodepth_resnet18_001.pth',
                                    'output_directory':'data/output/',
                                    'input_height':256,
                                    'input_width':512,
                                    'model':'resnet18_md',
                                    'pretrained':False,
                                    'mode':'test',
                                    'device':'cuda:0',
                                    'input_channels':3,
                                    'num_workers':0,
                                    'use_multiple_gpu':False})
        self.depth_model = DepthModel(dict_parameters_test)
        self.depth_model.test()
        t1 = time.time()
        total_time = round((t1-t0)/60, 2)
        self.log.info("1 - Initiated DepthModel. -- {} minutes".format(total_time))


    def test_image(self, img_path):
        # Bounding box
        
        # Depth estimation
        disp, disp_pp, original_size = self.depth_model.retest(img_path)
        disp = skimage.transform.resize(disp.squeeze(), original_size, mode='constant')
        disp_pp = skimage.transform.resize(disp_pp.squeeze(), original_size, mode='constant')