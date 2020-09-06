import os
import cv2
import numpy as np
from sightai.module import SightAI


class SightVideo():
    def __init__(self, vidpath, input_fps = 30.0, max_output_frames = None):
        self.input_path = vidpath 
        self.output_path = os.path.join("output_video", "out_" + os.path.basename(vidpath).split(".")[0] + ".avi")
        self.run_inference(input_fps = input_fps, max_output_frames=max_output_frames)

    def run_inference(self, input_fps = 30.0, max_output_frames  = None):
        # initialize inference module
        S = SightAI(use_cuda = True)

        vidcap = cv2.VideoCapture(self.input_path)
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out4 = cv2.VideoWriter(self.output_path, fourcc, input_fps, (frame_width*2,frame_height*2))

        if (vidcap.isOpened() == False): 
            print("Error opening video stream or file")

        success,image = vidcap.read()
        count = 0
        play = 0
        while(vidcap.isOpened()):
            success,image = vidcap.read()
            
            if success:
                imgfile = "./output/original_{}.jpg".format(count)
                dimgfile = "./output/dmap_{}.jpg".format(count)
                dimgfile2 = "./output/dmap2_{}.jpg".format(count)
                dimgfile3 = "./output/dmap3_{}.jpg".format(count)
                dimgfile4 = "./output/dmap4_{}.jpg".format(count)
                cv2.imwrite(imgfile, image)

                # run inference on current frame
                dmap, img, grid_dmap, sensor_gimg = S.inference(imgfile, plot = False, j = count)  

                # Aggregate (original with bbox, depth map, depth map with grid, vibrational mappings) on (top left, top right, bottom left, bottom right) in one image.
                top = np.hstack((img, dmap))
                bottom = np.hstack((grid_dmap, sensor_gimg))
                result = np.vstack((top, bottom))

                # Output current frame
                # cv2.imwrite(imgfile, image) # original with bbox
                # cv2.imwrite(dimgfile2, grid_dmap) # depth map with grid
                # cv2.imwrite(dimgfile3, sensor_gimg) # vibrational mappings
                cv2.imwrite(dimgfile4, result) # aggregated frame

                # Read the aggregated frame
                o4 = cv2.imread(dimgfile4)

                # Write the aggregated frame to video
                out4.write(o4)

                # Press "Q" to stop
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

            count += 1
            play += 1
            
            # Limit number of frames 
            if max_output_frames is not None:
                if count > max_output_frames:
                    break

        vidcap.release()
        out4.release()

        cv2.destroyAllWindows()
        print("Done: inference on video.")

