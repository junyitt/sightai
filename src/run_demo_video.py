import cv2
import numpy as np
from sightai.module import SightAI


S = SightAI(use_cuda = True)

filename = "two_way"
vidcap = cv2.VideoCapture('media/{}.mp4'.format(filename))
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out4 = cv2.VideoWriter('output_video/out_{}.avi'.format(filename), fourcc, 30.0, (frame_width*2,frame_height*2))

if (vidcap.isOpened()== False): 
    print("Error opening video stream or file")

max_count = 20 # no limit
success,image = vidcap.read()
count = 0
play = 0
while(vidcap.isOpened()):
    success,image = vidcap.read()
    
    if success:
        imgfile = "./output/fr{}.jpg".format(count)
        dimgfile = "./output/dmap_{}.jpg".format(count)
        dimgfile2 = "./output/dmap2_{}.jpg".format(count)
        dimgfile3 = "./output/dmap3_{}.jpg".format(count)
        dimgfile4 = "./output/dmap4_{}.jpg".format(count)
        cv2.imwrite(imgfile, image)

        dmap, img, grid_dmap, sensor_gimg = S.inference(imgfile, plot = False, j = count)  

        top = np.hstack((img, dmap))
        bottom = np.hstack((grid_dmap, sensor_gimg))
        result = np.vstack((top, bottom))


        # cv2.imwrite(imgfile, image)
        # cv2.imwrite(dimgfile2, grid_dmap)
        # cv2.imwrite(dimgfile3, sensor_gimg)
        cv2.imwrite(dimgfile4, result)

        # o1 = cv2.imread(imgfile)
        # o2 = cv2.imread(dimgfile2)
        # o3 = cv2.imread(dimgfile3)
        o4 = cv2.imread(dimgfile4)

        
        # cv2.imshow('Frame',dmap2)
        # out1.write(o1)
        # out2.write(o2)
        # out3.write(o3)
        out4.write(o4)
        # out.write(img)
        # cv2.imshow('image',dmap)
        
        # out.write(dmap)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

    count += 1
    play += 1
    
    if max_count is not None:
        if count > max_count:
            break

vidcap.release()
# out1.release()
# out2.release()
# out3.release()
out4.release()

cv2.destroyAllWindows()
print("done")

