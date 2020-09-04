import cv2
from sightai.module import SightAI
import gtts
from playsound import playsound
import pyttsx3
import threading
import numpy as np



engine = pyttsx3.init()
voices = engine.getProperty("voices")

S = SightAI(use_cuda = True)

def play_sound(text):
    engine.setProperty("voice", voices[1].id)
    engine.say(text)
    engine.runAndWait()

def run_infer(imgfile, j, plot=False):
    msg, df = S.inference(imgfile, plot = plot, j = j)
    print(df)
    for text in msg:
        engine.setProperty("voice", voices[1].id)
        engine.say(text)
        engine.runAndWait()

# vidcap = cv2.VideoCapture('media/MALAYSIA Kuala Lumpur 15 -Walk- from KL Sentral Monorail Sta. to Muzium Negara MRT Sta.mp4')
# vidcap = cv2.VideoCapture('media/London Underground to Waterloo incl. South Bank Walking Tour.mp4')
filename = "confine_space"
vidcap = cv2.VideoCapture('media/{}.mp4'.format(filename))
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
# out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('out1.avi', fourcc, 20.0, (frame_width,frame_height))
# out2 = cv2.VideoWriter('out2.avi', fourcc, 20.0, (frame_width,frame_height))
# out3 = cv2.VideoWriter('out3.avi', fourcc, 20.0, (frame_width,frame_height))
out4 = cv2.VideoWriter('out_{}.avi'.format(filename), fourcc, 30.0, (frame_width*2,frame_height*2))


if (vidcap.isOpened()== False): 
    print("Error opening video stream or file")

max_count = None # no limit
success,image = vidcap.read()
count = 0
play = 0
while(vidcap.isOpened()):
    success,image = vidcap.read()
    
    # if count % 15 == 0 and play % 45 == 0:
    #     # imgfile = "./frame/frame{}.jpg".format(count)
    #     imgfile = "./frame/frame{}.jpg".format(count)
    #     cv2.imwrite(imgfile, image)     # save frame as JPEG file  
    #     run_infer(imgfile, count, plot = True)    
    #     # t = threading.Thread(target=run_infer, args=(imgfile,))
    #     # t.start()  


    if success:
        imgfile = "./frame/fr{}.jpg".format(count)
        dimgfile = "./frame/dmap_{}.jpg".format(count)
        dimgfile2 = "./frame/dmap2_{}.jpg".format(count)
        dimgfile3 = "./frame/dmap3_{}.jpg".format(count)
        dimgfile4 = "./frame/dmap4_{}.jpg".format(count)
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

