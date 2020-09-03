import cv2
from sightai.module import SightAI
import gtts
from playsound import playsound
import pyttsx3
import threading

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
vidcap = cv2.VideoCapture('media/London Underground to Waterloo incl. South Bank Walking Tour.mp4')
if (vidcap.isOpened()== False): 
    print("Error opening video stream or file")

success,image = vidcap.read()
count = 0
play = 0
while(vidcap.isOpened()):
    success,image = vidcap.read()

    if count % 15 == 0 and play % 45 == 0:
        # imgfile = "./frame/frame{}.jpg".format(count)
        imgfile = "./frame/frame{}.jpg".format(count)
        cv2.imwrite(imgfile, image)     # save frame as JPEG file  
        run_infer(imgfile, count, plot = True)    
        # t = threading.Thread(target=run_infer, args=(imgfile,))
        # t.start()  

    if success:
        cv2.imshow('Frame',image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

    count += 1
    play += 1
    
    # if count > 2000:
    #     break

vidcap.release()
cv2.destroyAllWindows()
print("done")

