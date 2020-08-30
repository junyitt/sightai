from sightai.module import SightAI

S = SightAI(use_cuda = True)
# S.inference("image/image/init.png", plot = True)
S.inference("media/001_L.png", plot = True)
S.inference("media/165_R.png", plot = False)
S.inference("media/001_L.png", plot = False)