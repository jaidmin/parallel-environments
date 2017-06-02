import PIL.Image as Image
import numpy as np

class AtariEnvironment:
    def __init__(self,env):
        self.env = env

    @staticmethod
    def preprocess(image_arr):
        img = Image.fromarray(image_arr, mode="RGB")
        img = img.convert("L")
        img = img.resize((84, 110))
        img = img.crop((0, 13, 84, 97))
        img_arr = np.asarray(img.getdata(), dtype=np.uint8).reshape((84, 84))
        return img_arr

    def reset(self):
        return self.preprocess(self.env.reset())

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        preproc = self.preprocess(next_state)
        return preproc, reward, done
