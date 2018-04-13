import warnings
warnings.filterwarnings("ignore")
import os
import keras
from keras import backend as K
from keras.models import load_model
from keras import activations
from config import *
import scipy
from vis.visualization import visualize_activation
from vis.utils import utils
import numpy as np
import matplotlib as M
M.use('TkAgg') 
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")
print(M.__version__, keras.__version__)

model = load_model(MODEL_SAVING_PATH)
print(model.summary())

img_path = input('Please enter the path of your testing image:').strip()
true_class = input('Please enter the true class your testing image:').strip()


# Show the selected image
from skimage import io
img = io.imread(img_path, as_grey=False)
imgplot = plt.imshow(img)
plt.show()

img = scipy.misc.imresize(img, (224, 224))


def iter_occlusion(image, size=8):

    occlusion = np.full((size * 5, size * 5, 1), [0.5], np.float32)
    occlusion_center = np.full((size, size, 1), [0.5], np.float32)
    occlusion_padding = size * 2

    # print('padding...')
    image_padded = np.pad(image, ( \
                        (occlusion_padding, occlusion_padding), (occlusion_padding, occlusion_padding), (0, 0) \
                        ), 'constant', constant_values = ((0,0),(0,0),(0,0)))

    for y in range(occlusion_padding, image.shape[0] + occlusion_padding, size):

        for x in range(occlusion_padding, image.shape[1] + occlusion_padding, size):
            tmp = image_padded.copy()

            tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding, \
                x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding] \
                = occlusion

            tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center

            yield x - occlusion_padding, y - occlusion_padding, \
                  tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding]


occlusion_size = OCC_SIZE
img_size = 224
correct_class = true_class

print('occluding...')
heatmap = np.zeros((img_size, img_size), np.float32)
class_pixels = np.zeros((img_size, img_size), np.int16)

# Plot Occluding Attention
from collections import defaultdict
counters = defaultdict(int)
for n, (x, y, img_float) in enumerate(iter_occlusion(img, size=occlusion_size)):
    sz = img_float.shape
    X = img_float.reshape(1, sz[0], sz[1], sz[2])
    out = model.predict(X)

    heatmap[y:y + occlusion_size, x:x + occlusion_size] = out[0][correct_class]
    class_pixels[y:y + occlusion_size, x:x + occlusion_size] = np.argmax(out)
    counters[np.argmax(out)] += 1


from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm

fig = plt.figure(figsize=(8, 8))

ax1 = plt.subplot(1, 2, 1, aspect='equal')
hm = ax1.imshow(heatmap)
ax2 = plt.subplot(1, 2, 2, aspect='equal')
ori = ax2.imshow(img)

vals = np.unique(class_pixels).tolist()
bounds = vals + [vals[-1] + 1]  # add an extra item for cosmetic reasons

custom = cm.get_cmap('Pastel1', len(bounds)) # discrete colors
norm = BoundaryNorm(bounds, custom.N)


divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = plt.colorbar(hm, cax=cax1)

fig.tight_layout()
plt.show()

# Get Occlusion with Original Image
plt.figure(figsize=(6, 6))

plt.imshow(img, cmap=cm.gray)

plt.pcolormesh(heatmap, cmap=plt.cm.jet, alpha=0.50)
plt.colorbar().solids.set(alpha=1)

plt.show()