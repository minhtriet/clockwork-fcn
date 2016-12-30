#import pdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# random color map for segmentation
segm_cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))

def segsave(im, label, out, index, n_cl=None):

    out.save("{}.png".format(index))
	
#    plt.figure()
#    plt.subplot(1,3,1)
#    plt.imshow(im)
#    plt.axis('off')
#    plt.tight_layout()
#    plt.subplot(1,3,2)
#    plt.imshow(label)
#    if n_cl:
#        plt.imshow(label, cmap=segm_cmap, vmin=0, vmax=n_cl - 1)
#    plt.savefig(label, cmap=segm_cmap, vmin=0, vmax=n_cl - 1)
#   else:
#        plt.imshow(label)
#        ipdb.set_trace()
#        plt.savefig(label)
#    plt.axis('off')
#    plt.tight_layout()
#    plt.subplot(1,3,3)
#    plt.imshow(out)
#    if n_cl:
#        plt.imshow(out, cmap=segm_cmap, vmin=0, vmax=n_cl - 1)
#    plt.savefig(out, cmap=segm_cmap, vmin=0, vmax=n_cl - 1)
#    else:
#        plt.imshow(out)
#    plt.savefig("{}.png".format(index))
#    plt.axis('off')
#    plt.tight_layout()
