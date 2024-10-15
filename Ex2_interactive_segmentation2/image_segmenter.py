import numpy as np

from kmeans import (
    compute_distance,
    kmeans_fit,
    kmeans_predict_idx
)

from extract_patches import extract_patches
from advanced_methods import perform_min_cut


class ImageSegmenter:
    def __init__(self, k_fg=15, k_bg=15, mode='kmeans'):
        """ Feel free to add any hyper-parameters to the ImageSegmenter.
            
            But note:
            For the final submission the default hyper-parameteres will be used.
            In particular the segmetation will likely crash, if no defaults are set.
        """
        
        # Number of clusters in FG/BG
        self.k_fg = k_fg
        self.k_bg = k_bg
        self.patch = 1
        self.mode = mode
        
    #def extract_features_(self, sample_dd):
        """ Extract features, e.g. p x p neighborhood of pixel, from the RGB image """
        
        img = sample_dd['img']
        
        #
        # TO IMPLEMENT
        
        patches = extract_patches(img, self.patch)
        # For now this only extracts intensities
        return patches

    
    def segment_image_dummy(self, sample_dd):
        return sample_dd['scribble_fg']

    def segment_image_kmeans(self, sample_dd):
        """ Segment images using k means """
        H, W, C = sample_dd['img'].shape
        features = self.extract_features_(sample_dd)
        
        #
        # TO IMPLEMENT
        #
        data = features.reshape(-1, C*self.patch**2)

        fg = (sample_dd['scribble_fg'] > 0)
        bg = (sample_dd['scribble_bg'] > 0)

        fg_centroids = kmeans_fit(features[fg],self.k_fg)
        bg_centroids = kmeans_fit(features[bg],self.k_bg)

        fg_labels = kmeans_predict_idx(data, fg_centroids)
        bg_labels = kmeans_predict_idx(data, bg_centroids)
        dist_fg = np.linalg.norm(data - fg_centroids[fg_labels], axis=1)
        dist_bg = np.linalg.norm(data - bg_centroids[bg_labels], axis=1)

        segmentation_mask = (dist_fg < dist_bg).reshape(H, W)

        return segmentation_mask
        #return self.segment_image_dummy(sample_dd)

    def segment_image_grabcut(self, sample_dd):
        """ Segment via an energy minimisation """

        # Foreground potential set to 1 inside box, 0 otherwise
        unary_fg = sample_dd['scribble_fg'].astype(np.float32) / 255

        # Background potential set to 0 inside box, 1 everywhere else
        unary_bg = 1 - unary_fg

        # Pairwise potential set to 1 everywhere
        pairwise = np.ones_like(unary_fg)

        # Perfirm min cut to get segmentation mask
        im_mask = perform_min_cut(unary_fg, unary_bg, pairwise)
        
        return im_mask

    def segment_image(self, sample_dd):
        """ Feel free to add other methods """
        if self.mode == 'dummy':
            return self.segment_image_dummy(sample_dd)
        
        elif self.mode == 'kmeans':
            return self.segment_image_kmeans(sample_dd)
        
        elif self.mode == 'grabcut':
            return self.segment_image_grabcut(sample_dd)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")