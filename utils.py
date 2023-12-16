import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--content_path", type=str, default="data/content/c1.jpg",
                        help="Path to a single content img")
    parser.add_argument("-m", "--mask_path", type=str, default=None,
                        help="Path to a single mask img")
    parser.add_argument("-s", "--style_path", type=str, default="data/style/vg_starry_night.jpg",
                        help="Path to a single style img")
    parser.add_argument("--checkpoint_path", type=str, default="ckpt/pretrained",
                        help="Path to the checkpoint drectory")
    parser.add_argument("-o", "--output_name", type=str, default='result',
                        help="Output file name")
    parser.add_argument("--resize", action='store_true',
                        help="Whether resize images to the 512 scale, which is the training resolution "
                            "of the model and may yield better performance")
    parser.add_argument("--keep_ratio", action='store_true',
                        help="Whether keep the aspect ratio of original images while resizing")

    return parser.parse_args()

def train_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_path", type=str, default="dataset/PhraseCut_nano",
                        help="Path to the dataset")
    parser.add_argument("-c", "--checkpoint_path", type=str, default="ckpt/pretrained",
                        help="Path to the checkpoint drectory")
    parser.add_argument("-l", "--log_dir", type=str, default='logs/',
                        help="Path to the log directory")
    parser.add_argument("-p", "--log_name", type=str, default='trial_logs',
                        help="name of the log file")
    parser.add_argument("-b", "--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("-e", "--num_epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--msg", type=str, default="",
                        help="Message/Description of experiment")

    return parser.parse_args()


class Logger:

    import os
    import time
    import matplotlib.pyplot as plt

    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not self.os.path.exists(log_dir):
            self.os.makedirs(log_dir)

        self.plot_dir = self.os.path.join(log_dir, 'plots')
        if not self.os.path.exists(self.plot_dir):
            self.os.mkdir(self.plot_dir)

        self.media_dir = self.os.path.join(log_dir, 'media')
        if not self.os.path.exists(self.media_dir):
            self.os.mkdir(self.media_dir)

        self.log_file_path = self.log_dir + '/logs.txt'
        self.log_file = open(self.log_file_path, 'w')
        self.log_file.write('Logs date and time: '+self.time.strftime("%d-%m-%Y %H:%M:%S")+'\n\n')

        self.train_data = []
        self.val_data = []

    def log(self, tag, **kwargs):

        self.log_file = open(self.log_file_path, 'a')

        if tag == 'args':
            self.log_file.write('Training Args:\n')
            for k, v in kwargs.items():
                self.log_file.write(str(k)+': '+str(v)+'\n')
            self.log_file.write('#########################################################\n\n')
            self.log_file.write(f'Starting Training... \n')

        elif tag == 'train':
            self.train_data.append([kwargs['loss']])
            self.log_file.write(f'Epoch: {kwargs["epoch"]} \t Train Loss: {kwargs["loss"]} \t Avg Time: {kwargs["time"]} secs\n')

        elif tag == 'val':
            self.val_data.append([kwargs['loss']])
            self.log_file.write(f'Epoch: {kwargs["epoch"]} \t Val Loss: {kwargs["loss"]} \t Avg Time: {kwargs["time"]} secs\n')

        elif tag == 'model':
            self.log_file.write('#########################################################\n')
            self.log_file.write(f'Saving best model... Val Loss: {kwargs["loss"]}\n')
            self.log_file.write('#########################################################\n')

        elif tag == 'plot':
            self.plot(self.train_data, name='Train Loss', path=self.plot_dir)
            self.plot(self.val_data, name='Val Loss', path=self.plot_dir)
            self.plot_both(self.train_data, self.val_data, name='Loss', path=self.plot_dir)

        self.log_file.close()

    def draw(self, epoch, img):

        cv2.imwrite(self.media_dir+'/'+str(epoch)+'.png', img)

    def plot(self, data, name, path):

        self.plt.plot(data)
        self.plt.xlabel('Epochs')
        self.plt.ylabel(name)
        self.plt.title(name+' vs. Epochs')
        self.plt.savefig(self.os.path.join(path, name+'.png'), dpi=1200 ,bbox_inches='tight')
        self.plt.close()

    def plot_both(self, data1, data2, name, path):

        self.plt.plot(data1, label='Train Loss')
        self.plt.plot(data2, label='Val Loss')
        self.plt.xlabel('Epochs')
        self.plt.ylabel(name)
        self.plt.title(name+' vs. Epochs')
        self.plt.legend()
        self.plt.savefig(self.os.path.join(path, name+'.png'), dpi=1200 ,bbox_inches='tight')
        self.plt.close()

def img_to_tensor(img):
    """
    Converts image to tensor
    First takes in the image , converts to numpy.
    Changes the dimensions from (H,W,C) -> (C,H,W)
    Changes the datatype to float and then normalizes the image intensities between 0 and 1.
    Finally, it changes the dimension by adding a dimension at the start allowing for a batch accomodation
    """

    np_array = np.array(img).transpose((2,0,1))
    np_array = np_array.float()/255.
    tensor = torch.from_numpy(np_array).unsqueeze(0)

    return tensor

def tensor_to_img(tensor):
    """
    Converts tensor to image
    Takes the tensor to cpu, converts to numpy array
    Changes the dimensions from (C,H,W) -> (H,W,C)
    Clips the tensor values between 0 and 1, scales values till 255, adds 0.5 to it
    Changes datatype to integer.
    """

    img = tensor[0].data.cpu().numpy()
    img = img.transpose((1,2,0)).clip(0,1)*255
    img = (img + 0.5).astype(np.uint8)

    return img

def padding(image, factor=32):
    """
    Helper function to pad a particular image by some factor
    We calculate the padding in each dimension by factor related calculations.
    Padding in height can be maximum- factor. It can be a minimum of 0.
    Essentially, it is calculating padding so that it would be minimum padding required to make the height divisible by factor.
    Same for width.
    """
    height, width = image.shape[0], image.shape[1]

    # Calculates the padding in height and width
    padding_height = (factor - height % factor) % factor
    padding_width = (factor - width % factor) % factor

    # Initialize a numpy arrays of zeros with the new Image height and width
    new_image = np.zeros((height + padding_height, width + padding_width, image.shape[2]), dtype=image.dtype)

    # Fills in the original image into the top left corner of the new array. Rest of the pixels are 0 - padded.
    new_image[:height, :width, :] = image
    return new_image

def resize_img(img, long_side=512, keep_aspect_ratio=True):
    """
    Helper function for resizing images.
    If we want to keep the aspect ratio the same we resize according to the aspect ratio.
    If the height is greater than the width, the height is made equal to long_side, the width is increased
    in the same factor as the increase in height, thus width is multiplied by long_side/height.
    Similar is for the other case when width is greater than height.
    Otherwise we directly resize the image to acheive the target dimensions
    """

    if keep_aspect_ratio:
        height, width = img.shape[0],img.shape[1]
        if height > width:
            # Height is made to be equal to long side.
            # The same multiplication factor is used for the width too.
            new_height = int(long_side)
            new_width = int(long_side * width / height)

        else:
            # Width is made to be equal to long side.
            # The same multiplication factor is used for the height too.
            new_width = int(long_side)
            new_height = int(long_side * height / width)

        # Actually resizing by cv2
        return cv2.resize(img, (new_width, new_height))
    else:
        # Actually resizing by cv2
        return cv2.resize(img, (long_side, long_side))



def calc_mean_std(features, eps=1e-5):
    """
    Function that calculates the mean and standard deviation across different channels of the tensor.
    """
    # Get the feature tensors size, this tensor should be of length 4 since (N,C,H,W)
    feature_size = features.size()
    assert (len(feature_size) == 4)

    # Record batch and channel dimension
    N, C = feature_size[0],feature_size[1]

    # Calculates the mean by flattening the H,W dimension. We take the mean in the 3rd dimension.
    # Reshape the output to be (N,C,1,1)
    features_mean = features.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    # Calculate the variance of the tensor by flattening the H,W dimension.
    # eps is a small value added to the variance to avoid divide-by-zero.
    # Calculates the dimension at dimension 3
    features_variance = features.view(N, C, -1).var(dim=2) + eps

    # Calculates the standard deviation of the tensor by taking a squareroot, reshaping the result to (N,C,1,1)
    features_std = features_variance.sqrt().view(N, C, 1, 1)

    return features_mean, features_std

def mean_variance_norm(features):
    """
    Function that normalizes a tensor according to the mean and standard deviation
    Calculates the mean and standard deviation.
    Normalizes the tensor by (feature - mean(feature)/std(feature))
    """
    # Get size of the features tensor
    size = features.size()
    # Calculate mean and standard deviation of the features vector by calling the earlier function
    mean, std = calc_mean_std(features)

    # Expand the mean tensors shape to match with the features dimension
    expanded_mean = mean.expand(size)
    expanded_std = std.expand(size)

    # Normalize by applying the standard normalizing formula.
    normalized_features = (features - expanded_mean) / expanded_std
    return normalized_features

def get_key(features, last_layer_index, need_shallow=True):
    """
    The features is a collection of feature maps from different layers of the final encoder network

    This function calculates the normalized features for the entirity of layers if the need_shallow is True.
    Else it just returns the normalized tensor for the last layer of the model.
    """

    # If need_shallow is True, we extract features from shallower layers of the network.
    if need_shallow and last_layer_index > 0:

        # Initialize the list to store the different layers of the final encoder
        tensors = []
        # Recording the Height, width of the final layer embedding of the VGG Encoder
        _, _, height, width = features[last_layer_index].shape

        # Loop over the different layers of the encoder until the last_layer_index given to this function
        for idx in range(last_layer_index):
            # Interpolate the features so that the feature dimensions would be height and width, the same dimension of the
            #final encoder layer.
            temp_features = F.interpolate(features[idx],(height,width))
            #Normalize the Feature maps by calling the above written function
            normalized_temp_features = mean_variance_norm(temp_features)
            # Store this feature in the tensor list
            tensors.append(normalized_temp_features)

        # For the final layer append the normalized tensor to the tensors list. The tensors list will now have features from
        # all layers.
        tensors.append(mean_variance_norm(features[last_layer_index]))

        # Finally concatenate the tensor in the dimension 1.
        # Example:
        #x=  tensor([[ 0.6580, -1.0969, -0.4614],
        #[-0.1034, -0.5790,  0.1497]])
        # torch.cat((x, x, x), 1)= tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
        #  -1.0969, -0.4614],
        # [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
        #  -0.5790,  0.1497]])
        # Basically this will concatenate along the channel dimension, along the channel dimension , the tensor would
        # concatenated so that along the channel dimension there would different layers.

        return torch.cat(tensors, dim=1)
    else:
        # Else just return the normalized features from the last layer.
        # This would not be called in this implementation.
        return mean_variance_norm(features[last_layer_index])

