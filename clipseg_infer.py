import cv2
import torch
import numpy as np
from PIL import Image

from torchvision import transforms
from torch import nn

from utils import *
from models.decoder import Decoder
from models.vgg_encoder import ATA_Encoder
from models.AdaAttN import AdaAttN, Transformer
from models.clipseg import CLIPDensePredT


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')


class InferClipSeg:
    """
    Class to infer CLIPSeg
    """
    def __init__(self):
        # Load the CLIPSeg model
        self.model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
        self.model.eval()
        self.model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)

    def get_segmentation_masks(self, image_path, prompts):
        """
        Get segmentation masks for the given image and prompts

        :param image_path: Path to the image
        :type image_path: str
        :param prompts: List of prompts
        :type prompts: list
        :return: Segmentation masks for the given image and corresponding prompts
        :rtype: list
        """
        # Load and normalize image
        input_image = Image.open(image_path)
        original_size = input_image.size  # Store the original size of the image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352)),
        ])
        img = transform(input_image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            preds = self.model(img.repeat(len(prompts), 1, 1, 1), prompts)[0]

        # Thresholding and binarizing the predictions to create masks
        threshold = 0.5
        segmentation_masks = (torch.sigmoid(preds) > threshold).float()

        # Define dilation function
        def dilate(tensor, kernel_size):
            padding = kernel_size // 2
            kernel = torch.ones((kernel_size, kernel_size))
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(1)
            tensor = nn.functional.pad(tensor, (padding, padding, padding, padding), mode='constant', value=0)
            dilation = nn.functional.conv2d(tensor, kernel.unsqueeze(0).unsqueeze(0), padding=0)
            dilated_mask = (dilation > 0).float()
            return dilated_mask.squeeze(0).squeeze(0)

        # Apply dilation and resize masks
        kernel_size = 10
        resized_masks = []
        for mask in segmentation_masks:
            dilated_mask = dilate(mask, kernel_size)

            # Resize the mask to match the content image size
            resized_mask = transforms.Resize(original_size[::-1])(dilated_mask.unsqueeze(0)).squeeze(0)
            resized_masks.append(resized_mask)

        return resized_masks


class InferStyleTransfer:
    """
    Class to infer style transfer
    """

    def __init__(self,):
        self.seg_mask = None
        self.style_img = None
        self.content_img = None
        self.content_shape = None

        self.image_encoder = ATA_Encoder().to(DEVICE)
        self.decoder = Decoder().to(DEVICE)
        self.ada_attn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=64 * 64).to(DEVICE)
        self.transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64).to(DEVICE)

    def build_models(self, checkpoint_path):
        """
        Build the models for inference

        :param checkpoint_path: Path to the checkpoint
        :type checkpoint_path: str
        """
        # Load the models
        self.transformer.load_state_dict(torch.load(checkpoint_path+'/transformer.pth'))
        self.decoder.load_state_dict(torch.load(checkpoint_path+'/decoder.pth'))
        self.ada_attn_3.load_state_dict(torch.load(checkpoint_path+'/adaattn.pth'))

        # Set the models to eval mode
        self.image_encoder.eval()
        self.transformer.eval()
        self.decoder.eval()
        self.ada_attn_3.eval()

        # Freeze the models
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.ada_attn_3.parameters():
            p.requires_grad = False

    def load_images(self, content_path, mask_path, style_path, resize=True, keep_ratio=True):
        """
        Load the content, mask and style images

        :param content_path: Path to the content image
        :type content_path: str
        :param mask_path: Path to the mask image
        :type mask_path: str
        :param style_path: Path to the style image
        :type style_path: str
        :param resize: Resize image to 512, defaults to True
        :type resize: bool, optional
        :param keep_ratio: Maintain aspect ratio, defaults to True
        :type keep_ratio: bool, optional
        """
        self.content_img = cv2.imread(content_path)
        self.content_shape = self.content_img.shape[:2]
        self.style_img = cv2.imread(style_path)
        self.seg_mask = mask_path.numpy()

        # Resize images
        if resize:
            self.content_img = resize_img(self.content_img, 512, keep_ratio)
            self.style_img = resize_img(self.style_img, 512, keep_ratio)


    def run(self, content_path, mask_path, style_path, checkpoint_path, resize=True, keep_ratio=True):
        """
        Run the style transfer

        :param content_path: Path to the content image
        :type content_path: str
        :param mask_path: Path to the mask image
        :type mask_path: str
        :param style_path: Path to the style image
        :type style_path: str
        :param checkpoint_path: Path to the checkpoint
        :type checkpoint_path: str
        :param resize: Resize image to 512, defaults to True
        :type resize: bool, optional
        :param keep_ratio: Maintain aspect ratio, defaults to True
        :type keep_ratio: bool, optional
        :return: The stylized image
        :rtype: np.ndarray
        """

        self.load_images(content_path, mask_path, style_path, resize, keep_ratio)
        self.build_models(checkpoint_path)

        # Convert the images to tensors
        with torch.no_grad():
            style = img_to_tensor(cv2.cvtColor(padding(self.style_img, 32), cv2.COLOR_BGR2RGB)).to(DEVICE)
            content = img_to_tensor(cv2.cvtColor(padding(self.content_img, 32), cv2.COLOR_BGR2RGB)).to(DEVICE)
            c_feats = self.image_encoder(content)
            s_feats = self.image_encoder(style)
            c_adain_feat_3 = self.ada_attn_3(c_feats[2], s_feats[2], get_key(c_feats, 2), get_key(s_feats, 2))
            cs = self.transformer(c_feats[3], s_feats[3], c_feats[4], s_feats[4], get_key(c_feats, 3), get_key(s_feats, 3),
                             get_key(c_feats, 4), get_key(s_feats, 4))
            cs = self.decoder(cs, c_adain_feat_3)
            cs = tensor_to_img(cs[:, :, :int(self.content_shape[0]), :int(self.content_shape[1])])
            cs = cv2.cvtColor(cs, cv2.COLOR_RGB2BGR)

        # Resize the image to the original size
        if resize:
            cs = cv2.resize(cs, (int(self.content_shape[1]), int(self.content_shape[0])))
            self.content_img = cv2.resize(self.content_img, (int(self.content_shape[1]), int(self.content_shape[0])))

        # Apply the mask
        self.seg_mask = np.repeat(self.seg_mask[:, :, np.newaxis], 3, axis=2)
        cs = cs * self.seg_mask + self.content_img * (1 - self.seg_mask)

        return cs

if __name__ == '__main__':

    # Parse arguments
    args = infer_args()
    clip_seg = InferClipSeg() # Load the CLIPSeg model
    prompts = [args.prompts]
    mask = clip_seg.get_segmentation_masks(args.content_path, prompts) # Get the segmentation masks

    # Use the first generated mask for style transfer
    style_transfer = InferStyleTransfer() # Load the style transfer model
    # Run the style transfer
    result = style_transfer.run(content_path=args.content_path, mask_path=mask[0], style_path=args.style_path, checkpoint_path=args.checkpoint_path, resize=args.resize, keep_ratio=args.keep_ratio)
    cv2.imwrite("/home/megatron/workspace/WPI/CS541-DL/project/repos/PR/Zero-shot_Semantic_NST/output/result.png", result)