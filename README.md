# Zero-shot_Semantic_NST
Repository for "Zero-shot Semantic Neural Style Transfer for Images", course project for Deep Learning (CS541/Fall 2023).

## Steps to run the code:

To perform style transfer on the entire image in a zero-shot manner, use the following command:

```bash
python infer.py --content_path CONTENT_PATH --style_path STYLE_PATH --resize --keep_ratio
```

Example:

```bash
python infer.py --content_path data/content/c1.jpg --style_path data/style/candy.jpg --resize --keep_ratio
```

To perform style transfer on a specific region of the image (semantic segmented mask) using the CLIPSeg Segmentation model, use the following command:

```bash
python clipseg_infer.py --content_path CONTENT_PATH --style_path STYLE_PATH --prompts PROMPTS
```

Example:

```bash
python clipseg_infer.py --content_path data/content/parked_car.jpg --style_path data/style/candy.jpg --prompts "car"
```
![image](https://github.com/rane-tejas/Zero-shot_Semantic_NST/assets/55713396/1036acff-bcf6-4431-a2d9-9acba2d77e99)

A demo of the approach on a chair in the Unity Hall at WPI. 
The style images are towards the bottom left corner of the resultant images.
