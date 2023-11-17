# Zero-shot_Semantic_NST
Repository for "Zero-shot Semantic Neural Style Transfer for Images", course project for Deep Learning (CS541/Fall 2023).

## Steps to run the code:

1. Clone the repository
    ``` bash
    git clone https://github.com/rane-tejas/Zero-shot_Semantic_NST.git
    ```

2. Create conda environment and activate it
    ```bash
    conda create --name ENV_NAME python=3.11
    pip install -r requirements.txt
    conda activate ENV_NAME
    ```

3. Download the model checkpoint of AdaAttN from [here](https://drive.google.com/file/d/1Lnl_1vWfCvF7ZzmWwkHZG4SexjaXuUc5/view?usp=sharing) and unzip it to directory of this repo:

    ```bash
    mv [DOWNLOAD_PATH]/ckpt.zip .
    unzip ckpt.zip
    rm ckpt.zip
    ```

4. Run the following command:

    ```bash
    python infer.py --content_path CONTENT_PATH --style_path STYLE_PATH --resize
    ```

    Example:

    ```bash
    python infer.py --content_path data/content/c1.jpg --style_path data/style/candy.jpg --resize
    ```