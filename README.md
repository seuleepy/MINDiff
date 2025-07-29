# MINDiff
MINDiff: Mask-Integrated Negative Attention for Controlling Overfitting in Text-to-Image Personalization

> 📄 Accepted at ICCV 2025 Workshop on P13N

**MINDiff** is an inference-time method that mitigates overfitting in text-to-image personalization models such as DreamBooth and DreamBooth+LoRA. It uses **mask-integrated negative attention** to suppress subject's influence in irrelevant regions. It also allows user control via a scale parameter (λ) to balance subject fidelity and prompt alignment during inference.

## Results
The following results show the effect of applying **MINDiff** to a DreamBooth model fine-tuned on **Stable Diffusion 1.4**.
<p align="center">
  <img src="https://github.com/user-attachments/assets/debea15b-1b1b-4bc6-9c1f-2de3c85b6117" width="600"/>
</p>

Varying the value of λ controls the balance between subject fidelity and prompt alignment. Higher λ values lead to stronger suppression of subject influence, resulting in generations that more closely follow the input text prompt.
<p align="center">
  <img src="https://github.com/user-attachments/assets/48dbe3da-0f50-46eb-9c67-ce370cfa89d8" width="300"/>
</p>

## Usage
1. **Install PyTorch**
   This project was tested with the following PyTorch environment:
   - `torch==2.3.0`
   - `CUDA 11.8`
   
   We recommend installing PyTorch using the official instructions:

   👉 [Torch](https://pytorch.org/)
3. **Clone the repository**
```bash
git clone https://github.com/seuleepy/MINDiff.git
cd MINDiff
pip install -r requirements.txt
```
3. **Prepare a fine-tuned DreamBooth model**
Use any existing DreamBooth model. MINDiff has been tested on models fine-tuned with **Stable Diffusion 1.4**, **2.1**, and **SDXL + LoRA**.
4. **Generate an image with MINDiff**
Use the following command:
```
bash inference.sh
```
Before running the script, you need to provide the following arguments:
- `CUSTOM_MODEL_DIR`: Path to your fine-tuned DreamBooth model.
- `modifier_token`: The token used during DreamBooth training (e.g., "sks")
- `mask_token`: A token from your prompt used to guide mask generation via attention maps. It must be included in the prompt.
- `attn_scale`: A float value that controls the strength of suppression. Higher values increases text alignment by reducing subject influence.
