# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Inspired by: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/examples/community/reference_only.py and https://github.com/Mikubill/sd-webui-controlnet

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.cross_attention import CrossAttention
from diffusers.models.transformer_2d import Transformer2DModelOutput
from diffusers.models.unet_2d_blocks import (
    ResnetBlock2D,
    Transformer2DModel,
    Upsample2D,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    logging,
    randn_tensor,
    replace_example_docstring,
)


EPS = 1e-6

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DiffusionPileline
        >>> from diffusers.utils import load_image
        >>> pipe = DiffusionPileline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, custom_pipeline="stable_diffusion_reference_only")
        >>> pipe.to("cuda")
        >>> image = load_image("https://raw.githubusercontent.com/Mikubill/sd-webui-controlnet/main/samples/dog_rel.png").resize((512, 512))
        >>> prompt = "a dog running on grassland, best quality"
        >>> image = pipe(prompt,
        ...     image=image,
        ...     width=512,
        ...     height=512,
        ...     control_name="refernce_only", # "none", "reference_only", "reference_adain", "reference_adain+attn"
        ...     attention_auto_machine_weight=1.0,
        ...     gn_auto_machine_weight=1.0,
        ...     style_fidelity=0.5).images[0]
        >>> image.save("refernce_only_dog.png")
        ```
"""


def self_attn_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
    attn_output = None

    if getattr(self, "enable_attn", False):
        assert attention_mask is None, "attention_mask must be None!"
        if self.attention_auto_machine_weight > self.attn_weight:
            do_classifier_free_guidance = len(self.current_uc_indices) > 0
            split_size = 2 if do_classifier_free_guidance else 1
            latent_hidden_states = hidden_states[:split_size]  # uc, c
            image_hidden_states = hidden_states[split_size:]  # uc, c

            image_self_attn1 = self.processor(
                self,
                hidden_states=image_hidden_states,
                encoder_hidden_states=image_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            latent_self_attn1_uc = self.processor(
                self,
                latent_hidden_states,
                encoder_hidden_states=torch.cat(
                    (latent_hidden_states,) + image_hidden_states.split(split_size),
                    dim=1,
                ),
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            if do_classifier_free_guidance and self.style_fidelity > 1e-5:
                latent_self_attn1_c = latent_self_attn1_uc.clone()
                latent_self_attn1_c[self.current_uc_indices] = self.processor(
                    self,
                    hidden_states=latent_hidden_states[self.current_uc_indices],
                    encoder_hidden_states=latent_hidden_states[self.current_uc_indices],
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                latent_self_attn1 = (
                    self.style_fidelity * latent_self_attn1_c + (1.0 - self.style_fidelity) * latent_self_attn1_uc
                )
            else:
                latent_self_attn1 = latent_self_attn1_uc

            attn_output = torch.cat([latent_self_attn1, image_self_attn1])

    if attn_output is None:
        attn_output = self.processor(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
    return attn_output


def transformer_2d_model_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    class_labels: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    x = self.original_forward(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        class_labels=class_labels,
        cross_attention_kwargs=cross_attention_kwargs,
        attention_mask=attention_mask,
        encoder_attention_mask=encoder_attention_mask,
        return_dict=return_dict,
    )[0]
    output = None
    if getattr(self, "enable_gn", False):
        if self.gn_auto_machine_weight > self.gn_weight:
            do_classifier_free_guidance = len(self.current_uc_indices) > 0
            split_size = 2 if do_classifier_free_guidance else 1

            latent_hidden_states = x[:split_size]  # uc, c
            image_hidden_states = x[split_size:]  # uc, c
            image_var, image_mean = torch.var_mean(image_hidden_states, dim=(2, 3), keepdim=True, unbiased=False)
            var, mean = torch.var_mean(latent_hidden_states, dim=(2, 3), keepdim=True, unbiased=False)
            std = torch.maximum(var, torch.zeros_like(var) + EPS) ** 0.5

            div_num = image_hidden_states.shape[0] // split_size
            mean_acc = sum(image_mean.split(split_size)) / div_num
            var_acc = sum(image_var.split(split_size)) / div_num

            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + EPS) ** 0.5
            y_uc = (((latent_hidden_states - mean) / std) * std_acc) + mean_acc
            if do_classifier_free_guidance and self.style_fidelity > 1e-5:
                y_c = y_uc.clone()
                y_c[self.current_uc_indices] = latent_hidden_states[self.current_uc_indices]
                latent_hidden_states = self.style_fidelity * y_c + (1.0 - self.style_fidelity) * y_uc
            else:
                latent_hidden_states = y_uc
            output = torch.cat([latent_hidden_states, image_hidden_states])

    if output is None:
        output = x
    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def resnet_block_2d_forward(self, input_tensor, temb):
    x = self.original_forward(input_tensor, temb=temb)
    output = None
    if getattr(self, "enable_gn", False):
        if self.gn_auto_machine_weight > self.gn_weight:
            do_classifier_free_guidance = len(self.current_uc_indices) > 0
            split_size = 2 if do_classifier_free_guidance else 1

            latent_hidden_states = x[:split_size]  # uc, c
            image_hidden_states = x[split_size:]  # uc, c
            image_var, image_mean = torch.var_mean(image_hidden_states, dim=(2, 3), keepdim=True, unbiased=False)
            var, mean = torch.var_mean(latent_hidden_states, dim=(2, 3), keepdim=True, unbiased=False)
            std = torch.maximum(var, torch.zeros_like(var) + EPS) ** 0.5

            div_num = image_hidden_states.shape[0] // split_size
            mean_acc = sum(image_mean.split(split_size)) / div_num
            var_acc = sum(image_var.split(split_size)) / div_num

            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + EPS) ** 0.5
            y_uc = (((latent_hidden_states - mean) / std) * std_acc) + mean_acc
            if do_classifier_free_guidance and self.style_fidelity > 1e-5:
                y_c = y_uc.clone()
                y_c[self.current_uc_indices] = latent_hidden_states[self.current_uc_indices]
                latent_hidden_states = self.style_fidelity * y_c + (1.0 - self.style_fidelity) * y_uc
            else:
                latent_hidden_states = y_uc
            output = torch.cat([latent_hidden_states, image_hidden_states])

    if output is None:
        output = x

    return output


def upsample_2d_forward(self, hidden_states, output_size=None):
    x = self.original_forward(hidden_states, output_size=output_size)
    output = None
    if getattr(self, "enable_gn", False):
        if self.gn_auto_machine_weight > self.gn_weight:
            do_classifier_free_guidance = len(self.current_uc_indices) > 0
            split_size = 2 if do_classifier_free_guidance else 1

            latent_hidden_states = x[:split_size]  # uc, c
            image_hidden_states = x[split_size:]  # uc, c
            image_var, image_mean = torch.var_mean(image_hidden_states, dim=(2, 3), keepdim=True, unbiased=False)
            var, mean = torch.var_mean(latent_hidden_states, dim=(2, 3), keepdim=True, unbiased=False)
            std = torch.maximum(var, torch.zeros_like(var) + EPS) ** 0.5

            div_num = image_hidden_states.shape[0] // split_size
            mean_acc = sum(image_mean.split(split_size)) / div_num
            var_acc = sum(image_var.split(split_size)) / div_num

            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + EPS) ** 0.5
            y_uc = (((latent_hidden_states - mean) / std) * std_acc) + mean_acc
            if do_classifier_free_guidance and self.style_fidelity > 1e-5:
                y_c = y_uc.clone()
                y_c[self.current_uc_indices] = latent_hidden_states[self.current_uc_indices]
                latent_hidden_states = self.style_fidelity * y_c + (1.0 - self.style_fidelity) * y_uc
            else:
                latent_hidden_states = y_uc
            output = torch.cat([latent_hidden_states, image_hidden_states])

    if output is None:
        output = x

    return output


try:
    from diffusers.models.attention_processor import Attention

    if not hasattr(Attention, "original_forward"):
        Attention.original_forward = Attention.forward
    Attention.forward = self_attn_forward
except ImportError:
    pass
if not hasattr(CrossAttention, "original_forward"):
    CrossAttention.original_forward = CrossAttention.forward
if not hasattr(Transformer2DModel, "original_forward"):
    Transformer2DModel.original_forward = Transformer2DModel.forward
if not hasattr(ResnetBlock2D, "original_forward"):
    ResnetBlock2D.original_forward = ResnetBlock2D.forward
if not hasattr(Upsample2D, "original_forward"):
    Upsample2D.original_forward = Upsample2D.forward
CrossAttention.forward = self_attn_forward
Transformer2DModel.forward = transformer_2d_model_forward
ResnetBlock2D.forward = resnet_block_2d_forward
Upsample2D.forward = upsample_2d_forward


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(axis=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(axis=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def preprocess(image, resize_mode, width, height):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = resize_image(resize_mode=resize_mode, im=image, width=width, height=height)
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        image = [resize_image(resize_mode=resize_mode, im=im, width=width, height=height) for im in image]

        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def resize_image(resize_mode, im, width, height, upscaler_name=None):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
           -1: do nothing.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """
    # ["Just resize", "Crop and resize", "Resize and fill", "Do nothing"]
    #         0              1                   2               -1
    def resize(im, w, h):
        if upscaler_name is None or upscaler_name == "None" or im.mode == "L":
            return im.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])

    if resize_mode == -1:
        return im
    elif resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(
                resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                box=(0, fill_height + src_h),
            )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(
                resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                box=(fill_width + src_w, 0),
            )

    return res


class StableDiffusionReferenceOnlyPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with refernce only.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`PNDMScheduler`], [`EulerDiscreteScheduler`], [`EulerAncestralDiscreteScheduler`]
            or [`DPMSolverMultistepScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )
        self.attn_modules = None
        self.gn_modules = None

    def _default_height_width(self, height, width, image):
        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    def set_reference_only(
        self,
        attention_auto_machine_weight: float = 1.0,
        gn_auto_machine_weight: float = 1.0,
        style_fidelity: float = 0.5,
        enable_attn: bool = True,
        enable_gn: bool = True,
        do_classifier_free_guidance: bool = True,
    ):
        assert 0.0 <= attention_auto_machine_weight <= 1.0
        assert 0.0 <= gn_auto_machine_weight <= 2.0
        assert 0.0 <= style_fidelity <= 1.0

        if self.attn_modules is not None:
            for module in self.attn_modules:
                module.enable_attn = enable_attn
                module.attention_auto_machine_weight = attention_auto_machine_weight
                module.style_fidelity = style_fidelity
                module.current_uc_indices = [0] if do_classifier_free_guidance else []

        if self.gn_modules is not None:
            for module in self.gn_modules:
                module.enable_gn = enable_gn
                module.gn_auto_machine_weight = gn_auto_machine_weight
                module.style_fidelity = style_fidelity
                module.current_uc_indices = [0] if do_classifier_free_guidance else []

        # init attn_modules
        if self.attn_modules is None:
            attn_modules = []
            self_attn_processors_keys = []
            for name in self.unet.attn_processors.keys():
                if not name.endswith("attn1.processor"):
                    continue
                name = name.replace(".processor", "")
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]
                self_attn_processors_keys.append([name, hidden_size])

            # sorted by (-hidden_size, name)，down -> mid -> up.
            for i, (name, _) in enumerate(sorted(self_attn_processors_keys, key=lambda x: (-x[1], x[0]))):
                module = self.unet.get_submodule(name)
                module.attn_weight = float(i) / float(len(self_attn_processors_keys))

                module.enable_attn = enable_attn
                module.attention_auto_machine_weight = attention_auto_machine_weight
                module.style_fidelity = style_fidelity
                module.current_uc_indices = [0] if do_classifier_free_guidance else []

                attn_modules.append(module)
            self.attn_modules = attn_modules

        # init gn_modules
        if self.gn_modules is None:
            gn_modules = [
                self.unet.mid_block.attentions[-1],
            ]
            self.unet.mid_block.attentions[-1].gn_weight = 0.0  # mid             0.0

            input_block_names = [
                ("down_blocks.1.resnets.0", "down_blocks.1.attentions.0"),  # 4   2.0
                ("down_blocks.1.resnets.1", "down_blocks.1.attentions.1"),  # 5   1.66
                ("down_blocks.2.resnets.0", "down_blocks.2.attentions.0"),  # 7   1.33
                ("down_blocks.2.resnets.1", "down_blocks.2.attentions.1"),  # 8   1.0
                ("down_blocks.3.resnets.0",),  # 10                               0.66
                ("down_blocks.3.resnets.1",),  # 11                               0.33
            ]
            for w, block_names in enumerate(input_block_names):
                module = self.unet.get_submodule(block_names[-1])
                module.gn_weight = 1.0 - float(w) / float(len(input_block_names))
                gn_modules.append(module)

            output_block_names = [
                ("up_blocks.0.resnets.0",),  # 0                                 0.0
                ("up_blocks.0.resnets.1",),  # 1                                 0.25
                ("up_blocks.0.resnets.2", "up_blocks.0.upsamplers.0"),  # 2      0.5
                ("up_blocks.1.resnets.0", "up_blocks.1.attentions.0"),  # 3      0.75
                ("up_blocks.1.resnets.1", "up_blocks.1.attentions.1"),  # 4      1.0
                ("up_blocks.1.resnets.2", "up_blocks.1.attentions.2"),  # 5      1.25
                ("up_blocks.2.resnets.0", "up_blocks.2.attentions.0"),  # 6      1.5
                ("up_blocks.2.resnets.1", "up_blocks.2.attentions.1"),  # 7      1.75
            ]
            for w, block_names in enumerate(output_block_names):
                module = self.unet.get_submodule(block_names[-1])
                module.gn_weight = float(w) / float(len(output_block_names))
                gn_modules.append(module)

            for module in gn_modules:
                module.gn_weight *= 2
                module.enable_gn = enable_gn
                module.gn_auto_machine_weight = gn_auto_machine_weight
                module.style_fidelity = style_fidelity
                module.current_uc_indices = [0] if do_classifier_free_guidance else []

            self.gn_modules = gn_modules

    def prepare_image_latents(
        self, image, batch_size, dtype, device, generator=None, do_classifier_free_guidance=False
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )
        image = image.to(device=device, dtype=dtype)

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        if do_classifier_free_guidance:
            init_latents = torch.cat([init_latents] * 2)

        return init_latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        # reference
        control_name: str = "reference_only",  # "none", "reference_only", "reference_adain", "reference_adain+attn"
        attention_auto_machine_weight: float = 1.0,
        gn_auto_machine_weight: float = 1.0,
        style_fidelity: float = 0.5,
        resize_mode: int = 0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            attention_auto_machine_weight (`float`, *optional*, defaults to `1.0`):
                Weight of using reference query for self attention's context.
                If attention_auto_machine_weight=1.0, use reference query for all self attention's context.
            gn_auto_machine_weight (`float`, *optional*, defaults to `1.0`):
                Weight of using reference adain. If gn_auto_machine_weight=2.0, use all reference adain plugins.
            style_fidelity (`float`, *optional*, defaults to `0.5`):
                style fidelity of ref_uncond_xt. If style_fidelity=1.0, control more important,
                elif style_fidelity=0.0, prompt more important, else balanced.
            control_name (`str`, *optional*, defaults to `reference_only`):
                Choose control_name in ["none", "reference_only", "reference_adain", "reference_adain+attn"].
            resize_mode (`int`, *optional*, defaults to `-1`):
                The mode to use when resizing the image.
                -1: do nothing.
                0: Resize the image to the specified width and height.
                1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and
                then center the image within the dimensions, cropping the excess.
                2: Resize the image to fit within the specified width and height, maintaining the aspect ratio,
                and then center the image within the dimensions, filling empty with data from image.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        assert control_name in ["none", "reference_only", "reference_adain", "reference_adain+attn"]
        assert num_images_per_prompt == 1

        # 0. Default height and width
        height, width = self._default_height_width(height, width, image)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        dtype = prompt_embeds.dtype

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. reference_only
        enable_attn = (
            "only" in control_name
            or "attn" in control_name
            and image is not None
            and attention_auto_machine_weight > 0
        )
        enable_gn = "adain" in control_name and image is not None and gn_auto_machine_weight > 0
        self.set_reference_only(
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            enable_attn,
            enable_gn,
            do_classifier_free_guidance,
        )

        if enable_attn or enable_gn:
            image = preprocess(image, resize_mode, width, height)
            image_latents = self.prepare_image_latents(
                image, batch_size, dtype, device, generator, do_classifier_free_guidance
            )
            prompt_embeds = prompt_embeds.repeat(1 + image.shape[0], 1, 1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if enable_attn or enable_gn:
                    image_noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=dtype)
                    image_latent_model_input = self.scheduler.scale_model_input(
                        self.scheduler.add_noise(image_latents, image_noise, t.reshape(1)), t
                    )
                    split_size = 2 if do_classifier_free_guidance else 1
                    noise_pred = self.unet(
                        torch.cat([latent_model_input, image_latent_model_input]),
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample[:split_size]
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
