import sys
import time

sys.path.append("/home/gdli7/SparkAgent")
sys.path.append("/home/gdli7/SparkAgent/ComfyUI")

import torch
from PIL import Image
import numpy as np
import ComfyUI.comfy
import ComfyUI.comfy.utils
import ComfyUI.comfy.sd
import ComfyUI.folder_paths as folder_paths
from ComfyUI.external import import_custom_nodes, NODE_CLASS_MAPPINGS, get_value_at_index

from chat_utils import load_image_pil

# comfyui load nodes
import_custom_nodes()


def run_acepp(image,
              # mask,
              # task,
              prompt,
              # negative,
              # history_to,
              output_height,
              output_width,
              # sampler,
              sample_steps,
              guide_scale,
              # guide_rescale,
              seed,
              # refiner_prompt,
              # refiner_scale
              ):
    # import pdb;pdb.set_trace()
    with torch.inference_mode():
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_144 = unetloader.load_unet(
            unet_name="flux1-fill-dev.safetensors", weight_dtype="default"
        )

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_145 = dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5xxl_fp16.safetensors",
            type="flux",
            device="default",
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_146 = vaeloader.load_vae(vae_name="ae.safetensors")

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_147 = cliptextencode.encode(
            text=prompt, clip=get_value_at_index(dualcliploader_145, 0)
        )

        cliptextencode_148 = cliptextencode.encode(
            text="", clip=get_value_at_index(dualcliploader_145, 0)
        )

        # loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        # loadimage_151 = loadimage.load_image(image="female1.jpg")
        loadimage_151 = load_image_pil(image)

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_230 = loraloadermodelonly.load_lora_model_only(
            lora_name="comfyui_portrait_lora64.safetensors",
            strength_model=1,
            model=get_value_at_index(unetloader_144, 0),
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_305 = ksamplerselect.get_sampler(sampler_name="euler")

        disablenoise = NODE_CLASS_MAPPINGS["DisableNoise"]()
        disablenoise_308 = disablenoise.get_noise()

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_152 = fluxguidance.append(
            guidance=guide_scale, conditioning=get_value_at_index(cliptextencode_147, 0)
        )

        imagegenresolutionfromimage = NODE_CLASS_MAPPINGS[
            "ImageGenResolutionFromImage"
        ]()
        imagegenresolutionfromimage_164 = imagegenresolutionfromimage.execute(
            image=get_value_at_index(loadimage_151, 0)
        )

        imagepadforoutpaint = NODE_CLASS_MAPPINGS["ImagePadForOutpaint"]()
        imagepadforoutpaint_165 = imagepadforoutpaint.expand_image(
            left=0,
            top=0,
            right=get_value_at_index(imagegenresolutionfromimage_164, 0),
            bottom=0,
            feathering=0,
            image=get_value_at_index(loadimage_151, 0),
        )

        inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
        inpaintmodelconditioning_309 = inpaintmodelconditioning.encode(
            noise_mask=False,
            positive=get_value_at_index(fluxguidance_152, 0),
            negative=get_value_at_index(cliptextencode_148, 0),
            vae=get_value_at_index(vaeloader_146, 0),
            pixels=get_value_at_index(imagepadforoutpaint_165, 0),
            mask=get_value_at_index(imagepadforoutpaint_165, 1),
        )

        differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
        basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        imagecrop = NODE_CLASS_MAPPINGS["ImageCrop"]()

        # for q in range(1):
        differentialdiffusion_155 = differentialdiffusion.apply(
            model=get_value_at_index(loraloadermodelonly_230, 0)
        )

        basicguider_307 = basicguider.get_guider(
            model=get_value_at_index(differentialdiffusion_155, 0),
            conditioning=get_value_at_index(inpaintmodelconditioning_309, 0),
        )

        basicscheduler_306 = basicscheduler.get_sigmas(
            scheduler="normal",
            steps=sample_steps,
            denoise=1,
            model=get_value_at_index(differentialdiffusion_155, 0),
        )

        samplercustomadvanced_304 = samplercustomadvanced.sample(
            noise=get_value_at_index(disablenoise_308, 0),
            guider=get_value_at_index(basicguider_307, 0),
            sampler=get_value_at_index(ksamplerselect_305, 0),
            sigmas=get_value_at_index(basicscheduler_306, 0),
            latent_image=get_value_at_index(inpaintmodelconditioning_309, 2),
        )

        vaedecode_310 = vaedecode.decode(
            samples=get_value_at_index(samplercustomadvanced_304, 0),
            vae=get_value_at_index(vaeloader_146, 0),
        )

        imagecrop_312 = imagecrop.crop(
            width=get_value_at_index(imagegenresolutionfromimage_164, 0),
            height=get_value_at_index(imagegenresolutionfromimage_164, 1),
            x=get_value_at_index(imagegenresolutionfromimage_164, 0),
            y=0,
            image=get_value_at_index(vaedecode_310, 0),
        )

        # import pdb;pdb.set_trace()
        image_batch = imagecrop_312[0]

        image_list = []
        for (i, image) in enumerate(image_batch):
            img_np = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            image_list.append(img)
        return image_list