import os
from typing import List
from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import PIL.ImageOps
import torch.cuda as cuda

MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V3.0_VAE",
            cache_dir=MODEL_CACHE,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = "",
        negative_prompt: str = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        image: Path = Input(
            description="Input image to in-paint. Width and height should both be divisible by 8. If they're not, the image will be center cropped to the nearest width and height divisible by 8",
        ),
        mask: Path = Input(
            description="Black and white image to use as mask. White pixels are inpainted and black pixels are preserved.",
        ),
        invert_mask: bool = Input(
            description="If this is true, then black pixels are inpainted and white pixels are preserved.",
            default=False,
        ),
        num_outputs :  int = Input(description="Number of images to create (maximum: 5)", ge=1, le=5, default=1),
        num_inference_steps : int = Input(description=" num_inference_steps (maximum: 100)", ge=0, le=100, default=20),
        guidance_scale : float = Input(description="Higher guidance scale encourages to generate images that are closely linked to the text prompt (maximum: 20)", default=7.5, le=20),
        strength: float = Input(default=0.75, ge=0.0, le=1.0, description="Choose strength factor, 0 for no init."), 
        seed : int = Input(description="Seed (0 = random, maximum: 2147483647)", default=0),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed == 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("RGB")

        if invert_mask:
            mask = PIL.ImageOps.invert(mask)

        if image.width % 8 != 0 or image.height % 8 != 0:
            if mask.size == image.size:
                mask = crop(mask)
            image = crop(image)

        if mask.size != image.size:
            print(
                f"WARNING: Mask size ({mask.width}, {mask.height}) is different from image size ({image.width}, {image.height}). Mask will be resized to image size."
            )
            mask = mask.resize(image.size)

        generator = torch.Generator("cuda").manual_seed(seed)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = self.pipe(
                    strength = strength,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    num_images_per_prompt=num_outputs,
                    mask_image=mask,
                    width=image.width,
                    height=image.height,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                )

        samples = []
        if output.nsfw_content_detected is not None:
            for i, nsfw_flag in enumerate(output.nsfw_content_detected):
                if not nsfw_flag:
                    samples.append(output.images[i])
        else:
            samples = output.images

        if len(samples) == 0:
            raise Exception(
                "NSFW content detected. Try running it again or try a different prompt."
            )

        if num_outputs > len(samples):
            print(
                f"NSFW content detected in {num_outputs - len(samples)} outputs, returning the remaining {len(samples)} images."
            )
        output_paths = []
        for i, sample in enumerate(samples):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths

def crop(image):
    height = (image.height // 8) * 8
    width = (image.width // 8) * 8
    left = int((image.width - width) / 2)
    right = left + width
    top = int((image.height - height) / 2)
    bottom = top + height
    image = image.crop((left, top, right, bottom))
    return image
