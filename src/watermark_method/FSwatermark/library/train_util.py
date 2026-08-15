
import argparse
import os
from transformers import CLIPTokenizer
from diffusers import DDPMScheduler, StableDiffusionPipeline

import library.model_util as model_util

TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"

EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

def add_sd_models_arguments(parser: argparse.ArgumentParser):
  
  parser.add_argument("--v2", default='True',
                      help='load Stable Diffusion v2.0 model / Stable Diffusion 2.0')
  parser.add_argument("--v_parameterization", action='store_true',
                      help='enable v-parameterization training / v-parameterization')
  
def verify_training_args(args: argparse.Namespace):
  if args.v_parameterization and not args.v2:
    print("v_parameterization should be with v2")
  
def load_tokenizer(args: argparse.Namespace):
  print("prepare tokenizer")
  if args.v2:
    tokenizer = CLIPTokenizer.from_pretrained(V2_STABLE_DIFFUSION_PATH, subfolder="tokenizer")
  else:
    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)
  if args.max_token_length is not None:
    print(f"update token length: {args.max_token_length}")
  return tokenizer

def load_target_model(args: argparse.Namespace, weight_dtype):
  name_or_path = args.pretrained_model_name_or_path
  name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path
  load_stable_diffusion_format = os.path.isfile(name_or_path)           
  if load_stable_diffusion_format:
    print("load StableDiffusion checkpoint")
    text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, name_or_path)
  else:
    print("error Diffusers pretrained models")
    pipe = StableDiffusionPipeline.from_pretrained(name_or_path, tokenizer=None, safety_checker=None)
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    del pipe

  if args.vae is not None:
    vae = model_util.load_vae(args.vae, weight_dtype)
    print("additional VAE loaded")

  return text_encoder, vae, unet, load_stable_diffusion_format
