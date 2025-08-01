"""Utils for evaluating policies in LIBERO simulation environments."""
import time
import math
import os

import imageio
import numpy as np
# import tensorflow as tf
import torch
import torchvision.io
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    # img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    # img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    # img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    # img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    # img = img.numpy()
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    # 1. Equivalent of `tf.image.encode_jpeg(img)`
    # `torchvision.io.encode_jpeg` takes a (C, H, W) uint8 tensor.
    jpeg_encoded_tensor = torchvision.io.encode_jpeg(img_tensor)
    # 2. Equivalent of `tf.io.decode_image(img, ...)`
    # `torchvision.io.decode_image` takes a 1D tensor of encoded bytes.
    decoded_tensor = torchvision.io.decode_image(jpeg_encoded_tensor, mode=torchvision.io.ImageReadMode.UNCHANGED)
    # For resizing, the tensor needs to be in float format in the [0.0, 1.0] range.
    # The decoded tensor is uint8, so we convert it.
    float_tensor = decoded_tensor.float() / 255.0
    # 3. Equivalent of `tf.image.resize(img, resize_size, method="lanczos3", antialias=True)`
    # `F.resize` performs the image resizing.
    resized_tensor = F.resize(
        float_tensor,
        list(resize_size),
        interpolation=InterpolationMode.LANCZOS,
        antialias=True
    )
    # 4. Equivalent of `tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)`
    # We perform these operations in sequence on the resized tensor.
    final_tensor = (resized_tensor * 255).round().clamp(0, 255).to(torch.uint8)
    # 5. Equivalent of `.numpy()`
    # Convert the final tensor back to a NumPy array, permuting the dimensions
    # from (C, H, W) back to the standard image format of (H, W, C).
    img = final_tensor.permute(1, 2, 0).numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den