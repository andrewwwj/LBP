import h5py
import argparse
import numpy as np
from tqdm import tqdm
import os
import cv2
# from PIL import Image
import io
def main(args):
    data_names = ['libero_10', 'libero_90', 'libero_goal', 'libero_object', 'libero_spatial']
    for data_name in data_names:
        if data_name != args.libero_task:
            continue
        obs_keys = ['agentview_rgb', 'eye_in_hand_rgb']#, 'joint_states', 'gripper_states']
        base_dir = f'/home/andrew/pyprojects/datasets/Libero/256x256/{data_name}'
        save_base_dir = f'/home/andrew/pyprojects/datasets/{data_name}'
        hdf5_path_list = os.listdir(base_dir)

        for hdf5_path in tqdm(hdf5_path_list):
            print(f'\nProcessing {data_name} {hdf5_path} ...')
            # open file
            hdf5_file = h5py.File(os.path.join(base_dir, hdf5_path), 'r', swmr=False, libver='latest')
            demos = list(hdf5_file["data"].keys())
            inds = np.argsort([int(elem[5:]) for elem in demos])
            demos_sorted = [demos[i] for i in inds]
            # print(demos)
            # print(demos_sorted)

            # language instruction
            lang_instruction = hdf5_path.split("_demo.hdf5")[0].replace("_", " ")
            lang_instruction = ''.join([char for char in lang_instruction if not (char.isupper() or char.isdigit())])
            lang_instruction = lang_instruction.lstrip(' ')
            print(lang_instruction)

            # get data
            all_data = dict()
            for ep in demos_sorted:
                all_data = {}
                all_data["third_image"] = hdf5_file["data/{}/obs/{}".format(ep, 'agentview_rgb')][()].astype('float32')
                all_data["wrist_image"] = hdf5_file["data/{}/obs/{}".format(ep, 'eye_in_hand_rgb')][()].astype('float32')
                all_data["action"] = hdf5_file["data/{}/actions/".format(ep)][()].astype('float32')
                all_data['state'] = hdf5_file["data/{}/robot_states/".format(ep)][()].astype('float32')
                all_data['lang'] = lang_instruction

                # compress images
                compressed_3rd_img = []
                compressed_wrist_img = []
                for i in range(all_data["third_image"].shape[0]):
                    # Get a single image
                    img_3rd = all_data["third_image"][i][::-1, ::-1]
                    # Compress the image to the target size
                    result, encimg_3rd = cv2.imencode('.jpg', img_3rd)
                    # Add the compressed image to the list
                    compressed_3rd_img.append(encimg_3rd)
                for i in range(all_data["wrist_image"].shape[0]):
                    # Get a single image
                    img_wrist = all_data["wrist_image"][i]
                    # Compress the image to the target size
                    result, encimg_wrist = cv2.imencode('.jpg', img_wrist)
                    # Add the compressed image to the list
                    compressed_wrist_img.append(encimg_wrist)
                # save
                save_dir = os.path.join(save_base_dir, hdf5_path.split('.')[0])
                os.makedirs(save_dir, exist_ok=True)

                f = io.BytesIO()
                h = h5py.File(f, 'w')
                g = h.create_group('observation')
                dset = g.create_dataset('third_image', (len(compressed_3rd_img),), dtype=h5py.vlen_dtype(np.uint8))
                for i, image in enumerate(compressed_3rd_img):
                    dset[i] = image

                dset_wrist = g.create_dataset('wrist_image', (len(compressed_wrist_img),), dtype=h5py.vlen_dtype(np.uint8))
                for i, image in enumerate(compressed_wrist_img):
                    dset_wrist[i] = image

                h['action'] = all_data['action']
                h['proprio'] = all_data['state']
                h['language_instruction'] = all_data['lang']

                file_save_path = os.path.join(save_dir, f"{ep}.hdf5")
                h.close()
                with open(file_save_path, 'wb') as f_out:
                    f_out.write(f.getvalue())

                # os.makedirs(f"{save_dir}/image0", exist_ok=True)
                # os.makedirs(f"{save_dir}/image1", exist_ok=True)

                # save action
                # action_path = f"{save_dir}/action.npy"
                # state_path = f"{save_dir}/state.npy"
                # action = all_data[ep]["action"]
                # state = all_data[ep]["state"]
                # np.save(action_path, action)
                # np.save(state_path, state)

                # # save lang
                # with open(f"{save_dir}/lang.txt", "w") as f:
                #     f.write(lang_instruction)

                # # save image
                # for idx in range(all_data[ep]["attrs"]["num_samples"]):
                #     D435_image = Image.fromarray(all_data[ep]["obs"]['agentview_rgb'][idx].astype(np.uint8))
                #     wrist_image = Image.fromarray(all_data[ep]["obs"]['eye_in_hand_rgb'][idx].astype(np.uint8))

                #     D435_image_path = f"{save_dir}/image0/{idx}.jpg"
                #     wrist_image_path = f"{save_dir}/image1/{idx}.jpg"

                #     D435_image.save(D435_image_path)
                #     wrist_image.save(wrist_image_path)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    args = parser.parse_args()

    # Start data regeneration
    main(args)