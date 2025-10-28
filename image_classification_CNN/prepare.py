import os
import shutil
import random


def prepare_data(source_dir, dest_dir, split_ratios):
    classes = ['tip', 'hole']
    for cls in classes:
        cls_dir = f"img_{cls}"
        files = os.listdir(os.path.join(source_dir, cls_dir))

        random.shuffle(files)
        train_split = int(split_ratios[0] * len(files))
        val_split = int(split_ratios[1] * len(files))

        train_files = files[:train_split]
        val_files = files[train_split:train_split + val_split]
        test_files = files[train_split + val_split:]

        for split, split_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            split_dir = os.path.join(dest_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for file in split_files:
                shutil.copy(os.path.join(source_dir, cls_dir, file), split_dir)


if __name__ == '__main__':
    source_dir = '/home/wsy/image_classification'
    dest_dir = '/home/wsy/image_classification_CNN/dataset'
    split_ratio = [0.7, 0.2, 0.1]

    prepare_data(source_dir, dest_dir, split_ratio)
