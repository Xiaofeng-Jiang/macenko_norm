import argparse
import os
from tqdm import tqdm
import stainNorm_Macenko
import cv2
from pathlib import Path


# target_path = '/Users/jiangxiaofeng/Desktop/MCO',
#  output_path = '/Users/jiangxiaofeng/Desktop/Macenko_MCO',
#  ref_path = '/Users/jiangxiaofeng/Desktop/github_test/macenko_norm/Ref.png'

def main(target_path, output_path, ref_path):
    ref_img = cv2.imread(str(ref_path))
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    normalizer = stainNorm_Macenko.Normalizer()
    normalizer.fit(ref_img)

    img_list = target_path.glob('**/*.jpg')
    img_list = list(img_list)

    for img_path in tqdm(img_list):

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        nor_img = normalizer.transform(img)

        new_img_path = str(img_path).replace(str(target_path), str(output_path))
        Path(new_img_path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(new_img_path):
            print('Exists, skipping...')
            continue
        cv2.imwrite(new_img_path, cv2.cvtColor(nor_img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Macenko Normlization')
    parser.add_argument('--target_path', '-t', type=Path)
    parser.add_argument('--output_path', '-o', type=Path)
    parser.add_argument('--ref_path', '-r', type=Path)
    args = parser.parse_args()
    main(**vars(args))
