# coding:utf-8
from PIL import Image
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

# ✕train ◯training

"""
Annotation
data_type: trainingとかtestとか
data_dir: データを入れてるディレクトリの名前。data/raw/(data_type)/まではprefix
ディレクトリ構造:
data---raw--------------------train-input
     |     |------------------test-input
     |     |------------------train-labels
     |-preprocessed_dataset---training------median_extract_training_dataset
     |                      |          |----pooled_training_dataset
     |                      |-test----------median_extract_test_dataset
     |                                 |----pooled_test_dataset
     |-training_dataset
     |-test-dataset
     |-lmdb
"""


class Preprocessing(object):

    def __init__(self, tif_data_path, data_name, mem_cgan_home=os.getcwd()):
        self.tif_data_path = tif_data_path
        self.mem_cgan_home = mem_cgan_home
        self.data_name = data_name
        self.save_data_path = '{0}/data/{1}/preprocessed/'.format(self.mem_cgan_home, self.data_name)

    def load_tif_images(self, data_name=''):
        os.chdir(self.tif_data_path + '/' + data_name)
        files = glob.glob('*.tif')
        os.chdir(self.mem_cgan_home)
        return files

    @staticmethod
    def image_to_array(file):
        raw_image = Image.open(file)
        raw_matrix = np.array(list(raw_image.getdata())).reshape(1024, 1024)
        return raw_matrix

    def patch_extract(self, data_dir, label_data_dir, prefix='', image_size=512, crop_size=256, stride=16):
        """
        1 stackをtraining 80枚、test20枚に分ける
        """
        files = self.load_tif_images(data_dir)
        labels = self.load_tif_images(label_data_dir)

        # train dataを置くディレクトリを作成
        if os.path.exists("{0}{1}train".format(self.save_data_path, prefix)):
            print "{0}{1}train already exist.".format(self.save_data_path, prefix)
            return
        else:
            os.mkdir("{0}{1}train".format(self.save_data_path, prefix))
            os.mkdir("{0}{1}train/input".format(self.save_data_path, prefix))
            os.mkdir("{0}{1}train/label".format(self.save_data_path, prefix))

        # test dataを置くディレクトリを作成
        if os.path.exists("{0}{1}test".format(self.save_data_path, prefix)):
            print "{0}{1}test already exist.".format(self.save_data_path, prefix)
            return
        else:
            os.mkdir("{0}{1}test".format(self.save_data_path, prefix))
            os.mkdir("{0}{1}test/input".format(self.save_data_path, prefix))
            os.mkdir("{0}{1}test/label".format(self.save_data_path, prefix))

        file_index = 0
        for _file, label in zip(files, labels):
            file_index += 1
            for h in xrange(int((image_size - crop_size) / stride)):
                for w in xrange(int((image_size - crop_size) / stride)):

                    # 画像のサイズを指定
                    patch_range = (w * stride, h * stride, w * stride + crop_size, h * stride + crop_size)
                    cropped_image = Image.open("{0}/{1}/{2}".format(self.tif_data_path, data_dir, _file)).crop(patch_range)
                    cropped_label = Image.open("{0}/{1}/{2}".format(self.tif_data_path, label_data_dir, label)).crop(patch_range)

                    canvas = Image.new('L', (512, 256), 255)
                    canvas.paste(cropped_label, (0, 0))
                    canvas.paste(cropped_image, (256, 0))

                    # 保存部分
                    if file_index <= int(len(files) * 0.9):
                        canvas.save("%s%strain/image_%03d%03d%03d.jpg" % (self.save_data_path, prefix, file_index, h, w))
                        cropped_image.save("%s%strain/input/input_%03d%03d%03d.jpg" % (self.save_data_path, prefix, file_index, h, w))
                        cropped_label.save("%s%strain/label/label_%03d%03d%03d.jpg" % (self.save_data_path, prefix, file_index, h, w))
                    else:
                        canvas.save("%s%stest/image_%03d%03d%03d.jpg" % (self.save_data_path, prefix, file_index, h, w))
                        cropped_image.save("%s%stest/input/input_%03d%03d%03d.jpg" % (self.save_data_path, prefix, file_index, h, w))
                        cropped_label.save("%s%stest/label/label_%03d%03d%03d.jpg" % (self.save_data_path, prefix, file_index, h, w))

            if file_index % 10 == 0:
                print "{0} images ended".format(file_index)

            # if file_index == crop_num * 0.8:
            #     print "%straining_dataset is created." % prefix
            #
            # if file_index == 100:
            #     print "%stest_dataset is created." % prefix

if __name__ == '__main__':
    try:
        tif_data_path = sys.argv[1]
        data_name = sys.argv[2]
        preprocessing = Preprocessing(tif_data_path=tif_data_path, data_name=data_name)
        preprocessing.patch_extract(data_dir='train/input', label_data_dir='train/label')
    except IndexError:
        quit()
    else:
        quit()

