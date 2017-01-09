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

    def __init__(self, tif_data_dir, mem_cgan_home=os.getcwd()):
        self.tif_data_dir = tif_data_dir
        self.mem_cgan_home = mem_cgan_home

    def load_tif_images(self, data_name=''):
        os.chdir(self.tif_data_dir + '/' + data_name)
        files = glob.glob('*.tif')
        os.chdir(self.mem_cgan_home)
        return files

    @staticmethod
    def image_to_array(file):
        raw_image = Image.open(file)
        raw_matrix = np.array(list(raw_image.getdata())).reshape(1024, 1024)
        return raw_matrix

    def make_median_extracted_dataset(self, data_type):
        files = self.load_tif_images()

        if os.path.exists("%s/data/preprocessed/%s/median_extract_%s_dataset" % (mem_cgan_home, data_type, data_type)) != True:
            os.mkdir("%s/data/preprocessed/%s/median_extract_%s_dataset" % (mem_cgan_home, data_type, data_type)) # データ置き場用意

        # スタック中全画像からmedianを求める(medianの平均値)
        # fixme: medianの平均でいいのか・・・？
        N, _sum = 0, 0
        for _file in files:
            raw_matrix = self.image_to_array("data/%s/%s" % (_file))
            median = np.median(raw_matrix)
            _sum += median
            N += 1
        stack_median = _sum / N

        file_num = 1
        for _file in files:
            raw_matrix = self.image_to_array("data/%s/%s" % (_file))
            median = np.median(raw_matrix) #中央値
            # スタックのmedianに各画像のmedianを合わせる
            median_extract_matrix = (raw_matrix - (median - stack_median))

            # 負の画素値を0に補正
            # fixme: こんな処理を入れずにスマートにやりたい
            for i in range(1024):
                for j in range(1024):
                    if median_extract_matrix[i][j] < 0:
                        median_extract_matrix[i][j] = 0

            median_extract_image = Image.fromarray(np.uint8(median_extract_matrix).reshape(1024, 1024))
            median_extract_image.save("%s/data/preprocessed/%s/median_extract_%s_dataset/median_extract_image_%03d.tif" % (mem_cgan_home, data_type, data_type, file_num))
            file_num += 1
            if file_num % 10 == 0:
                print "%s images ended" % file_num
        print "median_extract_%s_dataset is created." % data_type

    def make_average_pooled_dataset(self, data_type, data_dir):
        filelist = self.load_tif_images(data_dir)

        if os.path.exists("%s/data/preprocessed/%s/pooled_%s_dataset" % (mem_cgan_home, data_type, data_type)) != True:
            os.mkdir("%s/data/preprocessed/%s/pooled_%s_dataset" % (mem_cgan_home, data_type, data_type)) # データ置き場用意

        file_num = 1
        for file in filelist:
            raw_matrix = self.image_to_array("data/%s/%s" % (data_dir, file))
            pooled_matrix = []
            for i in range(int(1024 / 4)):
                for j in range(int(1024 / 4)):
                    _sum = 0
                    for k in range(4):
                        for l in range(4):
                            _sum += raw_matrix[4 * i + k, 4 * j + l]
                    pooled_pixel = _sum / 16
                    pooled_matrix.append(pooled_pixel)
            pooled_image = Image.fromarray(np.uint8(pooled_matrix).reshape(256, 256))
            pooled_image.save("%s/data/preprocessed/%s/pooled_%s_dataset/pooled_image_%03d.tif" % (mem_cgan_home, data_type, data_type, file_num))
            file_num += 1
            if file_num % 10 == 0:
                print "%s images ended" % file_num
        print "pooled_%s_dataset is created." % data_type

    def patch_extract(self, data_dir, label_data_dir, prefix='', image_size=512, crop_size=256, stride=16):
        """
        1 stackをtraining 80枚、test20枚に分ける
        """
        files = self.load_tif_images(data_dir)
        labels = self.load_tif_images(label_data_dir)

        # train dataを置くディレクトリを作成
        if os.path.exists("%s/data/preprocessed/%strain" % (self.mem_cgan_home, prefix)):
            print "%s/data/preprocessed/%strain already exist." % (self.mem_cgan_home, prefix)
            return
        else:
            os.mkdir("%s/data/preprocessed/%strain" % (self.mem_cgan_home, prefix))
            os.mkdir("%s/data/preprocessed/%strain/input" % (self.mem_cgan_home, prefix))
            os.mkdir("%s/data/preprocessed/%strain/label" % (self.mem_cgan_home, prefix))

        # test dataを置くディレクトリを作成
        if os.path.exists("%s/data/preprocessed/%stest" % (self.mem_cgan_home, prefix)):
            print "%s/data/preprocessed/%stest already exist." % (self.mem_cgan_home, prefix)
            return
        else:
            os.mkdir("%s/data/preprocessed/%stest" % (self.mem_cgan_home, prefix))
            os.mkdir("%s/data/preprocessed/%stest/input" % (self.mem_cgan_home, prefix))
            os.mkdir("%s/data/preprocessed/%stest/label" % (self.mem_cgan_home, prefix))

        file_index = 0
        crop_num = int(((image_size - crop_size) / stride + 1)**2)
        for _file, label in zip(files, labels):
            file_index += 1
            # 縦横の切り取り回数を計算(適当)
            for h in xrange(crop_num):
                for w in xrange(int((image_size - crop_size) / stride)):

                    # 画像のサイズを指定
                    patch_range = (w * stride, h * stride, w * stride + crop_size, h * stride + crop_size)
                    cropped_image = Image.open("%s/%s/%s" % (self.tif_data_dir, data_dir, _file)).crop(patch_range)
                    cropped_label = Image.open("%s/%s/%s" % (self.tif_data_dir, label_data_dir, label)).crop(patch_range)

                    # 保存部分
                    if file_index <= crop_num * 0.8:
                        cropped_image.save("%s/data/preprocessed/%strain/input/image_%03d%03d%03d.jpg" % (self.mem_cgan_home, prefix, file_index, h, w))
                        cropped_label.save("%s/data/preprocessed/%strain/label/label_%03d%03d%03d.jpg" % (self.mem_cgan_home, prefix, file_index, h, w))
                    else:
                        cropped_image.save("%s/data/preprocessed/%stest/input/image_%03d%03d%03d.jpg" % (self.mem_cgan_home, prefix, file_index, h, w))
                        cropped_label.save("%s/data/preprocessed/%stest/label/label_%03d%03d%03d.jpg" % (self.mem_cgan_home, prefix, file_index, h, w))

            if file_index % 10 == 0:
                print "%s images ended" % file_index

            # if file_index == crop_num * 0.8:
            #     print "%straining_dataset is created." % prefix
            #
            # if file_index == 100:
            #     print "%stest_dataset is created." % prefix

if __name__ == '__main__':
    try:
        tif_data_dir = sys.argv[1]
        preprocessing = Preprocessing(tif_data_dir=tif_data_dir)
        preprocessing.patch_extract(data_dir='train/input', label_data_dir='train/label')
    except IndexError:
        quit()
    else:
        quit()

