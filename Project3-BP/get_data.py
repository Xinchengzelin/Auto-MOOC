import zipfile
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


IMG_SIZE = 32


def read_data(path, _prefix_):
    with zipfile.ZipFile(path, 'r') as z:
        # z.printdir()
        images = []
        labels = []
        # 总共有62个类别
        for c in range(0, 62):
            # 得到当前类别文件夹路径
            prefix = _prefix_ + format(c, '05d') + '/'
            # 打开annotation文件
            gtFile = z.open(prefix + 'GT-' + format(c, '05d') + '.csv')
            next(gtFile)
            # 根据annotation读取每一张图片
            for line in gtFile:
                annotation = line.decode('utf8')
                annotation = annotation.split(';')
                # 打开图片
                image = z.read(prefix + annotation[0])
                #print(image)
                # 使用cv2模块将zipfile读取的文件解码为正确格式
                image_decode = cv2.imdecode(np.frombuffer(image, np.uint8), 1)
                #print(image_decode)
                # 将大小调整为指定的值
                image_resize = cv2.resize(image_decode, (IMG_SIZE, IMG_SIZE))
                #print(image_resize)
                #print('*********************************************')
                images.append(image_resize)
                labels.append(int(annotation[7]))
            gtFile.close()
    return np.asarray(images), np.asarray(labels)


# 灰度化
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# 正规化
def normalise_images(imgs, dist):
    max = np.max(dist)
    min = np.min(dist)
    return (imgs - min)/(max-min)
    # std = np.std(dist)
    # mean = np.mean(dist)
    # return (imgs - mean) / std


def main():
    train_x, train_y = read_data('./data/BelgiumTSC_Training.zip', 'Training/')
    print(len(train_x))
    train_x_grayscale = np.asarray(list(map(lambda img:to_grayscale(img), train_x)))
    train_x_normalised = normalise_images(train_x_grayscale, train_x_grayscale)
    i = 0
    for sample in random.sample(range(0, len(train_x_normalised)), 20):
        plt.subplot(4, 5, i + 1)
        plt.axis('off')
        plt.imshow(train_x_normalised[sample])
        plt.title("class:" + str(train_y[sample]))
        plt.subplots_adjust(wspace=0.5)
        i += 1
    plt.show()
    print(train_x_normalised[sample])


if __name__ == '__main__':
    main()
