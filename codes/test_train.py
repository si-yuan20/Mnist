# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from mynn.optimizer import MultiStepLR  # Assume you implement this

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


def data_augmentation(images, labels, augment_ratio=1.0):
    """
    对输入图像进行数据增强
    :param images: 原始图像矩阵 (num_samples, 784)
    :param labels: 对应标签 (num_samples,)
    :param augment_ratio: 增强比例 (0.0-1.0)，1.0表示数据量翻倍
    :return: 增强后的图像和标签 (增强后的数据追加在原数据之后)
    """
    # 转换为图像格式 (num_samples, 28, 28)
    original_images = images.reshape(-1, 28, 28)
    num_augment = int(len(original_images) * augment_ratio)

    # 随机选择要增强的样本索引
    augment_indices = np.random.choice(len(original_images), num_augment, replace=False)

    augmented_images = []
    for idx in augment_indices:
        img = original_images[idx].copy()

        # 随机选择增强方式
        choice = np.random.choice(['translation', 'rotation', 'scaling'], p=[0.4, 0.3, 0.3])

        if choice == 'translation':
            # 随机平移 (±2像素)
            dx, dy = np.random.randint(-2, 3, size=2)
            img = np.roll(img, dx, axis=1)
            img = np.roll(img, dy, axis=0)

        elif choice == 'rotation':
            # 随机旋转 (±15度)
            angle = np.random.uniform(-15, 15)
            rows, cols = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

        elif choice == 'scaling':
            # 随机缩放 (0.9-1.1倍)
            scale = np.random.uniform(0.9, 1.1)
            h, w = img.shape
            new_h, new_w = int(h * scale), int(w * scale)

            # 缩放后裁剪回原尺寸
            img_resized = cv2.resize(img, (new_w, new_h))
            if scale < 1:
                # 填充
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                img = np.pad(img_resized, ((pad_h, h - new_h - pad_h),
                                           (pad_w, w - new_w - pad_w)),
                             mode='constant')
            else:
                # 裁剪
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                img = img_resized[start_h:start_h + h, start_w:start_w + w]

        augmented_images.append(img.flatten())

    # 合并增强数据
    augmented_images = np.array(augmented_images)
    augmented_labels = labels[augment_indices]

    return np.vstack([images, augmented_images]), np.hstack([labels, augmented_labels])


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
scheduler = MultiStepLR(
    optimizer=optimizer,
    milestones=[800, 2400, 4000],
    gamma=0.5
)

loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()