import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
import pickle

def show_img(image):
    plt.imshow(image)
    plt.show()

def pickle2np(dicts):
    images = []
    labels = []
    for data_dict in dicts:
        images.append(data_dict['features'])
        labels.append(data_dict['labels'])
    return np.concatenate([img for img in images],axis=0), np.concatenate([lbl for lbl in labels], axis=0)

train_dicts = []
for i in range(1,8):
    path = '../downloads/train_set'+str(i)+'.p'
    with open(path, mode='rb') as f:
        train_dicts.append(pickle.load(f))
train_images, train_labels = pickle2np(train_dicts)

print("Imported Datas")
print("Training Shape: " + str(train_images.shape))


valid_dicts = []
for i in range(1,3):
    path = '../downloads/valid_set'+str(i)+'.p'
    with open(path, mode='rb') as f:
        valid_dicts.append(pickle.load(f))
valid_images, valid_labels = pickle2np(valid_dicts)
del valid_dicts

print("Validation Shape: " + str(valid_images.shape))

test_dicts = []
path = '../downloads/test.p'
with open(path, 'rb') as f:
    test_dicts.append(pickle.load(f))
test_images, test_ids = pickle2np(test_dicts)
del test_dicts

print("test Shape: " + str(test_images.shape))



def center_and_normalize(data, mu, dev):
    return (data-mu)/dev

def rotate(image, angle, fill_val):
    rot_matrix = cv2.getRotationMatrix2D((image.shape[0]//2,image.shape[1]//2), angle, 1)
    return cv2.warpAffine(image, rot_matrix, dsize=(image.shape[0], image.shape[1]),
                          flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS, borderValue=fill_val)

def rotate_data(images, labels, angles, fill_val):
    counter = images.shape[0]
    cancer_count = np.sum(labels)
    addons = cancer_count*4 + (labels.shape[0]-cancer_count)*2
    rotated_images = np.append(images,np.zeros([addons,images.shape[1],images.shape[2]],dtype=np.float32), axis=0)
    rotated_labels = []
    for i,img in enumerate(images):
        used_angles = angles
        if labels[i] == 1: used_angles = angles + [10,-10]
        for angle in used_angles:
            rotated_images[counter,:,:] = rotate(img,angle,fill_val)
            counter+=1
            rotated_labels.append(labels[i])
    labels = np.append(labels,rotated_labels,axis=0)
    return rotated_images, labels

def chop(image, img_size):
    limit = image.shape[0]//img_size
    chops = []
    for i in range(1,limit+1):
        for j in range(1,limit+1):
            chops.append(image[img_size*i-img_size:img_size*i,img_size*j-img_size:img_size*j])
    return chops

def chop_data(images, img_size):
    chopped_images = []
    print("Start Chopping")
    for i,img in enumerate(images):
        chopped_images.append(chop(img, img_size))
    print("End Chopping")
    return np.array(chopped_images, dtype=np.float32)

def one_hot_encode(labels, n_labels):
    encoded_labels = np.zeros((labels.shape[0], n_labels), dtype=np.float32)
    for i,label in enumerate(labels):
        encoded_labels[i,int(label)] = 1
    return encoded_labels

def preprocess(images, labels, mu, dev, angles=[-5,5], add_data=False):
    print("Start Initial Copies")
    images_copy = center_and_normalize(images, mu, dev)
    labels_copy = labels.copy()
    print("Finish Initial Copies")
    return images_copy, labels_copy

process_mu = np.mean(train_images)
process_dev = 500

train_images, train_labels = preprocess(train_images, train_labels, process_mu, process_dev)
# train_chops = chop_data(train_images,32)

valid_images, valid_labels = preprocess(valid_images, valid_labels, process_mu, process_dev)
# valid_chops = chop_data(valid_images, 32)

test_images = center_and_normalize(test_images, process_mu, process_dev)

train_images = train_images.reshape([train_images.shape[i] for i in range(3)]+[1])
valid_images = valid_images.reshape([valid_images.shape[i] for i in range(3)]+[1])
test_images = test_images.reshape([test_images.shape[i] for i in range(3)]+[1])
train_labels = one_hot_encode(train_labels,2)
valid_labels = one_hot_encode(valid_labels,2)

mu = 0
dev = 0.1

convweight_shapes = [
    # 512x512
    (5,5,1,64), # 508x508 stride=1
    (4,4,64,16), # 253x253 stride=2
    (5,5,16,50), # 125x125 stride=2
    (2,2,50,2), # 124x124 stride=1
    (2,2,2,2), # 62x62 stride=2
    (62,62,2,2)
]

convweights = [tf.Variable(tf.truncated_normal(shape=x,mean=mu,stddev=dev),name="reg_conv"+str(x[-1])) for x in convweight_shapes]
convbiases = [tf.Variable(tf.zeros([x[-1]]),name="reg_convbias"+str(x[-1])) for x in convweight_shapes]

def conv2d(data, weight, bias, stride=1, padding="VALID"):
    activations = tf.nn.bias_add(tf.nn.conv2d(data, weight,strides=[1,stride,stride,1],padding=padding),bias)
    return tf.nn.elu(activations)

def max_pool(data,k=2):
    return tf.nn.max_pool(data,ksize=[1,k,k,1],strides=[1,k,k,1],padding="VALID")

def conv_net(data, weights, biases, dropout_prob, strides=[]):
    if len(strides) == 0: strides = [1]*len(weights)
    logits = data
    for i,weight in enumerate(weights):
        logits = conv2d(logits, weight, biases[i],stride=strides[i])
#         logits = max_pool(logits)
        logits = tf.nn.dropout(logits, dropout_prob)
    return logits

def fc_net(data, weights, biases, dropout_prob):
    logits = data
    for i,weight in enumerate(weights):
        if i < len(weights)-1:
            logits = tf.matmul(logits, weight) + biases[i]
            logits = tf.nn.elu(logits)
    #         logits = tf.nn.dropout(logits, dropout_prob)
    return tf.matmul(logits,weights[-1])+biases[-1]

data_features = tf.placeholder(tf.float32, [None]+[train_images.shape[i] for i in range(1,len(train_images.shape))], name="data_features")
data_labels = tf.placeholder(tf.float32, [None, 2], name='data_labels')

convdropout = tf.placeholder(tf.float32, name="convdropout")
fcdropout = tf.placeholder(tf.float32, name="fcdropout")
momentum = tf.placeholder(tf.float32, name="momentum")
learning_rate = tf.placeholder(tf.float32, name="learning_rate")

## Convolution Only Architecture

logits = conv_net(data_features, convweights, convbiases, convdropout, strides=[1,2,2,1,2,1])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=data_labels))
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum).minimize(cost)

equals_list = tf.equal(tf.argmax(logits,1), tf.argmax(data_labels,1))
accuracy = tf.reduce_mean(tf.cast(equals_list,tf.float32))

save_file = './net.ckpt'
saver = tf.train.Saver()

init = tf.global_variables_initializer()

rate = .05
moment = .75
conv_dropout = .75
fc_dropout = .5
epochs = 6
batch_size = 100

with tf.Session() as sess:
    sess.run(init)
    print("Session Start")
    n_batches = int(train_images.shape[0]/batch_size)
    loss = []
    for epoch in range(epochs):
        print("Epoch: " + str(epoch+1))
        train_images, train_labels = shuffle(train_images, train_labels)
        running_cost = 0
        for batch in range(1,n_batches):
            batch_x = train_images[(batch-1)*batch_size:batch*batch_size]
            batch_y = train_labels[(batch-1)*batch_size:batch*batch_size]
            feed = {learning_rate: rate, momentum: moment,\
                                           convdropout: conv_dropout, fcdropout:fc_dropout,\
                                            data_features: batch_x, data_labels: batch_y}
            optcost = sess.run([optimizer, cost, accuracy], feed_dict=feed)
            running_cost += optcost[1]
            if batch % 10 == 0:
                print("Non-Cancer Percentage: " + str(1-np.sum(np.argmax(batch_y,1))/batch_size))
                print("Running Cost (Batch " + str(batch) + "): " + str(running_cost) + ", Acc: " + str(optcost[2]))
        loss.append(running_cost)
        valid_images, valid_labels = shuffle(valid_images, valid_labels)
        validation = {convdropout: 1.,fcdropout:1., data_features: valid_images[:batch_size], data_labels: valid_labels[:batch_size]}
        accost = sess.run([accuracy,cost], feed_dict=validation)
        print("\nActual Cancer Percentage: " + str(np.sum(np.argmax(valid_labels[:batch_size],1))/batch_size))
        print("Cost: " + str(accost[1]) + ", Accuracy: " + str(accost[0]))
        print("\n")
        saver.save(sess,save_file)

    feed = {learning_rate: rate, momentum: moment,\
                                   convdropout: 1., fcdropout: 1.,\
                                    data_features: valid_images, data_labels: valid_labels}
    acc = sess.run(accuracy, feed_dict=feed)
    print("Validation Accuracy: " + str(acc))

    plt.plot(np.arange(len(loss)), loss)
    plt.ylim([0,loss[0]])
    plt.title("Loss Over Training Epochs")
    plt.show()
