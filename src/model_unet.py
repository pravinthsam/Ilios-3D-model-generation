import os
import shutil
import numpy as np
from skimage import io
import tensorflow as tf
import tools
import glob

from config import config

import time

vox_res64 = 512
vox_rex256 = 256
batch_size = 4
GPU0 = '0'

class Network:
    def __init__(self, config=None):

        self.config = config
        if config is None:
            self.epochs = 10
            self.learning_rate = 0.01
            self.batch_size = 4
        else:
            self.epochs = self.config['train_epochs']
            self.learning_rate = self.config['learning_rate_unet']
            self.batch_size = self.config['batch_size']

        self.train_mod_dir = './models/unet/'
        self.train_sum_dir = './summaries/train_sum_u/'
        self.test_res_dir = './summaries/test_res_u/'
        self.test_sum_dir = './summaries/test_sum_u/'
        self.global_vars = './summaries/global_vars_u'
        self.demo_dir = './demo/'
        re_train = True
        print ("re_train:", re_train)

        if not os.path.exists(self.global_vars):
            os.makedirs(self.global_vars)
            print ('global_vars: created!')

        if os.path.exists(self.test_res_dir):
            if re_train:
                print ("test_res_dir and files kept!")
            else:
                shutil.rmtree(self.test_res_dir)
                os.makedirs(self.test_res_dir)
                print ('test_res_dir: deleted and then created!')
        else:
            os.makedirs(self.test_res_dir)
            print ('test_res_dir: created!')

        if os.path.exists(self.train_mod_dir):
            if not re_train:
                shutil.rmtree(self.train_mod_dir)
                os.makedirs(self.train_mod_dir)
                print ('train_mod_dir: deleted and then created!')
        else:
            os.makedirs(self.train_mod_dir)
            print ('train_mod_dir: created!')

        if os.path.exists(self.train_sum_dir):
            if re_train:
                print ("train_sum_dir and files kept!")
            else:
                shutil.rmtree(self.train_sum_dir)
                os.makedirs(self.train_sum_dir)
                print ('train_sum_dir: deleted and then created!')
        else:
            os.makedirs(self.train_sum_dir)
            print ('train_sum_dir: created!')

        if os.path.exists(self.test_sum_dir):
            if re_train:
                print ("test_sum_dir and files kept!")
            else:
                shutil.rmtree(self.test_sum_dir)
                os.makedirs(self.test_sum_dir)
                print ('test_sum_dir: deleted and then created!')
        else:
            os.makedirs(self.test_sum_dir)
            print ('test_sum_dir: created!')

    def conv2d(self, x, k, out_c, str, name,pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_c = x.get_shape()[3]
        w = tf.get_variable(name + '_w', [k, k, in_c, out_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
        stride = [1, str, str, 1]
        y = tf.nn.bias_add(tf.nn.conv2d(x, w, stride, pad), b)
        return y
    def conv2d_transpose(self, x, k, out_c, str, name,pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_c = x.get_shape()[3]
        w = tf.get_variable(name + '_w', [k, k, in_c, out_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
        stride = [1, str, str, 1]
        y = tf.nn.bias_add(tf.nn.conv2d_transpose(x, w, [self.batch_size, int(str*x.shape[1]), int(str*x.shape[2]), out_c], stride, pad), b)
        return y
    def triple_conv(self, X, out_channels, name, Training):
        y = self.conv2d(X, 3, out_channels, 1, name+'_1')
        y = tf.nn.relu(y)
        y = self.conv2d(y, 3, out_channels, 1, name+'_2')
        y = tf.nn.relu(y)
        y = self.conv2d(y, 3, out_channels, 1, name+'_3')
        y = tf.nn.relu(y)
        y = tf.layers.batch_normalization(y,training=Training,
                        momentum=self.config['bn_momentum'])
        return y

    def unet_forward(self, X, Training):
        with tf.device('/gpu:'+GPU0):
            X = tf.reshape(X,[-1, vox_res64,vox_res64,4])

            conv1 = self.triple_conv(X, 64, 'conv_down1', Training)
            x = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            conv2 = self.triple_conv(x, 128, 'conv_down2', Training)
            x = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            conv3 = self.triple_conv(x, 256, 'conv_down3', Training)
            x = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            x = self.triple_conv(x, 512, 'conv_down4', Training)

            x = self.conv2d_transpose(x, 2, 512, 2, 'upsample3')
            x = tf.concat([x, conv3], axis=3)

            x = self.triple_conv(x, 256, 'conv_up3', Training)
            x = self.conv2d_transpose(x, 2, 256, 2, 'upsample2')
            x = tf.concat([x, conv2], axis=3)

            x = self.triple_conv(x, 128, 'conv_up2', Training)
            x = self.conv2d_transpose(x, 2, 128, 2, 'upsample1')
            x = tf.concat([x, conv1], axis=3)

            x = self.triple_conv(x, 64, 'conv_up1', Training)

            out = self.conv2d(x, 1, 1, 1, 'convlast')
            return out
    def build_graph(self):

        self.X = tf.placeholder(tf.float32, [None, vox_res64, vox_res64, 4], name='input')
        self.Y = tf.placeholder(tf.float32, [None, vox_res64, vox_res64, 1], name='target')
        self.Training = tf.placeholder(tf.bool, name='training_flag')

        with tf.device('/gpu:'+GPU0):
            self.Depth = self.unet_forward(self.X, self.Training)
            print(self.Depth.shape)
            mask = tf.reshape(self.X[:, :, :, 3], [-1, vox_res64, vox_res64, 1])
            mask = tf.greater(mask, 0.5)
            self.Depth = tf.where(mask, self.Depth, tf.ones_like(self.Depth, dtype=tf.float32))
            self.mse_loss = tf.reduce_mean(tf.squared_difference(self.Depth, self.Y))
            sum_mse_loss = tf.summary.scalar('mse_loss', self.mse_loss)

            self.global_step = tf.Variable(0, trainable=False)
            self.previous_step = tf.Variable(0, trainable=False)
            self.previous_epoch = tf.Variable(0, trainable=False)

            self.increment_prev_step_op = tf.assign(self.previous_step, self.previous_step+1)
            self.increment_prev_epoch_op = tf.assign(self.previous_epoch, self.previous_epoch+1)


            #learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
            #                                           1000, 0.96, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01,)
            self.train_op = optimizer.minimize(
                            self.mse_loss,
                            var_list=[var for var in tf.trainable_variables()],
                            global_step=self.global_step
                        )
            #self.train_op = optimizer.minimize(self.mse_loss)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group([self.train_op, update_ops])
            #self.train_op = train_op

        self.sum_merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=1)

        cfg = tf.ConfigProto(allow_soft_placement=True)
        cfg.gpu_options.visible_device_list = GPU0

        self.sess = tf.Session(config=cfg)
        self.sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, self.sess.graph)
        self.sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

        path = self.train_mod_dir
        model_path = glob.glob(path + 'model_*.cptk.data*')
        print(model_path)

        if len(model_path)>0:
            print ('restoring saved model')
            model_path.sort()
            self.saver.restore(self.sess, '.'.join(model_path[-1].split('.')[:-1]))
        else:
            print ('initilizing model')
            self.sess.run(tf.global_variables_initializer())


        return 0

    def train(self, data):
        [previous_step, previous_epoch] = self.sess.run(
                        [self.previous_step, self.previous_epoch])
        print('The model has been trained for {} epochs'.format(previous_epoch))
        for epoch in range(self.epochs):
            #data.shuffle_train_files()
            total_train_batch_num = data.total_train_batch_num
            print ('total_train_batch_num:', total_train_batch_num)
            print ('epochs:', self.epochs)

            ##### TRAINING ######self.train_op,
            for i in range(total_train_batch_num):
                X_train_batch, Y_train_batch = data.queue_train.get()
                self.sess.run(self.train_op, feed_dict={
                                        self.X:X_train_batch,
                                        self.Y:Y_train_batch,
                                        self.Training:True
                                    })
                [mse_loss, sum_train, _] = self.sess.run([
                                    self.mse_loss, self.sum_merged, self.increment_prev_step_op],
                                feed_dict={
                                    self.X:X_train_batch,
                                    self.Y:Y_train_batch,
                                    self.Training:True
                                })

                self.sum_writer_train.add_summary(sum_train,
                        previous_step + epoch * total_train_batch_num + i)
                print ('ep:',epoch,'i:',i, 'train mse loss:',mse_loss)

            self.sess.run(self.increment_prev_epoch_op)

            ##### VALIDATION ######

            X_test_batch, Y_test_batch = data.load_test_next_batch(2)
            [mse_loss, sum_test, depth] = self.sess.run([
                                self.mse_loss, self.sum_merged, self.Depth],
                            feed_dict={
                                self.X:X_test_batch,
                                self.Y:Y_test_batch,
                                self.Training:False
                            })
            #to_save = {'X_test':X_test_batch, 'Y_test_pred':depth, 'Y_test_true':Y_test_batch}
            #scipy.io.savemat(self.test_res_dir+'depth_pred_'+str(epoch).zfill(2)+'_'+str(i).zfill(5)+'.mat',
            #    to_save, do_compression=True)
            print ('ep:',epoch, 'test mse loss:', mse_loss)

            ##### MODEL SAVING #####
            if epoch%1==0:
                self.saver.save(self.sess, save_path=self.train_mod_dir + 'model_'+str(previous_epoch + epoch+1).zfill(2)+'.cptk')
                print('Model saved to {}'.format(self.train_mod_dir + 'model_'+str(previous_epoch + epoch+1).zfill(2)+'.cptk'))
        data.stop_queue = True

    def demo(self):
        previous_epoch = self.sess.run(self.previous_epoch)
        print('The model has been trained for {} epochs'.format(previous_epoch))
        d = tools.Data_depth(config)

        if not os.path.exists(self.demo_dir+'input/'):
            print('Demo input folder not present!!!')
            return
        filenames = glob.glob(self.demo_dir+'input/*')

        if len(filenames) == 0:
            print('No files found in input folder!!')
            return

        if not os.path.exists(self.demo_dir+'depth/'):
            os.makedirs(self.demo_dir+'depth/')

        if len(filenames)%self.batch_size != 0:
            print('Number of images should be a multiple of batch size ({})'.format(self.batch_size))
            return

        for i in range(len(filenames)//self.batch_size):
            X_data_files = filenames[self.batch_size * i:self.batch_size * (i + 1)]
            Y_data_files = filenames[self.batch_size * i:self.batch_size * (i + 1)]

            X_test_batch, Y_test_batch = d.load_X_Y_images(X_data_files, Y_data_files)

            Y_pred_batch = self.sess.run(self.Depth,  feed_dict={
                                    self.X:X_test_batch,
                                    self.Training:False
                                })
            for i, filename in enumerate(X_data_files):
                io.imsave(filename.replace('/input/',
                                    '/depth/'),
                                Y_pred_batch[i, :, :, 0])






if __name__ == '__main__':
    data = tools.Data_depth(config)
    data.daemon = True
    data.start()
    net = Network(config)
    net.build_graph()
    start = time.time()
    net.train(data)

    end = time.time()
    print('Training took {}s...'.format(end-start))
