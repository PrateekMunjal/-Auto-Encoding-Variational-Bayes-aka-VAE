from scipy import misc
import matplotlib.pyplot as plt 
import config
import logging
import PIL,os,sys
import numpy as np
from PIL import Image
import tensorflow as tf
import random


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data',one_hot=True);

tf.set_random_seed(0);
random.seed(0);

learning_rate = 0.0005;
batch_size = 256;
n_epoch = 100;

encoder_role = tf.placeholder(tf.float32,[]);
discriminator_role = tf.placeholder(tf.float32,[]);

z_dim = 16;

tfd = tf.contrib.distributions

X = tf.placeholder(tf.float32,[None,784]);
epoch_number = tf.placeholder(tf.float32,[]);


#Used to initialize kernel weights
stddev = 0.02;#0.02;#99999;

def prior_z(latent_dim):
    z_mean = tf.zeros(latent_dim);
    z_var = tf.ones(latent_dim);
    return tfd.MultivariateNormalDiag(z_mean,z_var);

#assumed noise distribution N(0,1)
def epsilon_distribution(latent_dim):
    eps_mean = tf.zeros(latent_dim);
    eps_var = tf.ones(latent_dim);
    return tfd.MultivariateNormalDiag(eps_mean,eps_var);

def encoder(X,isTrainable=True,reuse=False,name='encoder'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables();

        X = tf.reshape(X,[-1,28,28,1]);

        conv1 = tf.layers.conv2d(X,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=16,kernel_size=[3,3],padding='SAME',strides=(2,2),name='enc_conv1_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv1 = tf.layers.batch_normalization(conv1,training=isTrainable,reuse=reuse,name='bn_1');
        conv1 = tf.nn.relu(conv1,name='leaky_relu_conv_1');

        #14x14x32
        conv2 = tf.layers.conv2d(conv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=32,kernel_size=[3,3],padding='SAME',strides=(2,2),name='enc_conv2_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv2 = tf.layers.batch_normalization(conv2,training=isTrainable,reuse=reuse,name='bn_2');
        conv2 = tf.nn.relu(conv2,name='leaky_relu_conv_2');
        
        #7x7x64
        conv3 = tf.layers.conv2d(conv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[3,3],padding='SAME',strides=(2,2),name='enc_conv3_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv3 = tf.layers.batch_normalization(conv3,training=isTrainable,reuse=reuse,name='bn_3');
        conv3 = tf.nn.relu(conv3,name='leaky_relu_conv_3');
    
        
        #4x4x128
        conv3_flattened = tf.layers.flatten(conv3);
        
        z_mean = tf.layers.dense(conv3_flattened,z_dim,name='enc_mean',trainable=isTrainable,reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev));
        z_variance = tf.layers.dense(conv3_flattened,z_dim,activation=tf.nn.softplus,name='enc_variance',trainable=isTrainable,reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev));
        epsilon_val = epsilon_distribution(z_dim).sample(tf.shape(X)[0]);
        z_sample = tf.add(z_mean,tf.multiply(z_variance,epsilon_val));

        dist = tfd.MultivariateNormalDiag(z_mean,z_variance);
        return dist,z_sample;

def decoder(z_sample,isTrainable=True,reuse=False,name='decoder'):
    with tf.variable_scope(name) as scope:  
        #decoder_activations = {};
        if reuse:
            scope.reuse_variables();

        z_sample = tf.layers.dense(z_sample,4*4*64,activation=None,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_first_layer',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev));
        z_sample = tf.layers.batch_normalization(z_sample,training=isTrainable,reuse=reuse,name='bn_0');
        z_sample = tf.nn.relu(z_sample);
        z_sample = tf.reshape(z_sample,[-1,4,4,64]);
        #7x7x128

        deconv1 = tf.layers.conv2d_transpose(z_sample,kernel_initializer=tf.random_normal_initializer(stddev=stddev),filters=64,kernel_size=[3,3],padding='SAME',activation=None,strides=(2,2),name='dec_deconv1_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv1 = tf.layers.batch_normalization(deconv1,training=isTrainable,reuse=reuse,name='bn_1');
        deconv1 = tf.nn.relu(deconv1,name='relu_deconv_1');
         
        # 14x14x64
        deconv2 = tf.layers.conv2d_transpose(deconv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=32,kernel_size=[3,3],padding='SAME',activation=None,strides=(2,2),name='dec_deconv2_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv2 = tf.layers.batch_normalization(deconv2,training=isTrainable,reuse=reuse,name='bn_2');
        deconv2 = tf.nn.relu(deconv2,name='relu_deconv_2');
        
        #28x28x32
        deconv3 = tf.layers.conv2d_transpose(deconv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=1,kernel_size=[3,3],padding='SAME',activation=None,strides=(2,2),name='dec_deconv3_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv3 = tf.layers.batch_normalization(deconv3,training=isTrainable,reuse=reuse,name='bn_3');
        deconv3 = tf.nn.relu(deconv3,name='relu_deconv_3');
        
        deconv_3_reshaped = tf.layers.flatten(deconv3);

        #28x28x1
        final_op = tf.layers.dense(deconv_3_reshaped,784,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=tf.nn.sigmoid,name='dec_final_layer',trainable=isTrainable,reuse=reuse);
        
        final_op = tf.reshape(final_op,[-1,784]);
        return final_op;

posterior_dist,z_sample = encoder(X);
X_reconstructed = decoder(z_sample);

_,z_sample_test = encoder(X,reuse=True);
test_reconstruction = decoder(z_sample_test,reuse=True);
reconstruction_loss = tf.reduce_mean(tf.pow(X - X_reconstructed,2));

prior_dist = prior_z(z_dim);
kl_weight = 0.0005;#1/(z_dim*batch_size);
KL_loss = tf.reduce_mean(tfd.kl_divergence(posterior_dist,prior_dist));
generated_sample = prior_dist.sample(batch_size);

loss = kl_weight*KL_loss + reconstruction_loss;

enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='encoder');
dec_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='decoder');

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate);
    gradsVars = optimizer.compute_gradients(loss,enc_params+dec_params);
    train_optimizer = optimizer.apply_gradients(gradsVars);

for g,v in gradsVars:  
    tf.summary.histogram(v.name,v)
    tf.summary.histogram(v.name+str('grad'),g)  

tf.summary.scalar("reconstruction_loss",reconstruction_loss);
tf.summary.scalar("Weighted KL_loss",kl_weight*KL_loss);

merged_all = tf.summary.merge_all();
log_directory = 'Plain-VAE-dir';
model_directory='Plain-VAE-model_dir';

if not os.path.exists(log_directory):
    os.makedirs(log_directory);
if not os.path.exists(model_directory):
    os.makedirs(model_directory);

if not os.path.exists('op-real'):
    os.makedirs('op-real');

if not os.path.exists('op-gen'):
    os.makedirs('op-gen');

if not os.path.exists('op-recons'):
    os.makedirs('op-recons');

def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        ###########################
        #DATA READING
        ###########################
       
        n_batches = mnist.train.num_examples/batch_size;
        n_batches = int(n_batches);
        #n_batches = 50;
        print('n_batches : ',n_batches,' when batch_size : ',batch_size);
        temp_batch = 1; #for plotting
        #for tensorboard
        saver = tf.train.Saver();
        
        writer = tf.summary.FileWriter(log_directory,sess.graph);
        
        train_list = tf.trainable_variables();

        print('----------------TRAINABLE_VARIABLES----------------');
        for it in train_list:
            print(it.name+"\t");

        print('---------------------------------------------------');
        
        for epoch in range(n_epoch):
            epoch_loss = 0;
            epoch_KL_loss = 0;
            epoch_reconstruction_loss = 0;
            for batch in range(n_batches):
                X_batch,_ = mnist.train.next_batch(batch_size);
                _,batch_cost,merged,batch_KL_loss,batch_reconstruction_loss = sess.run([train_optimizer,loss,merged_all,KL_loss,reconstruction_loss],feed_dict={X:X_batch,epoch_number:epoch});
                epoch_loss += batch_cost;
                epoch_KL_loss += batch_KL_loss;
                epoch_reconstruction_loss += batch_reconstruction_loss;
                #writer.add_summary(merged,epoch*n_batches+batch);
            print('At epoch #',epoch,' loss is ',batch_cost ,' where recons loss : ',batch_reconstruction_loss,' and KL_loss : ',batch_KL_loss);
            if(epoch % 5) == 0:
                save_path = saver.save(sess, model_directory+'/model_'+str(epoch));
                print("At epoch #",epoch," Model is saved at path: ",save_path);
                
                n = 5;
                
                reconstructed = np.empty((28*n,28*n));
                original = np.empty((28*n,28*n));
                generated = np.empty((28*n,28*n));
                
                
                for i in range(n):
                    
                    batch_X,_ = mnist.test.next_batch(n);
                    recons = sess.run(test_reconstruction,feed_dict={X:batch_X});
                    #print ('recons : ',recons.shape);
                    recons = np.reshape(recons,[-1,784]);
                    #print ('recons : ',recons.shape);

                    sample = tf.random_normal([n,z_dim]);
                    generation = sess.run(test_reconstruction,feed_dict={z_sample_test:sample.eval()});

                    for j in range(n):
                            original[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_X[j].reshape([28, 28]);

                    for j in range(n):
                        reconstructed[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons[j].reshape([28, 28]);

                    for j in range(n):
                            generated[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = generation[j].reshape([28, 28]);

                
                print("Generated Images");
                plt.figure(figsize=(n,n));
                plt.imshow(generated, origin="upper", cmap="gray");
                plt.title('Epoch '+str(epoch));
                plt.savefig('op-gen/gen-img-'+str(epoch)+'.png');
                plt.close();

                print("Original Images");
                plt.figure(figsize=(n, n));
                plt.imshow(original, origin="upper", cmap="gray");
                plt.title('Epoch '+str(epoch));
                plt.savefig('op-real/original_new_vae-'+str(epoch)+'.png');
                plt.close();

                print("Reconstructed Images");
                plt.figure(figsize=(n, n));
                plt.imshow(reconstructed, origin="upper", cmap="gray");
                plt.title('Epoch '+str(epoch));
                plt.savefig('op-recons/reconstructed_new_vae-'+str(epoch)+'.png');
                plt.close();
        print('Optimization Done !!');
        
train();