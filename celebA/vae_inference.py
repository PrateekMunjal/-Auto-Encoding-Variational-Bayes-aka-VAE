from scipy import misc
import matplotlib.pyplot as plt 
import config
import logging
import PIL,os,sys
import numpy as np
from PIL import Image
import tensorflow as tf
import random

#opts = config.config_celebA_toy
opts = config.config_celebA
crop_style = opts['celebA_crop'];
celeb_source = opts['dataset_dir'];

tf.set_random_seed(0);
random.seed(0);

learning_rate = 0.0002;

encoder_role = tf.placeholder(tf.float32,[]);
discriminator_role = tf.placeholder(tf.float32,[]);

batch_size = opts['batch_size'];
n_epoch = opts['num_epoch'];
z_dim = opts['z_dim'];

#model_params
img_height = opts['img_height'];
img_width = opts['img_width'];
num_channels = opts['num_channels'];
n_inputs = 64*64; #as images are of 64 x 64 dimension
n_outputs = 10;


tfd = tf.contrib.distributions

X = tf.placeholder(tf.float32,[None,img_height,img_width,num_channels]);
epoch_number = tf.placeholder(tf.float32,[]);


#Used to initialize kernel weights
stddev = 0.02;#99999;


#Given absolute image path, returns image in numpy
def read_image(path): 
    logging.debug('In read_image function for path : ',path);
    img = misc.imread(path);
    return img;

#Returns meta data about image i.e height,width,num_channels
def image_meta(image):
    logging.debug('In image_meta function');
    return image.shape[0],image.shape[1],image.shape[2];
    #return image.size[0],image.size[1],opts['num_channels'];

def denormalize_image(image):
    image /= 2. #rescale value from [-1,1] to [-0.5,0.5]
    image = image + 0.5 #rescale value from [-0.5,0.5] to [0,1]
    return image;

def plot_denormalized_image(image,title):
    image = denormalize_image(image);
    plt.figure();
    plt.title(title);
    plt.imshow(image);

def plot_image(image,title):
    plt.figure();
    plt.axis('off');
    plt.title(title);
    plt.imshow(image);
    
    #plt.show();
    #plt.close();

def crop(im):
    width = 178
    height = 218
    new_width = 140
    new_height = 140
    if crop_style == 'closecrop':
        # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height)/2
        im = im.crop((left, top, right, bottom))
        im = im.resize((64, 64), PIL.Image.ANTIALIAS)
    elif self.crop_style == 'resizecrop':
        # This method was used in ALI, AGE, ...
        im = im.resize((64, 78), PIL.Image.ANTIALIAS)
        im = im.crop((0, 7, 64, 64 + 7))
    else:
        raise Exception('Unknown crop style specified')
    return np.array(im).reshape(64, 64, 3) / 255.

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def normalize_image(image):
    normalized_image = image - 0.5;
    normalized_image *= 2;
    # normalized_image = (image - np.min(image))/(np.max(image)-np.min(image));
    # normalized_image = 2*normalized_image;
    # normalized_image -= 1;
    return normalized_image;

def get_random_batch(file_iter,batch_size = 3,):
    random_file_iter = np.random.choice(file_iter,batch_size,replace=False);
    #print("Random_file_iter");
    #print(random_file_iter);
    #sys.exit(0);
    X = np.zeros([len(random_file_iter),opts['img_height'],opts['img_width'],opts['num_channels']]);
    #print(X.shape);
    index = -1;
    for f in random_file_iter:
        index += 1;
        f = f + '.jpg';
        curr_img = Image.open(f);
        curr_img = crop(curr_img);
        #print(np.min(curr_img));
        curr_img = normalize_image(curr_img);
        #print(np.min(curr_img));
        #print (curr_img.shape);
        X[index] = curr_img;
    return X;


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

        conv1 = tf.layers.conv2d(X,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv1_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv1 = tf.layers.batch_normalization(conv1,training=isTrainable,reuse=reuse,name='bn_1');
        conv1 = tf.nn.relu(conv1,name='leaky_relu_conv_1');

        #32x32x64
        conv2 = tf.layers.conv2d(conv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv2_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv2 = tf.layers.batch_normalization(conv2,training=isTrainable,reuse=reuse,name='bn_2');
        conv2 = tf.nn.relu(conv2,name='leaky_relu_conv_2');
        
        #16x16x128
        conv3 = tf.layers.conv2d(conv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv3_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv3 = tf.layers.batch_normalization(conv3,training=isTrainable,reuse=reuse,name='bn_3');
        conv3 = tf.nn.relu(conv3,name='leaky_relu_conv_3');
        
        #8x8x256
        conv4 = tf.layers.conv2d(conv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=512,kernel_size=[5,5],padding='SAME',strides=(1,1),name='enc_conv4_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv4 = tf.layers.batch_normalization(conv4,training=isTrainable,reuse=reuse,name='bn_4');
        conv4 = tf.nn.relu(conv4,name='leaky_relu_conv_4');
        
        #8x8x512
        conv4_flattened = tf.layers.flatten(conv4);
        
        z_mean = tf.layers.dense(conv4_flattened,z_dim,name='enc_mean',trainable=isTrainable,reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev));
        z_variance = tf.layers.dense(conv4_flattened,z_dim,activation=tf.nn.softplus,name='enc_variance',trainable=isTrainable,reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev));
        epsilon_val = epsilon_distribution(z_dim).sample(tf.shape(X)[0]);
        z_sample = tf.add(z_mean,tf.multiply(z_variance,epsilon_val));

        dist = tfd.MultivariateNormalDiag(z_mean,z_variance);
        return dist,z_sample;

def decoder(z_sample,isTrainable=True,reuse=False,name='decoder'):
    with tf.variable_scope(name) as scope:  
        #decoder_activations = {};
        if reuse:
            scope.reuse_variables();

        z_sample = tf.layers.dense(z_sample,8*8*512,activation=None,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_first_layer',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev));
        z_sample = tf.layers.batch_normalization(z_sample,training=isTrainable,reuse=reuse,name='bn_0');
        z_sample = tf.nn.relu(z_sample);
        z_sample = tf.reshape(z_sample,[-1,8,8,512]);
        #8x8x512

        deconv1 = tf.layers.conv2d_transpose(z_sample,kernel_initializer=tf.random_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv1_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv1 = tf.layers.batch_normalization(deconv1,training=isTrainable,reuse=reuse,name='bn_1');
        deconv1 = tf.nn.relu(deconv1,name='relu_deconv_1');
         
        # #16x16x256
        deconv2 = tf.layers.conv2d_transpose(deconv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv2_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv2 = tf.layers.batch_normalization(deconv2,training=isTrainable,reuse=reuse,name='bn_2');
        deconv2 = tf.nn.relu(deconv2,name='relu_deconv_2');
        
        #32x32x128
        deconv3 = tf.layers.conv2d_transpose(deconv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv3_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv3 = tf.layers.batch_normalization(deconv3,training=isTrainable,reuse=reuse,name='bn_3');
        deconv3 = tf.nn.relu(deconv3,name='relu_deconv_3');
        
        #64x64x64 
        deconv4 = tf.layers.conv2d_transpose(deconv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=3,kernel_size=[5,5],padding='SAME',activation=None,strides=(1,1),name='dec_deconv4_layer',trainable=isTrainable,reuse=reuse); # 16x16    
        #deconv4 = tf.layers.dropout(deconv4,rate=keep_prob,training=True);
        deconv4 = tf.nn.tanh(deconv4);
        #64x64x3
        
        deconv_4_reshaped = tf.reshape(deconv4,[-1,img_height,img_width,num_channels]);
        return deconv_4_reshaped;

posterior_dist,z_sample = encoder(X);
X_reconstructed = decoder(z_sample);

_,z_sample_test = encoder(X,reuse=True);
test_reconstruction = decoder(z_sample_test,reuse=True);
reconstruction_loss = tf.reduce_mean(tf.pow(X - X_reconstructed,2));

prior_dist = prior_z(z_dim);
kl_weight = 0.0001;
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

def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        ###########################
        #DATA READING
        ###########################
        mode = 'train';
        train_min_file_num = opts['train_min_filenum'];
        train_max_file_num = opts['train_max_filenum'];
        train_files = range(train_min_file_num, 1+train_max_file_num);
        train_file_iter=[os.path.join(celeb_source, '%s' % str(i).zfill(6)) for i in train_files]

        val_min_file_num = opts['val_min_filenum'];
        val_max_file_num = opts['val_max_filenum'];
        val_files = range(val_min_file_num, 1+val_max_file_num);
        val_file_iter=[os.path.join(celeb_source, '%s' % str(i).zfill(6)) for i in val_files]

        #learning_rate = 0.0001;
        n_batches = 162770/batch_size;#mnist.train.num_examples/batch_size;
        n_batches = int(n_batches);
        #n_batches = 50;
        print('n_batches : ',n_batches,' when batch_size : ',batch_size);
        temp_batch = 1; #for plotting
        #for tensorboard
        saver = tf.train.Saver();
        writer = tf.summary.FileWriter(log_directory,sess.graph);
        iterations = 0;
        for epoch in range(n_epoch):
            for batch in range(n_batches):
                iterations += 1;
                dropout_rate = 0.5;
                X_batch = get_random_batch(train_file_iter,batch_size);
                fd = {X:X_batch};
                _,merged = sess.run([train_optimizer,merged_all],feed_dict=fd);

                if(iterations%20==0):
                    writer.add_summary(merged,iterations);

                if(batch%200 == 0):
                    print('Batch #',batch,' done!');

            if(epoch%2==0):
                num_val_img = 25;
                batch_X = get_random_batch(val_file_iter,num_val_img);
                
                recons = sess.run(test_reconstruction,feed_dict={X:batch_X});
                recons = np.reshape(recons,[-1,64,64,3]);

                n_gen = 25;
                sample = tf.random_normal([n_gen,z_dim]);
                generations = sess.run(test_reconstruction,feed_dict={z_sample_test:sample.eval()});
                generations = np.reshape(generations,[-1,64,64,3]);

                temp_index = -1;
                for s in range(generations.shape[0]):
                    temp_index += 1;
                    generations[temp_index] = denormalize_image(generations[temp_index]);

                temp_index = -1;
                for s in range(batch_X.shape[0]):
                    temp_index += 1;
                    batch_X[temp_index] = denormalize_image(batch_X[temp_index]);

                temp_index = -1;
                for s in range(recons.shape[0]):
                    temp_index += 1;
                    recons[temp_index] = denormalize_image(recons[temp_index]);

                n = 5;
                reconstructed = np.empty((64*n,64*n,3));
                original = np.empty((64*n,64*n,3));
                generated_images = np.empty((64*n,64*n,3));
                for i in range(n):
                    for j in range(n):
                        original[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = batch_X[i*n+j];#.reshape([32, 32,3]);
                        reconstructed[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = recons[i*n+j];
                        generated_images[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = generations[i*n+j];

                print("Original Images");
                plt.figure(figsize=(n, n));
                plt.imshow(original, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/orig-img-'+str(epoch)+'.png');
                plt.close();

                print("Reconstructed Images");
                plt.figure(figsize=(n, n));
                plt.imshow(reconstructed, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/recons-img-'+str(epoch)+'.png');
                plt.close();

                print("Generated Images");
                plt.figure(figsize=(n, n));
                plt.imshow(generated_images, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/gen-img-'+str(epoch)+'.png');
                plt.close();
                
            if(epoch%5==0): 
                save_path = saver.save(sess, model_directory+'/model_'+str(epoch));
                print("At epoch #",epoch," Model is saved at path: ",save_path);


            #writer.add_summary(merged,epoch);#*n_batches + batch);
            print('------------------------------------');
            print('=== Epoch #',epoch,' completed! === learning_rate : ',learning_rate);

def generateImages(model_number=15,n=1):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES);
        saver = tf.train.Saver(var_list=params);

        string = model_directory+'/model_'+str(model_number); 

        try:
            saver.restore(sess, string);
        except:
            print("Previous weights not found of decoder"); 
            sys.exit(0);

        print ("Model loaded successfully");

        n_batches = int(10000.0/n);
        #n_batches = 5;
        print('Total n_batches : ',n_batches);
        start_image_number = 0;
        for batch in range(n_batches):
            #start_image_number = n * batch;
            n_gen = n;
            sample = tf.random_normal([n_gen,z_dim]);
            #generations = np.zeros([n_gen,64,64,3]);
            #k=0;
            # while k<n_gen:
            #     temp_generations = sess.run(test_reconstruction,feed_dict={z_sample:sample[k:k+2000].eval()});
            #     for()
            #     k+ = 2000;
            
            generations = sess.run(test_reconstruction,feed_dict={z_sample_test:sample.eval()});
            generations = np.reshape(generations,[-1,64,64,3]);
            temp_index = -1;
            for s in range(generations.shape[0]):
                temp_index += 1;
                generations[temp_index] = denormalize_image(generations[temp_index]);
            #print('generations : ',generations[0].shape);
            output_directory = opts['vae_generated_crop_dir'];
            if not os.path.exists(output_directory):
                os.makedirs(output_directory);
            for i in range(generations.shape[0]):
                #if i%50==0:
                #    print('Generated image #',i);
                plt.figure(figsize=(0.64, 0.64))
                plt.axis('off');
                plt.imshow(generations[i], origin="upper",interpolation='nearest', cmap="gray",aspect='auto');
                plt.savefig(output_directory+str(start_image_number+1).zfill(6)+'.jpg');
                start_image_number += 1;
                plt.close();
            print("Batch #",batch," done !!");

#train();
generateImages(15,100);