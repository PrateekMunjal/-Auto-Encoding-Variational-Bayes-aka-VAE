config_celebA = {};

config_celebA['dataset_name'] = 'celebA';
config_celebA['dataset_split_name'] = 'train';
config_celebA['dataset_dir'] = '/home/akanksha/Desktop/prateek/git/Adversarial-VAE-/celebA-experiments/img_align_celeba/';#'./f1/'
config_celebA['output_dir'] = 'f1/'; #destination for TF Record files
config_celebA['shuffle_data'] = True; # accepts True or False
config_celebA['dataset_path'] = "./f1/*.jpg";
config_celebA['train_shards'] = 32;
config_celebA['num_readers'] = 2;

config_celebA['batch_size'] = 128;
config_celebA['tfRecord_batch_size'] = 16;

config_celebA['num_clones'] = 1;
config_celebA['clone_on_cpu'] = False;
config_celebA['task'] = 0;
config_celebA['worker_replicas']=1;
config_celebA['num_ps_tasks']=0;
config_celebA['model.name'] = 'adv_vae'
config_celebA['num_epoch'] = 100;#100;
config_celebA['z_dim'] = 128;
config_celebA['img_height'] = 64;
config_celebA['img_width'] = 64;
config_celebA['celebA_crop'] = 'closecrop'
config_celebA['num_channels'] = 3;
config_celebA['train_min_filenum'] = 1;
config_celebA['train_max_filenum'] = 162770;
config_celebA['val_min_filenum'] = 162771;
config_celebA['val_max_filenum'] = 182637;
config_celebA['original_crop_dir'] = './original1-crop/';
config_celebA['generated_crop_dir'] = './generated-crop/';
config_celebA['vae_generated_crop_dir'] = './vae_generated-crop/';
config_celebA['load_model_number'] = 60;

config_celebA_toy = {};

config_celebA_toy['dataset_name'] = 'celebA';
config_celebA_toy['dataset_split_name'] = 'train';
config_celebA_toy['dataset_dir'] = 'f1/';#'./f1/'
config_celebA_toy['output_dir'] = 'f2/'; #destination for TF Record files
config_celebA_toy['shuffle_data'] = True; # accepts True or False
config_celebA_toy['dataset_path'] = "./f1/*.jpg";
config_celebA_toy['train_shards'] = 32;
config_celebA_toy['num_readers'] = 2;

config_celebA_toy['batch_size'] = 128;
config_celebA_toy['tfRecord_batch_size'] = 16;

config_celebA_toy['num_clones'] = 1;
config_celebA_toy['clone_on_cpu'] = False;
config_celebA_toy['task'] = 0;
config_celebA_toy['worker_replicas']=1;
config_celebA_toy['num_ps_tasks']=0;
config_celebA_toy['model.name'] = 'adv_vae'
config_celebA_toy['num_epoch'] = 100;
config_celebA_toy['z_dim'] = 100;
config_celebA_toy['img_height'] = 64;
config_celebA_toy['img_width'] = 64;
config_celebA_toy['num_channels'] = 3;
config_celebA_toy['celebA_crop'] = 'closecrop'
config_celebA_toy['train_min_filenum'] = 1;
config_celebA_toy['train_max_filenum'] = 5;
config_celebA_toy['val_min_filenum'] = 5;
config_celebA_toy['val_max_filenum'] = 20;
config_cifar10 = {};

config_cifar10['dataset_name'] = 'cifar10';
config_cifar10['dataset_split_name'] = 'train';
config_cifar10['dataset_dir'] = 'f1/';#'./f1/'
config_cifar10['output_dir'] = 'f1/'; #destination for TF Record files
config_cifar10['shuffle_data'] = True; # accepts True or False
config_cifar10['dataset_path'] = "./f1/*.jpg";
config_cifar10['train_shards'] = 32;
config_cifar10['num_readers'] = 2;

config_cifar10['batch_size'] = 128;
config_cifar10['tfRecord_batch_size'] = 16;

config_cifar10['num_clones'] = 1;
config_cifar10['clone_on_cpu'] = False;
config_cifar10['task'] = 0;
config_cifar10['worker_replicas']=1;
config_cifar10['num_ps_tasks']=0;
config_cifar10['model.name'] = 'adv_vae'
config_cifar10['num_epoch'] = 50;
config_cifar10['z_dim'] = 100;
config_cifar10['img_height'] = 32;
config_cifar10['img_width'] = 32;
config_cifar10['num_channels'] = 3;

config_mnist = {};

config_mnist['dataset_name'] = 'mnist';
config_mnist['dataset_split_name'] = 'train';
config_mnist['dataset_dir'] = 'f1/';#'./f1/'
config_mnist['output_dir'] = 'f1/'; #destination for TF Record files
config_mnist['shuffle_data'] = True; # accepts True or False
config_mnist['dataset_path'] = "./f1/*.jpg";
config_mnist['train_shards'] = 32;
config_mnist['num_readers'] = 2;

config_mnist['batch_size'] = 128;
config_mnist['tfRecord_batch_size'] = 16;

config_mnist['num_clones'] = 1;
config_mnist['clone_on_cpu'] = False;
config_mnist['task'] = 0;
config_mnist['worker_replicas']=1;
config_mnist['num_ps_tasks']=0;
config_mnist['model.name'] = 'adv_vae'
config_mnist['num_epoch'] = 50;
config_mnist['z_dim'] = 100;
config_mnist['img_height'] = 28;
config_mnist['img_width'] = 28;
config_mnist['num_channels'] = 1;
