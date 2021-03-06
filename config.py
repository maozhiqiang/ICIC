#!/usr/bin/env python3

class config_train(object):
    mode = 'gan-train'
    num_st_epochs = 100
    num_epochs = 256
    batch_size = 30
    ema_decay = 0.999
    G_learning_rate = 2e-4
    D_learning_rate = 2e-4
    lr_decay_rate = 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    kernel_size = 3
    diagnostic_steps = 256

    # WGAN
    gradient_penalty = True
    lambda_gp = 10
    weight_clipping = False
    max_c = 1e-2
    n_critic_iterations = 20

    # Compression
    lambda_X = 12
    lambda_i = 6
    channel_bottleneck = 8
    sample_noise = False#True
    use_vanilla_GAN = False
    use_feature_matching_loss = False#True
    upsample_dim = 256
    multiscale = False#True
    feature_matching_weight = 10
    use_conditional_GAN = False
    use_pixel_shuffle = False
    
class config_test(object):
    mode = 'gan-test'
    num_epochs = 512
    batch_size = 1
    ema_decay = 0.999
    G_learning_rate = 2e-4
    D_learning_rate = 2e-4
    lr_decay_rate = 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    kernel_size = 3
    diagnostic_steps = 256

    # WGAN
    gradient_penalty = True
    lambda_gp = 10
    weight_clipping = False
    max_c = 1e-2
    n_critic_iterations = 5

   # Compression
    lambda_X = 12
    channel_bottleneck = 8
    sample_noise = True
    use_vanilla_GAN = False
    use_feature_matching_loss = True
    upsample_dim = 256
    multiscale = True
    feature_matching_weight = 10
    use_conditional_GAN = False

class directories(object):
    #train = 'data/cityscapes_paths_train.h5'
    #test = 'data/cityscapes_paths_test.h5'
    #val = 'data/cityscapes_paths_val.h5'
    train = 'data/CLIC_train.h5'
    test = 'data/CLIC_test.h5'
    val = 'data/CLIC_val.h5'
    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'
    checkpoints_best = 'checkpoints/best'
    samples = 'samples/CLIC'

