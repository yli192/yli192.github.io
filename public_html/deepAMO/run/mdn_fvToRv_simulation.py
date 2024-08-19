import sys,os
sys.path.append('/netscratch/garyli2/cnn_ho')
from keras.layers import Dense, Activation, Input, Lambda, BatchNormalization, Dropout,concatenate
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from utils.mdn import *
from math import floor, ceil
from keras.models import Model,load_model
import shutil
import scipy.stats as ss

def generate(output, testSize, numComponents=24, outputDim=1, M=1):
    out_pi = output[:,:numComponents]
    out_sigma = output[:,numComponents:2*numComponents]
    out_mu = output[:,2*numComponents:]
    out_mu = np.reshape(out_mu, [-1, numComponents, outputDim])
    out_mu = np.transpose(out_mu, [1,0,2])
# use softmax to normalize pi into prob distribution
    max_pi = np.amax(out_pi, 1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = np.exp(out_pi)
    normalize_pi = 1 / (np.sum(out_pi, 1, keepdims=True))
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    # out_sigma = np.exp(out_sigma)
    result = np.random.rand(testSize, M, outputDim)
    rn = np.random.randn(testSize, M)

    mu = 0
    std = 0
    idx = 0
    for j in range(0, M):
        for i in range(0, testSize):
            for d in range(0, outputDim):
                #print(out_pi[i])
                #print(out_pi[0])
                idx = np.random.choice(numComponents, 1, p=out_pi[i])  ##np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0]) #array([3, 3, 0])
                mu = out_mu[idx,i,d]
                std = out_sigma[i, idx]
                #std = 0.2
                #print(out_sigma)
                #print(std)
                result[i, j, d] = mu + rn[i, j]*std
                #result[i, j, d] = mu
    return result


def form_roc_distribution(label_all,pred_all):
    defect_present_distribution = []
    defect_absent_distribution = []
    for i in range(len(label_all)):
        if label_all[i] == 1:
            defect_present_distribution.append(pred_all[i])
        else:
            defect_absent_distribution.append(pred_all[i])
    return defect_present_distribution, defect_absent_distribution

def plot_pdf(distribution, legend_name, ax):
    ax.hist(distribution,50)
    ax.set_title(legend_name, fontsize=14)
    ax.set_xlim([-15, 15])
    #ax.set_ylabel('Counts', fontsize=8)
    #ax.set_xlabel('Observer Rating Value', fontsize=8)
    #ax.legend([legend_name])


def dense_block(units, dropout=0.2, activation='relu', name='fc1'):
    def layer_wrapper(inp):
        x = Dense(units, name=name)(inp)
        x = BatchNormalization(name='{}_bn'.format(name))(x)
        x = Activation(activation, name='{}_act'.format(name))(x)
        x = Dropout(dropout, name='{}_dropout'.format(name))(x)
        return x

    return layer_wrapper

def sixDim2OneDim():
    sampleSize=100 #1000
    numComponents=2
    num_epochs=50000
    size_dense = 128
    outputDim=1
    train_fraction = 0.5
    num_repeated_reading_per_image = 2
    total_num_images = sampleSize * num_repeated_reading_per_image * 3
    input_feature_vectors = []
    output_rating_values = []
    feature_vector_length = 6
    # generate a feature vector of class 1; say two of the 6 features is a small number
    large_feature_num = 3
    large_feature_num_t2 = 2
    large_feature_num_t3 = 1
    output_path = ('/netscratch/garyli2/cnn_ho/experiments/outputs/'
                   +'final_Class1_ValidThreeTypeFV_WSeparatedMeans_3n3PriorLayers_RatingsWithStd_WConfinedSigma_mdnSimuExp_sampleSize_'+str(sampleSize)+'numRepeats_'+str(num_repeated_reading_per_image)+'numComponents_'+str(numComponents)+'sizeDense_'+str(size_dense)+'numEpochs_'+str(num_epochs)+'/')
    if output_path is not None and not os.path.exists(output_path):
        os.makedirs(output_path)

    this_file = 'mdn_fvToRv_simulation.py'
    plot_file = 'plot_mdn_train_curves.py'
    shutil.copy(plot_file, output_path)
    shutil.copy(this_file, output_path)


    for m in range(sampleSize):
        fv_t1 = [] #all 3 segmentation sums are large
        fv_t2 = [] # only 2/3 segmentation sums are large
        fv_t3 = []
        for i in range(large_feature_num):
            large_feature = np.random.normal(25, scale=5)
            fv_t1.append(large_feature)
        #use this as the mean and sample from it
        st_t1_mean = fv_t1[1]*fv_t1[2]/100.0
        print(st_t1_mean,st_t1_mean/3.0)
        st_t1 = np.random.normal(st_t1_mean, scale=st_t1_mean/3.0)
        ct_t1_mean = fv_t1[0]*fv_t1[2]/100.0
        ct_t1 = np.random.normal(ct_t1_mean, scale=ct_t1_mean/3.0)
        cs_t1_mean = fv_t1[0]*fv_t1[1]/100.0
        cs_t1 = np.random.normal(cs_t1_mean, scale=cs_t1_mean/3.0)

        fv_t1.append(st_t1)
        fv_t1.append(ct_t1)
        fv_t1.append(cs_t1)
        print("fv_type1:",fv_t1)
        for n in range(num_repeated_reading_per_image):
            input_feature_vectors.append(fv_t1)

        #one of the segmentation sum is small
        for i in range(large_feature_num_t2):
            large_feature = np.random.normal(25, scale=5)
            fv_t2.append(large_feature)
        for j in range(large_feature_num-large_feature_num_t2):
            small_feature = np.random.normal(5, scale=1)
            fv_t2.append(small_feature)
        np.random.shuffle(fv_t2)
        # use this as the mean and sample from it
        st_t2_mean = fv_t2[1] * fv_t2[2] / 100.0
        st_t2 = np.random.normal(st_t2_mean, scale=st_t2_mean / 3.0)
        ct_t2_mean = fv_t2[0] * fv_t2[2] / 100.0
        ct_t2 = np.random.normal(ct_t2_mean, scale=ct_t2_mean / 3.0)
        cs_t2_mean = fv_t2[0] * fv_t2[1] / 100.0
        cs_t2 = np.random.normal(cs_t2_mean, scale=cs_t2_mean / 3.0)
        fv_t2.append(st_t2)
        fv_t2.append(ct_t2)
        fv_t2.append(cs_t2)
        print("fv_type2:",fv_t2)

        for n in range(num_repeated_reading_per_image):
            input_feature_vectors.append(fv_t2)

        # two of the segmentation sum is small
        for i in range(large_feature_num_t3):
            large_feature = np.random.normal(25, scale=5)
            fv_t3.append(large_feature)
        for j in range(large_feature_num - large_feature_num_t3):
            small_feature = np.random.normal(5, scale=1)
            fv_t3.append(small_feature)
        np.random.shuffle(fv_t3)
        # use this as the mean and sample from it
        st_t3_mean = fv_t3[1] * fv_t3[2] / 100.0
        st_t3 = np.random.normal(st_t3_mean, scale=st_t3_mean / 3.0)
        ct_t3_mean = fv_t3[0] * fv_t3[2] / 100.0
        ct_t3 = np.random.normal(ct_t3_mean, scale=ct_t3_mean / 3.0)
        cs_t3_mean = fv_t3[0] * fv_t3[1] / 100.0
        cs_t3 = np.random.normal(cs_t3_mean, scale=cs_t3_mean / 3.0)
        fv_t3.append(st_t3)
        fv_t3.append(ct_t3)
        fv_t3.append(cs_t3)
        print("fv_type3:", fv_t3)

        for n in range(num_repeated_reading_per_image):
            input_feature_vectors.append(fv_t3)


    # genearte rating values for this feature vector, say one vector has 6 different rating values (applicable for both inter- and intra- varibility simulation)
        n = num_repeated_reading_per_image # number of times the same image is rated
        #np.random.seed(0x5eed)
        # Parameters of the mixture components
        # norm_params = np.array([[-10, 0.1],
        #                         [-8, 0.3],
        #                         [-5, 1.3],
        #                         [1, 0.3],
        #                         [5, 0.2],
        #                         [10, 0.1]
        #                         ])

        norm_params = np.array([[7, 1.2],
                                [10, 0.2]
                                ])
        norm_params_t2 = np.array([[2, 1.2],
                                [4, 1.2]
                                ])
        norm_params_t3 = np.array([[-3, 0.2]
                                   ])
        n_components = norm_params.shape[0]
        n_components_t2 = norm_params_t2.shape[0]
        n_components_t3 = norm_params_t3.shape[0]

        # Weight of each component, in this case all of them are 1/3
        weights = np.ones(n_components, dtype=np.float64) / 2.0
        weights_t2 = np.ones(n_components_t2, dtype=np.float64) / 2.0
        weights_t3 = np.ones(n_components_t3, dtype=np.float64) / 1.0
        # A stream of indices from which to choose the component
        mixture_idx = np.random.choice(len(weights), size=n, replace=True, p=weights)
        mixture_idx_t2 = np.random.choice(len(weights_t2), size=n, replace=True, p=weights_t2)
        mixture_idx_t3 = np.random.choice(len(weights_t3), size=n, replace=True, p=weights_t3)

        # y is the mixture sample
        output_rating_values_per_fv = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                           dtype=np.float64)
        output_rating_values_per_fv_t2 = np.fromiter((ss.norm.rvs(*(norm_params_t2[i])) for i in mixture_idx_t2),
                                                  dtype=np.float64)
        output_rating_values_per_fv_t3 = np.fromiter((ss.norm.rvs(*(norm_params_t3[i])) for i in mixture_idx_t3),
                                                     dtype=np.float64)
        #print("output_rating_values_per_fv",np.asarray(output_rating_values_per_fv))
        #output_rating_values = np.concatenate((np.asarray(output_rating_values),np.asarray(output_rating_values_per_fv),np.asarray(output_rating_values_per_fv_t2)),axis=0)
        #output_rating_values = np.concatenate((np.asarray(output_rating_values),np.asarray(output_rating_values_per_fv)),axis=0)
        output_rating_values = np.concatenate((np.asarray(output_rating_values),np.asarray(output_rating_values_per_fv),np.asarray(output_rating_values_per_fv_t2),np.asarray(output_rating_values_per_fv_t3)),axis=0)


    print(np.asarray(input_feature_vectors))
    print(np.asarray(input_feature_vectors).shape)
    print(output_rating_values)
    x_data = np.asarray(input_feature_vectors)[:int(floor(train_fraction*total_num_images)),:]
    y_data = np.asarray(output_rating_values)[:int(floor(train_fraction*total_num_images))]
    # print(y)
    # # Theoretical PDF plotting -- generate the x and y plotting positions
    # xs = np.linspace(y.min(), y.max(), 200)
    # ys = np.zeros_like(xs)
    #
    # for (l, s), w in zip(norm_params, weights):
    #     ys += ss.norm.pdf(xs, loc=l, scale=s) * w
    #
    # plt.plot(xs, ys)
    # plt.hist(y, normed=True, bins="fd")
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.show()

    # generate a feature vector of class 0
    # x_data = np.float32(np.random.uniform(-2.5, 2.5, (1, sampleSize))).T
    # r_data = np.float32(np.random.normal(size=(sampleSize,1)))
    # y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)
    # #invert training data
    # temp_data = x_data
    # x_data = y_data
    # y_data = temp_data
    #
    # print(y_data.shape)

    input_shape = (6,)

    input = Input(input_shape)
    x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc1')(input)
    x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc2')(x)
    x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc3')(x)
    # x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc4')(x)
    # x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc5')(x)
    # x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc6')(x)
    # x = MixtureDensity(outputDim,numComponents)(x)

    priors_x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc5')(x)
    priors_x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc6')(priors_x)
    priors_x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc7')(priors_x)
    #priors_x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc8')(priors_x)

    priors = Dense(numComponents, activation='softmax', kernel_initializer='random_normal')(priors_x)
    #priors = Dense(numComponents, activation='softmax', kernel_initializer='random_normal')(x)
    sigmas = Dense(numComponents, kernel_initializer='random_normal')(x)
    mus = Dense(numComponents, kernel_initializer='random_normal')(x)
    sigmas = Lambda(lambda x: 1.2 * K.sigmoid(x))(sigmas)
    mus = Lambda(lambda x: 10 * K.tanh(x))(mus)
    # sigmas = Lambda(elu_plus_one_plus_epsilon)(sigmas)
    # sigmas = Lambda(lambda x: 0.5 * K.sigmoid(x))(sigmas)
    # sigmas = Lambda(lambda x: 0.3 * x)(sigmas)



    final = concatenate([priors, sigmas, mus])

    model = Model(inputs=input, outputs=final)


    print(model.summary())
    model.compile(optimizer = Adam(lr=0.0001), loss=get_mixture_loss_func(outputDim,numComponents))


    x_test = np.asarray(input_feature_vectors)[int(floor(train_fraction * total_num_images)):,:]
    y_test = np.asarray(output_rating_values)[int(floor(train_fraction * total_num_images)):]
    print("x_test shape", x_test.shape)
    print("y_test shape", y_test.shape)
    print("output_rating_values shape", output_rating_values.shape)
    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(output_path+'model.h5',
                                 monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    #reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.9, patience=100, verbose=1, mode='auto',
     #                                           epsilon=0.0001, cooldown=0, min_lr=0)
    logger = keras.callbacks.CSVLogger(output_path+'training.log',
                                       separator=',', append=False)

    callacks_list = [checkpoint,logger]
    model.fit(x_data, y_data, validation_data=(x_test, y_test),
     batch_size=y_data.size, epochs=num_epochs,callbacks=callacks_list, verbose=1)

    custom_objects = {
        'inner': lambda y_true, y_pre: y_pre,
        'tf': tf,
        'keras': keras,
        'get_mixture_loss_func': get_mixture_loss_func,
        "mixture_loss": lambda y_true, y_pre: y_pre,
        "mdn_loss_func": lambda y_true, y_pre: y_pre
    }

    loaded_model = load_model(output_path+'model.h5',custom_objects)
    print('trained_model_at_the_best_valLoss loaded')


    #do N predictions per test image
    y_pred_all = np.empty(0, dtype=int)
    y_pred_all_train = np.empty(0, dtype=int)
    for s in range(1):
        y_pred = generate(loaded_model.predict(x_test), y_test.size,numComponents)
        y_pred_train = generate(loaded_model.predict(x_data), y_data.size, numComponents)
        y_pred = np.squeeze(y_pred)
        y_pred_train = np.squeeze(y_pred_train)
        y_pred_all = np.concatenate((y_pred,y_pred_all))
        y_pred_all_train = np.concatenate((y_pred_train, y_pred_all_train))
    #y_pred_all = np.asarray(y_pred_all)
    #y_pred_all = np.squeeze(y_pred_all)
    label_simulated_HO = np.ones(y_test.shape)
    label_predicted = np.ones(y_pred_all.shape)
    print("number of test simulated:", y_test.shape)
    print("number of test predicted:", y_pred_all.shape)
    #print(y_test)
    #print(y_pred)
    defect_present_distribution_test, defect_absent_distribution_test = form_roc_distribution(label_simulated_HO, y_test)
    defect_present_distribution_pred, defect_absent_distribution_pred = form_roc_distribution(label_predicted, y_pred_all)
    defect_present_distribution_pred_learned, defect_absent_distribution_pred_learned = form_roc_distribution(label_predicted,
                                                                                              y_pred_all_train)

    defect_present_distribution_train, defect_absent_distribution_train = form_roc_distribution(label_simulated_HO, y_data)


    np.savez_compressed(os.path.join(output_path, 'labels_and_preds_ho.npz'), label_all=label_simulated_HO, pred_all=y_test)
    np.savez_compressed(os.path.join(output_path, 'labels_and_preds_cnn.npz'), label_all=label_predicted, pred_all=y_pred_all)
    np.savez_compressed(os.path.join(output_path, 'labels_and_preds_learning_ho.npz'), label_all=label_simulated_HO,
                        pred_all=y_data)
    np.savez_compressed(os.path.join(output_path, 'labels_and_preds_learning_cnn.npz'), label_all=label_simulated_HO,
                        pred_all=y_pred_all_train)

    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    plot_pdf(defect_present_distribution_pred, 'Pred-test', ax[0])
    plot_pdf(defect_present_distribution_test, 'Test', ax[1])
    plot_pdf(defect_present_distribution_pred_learned, 'Pred-train', ax[2])
    plot_pdf(defect_present_distribution_train, 'Train', ax[3])

    #plt.show()
    fig.savefig(output_path+'sampleSize_'+str(sampleSize)+'numRepeats_'+str(num_repeated_reading_per_image)+'numComponents_'+str(numComponents)+'numEpochs_'+str(num_epochs)+'.png')

sixDim2OneDim()