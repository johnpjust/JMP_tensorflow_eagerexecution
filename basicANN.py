# tensorflow_eagerexecuation

import pandas as pd
import numpy as np
import tensorflow as tf
import functools
import re
from scipy.optimize import fmin_l_bfgs_b
import os
import shutil
import sys
#from matplotlib import pyplot as plt

####################################################################################
def eval_loss_and_grads(x, loss_train, var_list, var_shapes, var_locs):
    ## x: updated variables from scipy optimizer
    ## var_list: list of trainable variables
    ## var_sapes: the shape of each variable to use for updating variables with optimizer values
    ## var_locs:  slicing indecies to use on x prior to reshaping

    ## update variables
    for i in range(len(var_list)):
        var_list[i].assign(np.reshape(x[var_locs[i]:(var_locs[i+1])], var_shapes[i]))

    ## calculate new gradient
    with tf.GradientTape() as tape:
        prediction_loss = loss_train(x=x)

    grad_list = []
    for p in tape.gradient(prediction_loss, var_list):
        grad_list.extend(np.array(tf.reshape(p, [-1])))

    grad_list = [v if v is not None else 0 for v in grad_list]

    return np.float64(prediction_loss), np.float64(grad_list)

class Evaluator(object):

    def __init__(self, loss_train_fun, loss_val, global_step, early_stop_limit, var_list, var_shapes, var_locs):
        self.loss_train_fun = loss_train_fun #func_tools partial function with model, features, and labels already loaded
        self.predLoss_val = loss_val #func_tools partial function with model, features, and labels already loaded
        self.global_step = global_step #tf variable for tracking update steps from scipy optimizer step_callback
        self.predLoss_val_prev = np.float64(loss_val()) + 10.0 #state variable of loss for early stopping
        self.predLoss_val_cntr = 0 #counter to watch for early stopping
        self.early_stop_limit = early_stop_limit #number of cycles of increasing validation loss before stopping
        self.var_shapes = var_shapes #list of shapes of each tf variable
        self.var_list = var_list #list of trainable tf variables
        self.var_locs = var_locs
        self.loss_value = self.loss_train_fun(x=np.float64(0))
        self.grads_values = None

    def loss(self, x):
        # assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x, self.loss_train_fun, self.var_list, self.var_shapes, self.var_locs) #eval_loss_and_grads
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        # assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        # self.loss_value = None
        # self.grad_values = None
        return grad_values

    def step_callback(self, x):
        ## early stopping tracking
        predLoss_val_temp = np.float64(self.predLoss_val())
        if predLoss_val_temp > self.predLoss_val_prev:
            self.predLoss_val_cntr += 1
        else:
            self.predLoss_val_cntr = 0
            self.predLoss_val_prev = predLoss_val_temp

        #write tensorboard variables
        with tf.contrib.summary.always_record_summaries(): #with tf.contrib.summary.record_summaries_every_n_global_steps(100):
            tf.contrib.summary.scalar('loss_train', self.loss_value)
            tf.contrib.summary.scalar('loss_val', predLoss_val_temp)
            tf.contrib.summary.scalar('global_step', self.global_step)

        # print(self.global_step)
        # print(predLoss_val_temp)
        # sys.stdout.flush()

        # increment the global step counter
        self.global_step.assign_add(1)
        #return true to scipy optimizer to stop optimization loop
        if self.predLoss_val_cntr > self.early_stop_limit:
            return True
        else:
            return False

#
# def prediction_classifier(features, label, model):
#     predicted_label = model(features)
#     return tf.losses.softmax_cross_entropy(label, predicted_label)
#
# def prediction_loss_L2(features, label, model):
#     return tf.reduce_sum(tf.squared_difference(label, model(features)))
#
# def prediction_loss_L1(features, label, model):
#     #uses an abs function approximation
#     loss = tf.subtract(label,model(features))
#     return tf.reduce_sum(tf.square(loss) / tf.sqrt(tf.square(loss) + .01**2), axis=0)



def prediction_classifier(features, label, model, reg, x):
    predicted_label = model(features)
    return tf.add(tf.losses.softmax_cross_entropy(label, predicted_label), tf.multiply(reg, tf.reduce_sum(tf.square(x))))

def prediction_loss_L2(features, label, model, reg, x):
    return tf.add(tf.reduce_sum(tf.squared_difference(label, model(features)), axis=0),tf.multiply(reg, tf.reduce_sum(tf.square(x))))

def prediction_loss_L1(features, label, model, reg, x):
    #uses an abs function approximation
    loss = tf.subtract(label,model(features))
    return tf.add(tf.reduce_sum(tf.square(loss) / tf.sqrt(tf.square(loss) + .01**2), axis=0) , tf.multiply(reg, tf.reduce_sum(tf.square(x))))


"""
All operations need to be in 64 bit floats for use with scipy optimizers and also to
ensure full convergence with soft label classification

eager execution is enabled

Calling program must pass in the following:
df2:  features for predicting with
df2_gt:  ground truth to fit weights to
df2_groups:  train/validation split will be sorted and first index is used for training.
            If < two unique groups, 67%/33% random split will be used for training/validation

reg_in:  L2 regulation scale for weights...a value of zero results in no penalty on weights
early_stop_limit:  number of times in a row that validation loss can increase before stopping

n_hidden_units_in:  a zero will result in no hidden layer
loss_fun:  options are "Classifier", "L1", "L2"
activation_in:  options are 'elu', "lrelu", "relu", "tanh", "linear", "selu"
"""

## program testing data
df = pd.read_excel(r'') ##some general dataframe
df2 = pd.DataFrame(df.iloc[:,[*[i for i in range(145,458)]]]) ##some subset of features of main dataframe

#df2_gt = pd.DataFrame(df.iloc[:, [*[i for i in range(474, 477)]]] / 100)
df2_gt = pd.DataFrame(df.iloc[:,[-9]])  ##ground truth subset of main dataframe
df2_groups = pd.DataFrame(df.iloc[:, -1]) ##grouping for train/validation

if not (df2_groups.iloc[:,0].unique().shape[0] >= 2):
    df2_groups = pd.DataFrame(np.ceil(np.maximum(0,np.random.uniform(0,1,(df2.shape[0],1))-.667)))

a = ~pd.isnull(df2).any(1) & ~pd.isnull(df2_gt).any(1) & ~pd.isnull(df2_groups).any(1)

df2 = df2[a]
df2_gt = df2_gt[a]
df2_groups = df2_groups[a]

a = np.sort(df2_groups.iloc[:, 0].unique())
dfTrain = df2.loc[df2_groups.iloc[:,].values[:,0] == a[0]]
dfVal = df2.loc[df2_groups.iloc[:, ].values[:, 0] == a[1]]
dfTrain_gt = df2_gt.loc[df2_groups.iloc[:,].values[:,0] == a[0]]
dfVal_gt = df2_gt.loc[df2_groups.iloc[:, ].values[:, 0] == a[1]]

train_means = dfTrain.mean(axis=0)
dfTrain = (dfTrain - train_means)
dfVal = (dfVal - train_means)
train_sds = dfTrain.std(axis=0)

dfTrain = dfTrain / train_sds
dfVal = dfVal / train_sds

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config = config)

x_train_in = dfTrain.values.astype('float64')
y_train_in = dfTrain_gt.values.astype('float64')

x_val_in = dfVal.values.astype('float64')
y_val_in = dfVal_gt.values.astype('float64')

reg_in = np.float64(0)
early_stop_limit = 200

n_hidden_units_in = 0
loss_fun = "L1"#, "L2", "L1", "Classifier"
activation_in = 'elu' #options = "lrelu", "relu", "tanh", "linear", "selu", "elu"
feature_shape = (None, x_train_in.shape[1])
gt_shape = y_train_in.shape[1]

#error checking
assert x_train_in.ndim == 2
assert y_train_in.ndim == 2

if loss_fun == "Classifier":
    assert y_train_in.shape[1] > 1

#initialization based on activation function
if "selu" in activation_in:
    initializer = "lecun_normal"
elif "lu" in activation_in:
    initializer = "he_normal"
else:
    initializer = "glorot_uniform" #default

##note:  regularization is implemented in a more general way with the loss functions instead of through Keras
with tf.device("CPU:0"): ##L-BFGS is implemented on CPU with numpy so no use in storing model on GPU
    model = tf.keras.Sequential()
    if n_hidden_units_in > 0:
        model.add(tf.keras.layers.Dense(units = n_hidden_units_in, activation = activation_in, kernel_initializer=initializer, input_shape=feature_shape, dtype=tf.float64))
        model.add(tf.keras.layers.Dense(gt_shape, kernel_initializer=initializer, dtype=tf.float64)) #kernel_regularizer=tf.keras.regularizers.l2(reg_in)
    else:
        model.add(tf.keras.layers.Dense(units=gt_shape, input_shape=feature_shape, dtype=tf.float64))


if loss_fun == "Classifier":
    loss = prediction_classifier
elif loss_fun == "L1":
    loss = prediction_loss_L1
elif loss_fun == "L2":
    loss = prediction_loss_L2

loss_train = functools.partial(loss, features=x_train_in, label=y_train_in, model=model, reg = reg_in)
loss_val = functools.partial(loss, features=x_val_in, label=y_val_in, model=model, reg=np.float64(0), x=np.float64(0))

## get size and shape of trainable variables (reshaping required for input/output from scipy optimization)
## as well as the tf trainable variable list for updating
with tf.GradientTape() as tape:
    prediction_loss = loss_train(x=np.float64(0))
    var_list = tape.watched_variables()

var_shapes = []
var_locs = [0]
for v in var_list:
    var_shapes.append(np.array(tf.shape(v)))
    var_locs.append(np.prod(np.array(tf.shape(v))))
var_locs = np.cumsum(var_locs)

## setup tensorboard for monitoring training
basepath = os.path.join(os.path.expanduser("~/Desktop"), 'JMP_to_TF_basicANN')
try:
    os.makedirs(basepath)
except Exception as e:
        print(e)
        sys.stdout.flush()

TBbasePath = os.path.join(basepath, 'tensorboard')
CPbasePath = os.path.join(basepath, 'checkpoints')

re_runNum = re.compile(r'\d+$')
try:
    nextRun = max([int(re_runNum.search(f.name)[0]) for f in os.scandir(TBbasePath) if f.is_dir()]) + 1
except:
    nextRun = 0

TWpath = TBbasePath + r"/run_" + str(nextRun)
CPpath = CPbasePath + r"/run_" + str(nextRun)
# remove summary folder if already exists
# noinspection PyBroadException
try:
    shutil.rmtree(TWpath)
except:
    pass

global_step = tf.train.get_or_create_global_step()
global_step.assign(0)
writer = tf.contrib.summary.create_file_writer(TWpath)
writer.set_as_default()

x_init = []
for v in var_list:
    x_init.extend(np.array(tf.reshape(v, [-1])))

evaluator = Evaluator(loss_train, loss_val, global_step, early_stop_limit, var_list, var_shapes, var_locs)

x, min_val, info = fmin_l_bfgs_b(func=evaluator.loss, x0=np.float64(x_init),
                                 fprime=evaluator.grads, maxfun=500, callback=evaluator.step_callback)

# np.savetxt(CPpath + r'/checkpoint.txt', np.array(var_list))
#
# saver = tf.contrib.summary.Saver(var_list)
# saver.save(CPpath + r'/checkpoint.ckpt', global_step = global_step)

#### tensorboard -->  at command prompt or python terinal:   --logdir=C:\Users\justjo\PycharmProjects\basicANN_eagerExecution\tensorboard
