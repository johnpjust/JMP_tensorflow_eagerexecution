import pandas as pd
import numpy as np
import tensorflow as tf
import functools
import re
from scipy.optimize import fmin_l_bfgs_b
import os
import shutil
import sys
from scipy import stats
#from matplotlib import pyplot as plt

class Groupby:
    def __init__(self, keys):
        _, self.keys_as_int = np.unique(keys, return_inverse=True)
        self.n_keys = max(self.keys_as_int)
        self.keys_as_int_unique = list(dict.fromkeys(self.keys_as_int))
        self.set_indices()

    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys + 1)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
        #self.keys_as_int_unique = np.argsort([x[0] for x in self.indices])

    def apply(self, function, vector, broadcast):
        if broadcast:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = function(vector[idx])
        else:
            result = np.zeros(self.n_keys+1)
            for k in range(len(self.indices)):
                result[k] = function(vector[self.indices[self.keys_as_int_unique[k]]])
        return result

    def split(self, exists, vector, perc = 0.667):
        if not exists:
            if not (perc > 0 and perc < 1): perc = 0.667
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = np.ceil(np.maximum(0, np.random.uniform(0, 1) - perc))
        else:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = stats.mode(vector[idx])[0][0]
        return result


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
        # grad_values = np.copy(self.grad_values)
        # self.loss_value = None
        # self.grad_values = None
        return self.grad_values

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
            tf.contrib.summary.scalar('predLoss_val_cntr', self.predLoss_val_cntr)

        # increment the global step counter
        self.global_step.assign_add(1)
        #return true to scipy optimizer to stop optimization loop
        if self.predLoss_val_cntr > self.early_stop_limit:
            print("stop")
            sys.stdout.flush()
            return True
        else:
            return False

#
def prediction_classifier(features, label, model, reg, x, grouper=None):
    predicted_label = model(features)
    return tf.add(tf.losses.softmax_cross_entropy(label, predicted_label), tf.multiply(reg, tf.reduce_sum(tf.square(x))))

def prediction_loss_L2(features, label, model, reg, x, grouper=None, weights=1, var_reg=None):
    predicted_label = model(features)
    if grouper:  predicted_label = grouper.apply(np.mean, predicted_label, broadcast=False)
    if var_reg:
        predicted_var = grouper.apply(np.var, predicted_label, broadcast=False)
    return tf.reduce_sum(tf.add( tf.multiply(tf.add(tf.reduce_sum(tf.squared_difference(label, predicted_label), axis=0),tf.multiply(reg, tf.reduce_sum(tf.square(x)))), weights),  tf.multiply() ))

def prediction_loss_L1(features, label, model, reg, x, grouper=None, weights=1, var_reg=None):
    #uses an abs function approximation
    predicted_label = model(features)
    loss = tf.subtract(label,predicted_label)
    return tf.reduce_sum(tf.add(tf.reduce_sum(tf.square(loss) / tf.sqrt(tf.square(loss) + .01**2), axis=0) , tf.multiply(reg, tf.reduce_sum(tf.square(x)))))


def mode(df, key_cols, value_col, count_col):
    '''
    Pandas does not provide a `mode` aggregation function
    for its `GroupBy` objects. This function is meant to fill
    that gap, though the semantics are not exactly the same.

    The input is a DataFrame with the columns `key_cols`
    that you would like to group on, and the column
    `value_col` for which you would like to obtain the mode.

    The output is a DataFrame with a record per group that has at least one mode
    (null values are not counted). The `key_cols` are included as columns, `value_col`
    contains a mode (ties are broken arbitrarily and deterministically) for each
    group, and `count_col` indicates how many times each mode appeared in its group.
    '''
    return df.groupby(key_cols + [value_col]).size() \
             .to_frame(count_col).reset_index() \
             .sort_values(count_col, ascending=False) \
             .drop_duplicates(subset=key_cols)


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
maxiter:  # of LBFGS iterations
factr:  proportional to covergence tolerance (lower numbers on log scale are more stringent), defaults to 1E7 
"""

## program testing data
# df = pd.read_excel(r'C:\Users\justjo\PycharmProjects\CottonNIRfit_updated\data\pick_strip_data.xlsx')
df = pd.read_excel(r'C:\Users\justjo\PycharmProjects\HFRBmoistureFit\data\HF_RB_signalALLbyDiam.xlsx')
df2 = pd.DataFrame(df.iloc[:,[*[i for i in range(6,12)]]])
df2.res100k.loc[df2.res100k == 0] = 4E7
df2.res5k.loc[df2.res5k == 0] = 4E7
#df2 = pd.DataFrame(df.iloc[:,[*[i for i in range(145,458)]]])

#df2_gt = pd.DataFrame(df.iloc[:, [*[i for i in range(474, 477)]]] / 100)
#df2_gt = pd.DataFrame(df.iloc[:,[-9]])
df2_gt = pd.DataFrame(df.iloc[:,[19]])
#df2_groups = pd.DataFrame(df.iloc[:, -1])

df2_aggregate = pd.DataFrame(df.iloc[:,[1]])

try:
    df2_aggregate = pd.DataFrame(df2_aggregate)
    segment = True
except:
    segment = False

reg_in = 0
n_hidden_units_in = 3
loss_fun = "L1"#, "L2", "L1", "Classifier"
activation_in = 'elu' #options = "lrelu", "relu", "tanh", "linear", "selu", "elu"
early_stop_limit = 100
maxiter = 500
factr = 1E7

## include these in JMP implementation
# df2 = pd.DataFrame(df2)
# df2_gt = pd.DataFrame(df2_gt)
# df2_groups = pd.DataFrame(df2_groups)
# df2_aggregate = pd.DataFrame(df2_aggregate) #optional
# weights = pd.DataFrame(weights) #optional

try:
    weights = pd.DataFrame(weights, columns=['weights'])
except:
    weights = 1.0
    # weights = pd.DataFrame(np.ones(shape = (df2_groups.shape[0], 1)), columns=['weights'])
    #weights = pd.DataFrame(np.random.randint(low=0, high=5, size=(df2_groups.shape[0], 1)), columns=['weights']) ## for debug testing

if segment:
    if isinstance(weights, pd.DataFrame):
        a = ~pd.isnull(df2).any(1) & ~pd.isnull(df2_gt).any(1) & ~pd.isnull(df2_groups).any(1) & ~pd.isnull(df2_aggregate).any(1) & ~pd.isnull(weights).any(1)
    else:
        a = ~pd.isnull(df2).any(1) & ~pd.isnull(df2_gt).any(1) & ~pd.isnull(df2_groups).any(1) & ~pd.isnull(df2_aggregate).any(1)
else:
    if isinstance(weights, pd.DataFrame):
        a = ~pd.isnull(df2).any(1) & ~pd.isnull(df2_gt).any(1) & ~pd.isnull(df2_groups).any(1) & ~pd.isnull(weights).any(1)
    else:
        a = ~pd.isnull(df2).any(1) & ~pd.isnull(df2_gt).any(1) & ~pd.isnull(df2_groups).any(1)

df2 = df2[a]
df2_gt = df2_gt[a]
df2_groups = df2_groups[a]
if segment: df2_aggregate = df2_aggregate[a]
if isinstance(weights, pd.DataFrame): weights = weights[a]
### convert train/val groups to numeric if not already
df2_groups.columns = ['col1']
ip_addresses = np.sort(df2_groups.col1.unique())
ip_dict = dict(zip(ip_addresses, range(len(ip_addresses))))
df2_groups.replace(ip_dict, inplace=True)


## ensure consistency in train/val splits when there are groups
if segment:
    split_groups = Groupby(df2_aggregate.values)

if not (df2_groups.iloc[:,0].unique().shape[0] >= 2):
    #need to set the split level -- determine if grouped or not
    if segment:
        df2_groups = pd.DataFrame(split_groups.split(False, df2_groups.values, 0.667), columns=['split_levels'])
    else:
        df2_groups = pd.DataFrame(np.ceil(np.maximum(0, np.random.uniform(0, 1, (df2.shape[0], 1)) - .667)), columns=['split_levels'])
else:
    #just use the mode to ensure each group only has one split level
    df2_groups = pd.DataFrame(split_groups.split(True, df2_groups.values), columns=['split_levels'])


## break data into training and validation splits
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

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.enable_eager_execution(config = config)
tf.enable_eager_execution()

x_train_in = dfTrain.values.astype('float64')
y_train_in = dfTrain_gt.values.astype('float64')

x_val_in = dfVal.values.astype('float64')
y_val_in = dfVal_gt.values.astype('float64')

if isinstance(weights, pd.DataFrame):
    weights_train = weights.loc[df2_groups.iloc[:,].values[:,0] == a[0]].values.astype('float64')
    weights_val = weights.loc[df2_groups.iloc[:, ].values[:, 0] == a[1]].values.astype('float64')

if segment:
    #train
    df2_aggregate_train = Groupby(df2_aggregate.loc[df2_groups.iloc[:,].values[:,0] == a[0]].values)
    y_train_in = df2_aggregate_train.apply(np.mean, y_train_in, broadcast=False)
    #validation
    df2_aggregate_val = Groupby(df2_aggregate.loc[df2_groups.iloc[:,].values[:,0] == a[1]].values)
    y_val_in = df2_aggregate_val.apply(np.mean, y_val_in, broadcast=False)

    if isinstance(weights, pd.DataFrame):
        weights_train = df2_aggregate_train.apply(np.mean, weights_train, broadcast=False)
        weights_val = df2_aggregate_val.apply(np.mean, weights_val, broadcast=False)

#reg_in = np.float64(0)
reg_in = np.float64(reg_in)
#early_stop_limit = 2

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

if segment:
    loss_train = functools.partial(loss, features=x_train_in, label=y_train_in, model=model, reg=reg_in, grouper = df2_aggregate_train)
    loss_val = functools.partial(loss, features=x_val_in, label=y_val_in, model=model, reg=np.float64(0), x=np.float64(0), grouper = df2_aggregate_val)
else:
    loss_train = functools.partial(loss, features=x_train_in, label=y_train_in, model=model, grouper = None, reg = reg_in)
    loss_val = functools.partial(loss, features=x_val_in, label=y_val_in, model=model, reg=np.float64(0), grouper = None, x=np.float64(0))

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
        print(str(e))
        #sys.stdout.flush()

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
                                 fprime=evaluator.grads, maxiter=maxiter, factr=factr, callback=evaluator.step_callback)

print(info)

var_list_array = [x.numpy() for x in var_list]
a1=var_list_array[0]
a2=var_list_array[1]
b1=0
b2=0
if len(var_list_array) == 4:
    b1 = var_list_array[2]
    b2 = var_list_array[3]
train_means_array = train_means.values
train_sds_array = train_sds.values

# np.savetxt(CPpath + r'/checkpoint.txt', np.array(var_list))
#
# saver = tf.contrib.summary.Saver(var_list)
# saver.save(CPpath + r'/checkpoint.ckpt', global_step = global_step)

#### tensorboard -->  at command prompt or python terinal:   --logdir=C:\Users\justjo\PycharmProjects\basicANN_eagerExecution\tensorboard
