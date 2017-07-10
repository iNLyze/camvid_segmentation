from importlib import reload
from keras.regularizers import l2
import DL_utils.dl as dl; reload(dl) #  makes sure dl is always reloaded when changed\n",
from DL_utils.dl import * # access all imports of dl"
import DL_utils.utils2 as utils2; reload(utils2)
from DL_utils.utils2 import *
from tensorflow.python.ops.variables import Variable
from tqdm import tqdm, tqdm_notebook



def parse_code(l):
    c = l.strip().split("\t")
    a, b = [s for i, s in enumerate(c) if s is not '' ] # Strip entries with double \t
    return tuple(int(o) for o in a.split(' ')), b


def open_image(fn, img_sz): return np.array(Image.open(fn).resize(img_sz, Image.NEAREST))

def create_bc(files, target_size, bcolz_dir, is_label = False):
    files.sort()
    nb_files = len(files)
    shape = (0, target_size[1], target_size[0], 3)
    print('carray shape: {}'.format((nb_files,)+shape[1:]))
    c = bcolz.carray(np.zeros(shape), dtype='float32', 
                 rootdir=bcolz_dir, 
                 mode='w', 
                 expectedlen=nb_files)
    filenames = []
    #print(files)
    for i, fn in enumerate(tqdm_notebook(files)):
        #print(fn)
        filenames.extend(fn)
        img_buffer = open_image(fn, target_size)
        if not is_label:
            img_buffer = preprocess(img_buffer)
        #show(img_buffer, do_deprocess=True)
        #print(img_buffer.shape)
        c.append(img_buffer)
        if i % 10 == 0: # empty buffer
            c.flush()
    c.attrs['filenames'] = filenames
    c.flush()

def conv_one_label(img, c2g_dict, failed_code):
    r = img.shape[0]
    c = img.shape[1]
    res = np.zeros((r,c), 'uint8')
    for j in range(r): 
        for k in range(c):
            #print(img[j, k])
            try: res[j,k] = c2g_dict[tuple(img[j,k])]
            except: res[j,k] = failed_code
    return res

def conv_all_labels(labels, c2g_dict, failed_code):
    #ex = ProcessPoolExecutor(1)
    return np.stack([conv_one_label(labels[i], c2g_dict, failed_code) for i in range(len(labels))])

# Generate batches of indices for the segment generator
class BatchIndices(object):
    def __init__(self, n, bs, shuffle=False):
        self.n,self.bs,self.shuffle = n,bs,shuffle
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.idxs = (np.random.permutation(self.n) 
                     if self.shuffle else np.arange(0, self.n))
        self.curr = 0

    def __next__(self):
        with self.lock:
            if self.curr >= self.n: self.reset()
            ni = min(self.bs, self.n-self.curr)
            res = self.idxs[self.curr:self.curr+ni]
            self.curr += ni
            return res

# Do random 224, 224 crops from the original 640, 480 images
class segm_generator(object):
    def __init__(self, x, y, bs=64, out_sz=(224,224), train=True):
        self.x, self.y, self.bs, self.train = x,y,bs,train
        self.n, self.ri, self.ci, _ = x.shape
        self.idx_gen = BatchIndices(self.n, bs, train) # generate indices
        self.ro, self.co = out_sz
        self.ych = self.y.shape[-1] if len(y.shape)==4 else 1

    # Create random crop called a slice
    def get_slice(self, i,o):
        start = np.random.randint(0, i-o) if self.train and i-o > 0 else (i-o)
        return slice(start, start+o)
    
    # Slice x and y in the same way; x..images, y..labels
    def get_item(self, idx):
        slice_r = self.get_slice(self.ri, self.ro)
        slice_c = self.get_slice(self.ci, self.co)
        x = self.x[idx, slice_r, slice_c]
        y = self.y[idx, slice_r, slice_c]
        if self.train and (np.random.random()>0.5): # randomly flip images horizontally or vertically
            y = y[:,::-1]
            x = x[:,::-1]
        return x, y

    def __next__(self):
        idxs = next(self.idx_gen)
        items = (self.get_item(idx) for idx in idxs)
        xs,ys = zip(*items)
        return np.stack(xs), np.stack(ys).reshape(len(ys), -1, self.ych) # reshaping is for keras Activation('softmax')

# Do random 224, 224 crops from the original 640, 480 images
class segm_generator_hd(object):
    def __init__(self, x, y, bs=64, out_sz=(224,224), train=True):
        self.x, self.y, self.bs, self.train = x,y,bs,train
        self.n, self.ri, self.ci, _ = x.shape
        self.idx_gen = BatchIndices(self.n, bs, train) # generate indices
        self.ro, self.co = out_sz
        self.ych = self.y.shape[-1] if len(y.shape)==4 else 1

    # Create random crop called a slice
    def get_slice(self, i,o):
        start = np.random.randint(0, i-o) if self.train and i-o > 0 else (i-o)
        return slice(start, start+o)
    
    # Slice x and y in the same way; x..images, y..labels
    def get_item(self, idx):
        slice_r = self.get_slice(self.ri, self.ro)
        slice_c = self.get_slice(self.ci, self.co)
        x = self.x[idx, slice_r, slice_c]
        y = self.y[idx, slice_r, slice_c]
        if self.train and (np.random.random()>0.5): # randomly flip images horizontally or vertically
            y = y[:,::-1]
            x = x[:,::-1]
        return x, y

    def __next__(self):
        idxs = next(self.idx_gen)
        items = (self.get_item(idx) for idx in idxs)
        xs,ys = zip(*items)
        return np.stack(xs), np.stack(ys).reshape(len(ys), self.ro, self.co, 1) # adding one dimension


def relu(x): return Activation('relu')(x)
def dropout(x, p): return Dropout(p)(x) if p else x
def bn(x): return BatchNormalization(mode=2, axis=-1)(x)
def relu_bn(x): return relu(bn(x))
def concat(xs): return merge(xs, mode='concat', concat_axis=-1)

def conv(x, nf, sz, wd, p, stride=1): 
    x = Convolution2D(nf, sz, sz, init='he_uniform', border_mode='same', 
                      subsample=(stride,stride), W_regularizer=l2(wd))(x)
    return dropout(x, p)

def initial_conv(input_shape, nf, sz, wd, p, stride=1):
    x = Convolution2D(nf, sz, sz, init='he_uniform', border_mode='same', 
                      subsample=(stride,stride), W_regularizer=l2(wd), input_shape=input_shape)

def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1): 
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)

def dense_block(n,x,growth_rate,p,wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd) # create conv with some 12 or 16 filters at a time and concatenate them
        x = concat([x, b])
        added.append(b)
    ## For trouble-shooting ##
    #print('dense_block')
    #for k, v in locals().items():
    #    if type(v) is Variable or type(v) is tf.Tensor:
    #        print("{0}: {1}".format(k, v))
    ##
    return x,added

# Note: Jeremy says conv with stride 2 works better than conv followed by 2,2 MaxPooling
def transition_dn(x, p, wd):
#     x = conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd)
#     return MaxPooling2D(strides=(2, 2))(x)
    print('trans_dn {}'.format(x.get_shape().as_list()))
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)

def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i,n in enumerate(nb_layers):
        x,added = dense_block(n,x,growth_rate,p,wd)
        skips.append(x) # Keep track of skip connection in this list, pass it to up_path()
        x = transition_dn(x, p=p, wd=wd)
    return skips, added

def transition_up(added, wd=0):
    x = concat(added)
    _,r,c,ch = x.get_shape().as_list()
    return Deconvolution2D(ch, 3, 3, (None,r*2,c*2,ch), init='he_uniform', 
               border_mode='same', subsample=(2,2), W_regularizer=l2(wd))(x)
#     x = UpSampling2D()(x) # Deconvolution works better than UpSampling2D says Jeremy
#     return conv(x, ch, 2, wd, 0)

def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i,n in enumerate(nb_layers):
        ## For trouble-shooting ##
        print('up_path_before')
        for k, v in locals().items():
            if type(v) is Variable or type(v) is tf.Tensor:
                print("{0}: {1}".format(k, v))
        #
        x = transition_up(added, wd)
        ## For trouble-shooting ##
        print('up_path_transition_up')
        for k, v in locals().items():
            if type(v) is Variable or type(v) is tf.Tensor:
                print("{0}: {1}".format(k, v))
        #
        x = concat([x,skips[i]])
        x,added = dense_block(n,x,growth_rate,p,wd)
    return x


#def hidim_softmax(tensor):
#    """Softmax function for > 3D Tensors
#    
#    """
#    sigmoid = lambda x: 1 / (1 + K.exp(-x))
#    softmax_tensor = sigmoid(tensor) / K.sum(sigmoid(tensor), axis=0)
#    return softmax_tensor

def hidim_softmax(x, axis=-1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply hidim_softmax to a tensor that is 1D')

def reverse(a): return list(reversed(a))

def create_tiramisu(nb_classes, img_input, nb_dense_block=6, 
    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):
    
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips,added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)
    
    x = conv(x, nb_classes, 1, wd, 0)
    _,r,c,f = x.get_shape().as_list()
    x = Reshape((-1, nb_classes))(x) # for Activation('softmax')
    return Activation('softmax')(x) # Lambda(hidim_softmax)(x)

def create_tiramisu_hd(nb_classes, img_input, nb_dense_block=6, 
    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):

    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips,added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)
    
    x = conv(x, nb_classes, 1, wd, 0)
    _,r,c,f = x.get_shape().as_list()
    #x = Reshape((-1, nb_classes))(x) # for Activation('softmax')
    #x = Activation('softmax')(x) # 
    #x = Reshape((r, c, f))(x)
    x = Lambda(hidim_softmax)(x)
    return x  

def create_tiramisu_hd_noshape(nb_classes, input_shape, nb_dense_block=6, 
    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block    
    inp_tensor = Input(input_shape)
    x = conv(inp_tensor, nb_filter, 3, wd, 0)
    #x = initial_conv(input_shape, nb_filter, 3, wd, 0)
    ## For trouble-shooting ##
    print('create_tiramisu')
    for k, v in locals().items():
        if type(v) is Variable or type(v) is tf.Tensor:
            print("{0}: {1}".format(k, v))
    ##
    skips,added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)
    
    x = conv(x, nb_classes, 1, wd, 0)
    _,r,c,f = x.get_shape().as_list()
    x = Lambda(hidim_softmax)(x)
    return x  


def create_tiramisu_hidim(nb_classes, img_input, nb_dense_block=6, 
    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):


    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips,added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)
    
    x = conv(x, nb_classes, 1, wd, 0)
    _,r,c,f = x.get_shape().as_list()
    ## For trouble-shooting ##
    print('create_tiramisu')
    for k, v in locals().items():
        if type(v) is Variable or type(v) is tf.Tensor:
            print("{0}: {1}".format(k, v))
    ##
    x = Lambda(hidim_softmax)(x)
    return x  

def vec_to_img(x, img_format=(224, 224)):
    return x.reshape(img_format[0], img_format[1])

def to_plot(labels, shape):
    return np.stack([vec_to_img(labels[:,:,i], shape) for i in range(labels.shape[-1])])

def show_preds(model, examples, labels, shape, figsize=(15.,15.)):
    out = model.predict(examples)
    nu = np.unique(labels)
    print(nu)
    plotGrid(to_plot(out[0,:,nu], shape), figsize)
    return out
