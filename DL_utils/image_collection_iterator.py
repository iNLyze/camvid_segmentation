class ImageCollectionIterator(object):
    def __init__(self, X=None, y=None, w=None, path=None, batch_size=32, shuffle=False, seed=None, load_func=None):
        if path is None and X is None:
            raise ValueError('Must provide either X=ImageCollection or path to image files')
        if y is not None and len(X) != len(y):
            raise ValueError('X (features) and y (labels) should have the same length'
                             'Found: X.shape = %s, y.shape = %s' % (X.shape, y.shape))
        if w is not None and len(X) != len(w):
            raise ValueError('X (features) and w (weights) should have the same length'
                             'Found: X.shape = %s, w.shape = %s' % (X.shape, w.shape))


        self.X = X if X is not None else io.ImageCollection(self.get_filenames(path))
        self.y = y if y is not None else None
        self.w = w if w is not None else None
        self.N = len(X)
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed
        
    def get_filenames(self, path):
        return glob.glob(path+'*', recursive=True)


    def reset(self): self.batch_index = 0
        
    def current_index_array(self):
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
            
        # Check if enough images left for an entire batch_size of images
        self.images_left = self.N-self.batch_index*self.batch_size
        if self.images_left >= self.batch_size:
            self.idx_array_size = self.batch_size
        else:
            self.idx_array_size = self.images_left
        # Generate index_array for iterating through collection
        if self.shuffle:
            self.index_array = (np.random.permutation(self.idx_array_size)+self.batch_size*self.batch_index)
        else:
            self.index_array = np.arange( self.idx_array_size ) + self.batch_size*self.batch_index    

    def next(self):
        with self.lock:
            self.current_index_array()
            print(self.index_array)
            if self.index_array == []:
                batch_index=0
                return -1
            batches_x, batches_y, batches_w = [],[],[]
            for current_index in self.index_array:
                batches_x.append(self.X[current_index])
                self.batch_index += 1
                self.total_batches_seen += 1
                if not self.y is None: batches_y.append(self.y[current_index])
                if not self.w is None: batches_w.append(self.w[current_index])

            
            batch_x = np.concatenate(batches_x) 
            if self.y is None: return batch_x

            batch_y = np.concatenate(batches_y)
            if self.w is None: return batch_x, batch_y

            batch_w = np.concatenate(batches_w)
            return batch_x, batch_y, batch_w


    def __iter__(self): return self

    def __next__(self, *args, **kwargs): return self.next(*args, **kwargs)
    
