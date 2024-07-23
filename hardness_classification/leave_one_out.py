from adaptive_sampling import *
from preprocessing import *
from models import *
import time
import argparse
from tensorflow.python.ops.numpy_ops import np_config
import tensorflow as tf

if __name__ == "__main__":
    # Start the timer
    start_time = time.time()

    def train_model_ConvLSTM(base_cls, test_cls, strat="variance", verbose=1, load_model_from=None):
            model = ConvLSTM()
            if load_model_from is None:
                X_dummy, y_dummy = manager.get_raw_with_classes(CLASSES, 1)
                y_dummy = manager.encoder.transform(y_dummy)
                input_shape = X_dummy.shape[1:]
                num_classes = y_dummy.shape[1]
                model.get_model(input_shape, num_classes)
            else:
                model.model = tf.keras.models.load_model(load_model_from)
            return adaptive_sampling(CLASSES, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=100, iterations=50, num_dropouts=10, epochs_per_iter=25, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=4, sample_per_iter_per_class=1, val_per_class=0, comp_size=10, GPU=True, verbose=verbose)
    
    def train_model_OFNN(base_cls, test_cls, strat="variance", verbose=1):
            np_config.enable_numpy_behavior() # Enable numpy behavior in tensorflow

            model = OpticalFlowNN()
            X_dummy, y_dummy = manager.get_raw_with_classes(CLASSES, 1)
            y_dummy = manager.encoder.transform(y_dummy)
            X_dummy = [model.compute_optical_flow(video) for video in X_dummy]
            X_dummy = np.array(X_dummy)
            input_shape = X_dummy.shape[1:]
            num_classes = y_dummy.shape[1]
            
            model.get_model(input_shape, num_classes)
            return adaptive_sampling(CLASSES, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=25, iterations=50, num_dropouts=10, epochs_per_iter=7, strat=strat,augment_factor=1, max_samples=200, init_sample_per_class=5, sample_per_iter_per_class=1, val_per_class=0, comp_size=10, GPU=True, verbose=verbose)
        
    def train_model_DinoNN(base_cls, test_cls, strat="variance", verbose=1):
        model = DINOv2NN()
        X_dummy, y_dummy = manager.get_raw_with_classes(CLASSES, 1)
        y_dummy = manager.encoder.transform(y_dummy)
        input_shape = X_dummy.shape[1:]
        num_classes = y_dummy.shape[1]
        model.get_model(input_shape, num_classes)
        return adaptive_sampling(CLASSES, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=1, iterations=1, num_dropouts=1, epochs_per_iter=1, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=1, sample_per_iter_per_class=1, val_per_class=0, comp_size=1, GPU=True)
    
    def train_model_debug(base_cls, test_cls, strat="variance", verbose=1):
            model = ConvLSTM()
            X_dummy, y_dummy = manager.get_raw_with_classes(CLASSES, 1)
            y_dummy = manager.encoder.transform(y_dummy)
            input_shape = X_dummy.shape[1:]
            num_classes = y_dummy.shape[1]
            model.get_model(input_shape, num_classes)
            return adaptive_sampling(CLASSES, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=1, iterations=1, num_dropouts=1, epochs_per_iter=1, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=1, sample_per_iter_per_class=1, val_per_class=0, comp_size=1, GPU=True)

    for i in range(5):
        print("Deform level ", i, "closest: ", closest(i, hardness_levels))
    # Relative Path
    manager = DatasetManager(["../../IP-Object-Hardness/data_no_noise_vel_02to07_force_1to5/data"], classes=CLASSES, seq_length=4)
    # Windows labsolute path "C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"
    # Since the dataset has been processed and cached in the working directory, we can load it directly
    # Already removed the dataset to elsewhere, if the cache is not found, it will raise an error and need to be moved back

    parser = argparse.ArgumentParser(description='Adaptive Sampling Script')
    parser.add_argument('--i', type=int, default=1, help='idx of the class to leave out')
    parser.add_argument('--strat', type=str, default="entropy", choices=['random', 'entropy', 'variance', 'YOTO'], help='Sampling strategy')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose level. 0: silent, 1: progress bar, 2: one line per epoch')

    args = parser.parse_args()
    i = args.i
    strat = args.strat
    verbose = args.verbose
    print(f"# leave {CLASSES[i]} out, strategy: {strat}")
    model = train_model_ConvLSTM(base_cls=[CLASSES[j] for j in range(5) if j != i], test_cls=[CLASSES[i]], strat=strat, verbose=verbose, load_model_from=None)

    print("**Time taken:**", time.time() - start_time)


