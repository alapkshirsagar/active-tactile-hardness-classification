from adaptive_sampling import *
from preprocessing import *
from models import *
import time
import argparse
from tensorflow.python.ops.numpy_ops import np_config
DATASET = "hardness" # "yuan" or "hardness

if DATASET == "yuan":
    CLASSES =  [8, 17, 27, 46, 62, 73, 81, 87, 95] # Use these for FLAT object, 9 classes
    hardness_levels = np.array([float(x) for x in CLASSES]) # Yuan Dataset
elif DATASET == "hardness":
    CLASSES = ["hardness" + str(i) for i in range(1, 6)] # Use these for hardness dataset, 5 classes
    hardness_levels = np.array([1.13, 1.02, 0.79, 0.68, 0.44]) # Hardness levels for the 5 classes
else: raise ValueError("Invalid dataset name")
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()

    def train_model_ConvLSTM(manager, base_cls, test_cls, strat="variance", verbose=1, load_model_from=None, run_prefix=""):
            model = ConvLSTM()
            if load_model_from is None:
                X_dummy, y_dummy = manager.get_raw_with_classes(CLASSES, 1)
                y_dummy = manager.encoder.transform(y_dummy)
                input_shape = X_dummy.shape[1:]
                num_classes = y_dummy.shape[1]
                model.get_model(input_shape, num_classes)
            else:
                model.model = tf.keras.models.load_model(load_model_from)
                
            iterations = 10
            init_epochs = 50 if DATASET == "yuan" else 100
            epochs_per_iter = 25 if DATASET == "yuan" else 50
            
            return adaptive_sampling(CLASSES, hardness_levels, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=init_epochs, iterations=iterations, num_dropouts=10, epochs_per_iter=epochs_per_iter, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=5, sample_per_iter_per_class=5, val_per_class=5, comp_size=5, GPU=True, verbose=verbose, run_prefix=run_prefix)

    
    def train_model_OFNN(manager, base_cls, test_cls, strat="variance", verbose=1, run_prefix=""):
        np_config.enable_numpy_behavior() # Enable numpy behavior in tensorflow

        model = OpticalFlowNN()
        X_dummy, y_dummy = manager.get_raw_with_classes(base_cls + test_cls, 1)
        X_dummy, y_dummy = manager.get_raw_with_classes(base_cls + test_cls, 1)
        y_dummy = manager.encoder.transform(y_dummy)
        X_dummy = [model.compute_optical_flow(video) for video in X_dummy]
        X_dummy = np.array(X_dummy)
        input_shape = X_dummy.shape[1:]
        num_classes = y_dummy.shape[1]
        
        model.get_model(input_shape, num_classes)
        
        iterations = 10
        init_epochs = 50 if DATASET == "yuan" else 100
        epochs_per_iter = 25 if DATASET == "yuan" else 50
        return adaptive_sampling(CLASSES, hardness_levels, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=init_epochs, iterations=iterations, num_dropouts=10, epochs_per_iter=epochs_per_iter, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=5, sample_per_iter_per_class=5, val_per_class=5, comp_size=5, GPU=True, verbose=verbose, run_prefix=run_prefix)
    def train_model_DinoNN(manager, base_cls, test_cls, strat="variance", verbose=1, run_prefix=""):
        model = DINOv2NN()
        X_dummy, y_dummy = manager.get_raw_with_classes(CLASSES, 1)
        y_dummy = manager.encoder.transform(y_dummy)
        input_shape = X_dummy.shape[1:]
        num_classes = y_dummy.shape[1]
        model.get_model(input_shape, num_classes)
        iterations = 10
        init_epochs = 50 if DATASET == "yuan" else 100
        epochs_per_iter = 25 if DATASET == "yuan" else 50
        
        return adaptive_sampling(CLASSES, hardness_levels, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=init_epochs, iterations=iterations, num_dropouts=10, epochs_per_iter=epochs_per_iter, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=5, sample_per_iter_per_class=5, val_per_class=5, comp_size=5, GPU=True, verbose=verbose, run_prefix=run_prefix)
    

    def train_model_debug(manager,  base_cls, test_cls, strat="variance", verbose=1, run_prefix=""):
        model = ConvLSTM()
        X_dummy, y_dummy = manager.get_raw_with_classes(CLASSES, 1)
        y_dummy = manager.encoder.transform(y_dummy)
        input_shape = X_dummy.shape[1:]
        num_classes = y_dummy.shape[1]
        model.get_model(input_shape, num_classes)
        return adaptive_sampling(CLASSES, hardness_levels, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=1, iterations=1, num_dropouts=1, epochs_per_iter=1, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=1, sample_per_iter_per_class=1, val_per_class=0, comp_size=1, GPU=True, verbose=verbose, run_prefix=run_prefix)

        # hardness_levels = np.array([1.13, 1.02, 0.79, 0.68, 0.44]) # Hardness levels for the 5 classes
        # hardness_levels = np.array([1.13, 1.02, 0.79, 0.68, 0.44]) # Yuan Dataset
    for i in range(len(CLASSES)):
        print("Deform level ", i, "closest: ", closest(i, hardness_levels))
    # Relative Path
    # manager = 
    # Windows labsolute path "C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"
    # Since the dataset has been processed and cached in the working directory, we can load it directly
    # Already removed the dataset to elsewhere, if the cache is not found, it will raise an error and need to be moved back

    manager = DatasetManagerYuan("E:/MLData/dataset-yuan/Data1/", objects = ["FLAT"], classes = CLASSES, seq_length=3) if DATASET == "yuan" else DatasetManager(["E:/MLData/data_no_noise_vel_02to07_force_1to5/data"], classes=CLASSES, seq_length=4)


    parser = argparse.ArgumentParser(description='Adaptive Sampling Script')
    parser.add_argument('--i', type=int, default=1, help='idx of the class to leave out')
    parser.add_argument('--strat', type=str, default="entropy", choices=['random', 'entropy', 'variance', 'YOTO'], help='Sampling strategy')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose level. 0: silent, 1: progress bar, 2: one line per epoch')
    parser.add_argument('--prefix', type=str, default="Comp_Models", help='Prefix for the model name')
    parser.add_argument('--model', type=str, default="DinoNN", choices=['ConvLSTM', 'DinoNN', 'OFNN'], help='model')
    args = parser.parse_args()
    i = args.i
    strat = args.strat
    verbose = args.verbose
    print(f"# leave {CLASSES[i]} out, strategy: {strat}")
    if(args.model == "ConvLSTM"):
        model = train_model_ConvLSTM(manager, base_cls=[CLASSES[j] for j in range(len(CLASSES))], test_cls=[CLASSES[i]], strat=strat, verbose=verbose, run_prefix=args.prefix+"_"+DATASET+"_")
    elif(args.model == "OFNN"):
        model = train_model_OFNN(manager, base_cls=[CLASSES[j] for j in range(len(CLASSES))], test_cls=[CLASSES[i]], strat=strat, verbose=verbose, run_prefix=args.prefix+"_"+DATASET+"_")
    elif(args.model == "DinoNN"):
        model = train_model_DinoNN(manager, base_cls=[CLASSES[j] for j in range(len(CLASSES))], test_cls=[CLASSES[i]], strat=strat, verbose=verbose, run_prefix=args.prefix+"_"+DATASET+"_")
    else:
        print("Invalid model name")
    model.save_model("./AdaptiveModels/OFNN/leave_" + str(CLASSES[i]) + "_out.h5")

    print("**Time taken:**", time.time() - start_time)


