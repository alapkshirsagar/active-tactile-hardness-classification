from adaptive_sampling import *
from preprocessing import *
from models import *
import time
import argparse
from tensorflow.python.ops.numpy_ops import np_config

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Adaptive Sampling Script')
    parser.add_argument('--comp_index', type=int, default=2, help='idx of the class to leave out')
    parser.add_argument('--strat', type=str, default="variance", choices=['random', 'entropy', 'variance', 'YOTO'], help='Sampling strategy')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose level. 0: silent, 1: progress bar, 2: one line per epoch')
    parser.add_argument('--prefix', type=str, default="test_Sep_6", help='Prefix for the model name')
    parser.add_argument('--model', type=str, default="ConvLSTM_Expansion", choices=['ConvLSTM', 'DinoNN', 'OFNN', 'Debug', 'ConvLSTM_Expansion', 'DinoNN_Expansion'], help='model')
    parser.add_argument('--dataset', type=str, default="hardness", choices=['yuan', 'hardness'], help='dataset name')
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout rate")

    args = parser.parse_args()
    comp_index = args.comp_index
    strat = args.strat
    verbose = args.verbose
    DATASET = args.dataset
    
    # Start the timer
    start_time = time.time()
    
    #Create dataset manager
    if DATASET == "yuan":
        CLASSES =  [8, 17, 27, 46, 62, 73, 81, 87, 95] # Use these for FLAT object, 9 classes
        hardness_levels = np.array([float(x) for x in CLASSES]) # Yuan Dataset
    elif DATASET == "hardness":
        CLASSES = ["hardness" + str(i) for i in range(1, 6)] # Use these for hardness dataset, 5 classes
        hardness_levels = np.array([1.13, 1.02, 0.79, 0.68, 0.44]) # Hardness levels for the 5 classes
    else: raise ValueError("Invalid dataset name")
    
    manager = DatasetManagerYuan("E:/MLData/Data1", objects = ["FLAT"], classes = CLASSES, seq_length=3) if DATASET == "yuan" else DatasetManager(["../../IP-Object-Hardness/data_no_noise_vel_02to07_force_1to5/data"], classes=CLASSES, seq_length=4)
    
    # Create hyperparameters according to dataset
    iterations = 5
    init_epochs = 50 if DATASET == "yuan" else 100
    epochs_per_iter = 25 if DATASET == "yuan" else 50
    sample_per_iter_per_class = 5 if DATASET == "yuan" else 3
    
    def train_model_ConvLSTM(manager, base_cls, test_cls, strat="variance", verbose=1, load_model_from=None, run_prefix="", dropout=0.2):  
            model = ConvLSTM()
            if load_model_from is None:
                X_dummy, y_dummy = manager.get_raw_with_classes(CLASSES, 1)
                y_dummy = manager.encoder.transform(y_dummy)
                input_shape = X_dummy.shape[1:]
                num_classes = y_dummy.shape[1]
                model.get_model(input_shape, num_classes, dropout_rate=dropout)
            else:
                model.model = tf.keras.models.load_model(load_model_from)
                
            
            return adaptive_sampling(CLASSES, hardness_levels, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=init_epochs, iterations=iterations, num_dropouts=10, epochs_per_iter=epochs_per_iter, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=5, sample_per_iter_per_class=5, val_per_class=5, comp_size=5, GPU=True, verbose=verbose, run_prefix=run_prefix,  dropout=dropout)

    
    def train_model_OFNN(manager, base_cls, test_cls, strat="variance", verbose=1, run_prefix="", dropout=0.2):
        np_config.enable_numpy_behavior() # Enable numpy behavior in tensorflow

        model = OpticalFlowNN()
        X_dummy, y_dummy = manager.get_raw_with_classes(base_cls + test_cls, 1)
        X_dummy, y_dummy = manager.get_raw_with_classes(base_cls + test_cls, 1)
        y_dummy = manager.encoder.transform(y_dummy)
        X_dummy = [model.compute_optical_flow(video) for video in X_dummy]
        X_dummy = np.array(X_dummy)
        input_shape = X_dummy.shape[1:]
        num_classes = y_dummy.shape[1]
        
        model.get_model(input_shape, num_classes, dropout_rate=dropout)
        

        return adaptive_sampling(CLASSES, hardness_levels, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=init_epochs, iterations=iterations, num_dropouts=10, epochs_per_iter=epochs_per_iter, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=5, sample_per_iter_per_class=5, val_per_class=5, comp_size=5, GPU=True, verbose=verbose, run_prefix=run_prefix, dropout=dropout)

    def train_model_DinoKeras(manager, base_cls, test_cls, strat="variance", verbose=1, load_model_from=None, run_prefix="", dropout=0.2):
            np_config.enable_numpy_behavior() # Enable numpy behavior in tensorflow

            model = DINOKeras()
            if load_model_from is None:
                X_dummy, y_dummy = manager.get_raw_with_classes(CLASSES, 1)
                y_dummy = manager.encoder.transform(y_dummy)
                input_shape = X_dummy.shape[1:]
                num_classes = y_dummy.shape[1]
                model.get_model(input_shape, num_classes, dropout_rate=dropout)
            else:
                model.model = tf.keras.models.load_model(load_model_from)
                
            
            return adaptive_sampling(CLASSES, hardness_levels, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=init_epochs, iterations=iterations, num_dropouts=10, epochs_per_iter=epochs_per_iter, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=5, sample_per_iter_per_class=5, val_per_class=5, comp_size=5, GPU=True, verbose=verbose, run_prefix=run_prefix,dropout=dropout)

    def train_model_debug(manager,  base_cls, test_cls, strat="variance", verbose=1, run_prefix="", dropout=0.2):
        model = ConvLSTM()
        X_dummy, y_dummy = manager.get_raw_with_classes(CLASSES, 1)
        y_dummy = manager.encoder.transform(y_dummy)
        input_shape = X_dummy.shape[1:]
        num_classes = y_dummy.shape[1]
        model.get_model(input_shape, num_classes, dropout_rate=dropout)
        return adaptive_sampling(CLASSES, hardness_levels, model, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=1, iterations=1, num_dropouts=1, epochs_per_iter=1, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=1, sample_per_iter_per_class=1, val_per_class=0, comp_size=1, GPU=True, verbose=verbose, run_prefix=run_prefix, dropout=dropout)
    
    # Start execution
    print(f"# leave {CLASSES[comp_index]} out, strategy: {strat}")
    if(args.model == "ConvLSTM"):
        model = train_model_ConvLSTM(manager, base_cls=[CLASSES[j] for j in range(len(CLASSES))], test_cls=[CLASSES[comp_index]], strat=strat, verbose=verbose, run_prefix=args.prefix+"_"+DATASET+"_", dropout=args.dropout)
    elif(args.model == "OFNN"):
        model = train_model_OFNN(manager, base_cls=[CLASSES[j] for j in range(len(CLASSES))], test_cls=[CLASSES[comp_index]], strat=strat, verbose=verbose, run_prefix=args.prefix+"_"+DATASET+"_", dropout=args.dropout)
    elif(args.model == "DinoNN"):
        model = train_model_DinoKeras(manager, base_cls=[CLASSES[j] for j in range(len(CLASSES))], test_cls=[CLASSES[comp_index]], strat=strat, verbose=verbose, run_prefix=args.prefix+"_"+DATASET+"_", dropout=args.dropout)
    elif(args.model=="Debug"):
        model = train_model_debug(manager, base_cls=[CLASSES[j] for j in range(len(CLASSES))], test_cls=[CLASSES[comp_index]], strat=strat, verbose=verbose, run_prefix=args.prefix+"_"+DATASET+"_", dropout=args.dropout)
    elif(args.model == "ConvLSTM_Expansion"):
        model = train_model_ConvLSTM(manager, base_cls = [x for i, x in enumerate(CLASSES) if i != comp_index], test_cls=[CLASSES[comp_index]], strat=strat, verbose=verbose, run_prefix=args.prefix+"_"+DATASET+"_", dropout=args.dropout)
    elif(args.model == "DinoNN_Expansion"):
        model = train_model_DinoKeras(manager, base_cls = [x for i, x in enumerate(CLASSES) if i != comp_index], test_cls=[CLASSES[comp_index]], strat=strat, verbose=verbose, run_prefix=args.prefix+"_"+DATASET+"_", dropout=args.dropout)
    model.save_model("./AdaptiveModels/OFNN/leave_" + str(CLASSES[comp_index]) + "_out.h5")

    print("**Time taken:**", time.time() - start_time)


