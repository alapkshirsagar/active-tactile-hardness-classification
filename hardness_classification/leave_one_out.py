from adaptive_sampling import *
from preprocessing import *
from models import *
import time
import argparse


if __name__ == "__main__":
    # Start the timer
    start_time = time.time()

    def train_model_ConvLSTM(base_cls, test_cls, strat="variance", verbose=1):
            return adaptive_sampling(CLASSES, ConvLSTM(), manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=100, iterations=10, num_dropouts=10, epochs_per_iter=25, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=4, sample_per_iter_per_class=1, val_per_class=0, comp_size=10, GPU=True, verbose=verbose)
    def train_model_SVM(base_cls, test_cls, strat="variance", verbose=1):
        return adaptive_sampling(CLASSES, OpticalFlowSVM(), manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=100, iterations=10, num_dropouts=10, epochs_per_iter=25, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=4, sample_per_iter_per_class=1, val_per_class=0, comp_size=10, GPU=True, verbose=verbose)

    def train_model_debug(base_cls, test_cls, strat="variance", verbose=1):
            return adaptive_sampling(CLASSES, ConvLSTM(), manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=1, iterations=1, num_dropouts=1, epochs_per_iter=1, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=1, sample_per_iter_per_class=1, val_per_class=0, comp_size=1, GPU=True)

    for i in range(5):
        print("Deform level ", i, "closest: ", closest(i, hardness_levels))
    # Relative Path
    manager = DatasetManager(["../../IP-Object-Hardness/data_no_noise_vel_02to07_force_1to5/data"], classes=CLASSES, seq_length=4)
    # Windows labsolute path "C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"
    # Since the dataset has been processed and cached in the working directory, we can load it directly
    # Already removed the dataset to elsewhere, if the cache is not found, it will raise an error and need to be moved back

    parser = argparse.ArgumentParser(description='Adaptive Sampling Script')
    parser.add_argument('--i', type=int, default=0, help='idx of the class to leave out')
    parser.add_argument('--strat', type=str, default="variance", choices=['random', 'entropy', 'variance', 'YOTO'], help='Sampling strategy')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose level. 0: silent, 1: progress bar, 2: one line per epoch')

    args = parser.parse_args()
    i = args.i
    strat = args.strat
    verbose = args.verbose
    print(f"# leave {CLASSES[i]} out, strategy: {strat}")
    model = train_model_ConvLSTM(base_cls=[CLASSES[j] for j in range(5) if j != i], test_cls=[CLASSES[i]], strat=strat, verbose=verbose)
    model.save_model("./AdaptiveModels/SVM/leave_" + CLASSES[i] + "_out.h5")

    print("**Time taken:**", time.time() - start_time)


