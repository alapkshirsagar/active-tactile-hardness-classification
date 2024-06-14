import datetime
import time

class Logger():
    def __init__(self, filename, time_stamp, data_set, exp, model, model_save, lossfunction = "categorical_crossentropy", epochs = 100) -> None:
        self.filename = filename
        with open(self.filename, "w") as f:
            f.write("====================================================\n")
            f.write(datetime.datetime.fromtimestamp(time_stamp).ctime() + "\n")
            f.write("Dataset: " + data_set + "\n")
            f.write("Experiment: " + exp + "\n")
            f.write("Model: " + model + "\n")
            f.write("Model save file: " + model_save + "\n")
            f.write("Loss Function " + lossfunction + "\n")
            f.write("Epochs: " + str(epochs) + "\n")
            f.write("\n")

    def log(self,s):
        with open(self.filename, "a") as f:
            f.write(str(s) + "\n")

if __name__ == "__main__":
    logger = Logger("asdfasdfadsfasdfdsaf.txt", time.time(), "42", "full", "conv", "conv.h5")
    logger.log("Test")