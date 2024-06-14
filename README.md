- To run the training and evaluation:
```
python3 hardness_classification/train_eval.py
```

To apply new frankx code:
- activate projects/frankx/venv
- cd frankx
- pip install -e .

To remake the library (not necessary):
- cd frankx/build
- ccmake ..
- make
- make install
