Roadmap

- Check for errors in Confusion matrix and frame selection process
- add uncertainty estimation
- Try training on two samples, and estimating similarity to an unknown third
- Try preprocessing to make frames only show difference to neutral position, and grayscale
- Explore other network architectures
- Outline and draft the paper
- take data of pushes on the edge of objects

Deadlines
- IROS 1st of March

To apply new frankx code:
- activate projects/frankx/venv
- cd frankx
- pip install -e .

To remake the library (not necessary):
- cd frankx/build
- ccmake ..
- make
- make install
