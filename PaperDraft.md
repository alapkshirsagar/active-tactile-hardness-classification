Key Points:

Explaination of Gelsight sensor

Key Goals (depending on what works):
-> Sample efficent hardness Classification:
    -> Generic Network can learn different hardnesses without any prior works (Plug and Play)
      (3), Pretrain with full and augment with edge.
    -> Choose sample with most similar hardness (see Cognitive Science Exp.)



Methods:
-> Frame Selection (different approches if relevant)
-> Network Architectures:
    -> CNN LSTM
    -> 3D Convolutional Neural Network
    -> Simularity network  (Siamise Network) (TODO)
-> Different Loss Functions:
    -> Categorical cross entropy
    -> Ordinal categorical cross entropy (1)

-> Make it a regression problem (linear assumption on training hardnesses, but assumes order of samples)
    (2)

References:
List of Relevant Papers:
- Estimating object hardness with a gelsight touch sensor: https://ieeexplore.ieee.org/document/7759057
- Shape-independent hardness estimation using deep learning and a gelsight tactile sensor: https://ieeexplore.ieee.org/abstract/document/7989116
- Use of tactile feedback to control exploratory movements to characterize object compliance: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3405524/
- Tactual Recognition of Soft Objects From Deformation Cues: https://ieeexplore.ieee.org/abstract/document/9568739
- Non-destructive Fruit Firmness Evaluation Using Vision-Based Tactile Information: https://ieeexplore.ieee.org/abstract/document/9811920
- TactONet: Tactile Ordinal Network Based on Unimodal Probability for Object Hardness Classification: https://ieeexplore.ieee.org/abstract/document/9866706
- Hardness Recognition of Robotic Forearm Based on Semi-supervised Generative Adversarial Networks: https://www.frontiersin.org/articles/10.3389/fnbot.2019.00073/full
- Exploring Hardness and Geometry Information through Active Perception: https://ieeexplore.ieee.org/abstract/document/8931910
- Object stiffness recognition with descriptors given by an FPGA-based tactile sensor: https://ieeexplore.ieee.org/abstract/document/9152575

- CNN LSTM Video Classifacation Network: Shi, X., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. Advances in neural information processing systems, 28.
https://arxiv.org/abs/1506.04214v1
- Spatiotemporal Convolution (3D Convolution, 2D space, 1D time) https://arxiv.org/abs/1711.11248v3 
