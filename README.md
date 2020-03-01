# AgingGAN

This is a face aging deep learning model. Given an input image and a desired age range, the model ages the face to the desired age group. The age groups are:
1. 10-19 (encoded as the integer 0).
2. 20-29 (encoded as the integer 1).
3. 30-39 (encoded as the integer 2).
4. 40-49 (encoded as the integer 3).
5. 50+ (encoded as the integer 4).

It's mostly inspired by the [Identity Preserved Face Aging](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Face_Aging_With_CVPR_2018_paper.pdf). Most changes in this repo are geared towards making the model fast enough for running on a mobile device. It can achieve 30fps on an iPhone X.

# Design
The following image shows the adversarial training setup. The orange arrows denote the path where backprop to the generator happens.

<p align="center">
  
![Fast-AgingGAN](https://user-images.githubusercontent.com/4294680/71646087-5fd13a80-2ce2-11ea-8d5b-055d202ad1f1.png)

</p>

# Contributing
If you have ideas on improving model performance, adding metrics, or any other changes, please make a pull request or open an issue. I'd be happy to accept any contributions.
