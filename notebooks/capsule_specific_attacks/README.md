# Capsule Specific Attack Visualization Analysis
In capsule-specific attacks, we retrained a _CNN_ model and a full _CapsNet_
with sub-network on MNIST and CIFAR10 for 500 epochs. There were no data
deformations applied except some cropping operations in this set of experiments.
For MNIST, the image size is 28 $\times$ 28. For CIFAR10, the image size is
cropped to 28 $\times$ 28 $\times$ 3 (random cropping for training set and
central cropping for test set).

In this file, we include two types of layouts, **Loose Layout** and
**Tight Layout**, for possible combinations of different **dataset**, 
**model type**, and **attack methods**. For detailed image-distribution
visualizations, refer to **Loose Layout** sections. For quick overview of
visualizations, navigate to **Tight Layout** sections.

To keep visualizations of different methods comparable, the following discussions all use the same instance number and capsule selection settings (instance_num=3, cap_idx=7), that means, the figures you see below are mostly using the same original images. 

## Abbreviations

### Methods
#### Norm Based
- **NMN**: Naively maximizing target capsule's norm.
- **MND**: Maximizing the difference between target capsule's norm and the rest capsules' norm.
#### Dimension Based
- **NMCD**: Naively maximizing a dimension of the target capsule.
- **MCDD**: Maximizing the different between a dimension and the rest dimensions of the target capsule.

### Comparison Types
- **Same Origins vs Different Targets**: we use the same original image as the base image, then set different capsules as target class to push the base image into different target classes.
- **Different Origins vs Same Target**: we use different images as base images, then set a same capsule as target class to push the different base images into the same target class.

Table of Content
<!--ts-->
- [Capsule Specific Attack Visualization Analysis](#capsule-specific-attack-visualization-analysis)
  - [Abbreviations](#abbreviations)
    - [Methods](#methods)
      - [Norm Based](#norm-based)
      - [Dimension Based](#dimension-based)
    - [Comparison Types](#comparison-types)
  - [Norm Based Attacks](#norm-based-attacks)
  - [![](#norm-based-attacks)](#norm-based-attacks)
  - [![](#norm-based-attacks)](#norm-based-attacks-1)
  - [Dimension Based Attacks](#dimension-based-attacks)
  - [![](#dimension-based-attacks)](#dimension-based-attacks)
    - [Tight Layout](#tight-layout)
      - [MNIST CAPS NMCD (DBA TL)](#mnist-caps-nmcd-dba-tl)
      - [MNIST CAPS MCDD (DBA TL)](#mnist-caps-mcdd-dba-tl)
      - [CIFAR10 CAPS NMCD (DBA TL)](#cifar10-caps-nmcd-dba-tl)
      - [CIFAR10 CAPS MCDD (DBA TL)](#cifar10-caps-mcdd-dba-tl)
  - [Method Wise Comparison](#method-wise-comparison)
    - [MNIST CAPS (MWC)](#mnist-caps-mwc)
    - [CIFAR10 CAPS (MWC)](#cifar10-caps-mwc)
<!--te-->

---
**NOTE**: we seperated out **Loose Layout** visualizations of CIFAR10 in [More-Images.md](More-Images.md)
to keep this file as neat and general as possible.

**A Convenient Tip to Navigate between Sections**: In order to easily navigate between discussions
and corresponding visualizations, I added _hyperlink_ to figures so that readers can easily
click back to the discussion at the beginning of the section.

E.g., readers can click any figures under [Norm Based Attacks](#norm-based-attacks) and the web
browser will automatically reset page to the corresponding section discussion.

---

## Norm Based Attacks
Intuition on difference between _CNNs_ and _CapsNets_. Because the nature of norm based attacks, which
takes advantages of norms of vectors, one can also apply the same methods to _CNNs_ by directly
using the scalar values as ''norms'' (though here these ''norms'' can be negative, it doesn't
affect the gradient computation). The results shed light on how _CNNs_ and _CapsNets_ are
structurally different from each other. In Figures below,

---
Collection 1
Original Class = 7, Target Class = 2
- CNN NMN: 
[![](norm_based/loose_layout/ins3_cap7/Same_Ori-Diff_Tar/mnist_cnn_naive_max_norm/2.png)](#norm-based-attacks)
- CNN MND: 
[![](norm_based/loose_layout/ins3_cap7/Same_Ori-Diff_Tar/mnist_cnn_max_norm_diff/2.png)](#norm-based-attacks)
- Caps NMN: 
[![](norm_based/loose_layout/ins3_cap7/Same_Ori-Diff_Tar/mnist_cap_naive_max_norm/2.png)](#norm-based-attacks)
- Caps MND: 
[![](norm_based/loose_layout/ins3_cap7/Same_Ori-Diff_Tar/mnist_cap_max_norm_diff/2.png)](#norm-based-attacks)
---
it is not hard to observe that the results from _CNNs_ are much noiser than those from _CapsNets_,
which can partially result from the fact we did not adjust the step size for _CNNs_.
Nevertheless, we can observe that the learned 'features' are eventually added to the whole
region of the image in _CNN_ models, as shown in Figure (CNN NMN) and Figure (CNN MND).
While on results from _CapsNets_ (Figure (Caps NMN) and Figure (Caps MND) in Collection 1), new 'features' are
added in the vicinity of original strong 'features', e.g. tilted vertical line of this given
digit '7'. Why there is such difference? As mentioned earlier, the accumulated perturbation
that results from step size and number of iteration settings of adversarial perturbation
could result in this difference. For images in Figure (Caps NMN) and Figure (Caps MND) in Collection 1, if
kept applying a sufficiently large number of iterations, the white noise (or added 'features')
might eventually cover the whole region as well such as examples in Figure (Caps MND) in Collection 2.

We notice that white noise from _CNNs_ are more adhesive to each other, e.g. columns from 1 to 6 in
Figure (CNN NMN) in Collection 2, while those from _Caps_ are relatively independent to each other,
e.g. images in Figure (Caps NMN) and Figure (Caps MND). Therefore, we also believe the part-whole agreement of _CapsNets_ plays a substantial role here. _CapsNets_ are also applied with a variation of
convolutional operation --- capsule
convolution (as introduced in the original paper). Hence, while processing pixel intensity
values into probabilities of classes, both _CNNs_ and _CapsNets_ use shared kernels/capsule
kernels to slide through every fixed size subregion to detect features, by which they achieve
translational-equivariance. When generating adversarial examples, translational-equivariance
helps _CNNs_ send feedbacks no matter what feature it detected. But for _CapsNets_, those
feedbacks are further verified by the part-whole agreement mechanism, e.g. if a potential low
activity of feature 'A' (that belongs to object 'O') is detected a location, but 'A''s location
does not likely satisfy the part-whole agreement between 'A' and 'O', then its gradient signal
will be filtered out by the agreement mechnism.


---
Collection 2
Original Class = 7, Target Class = 0 ~ 9 (each row is targeting one class)
- CNN NMN:
  [![](norm_based/tight_layout/ins3_cap7/Same_Ori-Diff_Tar/mnist_cnn_naive_max_norm.png)](#norm-based-attacks)
- CNN MND
  [![](norm_based/tight_layout/ins3_cap7/Same_Ori-Diff_Tar/mnist_cnn_max_norm_diff.png)](#norm-based-attacks)
- Caps NMN
  [![](norm_based/tight_layout/ins3_cap7/Same_Ori-Diff_Tar/mnist_cap_naive_max_norm.png)](#norm-based-attacks)
- Caps MND
  [![](norm_based/tight_layout/ins3_cap7/Same_Ori-Diff_Tar/mnist_cap_max_norm_diff.png)](#norm-based-attacks)
---

## Dimension Based Attacks
We also probed the impact of optimizing individual dimensions of the winning capsule upon
the model's prediction. The results showed that even though in dimension based attacks
we did not explicitly maximize the norm of another capsule to push the original image
away from its label to make the model misclassify the category, after several iterations
of perturbations, the attacks still had successfully shifted the model's predictions from
correct ones to wrong categories, as shown in Collection 3. This behaviour makes sense under
the context of dynamic routing. We think when maximizing one dimension of the target capsule, 
it also impacts the agreements between other lower-level capsules and current layer capsules
by changing some specific groups of features in the image.

---
Collection 3, Same Origin = 7, Target Dimension of Capsule '7' = 2
- Caps NMCD:
  [![](dim_based/loose_layout/ins3_cap7/mnist_cap_naive_max_caps_dim/2.png)](#dimension-based-attacks)
  [![](dim_based/loose_layout/ins3_cap7/mnist_cap_naive_max_caps_dim/2_distr.png)](#dimension-based-attacks)
- Caps MCDD:
  [![](dim_based/loose_layout/ins3_cap7/mnist_cap_max_caps_dim_diff/2.png)](#dimension-based-attacks)
  [![](dim_based/loose_layout/ins3_cap7/mnist_cap_max_caps_dim_diff/2_distr.png)](#dimension-based-attacks)
---
 
Further, we notice that the resultant predicted labels are random. For example, in
Collection 4, model's predictions shifted to class '4' and class '5' respectively.

---
Collection 4, Same Origin = 7
- Caps NMCD (Target Dimension of Capsule '7' = 5):
  [![](dim_based/loose_layout/ins3_cap7/mnist_cap_naive_max_caps_dim/5.png)](#dimension-based-attacks)
  [![](dim_based/loose_layout/ins3_cap7/mnist_cap_naive_max_caps_dim/5_distr.png)](#dimension-based-attacks)
- Caps NMCD (Target Dimension of Capsule '7' = 10):
  [![](dim_based/loose_layout/ins3_cap7/mnist_cap_naive_max_caps_dim/10.png)](#dimension-based-attacks)
  [![](dim_based/loose_layout/ins3_cap7/mnist_cap_naive_max_caps_dim/10_distr.png)](#dimension-based-attacks)

---


### Tight Layout
#### MNIST CAPS NMCD (DBA TL)
[![](dim_based/tight_layout/ins3_cap7/mnist_cap_naive_max_caps_dim.png)](#dimension-based-attacks)
#### MNIST CAPS MCDD (DBA TL)
[![](dim_based/tight_layout/ins3_cap7/mnist_cap_max_caps_dim_diff.png)](#dimension-based-attacks)
#### CIFAR10 CAPS NMCD (DBA TL)
[![](dim_based/tight_layout/ins3_cap7/cifar10_cap_naive_max_caps_dim.png)](#dimension-based-attacks)
#### CIFAR10 CAPS MCDD (DBA TL)
[![](dim_based/tight_layout/ins3_cap7/cifar10_cap_max_caps_dim_diff.png)](#dimension-based-attacks)

## Method Wise Comparison

### MNIST CAPS (MWC)
- NMN: [![](norm_based/tight_layout/ins3_cap7/Same_Ori-Diff_Tar/mnist_cap_naive_max_norm.png)](#method-wise-comparison)
- MND: [![](norm_based/tight_layout/ins3_cap7/Same_Ori-Diff_Tar/mnist_cap_max_norm_diff.png)](#method-wise-comparison)
- NMCD: [![](dim_based/tight_layout/ins3_cap7/mnist_cap_naive_max_caps_dim.png)](#method-wise-comparison)
- MCDD: [![](dim_based/tight_layout/ins3_cap7/mnist_cap_max_caps_dim_diff.png)](#method-wise-comparison)

### CIFAR10 CAPS (MWC)
- NMN: [![](norm_based/tight_layout/ins3_cap7/Same_Ori-Diff_Tar/cifar10_cap_naive_max_norm.png)](#method-wise-comparison)
- MND: [![](norm_based/tight_layout/ins3_cap7/Same_Ori-Diff_Tar/cifar10_cap_max_norm_diff.png)](#method-wise-comparison)
- NMCD: [![](dim_based/tight_layout/ins3_cap7/cifar10_cap_naive_max_caps_dim.png)](#method-wise-comparison)
- MCDD: [![](dim_based/tight_layout/ins3_cap7/cifar10_cap_max_caps_dim_diff.png)](#method-wise-comparison)