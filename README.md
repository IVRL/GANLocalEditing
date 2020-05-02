
## Localized Semantic Editing of StyleGAN outputs

![ ](https://github.com/IVRL/GANLocalEditing/blob/master/teaser.gif)

Paper: https://arxiv.org/abs/2004.14367<br>
Video: https://youtu.be/9mXVPaT9Ryg<br>

This demo illustrates a simple and effective method for making local, semantically-aware edits to a *target* GAN output image. This is accomplished by borrowing styles from a *reference* image, also a GAN output.

The method requires neither supervision from an external model, nor involves complex spatial morphing operations. Instead, it relies on the emergent disentanglement of semantic objects that is learned by StyleGAN during its training, which we detect using Spherical *k*-means.

The implementation below relies on PyTorch and requires downloading [additional parameter files ](https://drive.google.com/open?id=1GYzEzOCaI8FUS6JHdt6g9UfNTmpO08Tt).
 

## Citation

```
@inproceedings{Collins20,
  title   = {Editing in Style: Uncovering the Local Semantics of {GANs}},
  author  = {Edo Collins and Raja Bala and Bob Price and Sabine S{\"u}sstrunk},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
  year = {2020},
}
```
