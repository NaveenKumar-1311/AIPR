# AIPR
Black-box Adversarial Attacks in Autonomous Vehicle Technology, published in AIPR workshop

Abstract: Despite the high quality performance of the deep
neural network in real-world applications, they are susceptible
to minor perturbations of adversarial attacks. This is mostly
undetectable to human vision. The impact of such attacks has
become extremely detrimental in autonomous vehicles with realtime “safety” concerns. The black-box adversarial attacks cause
drastic misclassification in critical scene elements such as road
signs and traffic lights leading the autonomous vehicle to crash
into other vehicles or pedestrians. In this paper, we propose a
novel query-based attack method called Modified Simple blackbox attack (M-SimBA) to overcome the use of a white-box
source in transfer based attack method. Also, the issue of late
convergence in a Simple black-box attack (SimBA) is addressed
by minimizing the loss of the most confused class which is the
incorrect class predicted by the model with the highest probability, instead of trying to maximize the loss of the correct class.
We evaluate the performance of the proposed approach to the
German Traffic Sign Recognition Benchmark (GTSRB) dataset.
We show that the proposed model outperforms the existing
models like Transfer-based projected gradient descent (T-PGD),
SimBA in terms of convergence time, flattening the distribution
of confused class probability, and producing adversarial samples
with least confidence on the true class.

If you use this work, please cite:  

```
@inproceedings{kumar2020black,
  title={Black-box adversarial attacks in autonomous vehicle technology},
  author={Kumar, K Naveen and Vishnu, Chalavadi and Mitra, Reshmi and Mohan, C Krishna},
  booktitle={2020 IEEE Applied Imagery Pattern Recognition Workshop (AIPR)},
  pages={1--7},
  year={2020},
  organization={IEEE}
}
```
