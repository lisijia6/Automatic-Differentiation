# Team01

[![.github/workflows/test.yml](https://code.harvard.edu/CS107/team01/actions/workflows/test.yml/badge.svg)](https://code.harvard.edu/CS107/team01/actions/workflows/test.yml)
[![.github/workflows/coverage.yml](https://code.harvard.edu/CS107/team01/actions/workflows/coverage.yml/badge.svg)](https://code.harvard.edu/CS107/team01/actions/workflows/coverage.yml)

The full documentation with examples is [here](https://code.harvard.edu/pages/CS107/team01/).

# Installation
## User Installation Guide
### Manual Installation
We publish `AutoDiff` on the testPyPI, and user can simply install `AutoDiff` and its dependency by the following command

```shell
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple AutoDiff-Team01
```

We have additional feature to visualize computation graph. To use this functionality user can install graphviz [here](https://graphviz.org/download/) and install python wrapper by the following command 
```
pip install graphviz
```
### Installation Using Custom Docker
Alternatively, users can skip the above installation (for example if graphviz is not available in the machine). Users can download the Docker file named `Dockerfile` in our repo. Inside the directoy that contains the `Dockerfile`, users can build docker container, demonstrated by the following commands

```shell
# go to the directory that contains our Dockerfile
cd <directory with Dockerfile>
docker build .
# for example, if the above command ends with
#    Successfully built d8b31c5835d6  
# the container id should be d8b31c5835d6
docker run -it <container id>
# then you can use our package in docker!
```

### Installation Using Our Docker
We also build the container and provide the built container that contains all depdendencies and AutoDiff on docker hub: [13052423200/autodiff](https://hub.docker.com/repository/docker/13052423200/autodiff). Please run 
```shell
docker pull 13052423200/autodiff
# then you can use our package in docker!
```

## Developer Installation Guide

Developers can git clone our repo and install from the source using the following command
```shell
git clone git@code.harvard.edu:CS107/team01.git
```

Then excute the following command
```shell
cd team01
# for main functionalty
pip install .
```
If developers would like to  explore additional features such as computational graph plotting, as well as testing:

```shell
# for all features including test and plot
pip install .[all]
```

Developers can also build or pull docker, please see user Installation guide above.

# Usage
After the installation, users can now explore functionalities of `AutoDiff`!

Please see examples of how to use our package in the `example` folder. Forward Mode AD and Reverse Mode AD examples are given in `forward_mode.py` and `reverse_mode.py` respectively. Computation graph example is given in `plot_graph.py`. And driver code for optimization methods is provided in `newton.py` and `sgd.py` corresponding to Newton's Method and Stochastic Gradient Descent respectively.

For more examples, please refer to `docs/documentation.pdf` section **3.0 How to Use the `AutoDiff` Package**, or view the full documentation with examples [here](https://code.harvard.edu/pages/CS107/team01/).

For developers, checkout `test` folder for more examples and usages.

# Broader Impact 
The potential broader impacts and implications of AutoDiff could be significant, as it has the potential to make it easier for researchers and developers to implement automatic differentiation in their own projects. Automatic differentiation is a powerful tool for optimizing machine learning algorithms, which can have a wide range of applications in various fields. 

One potential way that people could use AutoDiff responsibly is by using it to improve the performance of machine learning algorithms in a way that is transparent and explainable. For example, if AutoDiff is used to optimize a predictive model in healthcare, finance, or environmental science, the results of the optimization should be clearly communicated and explained to clinicians and patients, so that they can understand the model's predictions and make informed decisions.

On the other hand, there could be potential ethical implications if AutoDiff is used irresponsibly or without proper oversight. For example, if a predictive model optimized with AutoDiff is used to make decisions that have a significant impact on people's lives, such as in hiring or loan applications, it is important that the model is fair and unbiased. If the model is not properly validated or checked for bias, it could have negative consequences for the individuals who are affected by its decisions.

Overall, it is important for users of AutoDiff to use the software responsibly and take into account the potential broader impacts and ethical implications of their work. By doing so, they can help to ensure that the technology is used for the benefit of society, rather than causing harm.


# Inclusivity Statement

In terms of inclusivity, one potential way that AutoDiff could be inclusive to the broader community is by making automatic differentiation more accessible to a wider range of users. Currently, automatic differentiation can be difficult to implement, especially for users who do not have a strong background in mathematics or computer science. By providing an easy-to-use software package, AutoDiff could make automatic differentiation more accessible to a broader range of users, including those from underrepresented groups. This could help to promote diversity and inclusion in the fields of machine learning and artificial intelligence.

Additionally, AutoDiff could be inclusive by providing documentation and user support in multiple languages, which would make it more accessible to users who speak languages other than English. In the future we plan to support more languages such as Chinese and Spanish. This could be especially useful for users in non-English speaking countries, where access to resources and support for machine learning and artificial intelligence can be more limited. By providing support in multiple languages, AutoDiff could help to break down language barriers and make it easier for users from diverse backgrounds to access and use the software.

For any developers who would like to build on top of the `AutoDiff` package, they can fork the `AutoDiff` GitHub repository. If they find an issue with the implementation of the `AutoDiff` package or have any suggestions for improvements, they can send a pull request from the forked GitHub respository. All developers of the `AutoDiff` package (i.e., our AC207 project team) will review the pull request and decide whether or not to integrate the code change into the `AutoDiff` package source code. Once all team members review the pull request and confirm that there are no more issues, one of the team members will approve the pull request and merge the pull request into the `main` branch for `AutoDiff`.
