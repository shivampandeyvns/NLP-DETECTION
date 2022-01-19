
# MultiLingual Undesired Speech Detection

This is a Natural Language Processing project which detects hate speech
on Online forums. This project uses Support Vector Machine(SVM) for the classiification 
of the hate speech. Since this is a tpyical Machine Learning problem so Python will be our language of development.
Our Front-end GUI is developed using streamlit which is the topmost choice of all machine Learning problem.
This model can be used by any online forums and discussions websites so that a clean and disciplined environment 
can be maintained.


![Logo](https://i.ibb.co/kM9dBJ6/logo.jpg)


## Features

- Easy to use GUI
- Supports Multiple Language
- Uses Machine Learning for detection
- Cross platform


## Technologies Used
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)]()
[![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)]()
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)]()
[![streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)]()

## Dataset
The Dataset for the model development contains more than 50k comments lablelled as fine or toxic.
This is same for both english and hindi language. 
We had to clean the data to remove the special characters and include padding to maintain a certain length.
Then Using the nltk library we used countVectorizer for the vectorization of our text data.
Further using the data we trained the SVM model and deployed the app using streamlit cloud.


## Machine Learning Model
Support Vector Machine(SVM) is a supervised machine Learning algorithm used for classification problems.
SVM is also called as large margin classifiers.
The objective of SVM algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points. The dimension of the hyperplane depends upon the number of features. If the number of input features is two, then the hyperplane is just a line. If the number of input features is three, then the hyperplane becomes a 2-D plane. It becomes difficult to imagine when the number of features exceeds three.

![Logo](https://static.javatpoint.com/tutorial/machine-learning/images/support-vector-machine-algorithm.png)

## Video Demonstration

https://user-images.githubusercontent.com/77262548/150108802-727ce95e-0f8f-40bb-a0d7-846cb2ad91ec.mp4


## Authors

- [@Mayank Sinha](https://github.com/willywonka32)
- [@Amit Prakash](https://github.com/amitpr07)
- [@Shivam Pandey](https://github.com/shivampandeyvns)
- [@Vibhav Bhriguvanshi](https://github.com/vibhav0710)
