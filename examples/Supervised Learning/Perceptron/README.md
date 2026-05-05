# Perceptron — Binary Note Detection on GuitarSet

## Algorithm

The **Perceptron** is a linear binary classifier. It learns a hyperplane that separates two classes by iteratively adjusting weights for every misclassified sample:

```
w ← w − η (ŷ − y) x
b ← b − η (ŷ − y)
```

where η is the learning rate, y ∈ {−1, +1} is the true label, and ŷ is the prediction.  

Despite its simplicity, the perceptron laid the foundation for more complex models like the multilayered perceptron and convolutional neural nets, and plays a huge role in advancing the field of machine learning. 

In this notebook, we will build our own perceptron and demonstrate how the perceptron learns on our Guitar dataset to determine whether a string is playing or not at a certain point in time. We will also analyze the results and any limitations, discussing how the perceptron behaves on our data and when we would potentially need more complex models.