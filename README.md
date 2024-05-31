# Neural Network

A simple neural network implementation with a single hidden layer, capable of training on a dataset and visualizing the results using JFreeChart.

## Features

- Single hidden layer neural network
- Sigmoid activation function
- Backpropagation for training
- Mean Squared Error (MSE) for loss calculation
- Visualization of actual vs. predicted values using JFreeChart

## Requirements

- Java Development Kit (JDK) 8 or higher
- JFreeChart library

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Aminmalek/sine-neural-net
   cd 
   mvn clean install
   mvn compile exec:java -Dexec.mainClass="org.example.Main"


## Usage
Example Code
Here is an example usage of the NeuralNetwork class:
   ```java
package org.example;

public class Main {
    public static void main(String[] args) {
        // Define the neural network parameters
        int inputSize = 1;
        int hiddenSize = 10;
        int outputSize = 1;
        double learningRate = 0.01;
        int epochs = 1000;

        // Create the neural network
        NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSize, outputSize, learningRate);

        // Generate training data
        double[] xValues = new double[100];
        double[] yValues = new double[100];
        for (int i = 0; i < 100; i++) {
            xValues[i] = i / 10.0;
            yValues[i] = Math.sin(xValues[i]);
        }

        // Train the neural network
        nn.train(xValues, yValues, epochs);

        // Generate predictions
        double[] predictedValues = new double[xValues.length];
        for (int i = 0; i < xValues.length; i++) {
            predictedValues[i] = nn.forward(xValues[i]);
        }

        // Plot the results
        NeuralNetwork.plotData(xValues, yValues, predictedValues);
    }
}
```
Build and Run the Project: Compile and run the project using Maven. Follow the instructions in the README file.

View Results: Once the program finishes execution, you'll see a visualization comparing the actual sine function values with the predicted values.

### Why This Project?
This project is designed for educational purposes, aiming to provide a hands-on understanding of neural networks, backpropagation, and gradient descent. By implementing a simple neural network from scratch and training it to learn the sine function, users can gain insights into the inner workings of neural networks and deepen their understanding of machine learning concepts.


