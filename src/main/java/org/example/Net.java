package org.example;

import java.awt.Dimension;
import java.util.Random;
import javax.swing.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class Net {
    private double[][] weightsInputHidden;
    private double[] weightsHiddenOutput;
    private double[] hiddenLayer;
    private double output;

    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    private final double learningRate;
    private final Random random;

    public Net(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.random = new Random();

        this.weightsInputHidden = new double[hiddenSize][inputSize];
        this.weightsHiddenOutput = new double[hiddenSize];
        this.hiddenLayer = new double[hiddenSize];
        this.weightsHiddenOutput = new double[hiddenSize];

        initializeWeights();
    }

    private void initializeWeights() {
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weightsInputHidden[i][j] = random.nextDouble() - 0.5; // Initialize weights between -0.5 and 0.5
            }
            weightsHiddenOutput[i] = random.nextDouble() - 0.5;
        }
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public void train(double[] inputs, double target) {
        // Forward pass
        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0;
            for (int j = 0; j < inputSize; j++) {
                sum += inputs[j] * weightsInputHidden[i][j];
            }
            hiddenLayer[i] = sigmoid(sum);
        }

        double sum = 0;
        for (int i = 0; i < hiddenSize; i++) {
            sum += hiddenLayer[i] * weightsHiddenOutput[i];
        }
        output = sigmoid(sum);

        // Backpropagation with SGD
        double outputError = output - target;
        for (int i = 0; i < hiddenSize; i++) {
            double hiddenError = outputError * weightsHiddenOutput[i] * (output * (1 - output));
            for (int j = 0; j < inputSize; j++) {
                // Update weights with SGD
                weightsInputHidden[i][j] -= learningRate * hiddenError * inputs[j];
            }
            // Update weights with SGD
            weightsHiddenOutput[i] -= learningRate * outputError * hiddenLayer[i];
        }
    }

    public double predict(double[] inputs) {
        double sum = 0;
        for (int i = 0; i < hiddenSize; i++) {
            double hiddenNeuronOutput = 0;
            for (int j = 0; j < inputSize; j++) {
                hiddenNeuronOutput += inputs[j] * weightsInputHidden[i][j];
            }
            hiddenLayer[i] = sigmoid(hiddenNeuronOutput);
            sum += hiddenLayer[i] * weightsHiddenOutput[i];
        }
        output = sigmoid(sum);
        return output;
    }

    public static void main(String[] args) {
        // Example usage
        int inputSize = 1;
        int hiddenSize = 6;
        int outputSize = 2;
        double learningRate = 1.7;

        // Create the neural network
        Net nn = new Net(inputSize, hiddenSize, outputSize, learningRate);

        // Train the neural network with sine data
        int dataSize = 100;
        double[] sinX = new double[dataSize];
        double[] sinY = new double[dataSize];
        double[] predictedY = new double[dataSize];
        for (int i = 0; i < dataSize; i++) {
            sinX[i] = i * 2 * Math.PI / dataSize;
            sinY[i] = Math.sin(sinX[i]);
            nn.train(new double[]{sinX[i]}, sinY[i]);
            predictedY[i] = nn.predict(new double[]{sinX[i]});
        }

        // Plot the results
        plotData(sinX, sinY, predictedY);
    }

    public static void plotData(double[] xValues, double[] yValues, double[] predictedValues) {
        XYSeries actualSeries = new XYSeries("Actual Sine");
        XYSeries predictedSeries = new XYSeries("Predicted Sine");

        // Add actual sine values to the dataset
        for (int i = 0; i < xValues.length; i++) {
            actualSeries.add(xValues[i], yValues[i]);
        }

        // Add predicted values to the dataset
        for (int i = 0; i < xValues.length; i++) {
            predictedSeries.add(xValues[i], predictedValues[i]);
        }

        // Combine the series into a collection
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(actualSeries);
        dataset.addSeries(predictedSeries);

        // Create the chart
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Actual vs Predicted Sine",
                "X",
                "Y",
                dataset
        );

        // Display the chart in a frame
        JFrame frame = new JFrame("Sine Plot");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(800, 600));
        frame.setContentPane(chartPanel);
        frame.pack();
        frame.setVisible(true);
    }
}
