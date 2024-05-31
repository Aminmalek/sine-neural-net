package org.example;

import javax.swing.JFrame;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.util.Random;

public class NeuralNetwork {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] hiddenWeights;
    private double[] hiddenBiases;
    private double[] outputWeights;
    private double outputBias;
    private double learningRate;
    private double[] hiddenLayerOutputs;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        hiddenWeights = new double[hiddenSize][inputSize];
        hiddenBiases = new double[hiddenSize];
        outputWeights = new double[hiddenSize];
        Random rand = new Random();

        // Initialize weights and biases
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                hiddenWeights[i][j] = rand.nextDouble();
            }
            hiddenBiases[i] = rand.nextDouble();
            outputWeights[i] = rand.nextDouble();
        }

        outputBias = rand.nextDouble();
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public double forward(double input) {
        hiddenLayerOutputs = new double[hiddenSize]; // Initializing hiddenLayerOutputs
        for (int j = 0; j < hiddenSize; j++) {
            hiddenLayerOutputs[j] = sigmoid(input * hiddenWeights[j][0] + hiddenBiases[j]);
        }

        double output = 0;
        for (int j = 0; j < hiddenSize; j++) {
            output += hiddenLayerOutputs[j] * outputWeights[j];
        }
        return sigmoid(output + outputBias);
    }

    private void backpropagate(double input, double output, double target) {
        double outputErrorSignal = (target - output) * sigmoidDerivative(output);
        for (int j = 0; j < hiddenSize; j++) {
            double hiddenErrorSignal = outputErrorSignal * outputWeights[j] * sigmoidDerivative(hiddenLayerOutputs[j]);

            outputWeights[j] += learningRate * hiddenLayerOutputs[j] * outputErrorSignal;
            hiddenWeights[j][0] += learningRate * input * hiddenErrorSignal;
            hiddenBiases[j] += learningRate * hiddenErrorSignal;
        }
        outputBias += learningRate * outputErrorSignal;
    }

    public void train(double[] xValues, double[] yValues, int epochs) {
        Random rand = new Random();
        for (int epoch = 0; epoch < epochs; epoch++) {
            double mse = 0;
            for (int i = 0; i < xValues.length; i++) {
                // Random selection of
                int randomIndex = rand.nextInt(xValues.length);
                double input = xValues[randomIndex];
                double target = yValues[randomIndex];

                // Forward pass
                double output = forward(input);

                // Compute the error
                double error = target - output;
                mse += Math.pow(error, 2);

                // Backpropagation (update weights and biases)
                backpropagate(input, output, target);
            }

            System.out.println("Epoch " + (epoch + 1) + ", MSE: " + mse);
        }
    }


    public static void plotData(double[] xValues, double[] yValues, double[] predictedValues) {
        XYSeries actualSeries = new XYSeries("Actual");
        XYSeries predictedSeries = new XYSeries("Predicted");

        for (int i = 0; i < xValues.length; i++) {
            actualSeries.add(xValues[i], yValues[i]);
            predictedSeries.add(xValues[i], predictedValues[i]);
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(actualSeries);
        dataset.addSeries(predictedSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Actual vs Predicted",
                "X-Value",
                "Y-Value",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        XYPlot plot = (XYPlot) chart.getPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesLinesVisible(0, true);
        renderer.setSeriesShapesVisible(0, false);
        renderer.setSeriesLinesVisible(1, true);
        renderer.setSeriesShapesVisible(1, false);
        plot.setRenderer(renderer);

        JFrame frame = new JFrame("Neural Network Results");
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 600));
        frame.setContentPane(chartPanel);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

}
