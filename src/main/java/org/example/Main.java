package org.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
public static void main(String[] args) {
    // Example data
    double[] sinX = new double[100];
    double[] sinY = new double[100];
    for (int i = 0; i < sinX.length; i++) {
        sinX[i] = i * 2 * Math.PI / 100;
        sinY[i] = Math.sin(sinX[i]);
    }

    // Scale input values to [-1, 1] and normalize output values to [0, 1]
    double[] normSinX = new double[sinX.length];
    double[] normSinY = new double[sinY.length];
    for (int i = 0; i < sinX.length; i++) {
        normSinX[i] = (sinX[i] / Math.PI) - 1.0;
        normSinY[i] = (sinY[i] + 1.0) / 2.0;
    }

    NeuralNetwork nn = new NeuralNetwork(1, 25, 1, 0.1);
    int epochs = 10000;
    nn.train(normSinX, normSinY, epochs);

    // Test the trained model
    double[] predictedY = new double[sinX.length];
    for (int i = 0; i < normSinX.length; i++) {
        predictedY[i] = nn.forward(normSinX[i]);
        // Denormalize the predicted output
        predictedY[i] = (predictedY[i] * 2.0) - 1.0;
    }

    // Plot actual vs predicted
    nn.plotData(sinX, sinY, predictedY);
}

    private static List<double[]> readDataFromCSV(String filePath) {
        List<double[]> data = new ArrayList<>();
        String line;

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            // Skip header if present
            br.readLine();

            // Read data line by line
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                data.add(new double[]{x, y});
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return data;
    }
}
