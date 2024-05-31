import org.example.NeuralNetwork;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class NeuralNetworkTest {

    @Test
    public void testSpecificSineValues() {
        NeuralNetwork nn = new NeuralNetwork(1, 20, 1, 0.1);
        int epochs = 10000;

        // Input values for sine functions
        double[] xValues = {1, 2, 3, 4};
        // Corresponding sine values
        double[] yValues = {Math.sin(Math.toRadians(30)), Math.sin(Math.toRadians(90)),
                            Math.sin(Math.toRadians(180)), Math.sin(Math.toRadians(45))};

        // Scale input values to [-1, 1] and normalize output values to [0, 1]
        for (int i = 0; i < xValues.length; i++) {
            xValues[i] = (xValues[i] / 180.0) - 1.0;
            yValues[i] = (yValues[i] + 1.0) / 2.0;
        }

        nn.train(xValues, yValues, epochs);

        // Test the trained model
        double[] predictedY = new double[xValues.length];
        for (int i = 0; i < xValues.length; i++) {
            predictedY[i] = nn.forward(xValues[i]);
            // Denormalize the predicted output
            predictedY[i] = (predictedY[i] * 2.0) - 1.0;
        }

        // Assert that the predicted values are close to the actual values
        double tolerance = 0.1; // Tolerance for difference between predicted and actual values
        for (int i = 0; i < xValues.length; i++) {
            assertEquals(yValues[i], predictedY[i], tolerance);
        }
    }
}
