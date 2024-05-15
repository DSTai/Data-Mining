package preprocessing;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class CorrelationCalculator {

    public static double[][] calculateCorrelationMatrix(Instances data) {
        int numAttributes = data.numAttributes();
        double[][] correlationMatrix = new double[numAttributes][numAttributes];

        for (int i = 0; i < numAttributes; i++) {
            for (int j = 0; j < numAttributes; j++) {
                correlationMatrix[i][j] = calculateCorrelation(data, i, j);
            }
        }

        return correlationMatrix;
    }

    private static double calculateCorrelation(Instances data, int attr1, int attr2) {
        double sumXY = 0.0;
        double sumX = 0.0;
        double sumY = 0.0;
        double sumXSquare = 0.0;
        double sumYSquare = 0.0;
        int numInstances = data.numInstances();

        Attribute attribute1 = data.attribute(attr1);
        Attribute attribute2 = data.attribute(attr2);

        for (int i = 0; i < numInstances; i++) {
            Instance instance = data.instance(i);
            double x = instance.value(attribute1);
            double y = instance.value(attribute2);

            sumXY += x * y;
            sumX += x;
            sumY += y;
            sumXSquare += x * x;
            sumYSquare += y * y;
        }

        double numerator = numInstances * sumXY - sumX * sumY;
        double denominator = Math.sqrt((numInstances * sumXSquare - sumX * sumX) * (numInstances * sumYSquare - sumY * sumY));

        if (denominator == 0) {
            return 0.0; // Return 0 if denominator is 0 to avoid division by zero
        }

        return numerator / denominator;
    }
}
