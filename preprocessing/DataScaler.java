package preprocessing;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;

public class DataScaler {

    public static double[] robustScale(double[] data) {
        double median = calculateMedian(data);
        double IQR = calculateIQR(data);
        return scaleAttribute(data, median, IQR);
    }

    private static double calculateMedian(double[] data) {
        // Sort the array
        int n = data.length;
        // If number of elements is odd, return the middle element
        if (n % 2 != 0) {
            return data[n / 2];
        } else { // If number of elements is even, return the average of the middle two elements
            return (data[(n - 1) / 2] + data[n / 2]) / 2.0;
        }
    }

    private static double calculateIQR(double[] data) {
        Percentile percentile = new Percentile();
        // Calculate the 75th percentile (Q3)
        double q3 = percentile.evaluate(data, 75.0);
        // Calculate the 25th percentile (Q1)
        double q1 = percentile.evaluate(data, 25.0);
        // Calculate the interquartile range (IQR)
        return q3 - q1;
    }

    private static double[] scaleAttribute(double[] data, double median, double IQR) {
        double[] scaledData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            scaledData[i] = (data[i] - median) / IQR;
        }
        return scaledData;
    }
}
