package preprocessing;

import weka.core.Attribute;
import weka.core.Instances;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RemoveOutlier {

    public static Instances removeOutliers(Instances data, String attributeName) {
        Attribute attribute = data.attribute(attributeName);
        double[] attributeValues = data.attributeToDoubleArray(attribute.index());

        // Sort the array
        Arrays.sort(attributeValues);
        // Calculate quartiles
        double q25 = calculateQuartile(attributeValues, 0.25);
        double q75 = calculateQuartile(attributeValues, 0.75);
        double iqr = q75 - q25;

        // Define cutoff values
        double cutOff = iqr * 3;
        double lower = q25 - cutOff;
        double upper = q75 + cutOff;


        // Remove instances with outliers
        List <Double> valueToRemove = new ArrayList<>();
        for (int i = 0; i < data.numInstances(); i++) {
            double value = attributeValues[i];
            if (value < lower || value > upper) {
                valueToRemove.add(value);
            }
        }

        // Check if there are instances to remove
        if (!valueToRemove.isEmpty()) {
            // Remove instances by value
            for (double value : valueToRemove) {
                for (int i = 0; i < data.numInstances(); i++) {
                    if (data.instance(i).value(attribute) == value) {
                        data.delete(i);
                        break; 
                    }
                }
            }
        }
        

        System.out.println("Number of Instances after outliers removal: " + data.numInstances());
        System.out.println(attributeName+" Lower: " + lower);
        System.out.println(attributeName+" Upper: " + upper);
        System.out.println(attributeName+" outliers: " + formatValues(valueToRemove));
        System.out.println(attributeName+" Feature Outliers for Fraud Cases: " + valueToRemove.size());
        
        return data;
    }

    private static double calculateQuartile(double[] values, double percentile) {
        // You can implement your own method to calculate quartiles
        // For simplicity, let's assume the data is sorted
        int index = (int) (percentile * (values.length - 1));
        return values[index];
    }

    private static String formatValues(List<Double> values) {
        if (values.isEmpty()) {
            return "No outliers";
        }
    
        StringBuilder formattedValues = new StringBuilder("[");
        for (double value : values) {
            formattedValues.append(value).append(", ");
        }
        formattedValues.setLength(formattedValues.length() - 2); // Remove trailing comma and space
        formattedValues.append("]");
        return formattedValues.toString();
    }
    
}
