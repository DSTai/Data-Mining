package preprocessing;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RemoveOutlier {

    public static Instances removeFraudOutliers(Instances data, String attributeName) {
        Attribute attribute = data.attribute(attributeName);

        // Filter instances with class label 1
        List<Double> classFraudValues = new ArrayList<>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            double value = instance.value(attribute);
            int classIndex = (int) instance.value(data.classAttribute());

            if (classIndex == 1) {
                classFraudValues.add(value);
            }
        }

        // Sort the array
        double[] sortedFraudValues = classFraudValues.stream().mapToDouble(Double::doubleValue).toArray();
        Arrays.sort(sortedFraudValues);
        // Calculate quartiles
        double q25 = calculateQuartile(sortedFraudValues, 0.25);
        double q75 = calculateQuartile(sortedFraudValues, 0.75);
        double iqr = q75 - q25;

        // Define cutoff values
        double cutOff = iqr * 3;
        double lower = q25 - cutOff;
        double upper = q75 + cutOff;


        // Remove instances with outliers
        List <Double> valueToRemove = new ArrayList<>();
        for (double value : sortedFraudValues) {
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
        

        System.out.println("Number of Instances after (Fraud Transactions) outliers removal: " + data.numInstances());
        System.out.println(attributeName+" Lower: " + lower);
        System.out.println(attributeName+" Upper: " + upper);
        System.out.println(attributeName+" outliers: " + formatValues(valueToRemove));
        System.out.println(attributeName+" Feature Outliers for Fraud Cases: " + valueToRemove.size());
        
        return data;
    }

    private static double calculateQuartile(double[] values, double percentile) {
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
        formattedValues.setLength(formattedValues.length() - 2); 
        formattedValues.append("]");
        return formattedValues.toString();
    }
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
}
