package preprocessing;

import java.text.DecimalFormat;

public class CorrelationHeatmap {

    public static void visualize(String[] attributeNames, double[][] corrMatrix) {
        DecimalFormat df = new DecimalFormat("#.####");

        System.out.println("Correlation Heatmap:");
        System.out.println("---------------------");
        System.out.printf("%-7.5s", "Attribute");

        // Print column headers
        for (String attributeName : attributeNames) {
            System.out.printf("%-7.5s", attributeName);
        }
        System.out.println();

        // Print correlation matrix
        for (int i = 0; i < attributeNames.length; i++) {
            System.out.printf("%-7.5s", attributeNames[i]);
            for (int j = 0; j < attributeNames.length; j++) {
                System.out.printf("%-7.5s", df.format(corrMatrix[i][j]));
            }
            System.out.println();
        }
    }
}
