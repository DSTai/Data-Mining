package preprocessing;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;
import weka.core.Instance;
import weka.core.Instances;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class BoxPlotVisualizer {
    public static void displayBoxPlots(Instances data, String attributeName) {
        // Create a panel with multiple subplots
        JPanel panel = new JPanel(new GridLayout(1, 2));

        // Create box plots for each class
        for (int classValue = 0; classValue <= 1; classValue++) {
            DefaultBoxAndWhiskerCategoryDataset dataset = createBoxPlotDataset(data, classValue, attributeName);
            JFreeChart chart = createBoxPlot(dataset, "Class " + classValue);
            ChartPanel chartPanel = new ChartPanel(chart);
            panel.add(chartPanel);
        }

        // Display the panel with box plots
        JFrame frame = new JFrame("Box Plot Visualizer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(panel);
        frame.pack();
        frame.setVisible(true);
    }

    private static DefaultBoxAndWhiskerCategoryDataset createBoxPlotDataset(Instances data, int classValue, String attributeName) {
        DefaultBoxAndWhiskerCategoryDataset dataset = new DefaultBoxAndWhiskerCategoryDataset();

        // Get the class attribute index
        int classIndex = data.classIndex();

        // List to store values for each class
        List<Double> values = new ArrayList<>();

        // Iterate over instances
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            // Check if the instance belongs to the specified class
            if ((int) instance.value(classIndex) == classValue) {
                // Add the specified attribute value to the list
                double value = instance.value(data.attribute(attributeName));
                values.add(value);
            }
        }

        // Add the list of values to the dataset
        dataset.add(values, "Class " + classValue, attributeName);

        return dataset;
    }

    private static JFreeChart createBoxPlot(DefaultBoxAndWhiskerCategoryDataset dataset, String title) {
        JFreeChart chart = ChartFactory.createBoxAndWhiskerChart(
                "Box Plot: " + title,
                "Class",
                "Attribute Value",
                dataset,
                true
        );

        chart.setBackgroundPaint(Color.white);
        chart.getTitle().setPaint(Color.black);

        return chart;
    }
}
