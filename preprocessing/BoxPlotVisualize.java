package preprocessing;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.renderer.category.BoxAndWhiskerRenderer;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;
import weka.core.Instance;
import weka.core.Instances;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class BoxPlotVisualize {
    public static void visualize(Instances data, String... attributeName) {
        DefaultBoxAndWhiskerCategoryDataset dataset = createBoxPlotDataset(data, attributeName);
        JFreeChart chart = createBoxPlot(dataset, "Box Plot: ");
       // ((CategoryPlot) chart.getPlot()).getRangeAxis().setFixedDimension(300); 
        JFrame frame = new JFrame("Box Plot Visualizer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(new ChartPanel(chart), BorderLayout.CENTER);
        frame.pack();
        frame.setVisible(true);
    }

    private static DefaultBoxAndWhiskerCategoryDataset createBoxPlotDataset(Instances data, String... attributeNames) {
        DefaultBoxAndWhiskerCategoryDataset dataset = new DefaultBoxAndWhiskerCategoryDataset();

        for (String attributeName : attributeNames) {
            List<Double> class0Values = new ArrayList<>();
            List<Double> class1Values = new ArrayList<>();

            for (int i = 0; i < data.numInstances(); i++) {
                Instance instance = data.instance(i);
                double value = instance.value(data.attribute(attributeName));
                int classIndex = (int) instance.value(data.classAttribute());

                if (classIndex == 0) {
                    class0Values.add(value);
                } else if (classIndex == 1) {
                    class1Values.add(value);
                }
            }

            dataset.add(class0Values, "Class 0", attributeName);
            dataset.add(class1Values, "Class 1", attributeName);
        }
        return dataset;
    }
    
    private static JFreeChart createBoxPlot(DefaultBoxAndWhiskerCategoryDataset dataset, String title) {
        CategoryAxis xAxis = new CategoryAxis("Class");
        NumberAxis yAxis = new NumberAxis("Value");
        BoxAndWhiskerRenderer renderer = new BoxAndWhiskerRenderer();
    
        CategoryPlot plot = new CategoryPlot(dataset, xAxis, yAxis, renderer);
        JFreeChart chart = new JFreeChart(title, JFreeChart.DEFAULT_TITLE_FONT, plot, true);
    return chart;    
    }
}
