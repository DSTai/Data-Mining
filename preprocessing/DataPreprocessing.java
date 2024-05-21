package preprocessing;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import javax.swing.JFrame;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.statistics.HistogramType;
import org.jfree.chart.plot.PlotOrientation;

import filter.NumerictoNominal;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class DataPreprocessing {
    public static void main(String[] args) throws Exception {
        // Convert CSV to ARFF
        String arffFilePath = convertCSVtoARFF("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\creditcard.csv");
        // Read the raw dataset
        DataSource source = new DataSource(arffFilePath);
        Instances data = source.getDataSet();
        // Convert numeric class attribute to nominal
        Instances nData = NumerictoNominal.NumClassToNomClas(data);

        // Data Analysis 
        System.out.println(nData.toSummaryString());
        Instances cleanData = checkMissingValue(nData);
        performDataAnalysis(cleanData);
 
        /*
        Scaling and Distributing
        In this phase of our kernel, we will first scale the columns comprise of Time and Amount. 
        Time and amount should be scaled as the other columns. 
        */
        displayDistributionPlot(cleanData,"Time");
        displayDistributionPlot(cleanData,"Amount");

        //Scaling Time and Amount
        Instances scaledData = scaleAndAddAttributes(cleanData);

        // Splitting the Data
        Instances[] splitData = splitData(scaledData);
        Instances trainData = splitData[0];
        Instances testData = splitData[1];
        printLabelDistributions(trainData,testData);

        //shuffleAndBalanceData
        Instances subData = shuffleAndBalanceData(scaledData); //sub-sample with 50/50 ratio 
        displayDistributionPlot(subData,"Class");


        // Calculate correlation matrices
        //use the subsample - use for reference
        double[][] corrMatrix1 = CorrelationCalculator.calculateCorrelationMatrix(subData);

        //dont use for reference (high class imbalance)
        double[][] corrMatrix2 = CorrelationCalculator.calculateCorrelationMatrix(scaledData);
        
        String[] attributeNames = {
            "scaled_amount","scaled_time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",  "Class"
        };
        CorrelationHeatmap.visualize(attributeNames,corrMatrix2);
        CorrelationHeatmap.visualize(attributeNames,corrMatrix1);

        //Class Negative Correlation
        BoxPlotVisualize.visualize(subData,"V10","V12","V14","V17");

        //Class Positive Correlation
        BoxPlotVisualize.visualize(subData,"V2","V4","V11","V19");


        //Topic focus on Fraud Transaction so we check and remove Fraud values
        // Threshold = 3
        Instances rmV10Data = RemoveOutlier.removeFraudOutliers(subData,"V10");// have 3 outliers
        Instances rmV12Data = RemoveOutlier.removeFraudOutliers(subData,"V12");
        Instances rmV14Data = RemoveOutlier.removeFraudOutliers(subData,"V14");

        // Splitting the sub Data
        Instances[] splitSubData = splitData(rmV10Data);
        Instances trainSubData = splitSubData[0];
        Instances testSubData = splitSubData[1];
        printLabelDistributions(trainSubData,testSubData);  

        // Save the preprocessed data as ARFF file
        saveAsARFF(rmV10Data, "C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\subsample.arff");
        saveAsARFF(scaledData, "C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\creditcard_scaled.arff");
    //    saveAsARFF(trainData, "C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\train_data.arff");
    //    saveAsARFF(testData, "C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\test_data.arff");
    //    saveAsARFF(trainSubData, "C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\train_sub_data.arff");
    //    saveAsARFF(testSubData, "C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\test_sub_data.arff");

        ////////////////////////////////////////////////////////////////////////////
        //    After Preprocessing step, we will have 4 data files:                //
        //                                                                        //        
        //        + train_data.arff   ( 80% original data,                        //
        //        + test_data.arff      20% original data ) have same ratio:      //
        //                           No Fraud: 99.83% , Fraud: 0.17%              //
        //        + train_sub_data.arff   (sub-sample 50/50 ratio)                //
        //        + test_sub_data.arff   (sub-sample 50/50 ratio)                 //
        ////////////////////////////////////////////////////////////////////////////
    }
    
    public static Instances shuffleAndBalanceData(Instances data) {
        // Set the class index if not set
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Separate instances based on class values
        List<Instance> class0Instances = new ArrayList<>();
        List<Instance> class1Instances = new ArrayList<>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            if (instance.classValue() == 0.0) {
                class0Instances.add(instance);
            } else {
                class1Instances.add(instance);
            }
        }

        // Shuffle the instances
        Collections.shuffle(class0Instances, new Random());
        Collections.shuffle(class1Instances, new Random());

        // Take the minimum count of instances from both classes
        int minCount = Math.min(class0Instances.size(), class1Instances.size());

        // Create a balanced dataset with the same number of instances from each class
        Instances balancedData = new Instances(data, minCount * 2);
        for (int i = 0; i < minCount; i++) {
            balancedData.add(class0Instances.get(i));
            balancedData.add(class1Instances.get(i));
        }

        // Shuffle the balanced dataset
        Collections.shuffle(balancedData, new Random());

        return balancedData;
    }

    public static void printLabelDistributions(Instances trainData, Instances testData) {
        Map<Double, Integer> trainLabelDistribution = calculateLabelDistribution(trainData);
        Map<Double, Integer> testLabelDistribution = calculateLabelDistribution(testData);

        System.out.println("Label Distributions:");
        System.out.println("Train Data:");
        printDistribution(trainLabelDistribution, trainData.numInstances());
        System.out.println("Test Data:");
        printDistribution(testLabelDistribution, testData.numInstances());
    }

    public static Map<Double, Integer> calculateLabelDistribution(Instances data) {
        data.setClassIndex(data.attribute("Class").index());
        Map<Double, Integer> labelCounts = new HashMap<>();
        // Iterate over the instances in the dataset and count occurrences of each class
        for (int i = 0; i < data.numInstances(); i++) {
            // Get the class value for the current instance
            double classValue = data.instance(i).classValue();
            // Increment the count for the corresponding class
            labelCounts.put(classValue, labelCounts.getOrDefault(classValue, 0) + 1);
        }
        return labelCounts;
    }

    public static void printDistribution(Map<Double, Integer> labelCounts, int totalInstances) {
        for (Map.Entry<Double, Integer> entry : labelCounts.entrySet()) {
            double classValue = entry.getKey();
            int count = entry.getValue();
            double percentage = (double) count / totalInstances * 100;
            System.out.println("Class " + classValue + ": " + count + " instances (" + percentage + "%)");
        }
    }


    public static Instances[] splitData(Instances data) throws Exception {
        Instances[] splitData = new Instances[2];

        // Set the class index if not set
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
    
        // Separate instances based on class values
        Instances class0Instances = new Instances(data);
        Instances class1Instances = new Instances(data);
        class0Instances.delete();
        class1Instances.delete();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            if (instance.classValue() == 0.0) {
                class0Instances.add(instance);
            } else {
                class1Instances.add(instance);
            }
        }
    
        // Shuffle the instances
        class0Instances.randomize(new Random());
        class1Instances.randomize(new Random());
    
        // Calculate the split sizes based on the proportion of instances
        int trainClass0Count = (int) (class0Instances.size() * 0.8);
        int testClass0Count = class0Instances.size() - trainClass0Count;
        int trainClass1Count = (int) (class1Instances.size() * 0.8);
        int testClass1Count = class1Instances.size() - trainClass1Count;
    
        // Create the training and testing sets based on the calculated counts
        Instances trainData = new Instances(data, trainClass0Count + trainClass1Count);
        Instances testData = new Instances(data, testClass0Count + testClass1Count);
    
        // Add instances to the training set
        for (int i = 0; i < trainClass0Count; i++) {
            trainData.add(class0Instances.instance(i));
        }
        for (int i = 0; i < trainClass1Count; i++) {
            trainData.add(class1Instances.instance(i));
        }
    
        // Add instances to the testing set
        for (int i = trainClass0Count; i < class0Instances.size(); i++) {
            testData.add(class0Instances.instance(i));
        }
        for (int i = trainClass1Count; i < class1Instances.size(); i++) {
            testData.add(class1Instances.instance(i));
        }
        splitData[0] = trainData;
        splitData[1] = testData;
        // Return the split data
        return splitData;
    }

    public static Instances scaleAndAddAttributes(Instances data) {
        // Extract 'Time' and 'Amount' attributes
        double[] timeValues = data.attributeToDoubleArray(data.attribute("Time").index());
        double[] amountValues = data.attributeToDoubleArray(data.attribute("Amount").index());
    
        // Scale 'Time' and 'Amount' attributes
        double[] scaledTime = DataScaler.robustScale(timeValues);
        double[] scaledAmount = DataScaler.robustScale(amountValues);
    
        // Remove 'Time' and 'Amount' attributes from data
        data.deleteAttributeAt(data.attribute("Time").index());
        data.deleteAttributeAt(data.attribute("Amount").index());
    
        // Add scaled 'Time' and 'Amount' attributes to data
        data.insertAttributeAt(new Attribute("scaled_amount"), 0);
        data.insertAttributeAt(new Attribute("scaled_time"), 1);
    
        for (int i = 0; i < data.numInstances(); i++) {
            data.instance(i).setValue(0, scaledAmount[i]);
            data.instance(i).setValue(1, scaledTime[i]);
        }
        return data;
    }

    private static Instances checkMissingValue(Instances data) throws Exception {
        // Check for missing values
        if (hasMissingValues(data)) {
            System.out.println("Dataset contains missing values.");

            // Replace missing values with mean
            ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(data);
            data = Filter.useFilter(data, replaceMissing);
        } else {
            System.out.println("Dataset does not contain missing values.");
        }
        return data;    
    }

    private static boolean hasMissingValues(Instances data) {
    for (int i = 0; i < data.numInstances(); i++) {
        Instance instance = data.instance(i);
        for (int j = 0; j < instance.numAttributes(); j++) {
            if (instance.isMissing(j)) {
                return true;
            }
        }
    }
    return false;
    }
   
    private static void performDataAnalysis(Instances data) {
        System.out.println("Data description:");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            System.out.println("Attribute: " + attribute.name());

            // Check if the attribute is numeric
            if (attribute.isNumeric()) {
                // Compute statistics for numeric attributes
                AttributeStats stats = data.attributeStats(i);
                System.out.println("  - Mean: " + stats.numericStats.mean);
                System.out.println("  - Standard Deviation: " + stats.numericStats.stdDev);
                System.out.println("  - Minimum: " + stats.numericStats.min);
                System.out.println("  - Maximum: " + stats.numericStats.max);
            } else if (attribute.isNominal()) {
                String[] values = new String[attribute.numValues()];
                Enumeration<Object> enumeration = attribute.enumerateValues();
                int index = 0;
                while (enumeration.hasMoreElements()) {
                    values[index] = enumeration.nextElement().toString();
                    index++;
                }
                // Compute statistics for nominal attributes
                System.out.println("  - Values: " + Utils.joinOptions(values));
            }
        }
        // Calculate the percentage of each class in the dataset
        data.setClassIndex(data.numAttributes() - 1);
        int numInstances = data.numInstances();
        int numFrauds = 0;
        int numNonFrauds = 0;

        // Count occurrences of each class
        for (int i = 0; i < numInstances; i++) {
            Instance instance = data.instance(i);
            double classValue = instance.classValue();
            if (classValue == 0.0) {
                numNonFrauds++;
            } else if (classValue == 1.0) {
                numFrauds++;
            }
        }

        // Calculate percentages
        double percentNonFrauds = (double) numNonFrauds / numInstances * 100;
        double percentFrauds = (double) numFrauds / numInstances * 100;

        // Print the results
        System.out.println("Percentage of Non-Frauds: " + percentNonFrauds + "%");
        System.out.println("Percentage of Frauds: " + percentFrauds + "%"); 
        // Display the bar chart
        displayDistributionPlot(data,"Class");      
    }
    private static void displayDistributionPlot(Instances data, String attributeName) {
        // Create a histogram dataset
        HistogramDataset dataset = new HistogramDataset();
        dataset.setType(HistogramType.FREQUENCY);

        // Get the attribute index
        int attributeIndex = data.attribute(attributeName).index();

        // Add data to the dataset
        double[] values = data.attributeToDoubleArray(attributeIndex);
        dataset.addSeries(attributeName, values, 20); // 20 bins

        // Create the chart
        JFreeChart chart = ChartFactory.createHistogram(
                attributeName, 
                attributeName, 
                "Frequency", 
                dataset, 
                PlotOrientation.VERTICAL, 
                true, 
                false, 
                false);

        // Create a chart panel and display it
        ChartPanel chartPanel = new ChartPanel(chart);
        JFrame frame = new JFrame(attributeName + " Distribution Plot");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.add(chartPanel);
        frame.pack();
        frame.setVisible(true);
    }

    public static void saveAsARFF(Instances data, String filename) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(filename)); // Corrected to pass File object
        saver.writeBatch();
        System.out.println("ARFF file saved successfully: " + filename);
    }
    public static String convertCSVtoARFF(String csvFile) throws Exception {
        // Load CSV data
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvFile));
        Instances data = loader.getDataSet();

        // Save as ARFF
        String arffFilePath = csvFile.replace(".csv", ".arff");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(arffFilePath));
        saver.writeBatch();
        System.out.println("ARFF file saved successfully: " + arffFilePath);
        return arffFilePath;
    }
    
}

