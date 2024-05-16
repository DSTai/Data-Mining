package classification;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.SerializationHelper;

public class ClassifierTreeJ48 {

    public static void main(String[] args) throws Exception {
        // Load the ARFF file for original data for cross-validation
        DataSource sourceOrg = new DataSource("D:\\Documents\\Downloads\\Weka project\\data\\creditcard_scaled.arff");
        Instances orgData = sourceOrg.getDataSet();
        orgData.setClassIndex(orgData.numAttributes() - 1);   
          
        // Load the ARFF file
        DataSource sourceTrain = new DataSource("D:\\Documents\\Downloads\\Weka project\\data\\train_data.arff");
        Instances trainData = sourceTrain.getDataSet();
        //set class attribute
        trainData.setClassIndex(trainData.numAttributes()-1);

        DataSource sourceTest = new DataSource("D:\\Documents\\Downloads\\Weka project\\data\\test_data.arff");
        Instances testData = sourceTest.getDataSet();
        //set class attribute
        testData.setClassIndex(testData.numAttributes()-1); 

        DataSource source = new DataSource("D:\\Documents\\Downloads\\Weka project\\data\\new_data.arff");
        Instances newData = source.getDataSet();
        //set class attribute
        newData.setClassIndex(newData.numAttributes()-1);
        //////////////////////////////////////////////////////////////////
        //               FIRST MODEL FOR TRAIN DATA                     //
        //////////////////////////////////////////////////////////////////         
         //TREE J48
        J48 tree =  new J48();
        tree.buildClassifier(trainData);
        System.out.println("Tree J48 Classifier:");
        System.out.println(tree.getCapabilities().toString());
        System.out.println(tree.graph());

         // Create the directory if it doesn't exist
        File modelDir = new File("D:\\Documents\\Downloads\\Weka project\\models");
        if (!modelDir.exists()) {
            modelDir.mkdirs();
        }
        // Save the trained model
        SerializationHelper.write("D:\\Documents\\Downloads\\Weka project\\models\\treeJ48_model.model", tree);
   
        // Load the trained model
        Classifier classifier = (Classifier) SerializationHelper.read("D:\\Documents\\Downloads\\Weka project\\models\\treeJ48_model.model");

        // Evaluate the model
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifier, testData);
        System.out.println(eval.toSummaryString("=== Summary ===\n", false));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));

        //////////////////////////////////////////////////////////////////
        //               SECOND MODEL FOR NEW DATA                     //
        /////////////////////////////////////////////////////////////////
        // Train the J48 model
        J48 tree2 = new J48();
        tree2.buildClassifier(newData);
        System.out.println("Tree J48 Classifier:");
        System.out.println(tree2.graph());
        // Save the trained model
        SerializationHelper.write("D:\\Documents\\Downloads\\Weka project\\models\\treeJ48_model2.model", tree2);

        // Load the trained model
        J48 loadedTree2 = (J48) SerializationHelper.read("D:\\Documents\\Downloads\\Weka project\\models\\treeJ48_model2.model");

        // Evaluate the model using cross-validation
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval2 = new Evaluation(newData);
        eval2.crossValidateModel(loadedTree2, newData, 10, new java.util.Random(1));
        long endTime = System.currentTimeMillis(); // Record end time

        // Calculate runtime
         long runtimeMillis = endTime - startTime;
         double runtimeSeconds = runtimeMillis / 1000.0;       
        // Print summary
        System.out.println(eval2.toSummaryString("=== Summary 2nd Model ===\n", false));
        System.out.println(eval2.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval2.toMatrixString("=== Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds);
        
        //////////////////////////////////////////////////////////////////
        //               MODEL EVALUATION USING 10FCV                   //
        //////////////////////////////////////////////////////////////////
        // Train the J48 model
        J48 tree3 = new J48();
        tree3.buildClassifier(orgData);
        System.out.println("Tree J48 Classifier:");
        System.out.println(tree3.graph());
        // Save the trained model
        SerializationHelper.write("D:\\Documents\\Downloads\\Weka project\\models\\treeJ48_10fcv_model.model", tree3);

        // Load the trained model
        J48 loadedTree3 = (J48) SerializationHelper.read("D:\\Documents\\Downloads\\Weka project\\models\\treeJ48_10fcv_model.model");

        // Evaluate the model using cross-validation
        long startTime2 = System.currentTimeMillis(); // Record start time
        Evaluation eval3 = new Evaluation(orgData);
        eval3.crossValidateModel(loadedTree3, orgData, 10, new java.util.Random(1));
        long endTime2 = System.currentTimeMillis(); // Record end time

        // Calculate runtime
        long runtimeMillis2 = endTime2 - startTime2;
        double runtimeSeconds2 = runtimeMillis2 / 1000.0;

        // Print summary for the model evaluation
        System.out.println(eval3.toSummaryString("=== Summary Model Evaluation ===\n", false));
        System.out.println(eval3.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval3.toMatrixString("=== Confusion Matrix ===\n"));  
        System.out.println("Runtime (seconds): " + runtimeSeconds2);   
     }
}
