package classification;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.SerializationHelper;

public class ClassifierJ48 {

    public static void main(String[] args) throws Exception {

        // Load the ARFF file for training data
        DataSource sourceTrain = new DataSource("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\train_data.arff");
        Instances trainData = sourceTrain.getDataSet();
        trainData.setClassIndex(trainData.numAttributes() - 1);

        // Load the ARFF file for test data
        DataSource sourceTest = new DataSource("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\test_data.arff");
        Instances testData = sourceTest.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);

        // Load the ARFF file for train sub data
        DataSource sourcetrainSub = new DataSource("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\train_sub_data.arff");
        Instances trainSubData = sourcetrainSub.getDataSet();
        trainSubData.setClassIndex(trainSubData.numAttributes() - 1);

        // Load the ARFF file for test sub data
        DataSource sourcetestSub = new DataSource("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\test_sub_data.arff");
        Instances testSubData = sourcetestSub.getDataSet();
        testSubData.setClassIndex(testSubData.numAttributes() - 1); 
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
        File modelDir = new File("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models");
        if (!modelDir.exists()) {
            modelDir.mkdirs();
        }
        // Save the trained model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\treeJ48_model.model", tree);
   
        // Load the trained model
        Classifier classifier = (Classifier) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\J48_model.model");

        // Evaluate the model
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifier, testData);
        System.out.println(eval.toSummaryString("=== Summary ===\n", false));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));

        //////////////////////////////////////////////////////////////////
        //               SECOND MODEL FOR TRAIN Sub DATA                //
        //////////////////////////////////////////////////////////////////         
         //TREE J48
         J48 tree2 =  new J48();
         tree2.buildClassifier(trainSubData);
         System.out.println("Tree J48 Classifier:");
         System.out.println(tree2.getCapabilities().toString());
         System.out.println(tree2.graph());
 
         // Save the trained model
         SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\treeJ48_model2.model", tree2);
    
         // Load the trained model
         Classifier classifier2 = (Classifier) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\J48_model2.model");
 
         // Evaluate the model
         Evaluation eval2 = new Evaluation(trainSubData);
         eval2.evaluateModel(classifier2, testSubData);
         System.out.println(eval2.toSummaryString("=== Summary 2nd Model ===\n", false));
         System.out.println(eval2.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
         System.out.println(eval2.toMatrixString("=== Confusion Matrix ===\n"));
        //////////////////////////////////////////////////////////////////
        //               3rd MODEL EVALUATION USING 10FCV              //
        /////////////////////////////////////////////////////////////////
        // Train the J48 model
        J48 tree3 = new J48();
        tree3.buildClassifier(trainData);
        System.out.println("Tree J48 Classifier:");
        System.out.println(tree3.graph());
        // Save the trained model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\J48_10fcv_model.model", tree3);

        // Load the trained model
        J48 loadedTree3 = (J48) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\J48_10fcv_model.model");

        // Evaluate the model using cross-validation
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval3 = new Evaluation(trainData);
        eval3.crossValidateModel(loadedTree3, trainData, 10, new java.util.Random(1));
        long endTime = System.currentTimeMillis(); // Record end time

        // Calculate runtime
         long runtimeMillis = endTime - startTime;
         double runtimeSeconds = runtimeMillis / 1000.0;       
        // Print summary
        System.out.println(eval3.toSummaryString("=== Summary 3rd Model Evaluation===\n", false));
        System.out.println(eval3.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval3.toMatrixString("=== Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds);
        
        //////////////////////////////////////////////////////////////////
        //              4th MODEL EVALUATION USING 10FCV                //
        //////////////////////////////////////////////////////////////////
        // Train the J48 model
        J48 tree4 = new J48();
        tree4.buildClassifier(trainSubData);
        System.out.println("Tree J48 Classifier:");
        System.out.println(tree4.graph());
        // Save the trained model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\J48_10fcv_model2.model", tree4);

        // Load the trained model
        J48 loadedTree4 = (J48) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\J48_10fcv_model2.model");

        // Evaluate the model using cross-validation
        long startTime2 = System.currentTimeMillis(); // Record start time
        Evaluation eval4 = new Evaluation(trainSubData);
        eval4.crossValidateModel(loadedTree4, trainSubData, 10, new java.util.Random(1));
        long endTime2 = System.currentTimeMillis(); // Record end time

        // Calculate runtime
        long runtimeMillis2 = endTime2 - startTime2;
        double runtimeSeconds2 = runtimeMillis2 / 1000.0;

        // Print summary for the model evaluation
        System.out.println(eval4.toSummaryString("=== Summary 4th Model Evaluation ===\n", false));
        System.out.println(eval4.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval4.toMatrixString("=== Confusion Matrix ===\n"));  
        System.out.println("Runtime (seconds): " + runtimeSeconds2);   
     }
}
