package classification;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;


public class ClassifierRandomForest {
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
        //RANDOM FOREST
        // Create a RandomForest classifier
        RandomForest randomForest = new RandomForest();
        randomForest.buildClassifier(trainData);

        // Save the trained random forest model
        SerializationHelper.write("D:\\Documents\\Downloads\\Weka project\\models\\randomForest_model.model", randomForest);

        // Load the trained random forest model
        RandomForest loadedRandomForest = (RandomForest) SerializationHelper.read("D:\\Documents\\Downloads\\Weka project\\models\\randomForest_model.model");
        
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(loadedRandomForest, testData);

        // Print summary
        System.out.println(eval.toSummaryString("=== Summary Random Forest ===\n", false));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));


        //////////////////////////////////////////////////////////////////
        //               SECOND MODEL FOR NEW DATA                     //
        /////////////////////////////////////////////////////////////////
        // Train the randomForest model
        RandomForest randomForest2 = new RandomForest();
        randomForest2.buildClassifier(newData);

        // Save the trained model
        SerializationHelper.write("D:\\Documents\\Downloads\\Weka project\\models\\randomForest_model2.model", randomForest2);

        // Load the trained model
        RandomForest loadedRandomForest2 = (RandomForest) SerializationHelper.read("D:\\Documents\\Downloads\\Weka project\\models\\randomForest_model2.model");

        // Evaluate the model using cross-validation
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval2 = new Evaluation(newData);
        eval2.crossValidateModel(loadedRandomForest2, newData, 10, new java.util.Random(1));
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
        // Train the randomForest model
        RandomForest randomForest3 = new RandomForest();
        randomForest3.buildClassifier(orgData);

        // Save the trained model
        SerializationHelper.write("D:\\Documents\\Downloads\\Weka project\\models\\randomForest_10cfv_model.model", randomForest3);

        // Load the trained model
        RandomForest loadedRandomForest3 = (RandomForest) SerializationHelper.read("D:\\Documents\\Downloads\\Weka project\\models\\randomForest_10cfv_model.model");

        // Evaluate the model using cross-validation
        long startTime2 = System.currentTimeMillis(); // Record start time
        Evaluation eval3 = new Evaluation(orgData);
        eval3.crossValidateModel(loadedRandomForest3, orgData, 10, new java.util.Random(1));
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
