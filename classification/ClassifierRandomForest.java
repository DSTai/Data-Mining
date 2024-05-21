package classification;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;


public class ClassifierRandomForest {
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
        //RANDOM FOREST
        // Create a RandomForest classifier
        RandomForest randomForest = new RandomForest();
        randomForest.buildClassifier(trainData);

        // Save the trained random forest model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\randomForest_model.model", randomForest);

        // Load the trained random forest model
        RandomForest loadedRandomForest = (RandomForest) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\randomForest_model.model");
        
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(loadedRandomForest, testData);

        // Print summary
        System.out.println(eval.toSummaryString("=== Summary Random Forest ===\n", false));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));

        //////////////////////////////////////////////////////////////////
        //               SECOND MODEL FOR TRAIN Sub DATA                //
        ////////////////////////////////////////////////////////////////// 
        //RANDOM FOREST
        // Create a RandomForest classifier
        RandomForest randomForest2 = new RandomForest();
        randomForest2.buildClassifier(trainSubData);

        // Save the trained random forest model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\randomForest_model2.model", randomForest2);

        // Load the trained random forest model
        RandomForest loadedRandomForest2 = (RandomForest) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\randomForest_model2.model");
        
        Evaluation eval2 = new Evaluation(trainSubData);
        eval2.evaluateModel(loadedRandomForest2, testSubData);

        // Print summary
        System.out.println(eval2.toSummaryString("=== Summary 2nd Model Random Forest ===\n", false));
        System.out.println(eval2.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval2.toMatrixString("=== Confusion Matrix ===\n"));
        //////////////////////////////////////////////////////////////////
        //               3rd MODEL EVALUATION USING 10FCV               //
        /////////////////////////////////////////////////////////////////
        // Train the randomForest model
        RandomForest randomForest3 = new RandomForest();
        randomForest3.buildClassifier(trainData);

        // Save the trained model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\randomForest_10cfv_model.model", randomForest3);

        // Load the trained model
        RandomForest loadedRandomForest3 = (RandomForest) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\randomForest_10cfv_model.model");

        // Evaluate the model using cross-validation
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval3 = new Evaluation(trainData);
        eval3.crossValidateModel(loadedRandomForest3, trainData, 10, new java.util.Random(1));
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
        //             4th  MODEL EVALUATION USING 10FCV                //
        //////////////////////////////////////////////////////////////////
        // Train the randomForest model
        RandomForest randomForest4 = new RandomForest();
        randomForest4.buildClassifier(trainSubData);

        // Save the trained model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\randomForest_10cfv_model2.model", randomForest4);

        // Load the trained model
        RandomForest loadedRandomForest4 = (RandomForest) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\randomForest_10cfv_model2.model");

        // Evaluate the model using cross-validation
        long startTime2 = System.currentTimeMillis(); // Record start time
        Evaluation eval4 = new Evaluation(trainSubData);
        eval4.crossValidateModel(loadedRandomForest4, trainSubData, 10, new java.util.Random(1));
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
