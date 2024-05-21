package classification;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk; // Import the I-Bk classifier
import weka.core.SerializationHelper;

public class ClassifierIBk {

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
        // I-Bk Classifier
        IBk ibk = new IBk(); // Create an instance of I-Bk
        ibk.buildClassifier(trainData);
        System.out.println("I-Bk Classifier:");
        System.out.println(ibk.getCapabilities().toString());

        // Save the trained model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\ibk_model.model", ibk);

        // Load the trained model
        Classifier classifier = (Classifier) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\ibk_model.model");

        // Evaluate the model
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifier, testData);
        System.out.println(eval.toSummaryString("=== Summary ===\n", false));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));

        //////////////////////////////////////////////////////////////////
        //               SECOND MODEL FOR TRAIN sub DATA                //
        ////////////////////////////////////////////////////////////////// 
        // I-Bk Classifier
        IBk ibk2 = new IBk(); // Create an instance of I-Bk
        ibk2.buildClassifier(trainSubData);
        System.out.println("I-Bk Classifier:");

        // Save the trained model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\ibk_model2.model", ibk2);

        // Load the trained model
        Classifier classifier2 = (Classifier) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\ibk_model2.model");

        // Evaluate the model
        Evaluation eval2 = new Evaluation(trainSubData);
        eval2.evaluateModel(classifier2, testSubData);
        System.out.println(eval2.toSummaryString("=== Summary 2nd  ===\n", false));
        System.out.println(eval2.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval2.toMatrixString("=== Confusion Matrix ===\n"));

        //////////////////////////////////////////////////////////////////
        //               3rd MODEL EVALUATION USING 10FCV               //
        /////////////////////////////////////////////////////////////////
        // Train a new I-Bk model
        IBk ibk3 = new IBk();
        ibk3.buildClassifier(trainData);

        // Save the trained model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\ibk_10fcv_model.model", ibk3);

        // Load the trained model
        IBk loadedIbk3 = (IBk) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\ibk_10fcv_model.model");

        // Evaluate the model using cross-validation
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval3 = new Evaluation(trainData);
        eval3.crossValidateModel(loadedIbk3, trainData, 10, new java.util.Random(1));
        long endTime = System.currentTimeMillis(); // Record end time

        // Calculate runtime
        long runtimeMillis = endTime - startTime;
        double runtimeSeconds = runtimeMillis / 1000.0;     

        // Print summary for the second model
        System.out.println(eval3.toSummaryString("=== Summary 3rd Model Evaluation===\n", false));
        System.out.println(eval3.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval3.toMatrixString("=== Confusion Matrix ===\n")); 
        System.out.println("Runtime (seconds): " + runtimeSeconds);  

        //////////////////////////////////////////////////////////////////
        //              4th MODEL EVALUATION USING 10FCV                //
        //////////////////////////////////////////////////////////////////
        // Train a new I-Bk model
        IBk ibk4 = new IBk();
        ibk4.buildClassifier(trainSubData);

        // Save the model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\ibk_10fcv_model2.model", ibk4);

        // Load the model
        IBk loadedIbk4 = (IBk) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\ibk_10fcv_model2.model");

        // Evaluate the model using cross-validation
        long startTime2 = System.currentTimeMillis(); // Record start time
        Evaluation eval4 = new Evaluation(trainSubData);
        eval4.crossValidateModel(loadedIbk4, trainSubData, 10, new java.util.Random(1));
        long endTime2 = System.currentTimeMillis(); // Record end time

        // Calculate runtime
        long runtimeMillis2 = endTime2 - startTime2;
        double runtimeSeconds2 = runtimeMillis2 / 1000.0;

        // Print summary for the model evaluation
        System.out.println(eval4.toSummaryString("===4th Summary Model Evaluation ===\n", false));
        System.out.println(eval4.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval4.toMatrixString("=== Confusion Matrix ===\n"));  
        System.out.println("Runtime (seconds): " + runtimeSeconds2);      

    }
}
