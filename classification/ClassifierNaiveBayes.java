package classification;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import preprocessing.DataPreprocessing;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.SerializationHelper;
 
public class ClassifierNaiveBayes {

    public static void main(String[] args) throws Exception {  
        DataSource sourceAtt = new DataSource("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\att_select_org.arff");
        Instances AttData = sourceAtt.getDataSet();
        AttData.setClassIndex(AttData.numAttributes() - 1);

        
        DataSource sourceAttSub = new DataSource("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\att_select_sub.arff");
        Instances AttSubData = sourceAttSub.getDataSet();
        AttSubData.setClassIndex(AttSubData.numAttributes() - 1);  

        // Splitting the sub Data
        Instances[] splitSubData = DataPreprocessing.splitData(AttData);
        Instances trainAtt = splitSubData[0];
        Instances testAtt = splitSubData[1];

        Instances[] splitSubData2 = DataPreprocessing.splitData(AttSubData);
        Instances trainAttSub = splitSubData2[0];
        Instances testAttSub = splitSubData2[1];
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
        // Naive Bayes Classifier
        NaiveBayes naiveBayes = new NaiveBayes(); // Create an instance of Naive Bayes
        naiveBayes.buildClassifier(trainAtt);
        System.out.println("Naive Bayes Classifier:");
        System.out.println(naiveBayes.getCapabilities().toString());

        // Save the trained model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\naiveBayes_model.model", naiveBayes);

        // Load the trained model
        Classifier classifier = (Classifier) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\naiveBayes_model.model");

        // Evaluate the model
        Evaluation eval = new Evaluation(trainAtt);
        eval.evaluateModel(classifier, testAtt);
        System.out.println(eval.toSummaryString("=== Summary ===\n", false));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));

        //////////////////////////////////////////////////////////////////
        //               SECOND MODEL FOR TRAIN Sub DATA                //
        //////////////////////////////////////////////////////////////////   
        // Naive Bayes Classifier
        NaiveBayes naiveBayes2 = new NaiveBayes(); // Create an instance of Naive Bayes
        naiveBayes2.buildClassifier(trainAttSub);
        System.out.println("Naive Bayes Classifier:");
        System.out.println(naiveBayes2.getCapabilities().toString());

        // Save the trained model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\naiveBayes_model2.model", naiveBayes2);

        // Load the trained model
        Classifier classifier2 = (Classifier) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\naiveBayes_model2.model");

        // Evaluate the model
        Evaluation eval2 = new Evaluation(trainAttSub);
        eval2.evaluateModel(classifier2, testAttSub);
        System.out.println(eval2.toSummaryString("=== Summary 2nd Model ===\n", false));
        System.out.println(eval2.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval2.toMatrixString("=== Confusion Matrix ===\n"));

        //////////////////////////////////////////////////////////////////
        //               3rd MODEL EVALUATION USING 10FCV              //
        /////////////////////////////////////////////////////////////////
        // Train a new Naive Bayes model
        NaiveBayes naiveBayes3 = new NaiveBayes();
        naiveBayes3.buildClassifier(trainAtt);

        // Save the trained model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\naiveBayes_10fcv1.model", naiveBayes3);

        // Load the trained model
        NaiveBayes loadedNaiveBayes3 = (NaiveBayes) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\naiveBayes_10fcv1.model");

        // Evaluate the model using cross-validation
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval3 = new Evaluation(trainAtt);
        eval3.crossValidateModel(loadedNaiveBayes3, trainAtt, 10, new java.util.Random(1));
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
        //               MODEL EVALUATION USING 10FCV                   //
        //////////////////////////////////////////////////////////////////
        // Train a new Naive Bayes model
        NaiveBayes naiveBayes4 = new NaiveBayes();
        naiveBayes4.buildClassifier(trainAttSub);

        // Save the model
        SerializationHelper.write("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\naiveBayes_10fcv2.model", naiveBayes4);

        // Load the trained model
        NaiveBayes loadedNaiveBayes4 = (NaiveBayes) SerializationHelper.read("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\models\\naiveBayes_10fcv2.model");

        // Evaluate the model using cross-validation
        long startTime1 = System.currentTimeMillis(); // Record start time
        Evaluation eval4 = new Evaluation(trainAttSub);
        eval4.crossValidateModel(loadedNaiveBayes4, trainAttSub, 10, new java.util.Random(1));
        long endTime1 = System.currentTimeMillis(); // Record end time

        // Calculate runtime
        long runtimeMillis1 = endTime1 - startTime1;
        double runtimeSeconds1 = runtimeMillis1 / 1000.0;

        // Print summary for the model evaluation
        System.out.println(eval4.toSummaryString("=== Summary 4th Model Evaluation ===\n", false));
        System.out.println(eval4.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval4.toMatrixString("=== Confusion Matrix ===\n"));  
        System.out.println("Runtime (seconds): " + runtimeSeconds1);              
    }
}
