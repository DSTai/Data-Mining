package classification;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes; // Import the Naive Bayes classifier
import weka.core.SerializationHelper;

public class ClassifierNaiveBayes {

    public static void main(String[] args) throws Exception {
        // Load the ARFF file for training data
        DataSource sourceTrain = new DataSource("D:\\Documents\\Downloads\\Weka project\\data\\train_data.arff");
        Instances trainData = sourceTrain.getDataSet();
        trainData.setClassIndex(trainData.numAttributes() - 1);

        // Load the ARFF file for test data
        DataSource sourceTest = new DataSource("D:\\Documents\\Downloads\\Weka project\\data\\test_data.arff");
        Instances testData = sourceTest.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);

        // Load the ARFF file for new data
        DataSource sourceNew = new DataSource("D:\\Documents\\Downloads\\Weka project\\data\\new_data.arff");
        Instances newData = sourceNew.getDataSet();
        newData.setClassIndex(newData.numAttributes() - 1);

        // Naive Bayes Classifier
        NaiveBayes naiveBayes = new NaiveBayes(); // Create an instance of Naive Bayes
        naiveBayes.buildClassifier(trainData);
        System.out.println("Naive Bayes Classifier:");
        System.out.println(naiveBayes.getCapabilities().toString());

        // Save the trained model
        SerializationHelper.write("D:\\Documents\\Downloads\\Weka project\\models\\naiveBayes_model.model", naiveBayes);

        // Load the trained model
        Classifier classifier = (Classifier) SerializationHelper.read("D:\\Documents\\Downloads\\Weka project\\models\\naiveBayes_model.model");

        // Evaluate the model
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifier, testData);
        System.out.println(eval.toSummaryString("=== Summary ===\n", false));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));

        // SECOND MODEL FOR NEW DATA
        // Train a new Naive Bayes model
        NaiveBayes naiveBayes2 = new NaiveBayes();
        naiveBayes2.buildClassifier(newData);

        // Save the trained model
        SerializationHelper.write("D:\\Documents\\Downloads\\Weka project\\models\\naiveBayes_model2.model", naiveBayes2);

        // Load the trained model
        NaiveBayes loadedNaiveBayes2 = (NaiveBayes) SerializationHelper.read("D:\\Documents\\Downloads\\Weka project\\models\\naiveBayes_model2.model");

        // Evaluate the model using cross-validation
        Evaluation eval2 = new Evaluation(newData);
        eval2.crossValidateModel(loadedNaiveBayes2, newData, 10, new java.util.Random(1));

        // Print summary for the second model
        System.out.println(eval2.toSummaryString("=== Summary 2nd Model ===\n", false));
        System.out.println(eval2.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval2.toMatrixString("=== Confusion Matrix ===\n"));
    }
}
