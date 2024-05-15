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

        // I-Bk Classifier
        IBk ibk = new IBk(); // Create an instance of I-Bk
        ibk.buildClassifier(trainData);
        System.out.println("I-Bk Classifier:");
        System.out.println(ibk.getCapabilities().toString());

        // Save the trained model
        SerializationHelper.write("D:\\Documents\\Downloads\\Weka project\\models\\ibk_model.model", ibk);

        // Load the trained model
        Classifier classifier = (Classifier) SerializationHelper.read("D:\\Documents\\Downloads\\Weka project\\models\\ibk_model.model");

        // Evaluate the model
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifier, testData);
        System.out.println(eval.toSummaryString("=== Summary ===\n", false));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));

        // SECOND MODEL FOR NEW DATA
        // Train a new I-Bk model
        IBk ibk2 = new IBk();
        ibk2.buildClassifier(newData);

        // Save the trained model
        SerializationHelper.write("D:\\Documents\\Downloads\\Weka project\\models\\ibk_model2.model", ibk2);

        // Load the trained model
        IBk loadedIbk2 = (IBk) SerializationHelper.read("D:\\Documents\\Downloads\\Weka project\\models\\ibk_model2.model");

        // Evaluate the model using cross-validation
        Evaluation eval2 = new Evaluation(newData);
        eval2.crossValidateModel(loadedIbk2, newData, 10, new java.util.Random(1));

        // Print summary for the second model
        System.out.println(eval2.toSummaryString("=== Summary 2nd Model ===\n", false));
        System.out.println(eval2.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval2.toMatrixString("=== Confusion Matrix ===\n"));
    }
}
