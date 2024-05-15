package classification;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassifierRandomForest {
    public static void main(String[] args) throws Exception {
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
        Evaluation eval2 = new Evaluation(newData);
        eval2.crossValidateModel(loadedRandomForest2, newData, 10, new java.util.Random(1));

        // Print summary
        System.out.println(eval2.toSummaryString("=== Summary 2nd Model ===\n", false));
        System.out.println(eval2.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval2.toMatrixString("=== Confusion Matrix ===\n"));    
    }
    
}
