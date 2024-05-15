package filter;

import preprocessing.DataPreprocessing;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class AttrSelection {
    public static Instances AttriSelection (Instances data) throws Exception {
        AttributeSelection filter = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();

        search.setSearchBackwards(true);
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);

        // apply
        Instances newData = Filter.useFilter(data, filter);
        DataPreprocessing.saveAsARFF(newData,"D:\\Documents\\Downloads\\Weka project\\data\\attselect_data.arff");
        return newData;
    }
    public static void main(String[] args) throws Exception{
        DataSource source = new DataSource("D:\\Documents\\Downloads\\Weka project\\data\\new_data.arff");
        Instances dataset = source.getDataSet();
        AttriSelection(dataset);
    }
}
