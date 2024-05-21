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

        search.setSearchBackwards(false);
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);

        // apply
        Instances newData = Filter.useFilter(data, filter);
        DataPreprocessing.saveAsARFF(newData,"C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\att_select_org.arff");
        return newData;
    }
    public static void main(String[] args) throws Exception{
        DataSource source = new DataSource("C:\\Users\\ttai2\\Documents\\GitHub\\Data-Mining\\data\\creditcard_scaled.arff");
        Instances dataset = source.getDataSet();
        AttriSelection(dataset);
    }
}
