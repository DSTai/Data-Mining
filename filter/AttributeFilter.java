package filter;
import preprocessing.DataPreprocessing;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class AttributeFilter {
    public static Instances AttriFilter (Instances data) throws Exception {
        //simple filter to remove attribute
        String[] opts = new String[]{"-R", "1"};
        Remove rm = new Remove();
        rm.setOptions(opts);
        rm.setInputFormat(data);
        Instances newData = Filter.useFilter(data, rm);
        DataPreprocessing.saveAsARFF(newData,"D:\\Documents\\Downloads\\Weka project\\data\\filte_data.arff");
        return newData;
    }
    public static void main(String[] args) throws Exception{
        DataSource source = new DataSource("D:\\Documents\\Downloads\\Weka project\\data\\new_data.arff");
        Instances dataset = source.getDataSet();
        AttriFilter(dataset);
    }
}