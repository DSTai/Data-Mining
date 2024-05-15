package filter;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;


public class NumerictoNominal {
    public static Instances NumClassToNomClas(Instances data) throws Exception{
                // Convert numeric class attribute to nominal
        NumericToNominal filter = new NumericToNominal();
        String[] options = new String[]{"-R", "last"};
        filter.setOptions(options);
        filter.setInputFormat(data);
        Instances nData = Filter.useFilter(data, filter);  
        return nData;
    }
}
