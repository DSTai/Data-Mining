package preprocessing;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import javax.swing.JPanel;

public class BarChart extends JPanel {
    private int numNonFrauds;
    private int numFrauds;
    private Color nonFraudColor;
    private Color fraudColor;

    public BarChart(int numNonFrauds, int numFrauds) {
        this.numNonFrauds = numNonFrauds;
        this.numFrauds = numFrauds;
        this.nonFraudColor = Color.BLUE;
        this.fraudColor = Color.RED;
        setPreferredSize(new Dimension(600, 500));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
    
        int total = numNonFrauds + numFrauds;
        int barWidth = getWidth() / 4;
        int nonFraudBarHeight = (int) ((double) numNonFrauds / total * getHeight());
        int fraudBarHeight = (int) ((double) numFrauds / total * getHeight());
    
        // Draw non-fraud bar
        g.setColor(nonFraudColor);
        g.fillRect(barWidth, getHeight() - nonFraudBarHeight, barWidth, nonFraudBarHeight);
        g.setColor(Color.BLACK);
        g.drawString(Integer.toString(numNonFrauds), barWidth + barWidth / 2 - 10, getHeight() - nonFraudBarHeight - 5);
        g.drawString("Non-Fraud", barWidth, getHeight() - 10);
    
        // Draw fraud bar
        g.setColor(fraudColor);
        g.fillRect(barWidth * 2, getHeight() - fraudBarHeight, barWidth, fraudBarHeight);
        g.setColor(Color.BLACK);
        g.drawString(Integer.toString(numFrauds), barWidth * 2 + barWidth / 2 - 10, getHeight() - fraudBarHeight - 5);
        g.drawString("Fraud", barWidth * 2, getHeight() - 10);
    
        // Draw x-axis label
        g.setFont(new Font("Arial", Font.BOLD, 12));
        g.drawString("Class", getWidth() / 2 - 10, getHeight() - 5);
    
        // Draw y-axis labels and values
        g.drawString("0", barWidth / 2, getHeight());
        g.drawString(Integer.toString(total), barWidth / 2, 10);
    }
    
}
