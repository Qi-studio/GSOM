/**
 * Created by Q. Li on 2017-09-02.
 * 
 */

import java.io.FileNotFoundException;
import java.io.File;
import java.util.Vector;
import java.util.Scanner;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;
import javafx.scene.shape.Line;
import javafx.scene.paint.Color;


public class GSOM_Main extends Application {

    public static void main(String[] args)  {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws FileNotFoundException {


        /*************************Input Parameters Start*************************/
        int sizeOfDataset = 7569; //Sice of input data set;
        int dimOfInputData = 225; //Dimension of input data vector and neuron weight vector;
        double[][] dataSet = new double[sizeOfDataset][dimOfInputData];
        int coor_unit = 30; //Size of 1 Unit on coordinate grid;
        double spread_factor = 0.5;
        double learning_rate_init = 0.1; //Initial learning rate;
        double neighborhood_radius_init = 8; //Initial neighborhood_radius;
        double alpha = 0.1; //learning_rate reduction factor,0 < alpha < 1;
        double R = 3.8; //Constant in learning_rate formula
        double gamma = 0.25; //Factor of Distribution (FD), used in error distribution stage, 0 < FD < 1;
        int max_iter_inputVector = 20; //Maximum number of iterations for each input vector;

        //Output File Location
        String input_file_dir = " ";
        String cluster_output_file = " ";
        /*************************Input Parameters End*************************/


        /** Read Input dataset **/
        File input_file = new File(input_file_dir);
        Scanner input = new Scanner(input_file);
        int inputDataIdx = 0;
        while(input.hasNext()) {
            for(int elementOfEachDim = 0; elementOfEachDim < dimOfInputData; elementOfEachDim++) {
                dataSet[inputDataIdx][elementOfEachDim] = input.nextDouble();
            }
            inputDataIdx += 1;
        }

        /** Training **/
        GSOM trial = new GSOM(dataSet, dimOfInputData, sizeOfDataset, coor_unit, spread_factor,
                learning_rate_init, neighborhood_radius_init, alpha, R, gamma, max_iter_inputVector, cluster_output_file);


        trial.train();

        /** Printout Results **/
        System.out.println("**********************************************************************");
        System.out.println("Final neuron list size: " + trial.neuron_list.size());


        /** Visualization of Result **/
        Scene scene = new Scene(new LinePane(trial.neuron_list), 800, 800);
        primaryStage.setTitle("Neural Grid");
        primaryStage.setScene(scene);
        primaryStage.show();
    }
}

class LinePane extends Pane {
    public LinePane(Vector<GSOM_Neuron> neuron_list_) {

        int multiplier = 10;
        int min_x = 0;
        int min_y = 0;
        for (GSOM_Neuron element : neuron_list_) {
            if(element.x < 0 && element.x < min_x) {
                min_x = element.x;
                //System.out.println("element.x" + element.x+" "+min_x);
            }
            if(element.y < 0 && element.y < min_y) {
                min_y = element.y;
            }
        }

        min_x = -min_x;
        min_y = -min_y;
        //System.out.println("min_x " + min_x + " min_y " + min_y);

        for (GSOM_Neuron element : neuron_list_) {
//            System.out.println(element.x+" "+element.y);
            if(element.left != null) {
//                System.out.println(element.x +" "+element.y+
//                        " left "+
//                        element.left.x+" "+element.left.y);

                Line line_ = new Line(element.x + min_x,
                        element.y + min_y,
                        element.left.x + min_x ,
                        element.left.y + min_y);
                line_.setStrokeWidth(3);
                line_.setStroke(Color.GREEN);
                getChildren().add(line_);
            }

            if(element.right != null) {
//                System.out.println(element.x +" "+element.y+
//                        " right "+
//                        element.right.x+" "+element.right.y);

                Line line_ = new Line(element.x + min_x,
                        element.y + min_y,
                        element.right.x + min_x,
                        element.right.y + min_y);
                line_.setStrokeWidth(3);
                line_.setStroke(Color.GREEN);
                getChildren().add(line_);
            }

            if(element.up != null) {
//                System.out.println(element.x +" "+element.y+
//                        " up "+
//                        element.up.x+" "+element.up.y);

                Line line_ = new Line(element.x + min_x,
                        element.y + min_y,
                        element.up.x + min_x,
                        element.up.y + min_y);
                line_.setStrokeWidth(3);
                line_.setStroke(Color.GREEN);
                getChildren().add(line_);
            }

            if(element.down != null) {
//                System.out.println(element.x +" "+element.y +
//                        " down "+
//                        element.down.x +" "+element.down.y);

                Line line_ = new Line(element.x + min_x,
                         element.y + min_y,
                         element.down.x + min_x,
                         element.down.y + min_y);
                line_.setStrokeWidth(3);
                line_.setStroke(Color.GREEN);
                getChildren().add(line_);
            }


        }
    }
}
