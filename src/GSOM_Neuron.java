/**
 * Created by Q. Li on 2017-08-31.
 */

import java.util.stream.IntStream;

/**Define a Neuron**/
public class GSOM_Neuron {

    int x; //x-coordinate of this neuron
    int y; //y-coordinate of this neuron
    int dim; //Dimension of weight vector

    double[] weights; //weights vector with dimension dim;
    double error; //Error of this neuron;

    int last_it; // Iteration number of this neuron selected as bmu;

    //Topology Neighborhood of this neuron
    GSOM_Neuron right;
    GSOM_Neuron left;
    GSOM_Neuron up;
    GSOM_Neuron down;

     public GSOM_Neuron(int dim, int x, int y) {

         this.dim = dim;

         //Assign coordinates.
         this.x = x;
         this.y = y;

         //Initialize weights vector of dimension "dim" with random values [0,1) ;
         weights = IntStream.range(0, dim).mapToDouble(i -> StrictMath.random()).toArray();

         error = 0;

         last_it = 0;

         right = null;
         left = null;
         up = null;
         down = null;

     }

     // Weight adjustment
     public void adjust_weights(double[] neuron_, double learning_rate_, double learning_efficiency_) {
         weights = IntStream.range(0, dim).mapToDouble(i -> weights[i] + learning_rate_ * learning_efficiency_ * (neuron_[i] - weights[i])).toArray();
     }

     /** Check if neuron is boundary neuron
     * Return true if it is a boundary neuron
     * */
     public boolean boundary_check() {

         return left == null || right == null || up == null || down == null;

     }

}
