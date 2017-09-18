/**
 * Created by Qi Li on 2017-08-31.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.security.SecureRandom;
import java.util.Vector;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import javafx.util.Pair;


public class GSOM {

    double[][] dataset;
    int dimOfInputData; //Dimension of input data vector and neuron weight vector;
    int sizeOfDataset; //Sice of input data set;
    int coor_unit; //Size of 1 Unit on coordinate grid;

    double growing_threshold; //GT;
    double learning_rate_init; //Initial learning rate;
    double neighborhood_radius_init; //Initial neighborhood_radius;
    double alpha; //learning_rate reduction factor,0 < alpha < 1;
    double R; //Constant in learning_rate formula;
    double gamma; //Factor of Distribution (FD), used in error distribution stage, 0 < FD < 1;
    GSOM_Neuron bmu;
    Vector<GSOM_Neuron> neuron_list; //Vector used to store neurons during training;
    int[] cluster; //Array used to store cluster information of each input vector;

    int global_iter; //global iteration number during training process;
    int local_iter; //Local iteration number; initialize to 0 each input vector;
    int epochs; //Maximum number of iterations for each input vector;

    String cluster_output_file_dir; //Output file location for cluster result;

    public GSOM(double[][] dataset_, int dimOfInputData_, int sizeOfDataset_, int coor_unit_, double spread_factor,
                double learning_rate_init_, double neighborhood_radius_init_, double alpha_, double R_, double gamma_, int max_iter_inputVector_,
                String cluster_output_file_dir_) throws FileNotFoundException {

        dataset = Arrays.copyOf(dataset_,dataset_.length);
        dimOfInputData = dimOfInputData_;
        sizeOfDataset = sizeOfDataset_;
        coor_unit = coor_unit_;

        growing_threshold = - dimOfInputData * StrictMath.log(spread_factor) / StrictMath.log(2);
        learning_rate_init = learning_rate_init_;
        neighborhood_radius_init = neighborhood_radius_init_;
        alpha = alpha_;
        R = R_;
        gamma = gamma_;
        neuron_list = new Vector<>();
        cluster = new int[sizeOfDataset];

        global_iter = 0;
        local_iter = 0;
        epochs = max_iter_inputVector_;

        cluster_output_file_dir = cluster_output_file_dir_;

        //Initialize 4-neuron grid
        GSOM_Neuron neuron_00 = new GSOM_Neuron(dimOfInputData, 0, 0);
        GSOM_Neuron neuron_01 = new GSOM_Neuron(dimOfInputData, 0, coor_unit);
        GSOM_Neuron neuron_10 = new GSOM_Neuron(dimOfInputData, coor_unit, 0);
        GSOM_Neuron neuron_11 = new GSOM_Neuron(dimOfInputData, coor_unit, coor_unit);

        //Add 4 initial neurons to the list
        neuron_list.add(neuron_00);
        neuron_list.add(neuron_01);
        neuron_list.add(neuron_10);
        neuron_list.add(neuron_11);

        //Connect initial 4 neurons
        neuron_00.right = neuron_10;
        neuron_00.up    = neuron_01;
        neuron_01.right = neuron_11;
        neuron_01.down  = neuron_00;
        neuron_10.up    = neuron_11;
        neuron_10.left  = neuron_00;
        neuron_11.left  = neuron_01;
        neuron_11.down  = neuron_10;

    }

    /**Get Distance between two vectors **/
    public double get_distance(double[] neuron_1, double[] neuron_2) {

        return StrictMath.sqrt(euclidean(neuron_1, neuron_2));

    }

    /**Calculate Euclidean Distance between two vectors**/
    public double euclidean(double[] weight_vector_1, double[] weight_vector_2) {

        return IntStream.range(0, dimOfInputData).mapToDouble(i -> (weight_vector_1[i] - weight_vector_2[i]) * (weight_vector_1[i] - weight_vector_2[i])).sum();

    }

    /**Find the BMU of the input data vector**/
    public GSOM_Neuron best_match_unit(double[] input_data_vector) {
    
        double dist = Double.POSITIVE_INFINITY;
        double dist_tempt = 0;
        GSOM_Neuron best_match_unit_ = null;
        for (GSOM_Neuron element : neuron_list) {

            dist_tempt = get_distance(input_data_vector, element.weights);

            if(dist_tempt < dist) {
                dist = dist_tempt;
                best_match_unit_ = element;
            }
        }

        return best_match_unit_;

    }


    /**Training GSOM map**/
    public void train() throws FileNotFoundException {

        global_iter = 0;

        //Record bmu neuron;
        Vector<GSOM_Neuron> bmu_record = new Vector<>();

        //Track presented input data, idx is stored
        Vector<Integer> presentedDataTracking = new Vector<>(sizeOfDataset);

        /** Training Loop
         * Randomly choose one input data vector from input database until the whole dataset has been traversed
         * */
        while(presentedDataTracking.size() < sizeOfDataset) {
            /* Randomly retrieve an input data vector */
            SecureRandom random = new SecureRandom();
            int seed = random.nextInt(sizeOfDataset);

            //Check if input data has been presented before
            if(presentedDataTracking.contains(seed)) {
                continue;
            }

            //Add used input data's idx in database into presentedDataTracking vector
            presentedDataTracking.add(seed);
            //Retrieve input data vector
            double[] input_data_vector = dataset[seed];

            /**Re-initialize learning-rate,  neighborhood function and iter number for each input vector
            * GSOM performs local update
            * */
            double learning_rate = learning_rate_init;
            local_iter = 0;

            //Vector<GSOM_Neuron> recalc_nodes = new Vector<>();

            /**Iterate each input_data_vector
            * Until either epochs number of iterations has been reached
            * Or neighborhood size is shrinked to size 1;
            * */
            for (int dummy_var = 0; dummy_var < epochs; dummy_var++) {

                /*Parameter update*/
                //Update iteration number
                local_iter += 1;
                global_iter += 1;

                //Find the BMU
                bmu = best_match_unit(input_data_vector);

                bmu.last_it = global_iter;

                if(!bmu_record.contains(bmu)) {
                    bmu_record.add(bmu);
                }

                //Store "bmu idx" of each input data vector into cluster; "bmu idx" is the index of its postion in bmu_record;
                cluster[seed] = bmu_record.indexOf(bmu);

                // Update learning_rate for each iteration
                learning_rate = learning_rate * alpha * (1 - R / neuron_list.size());
                //double neighborhood_radius = neighborhood_radius_init * StrictMath.exp( - (double)local_iter/neighborhood_radius_init );
                //double learning_efficiency = 0.5 * StrictMath.exp( - get_distance(input_data_vector, bmu.weights) * get_distance(input_data_vector, bmu.weights) /
                        //(2 * neighborhood_radius * neighborhood_radius));
                //System.out.println(neighborhood_radius + " radius ");
                double learning_efficiency = 0.5 * StrictMath.exp( - local_iter);

                /** Construct neighborhood region of bmu, and adjust weight of each neuron within the region **/
                Vector<GSOM_Neuron> bmu_neighbour_neurons = addNeighbors(bmu, dummy_var);

                //Update weight vector for each neuron in the neighborhood region
                for(int element = 0; element < bmu_neighbour_neurons.size(); element++) {
                    bmu_neighbour_neurons.get(element).adjust_weights(input_data_vector, learning_rate, learning_efficiency);

                }

                /**Track error of bmu **/
                //Calculate current error
                double err = get_distance(bmu.weights, input_data_vector);

                //Update accumulated error of bmu
                error_update(bmu, err);

                /**Neuron grow
                * Return Pair<Indicator of growth or not, New neurons>
                * */
                Pair<Boolean, Vector<GSOM_Neuron>> p = neuron_grow(bmu);

                /** Progress monitoring on screen **/
                if(global_iter % 50 == 0) {
                    System.out.println("Current neuron list size:  " + neuron_list.size() +
                            "   Progress percentage:  " +
                            Math.round((double)global_iter * 100 / (epochs * sizeOfDataset)) + "%");

                }
            }
        }

        //Trim neurons that have never been selected as bmu;
        neuro_trim();

        /** Output cluster results into output file located at cluster_output_file_dir
        ** Out put format "data_idx cluster_idx_in_neruon_list cluster_x_coord cluster_y_coord cluster_weight_vector"
        **/
        PrintWriter cluster_output = new PrintWriter(new File(cluster_output_file_dir));
        for (int element = 0; element < cluster.length; element++) {
            cluster_output.print(element + 1);
            cluster_output.print(" ");
            cluster_output.print(cluster[element] + 1);
            cluster_output.print(" ");
            cluster_output.print(bmu_record.get(cluster[element]).x);
            cluster_output.print(" ");
            cluster_output.print(bmu_record.get(cluster[element]).y);
            cluster_output.print(" ");

            for(int element_ = 0; element_ < dimOfInputData; element_++) {
                cluster_output.print(bmu_record.get(cluster[element]).weights[element_]);
                cluster_output.print(" ");
            }

            cluster_output.println(" ");

        }

        cluster_output.close();
        //System.out.println(bmu_record.size() + " bmu ");

    }

    /** Decaying Neighborhood Region
     * Different from orignial neighborhood function used in original SOM & GSOM
     * Using a shrinking squared neighborhood region
     * 1st 1/3 epochs include maximum 49 neurons, including bmu, which are 3 neurons range in each 4 directions of bmu
     * 2nd 1/3 epochs include maximum 25 neurons, including bmu, which are 2 neurons range in each 4 directions of bmu
     * 3rd 1/3 epochs include maximum 9 neurons, including bmu, which are 1 neruons range in each 4 directions of bmu
     * **/
    public Vector<GSOM_Neuron> addNeighbors(GSOM_Neuron bmu_, int dummy_var_) {

        Vector<GSOM_Neuron> bmu_neighbour_neurons_ = new Vector<>();

        //Add bmu neuron
        bmu_neighbour_neurons_.add(bmu);

        //Direct neruons
        if (bmu.left != null) {
            bmu_neighbour_neurons_.add(bmu.left);
        }

        if (bmu.right != null) {
            bmu_neighbour_neurons_.add(bmu.right);
        }

        if (bmu.up != null) {
            bmu_neighbour_neurons_.add(bmu.up);
        }

        if (bmu.down != null) {
            bmu_neighbour_neurons_.add(bmu.down);
        }

        //2 neurons range in each 4 directions of bmu
        if(dummy_var_ < 2 * epochs / 3) {
            if (bmu.left != null && bmu.left.left != null) {
                bmu_neighbour_neurons_.add(bmu.left.left);
            }

            if (bmu.right != null && bmu.right.right != null) {
                bmu_neighbour_neurons_.add(bmu.right.right);
            }

            if (bmu.up != null && bmu.up.up != null) {
                bmu_neighbour_neurons_.add(bmu.up.up);
            }

            if (bmu.down != null && bmu.down.down != null) {
                bmu_neighbour_neurons_.add(bmu.down.down);
            }

        }

        //3 neurons range in each 4 directions of bmu
        if(dummy_var_ < epochs / 3) {
            if (bmu.left != null && bmu.left.left != null && bmu.left.left.left != null) {
                bmu_neighbour_neurons_.add(bmu.left.left.left);
            }

            if (bmu.right != null && bmu.right.right != null && bmu.right.right.right != null) {
                bmu_neighbour_neurons_.add(bmu.right.right.right);
            }

            if (bmu.up != null && bmu.up.up != null && bmu.up.up.up != null) {
                bmu_neighbour_neurons_.add(bmu.up.up.up);
            }

            if (bmu.down != null && bmu.down.down != null && bmu.down.down.down != null) {
                bmu_neighbour_neurons_.add(bmu.down.down.down);
            }
        }

        return bmu_neighbour_neurons_;

    }


    /** Update error of bmu **/
    public void error_update(GSOM_Neuron bmu_neuron_, double current_error_) {

        //Accumulate error of bmu neuron
        bmu_neuron_.error += current_error_;

    }

    /**Neuron grow
    * Return Pair<Indicator of growth or not, New neurons>
    * */
    public Pair<Boolean, Vector<GSOM_Neuron>> neuron_grow(GSOM_Neuron bmu_neuron_) {

        /* Case 1 -- Error Distribution
        * Error >= growing_threshold GT and this BMU is NOT a boundary neuron;
        * Neuron-Growth = False;
        * */
        if (bmu_neuron_.error > growing_threshold && !bmu_neuron_.boundary_check()) {

            //Ripple outward error distribution to immediate 4 neighborhood neurons
            bmu_neuron_.error = 0.5 * growing_threshold; //Reduce this (BMU) neuron error to GT / 2
            bmu_neuron_.left.error += gamma * bmu_neuron_.left.error; //Increase immediate 4 neighbor-neurons error by gamma * GT
            bmu_neuron_.right.error += gamma * bmu_neuron_.right.error;
            bmu_neuron_.up.error += gamma * bmu_neuron_.up.error;
            bmu_neuron_.down.error += gamma * bmu_neuron_.down.error;

            return new Pair<Boolean, Vector<GSOM_Neuron>>(false,null);
        }

        /** Case 2 -- Weight Distribution
        * Error exceeds growing_threshold GT and this BMU IS a boundary neuron;
        * Neuron-Growth = True;
        * */
        if(bmu_neuron_.error >= growing_threshold && bmu_neuron_.boundary_check()) {

            Vector<GSOM_Neuron> new_neurons_ = add_neuron(bmu_neuron_);

            return new Pair<Boolean, Vector<GSOM_Neuron>>(true,new_neurons_);

        }

        /** Case 3
        * Error below growing_threshold GT;
        * Neuron-Growth = False;
        * */
        return new Pair<Boolean, Vector<GSOM_Neuron>>(false,null);
    }

    /** Add new neurons
    * In all free directions of immediate 4 neighborss
    * */
    public Vector<GSOM_Neuron> add_neuron(GSOM_Neuron parent_neuron_) {

        Vector<GSOM_Neuron> added_neuron=new Vector<>();

        GSOM_Neuron tempt_neuron;

        if(parent_neuron_.left == null) {
            tempt_neuron = insert_neuron(parent_neuron_.x - coor_unit, parent_neuron_.y, parent_neuron_, 1);
            added_neuron.add(tempt_neuron);

        }

        if(parent_neuron_.right == null) {
            tempt_neuron = insert_neuron(parent_neuron_.x + coor_unit, parent_neuron_.y, parent_neuron_, 2);
            added_neuron.add(tempt_neuron);
        }

        if(parent_neuron_.up == null) {
            tempt_neuron = insert_neuron(parent_neuron_.x, parent_neuron_.y + coor_unit, parent_neuron_, 3);
            added_neuron.add(tempt_neuron);

        }

        if(parent_neuron_.down == null) {
            tempt_neuron = insert_neuron(parent_neuron_.x, parent_neuron_.y - coor_unit, parent_neuron_, 4);
            added_neuron.add(tempt_neuron);

        }

        return added_neuron;
    }

    /** Insert a new neuron
    * from parent_neuron_
    * @location (x_,y_)
    * & Update topology neighbors of this new neuron.
    * */
    public GSOM_Neuron insert_neuron(int x_, int y_, GSOM_Neuron parent_neuron_, int directionIndicator) {

        GSOM_Neuron new_neuron = new GSOM_Neuron(dimOfInputData, x_, y_ );

        /**Update the current neuron list
        * neuron number is changed at this step
        * */
        neuron_list.add(new_neuron);

        //Connect available neurons
        for(GSOM_Neuron element : neuron_list) {

            if(element.x == x_ - coor_unit && element.y == y_) {
                new_neuron.left = element;
                element.right = new_neuron;
            }

            if(element.x == x_ + coor_unit && element.y == y_) {
                new_neuron.right = element;
                element.left = new_neuron;
            }

            if(element.x == x_ && element.y == y_ + coor_unit) {
                new_neuron.up = element;
                element.down = new_neuron;
            }

            if(element.x == x_ && element.y == y_ - coor_unit) {
                new_neuron.down = element;
                element.up = new_neuron;
            }
        }


        // Assign weights to newly inserted neuron;
        new_weights(new_neuron, parent_neuron_, directionIndicator);

        return new_neuron;
    }

    /** Calculate new weights to the newly inserted neuron;
     * Considering 4 scenarios;
     */
    public double[] new_weights(GSOM_Neuron new_neuron_, GSOM_Neuron parent_neuron_, int directionIndicator_) {

        /* new_neuron_ is to the LEFT direction of parent_neuron_ */
        if(directionIndicator_ == 1) {
            if(new_neuron_.left != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        (parent_neuron_.weights[i] + new_neuron_.left.weights[i]) / 2).toArray();
            }

            else if(parent_neuron_.right != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] -  parent_neuron_.right.weights[i]).toArray();
            }

            else if(parent_neuron_.up != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] - parent_neuron_.up.weights[i]).toArray();
            }

            else if(parent_neuron_.down != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] - parent_neuron_.down.weights[i]).toArray();
            }

            else if(parent_neuron_.up == null && parent_neuron_.down == null && parent_neuron_.right == null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i -> 0.5).toArray();
            }

        }

        /** new_neuron_ is to the RIGHT direction of parent_neuron_ */
        if(directionIndicator_ == 2) {
            if(new_neuron_.right != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        (parent_neuron_.weights[i] + new_neuron_.right.weights[i]) / 2).toArray();
            }

            else if(parent_neuron_.left != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] -  parent_neuron_.left.weights[i]).toArray();
            }

            else if(parent_neuron_.down != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] - parent_neuron_.down.weights[i]).toArray();
            }

            else if(parent_neuron_.up != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] - parent_neuron_.up.weights[i]).toArray();
            }

            else if(parent_neuron_.up == null && parent_neuron_.down == null && parent_neuron_.left == null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i -> 0.5).toArray();
            }
        }

        /** new_neuron_ is to the UP direction of parent_neuron_ */
        if(directionIndicator_ == 3) {
            if(new_neuron_.up != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        (parent_neuron_.weights[i] + new_neuron_.up.weights[i]) / 2).toArray();
            }

            else if(parent_neuron_.down != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] -  parent_neuron_.down.weights[i]).toArray();
            }

            else if(parent_neuron_.left != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] - parent_neuron_.left.weights[i]).toArray();
            }

            else if(parent_neuron_.right != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] - parent_neuron_.right.weights[i]).toArray();
            }

            else if(parent_neuron_.left == null && parent_neuron_.down == null && parent_neuron_.right == null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i -> 0.5).toArray();
            }
        }

        /** new_neuron_ is to the DOWN direction of parent_neuron_ */
        if(directionIndicator_ == 4) {
            if(new_neuron_.down != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        (parent_neuron_.weights[i] + new_neuron_.down.weights[i]) / 2).toArray();
            }

            else if(parent_neuron_.up != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] -  parent_neuron_.up.weights[i]).toArray();
            }

            else if(parent_neuron_.right != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] - parent_neuron_.right.weights[i]).toArray();
            }

            else if(parent_neuron_.left != null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i ->
                        2 * parent_neuron_.weights[i] - parent_neuron_.left.weights[i]).toArray();
            }

            else if(parent_neuron_.up == null && parent_neuron_.right == null && parent_neuron_.left == null) {
                new_neuron_.weights = IntStream.range(0, dimOfInputData).mapToDouble(i -> 0.5).toArray();
            }
        }

        return new_neuron_.weights;
    }

    /** Trip the neurons that has never been selected as bmu **/
    public void neuro_trim () {

        Vector<GSOM_Neuron> removed_neuron_list = new Vector<>();

        //Remove connection to all 4 direct neighborhood neurons
        for(GSOM_Neuron element : neuron_list) {

           if(element.last_it == 0) {
               if(element.left != null) {
                   element.left.right = null;
               }

               if(element.right != null) {
                   element.right.left = null;
               }

               if(element.up != null) {
                   element.up.down = null;
               }

               if(element.down != null) {
                   element.down.up = null;
               }

               //Record removed neuron
               removed_neuron_list.add(element);
           }

        }

        //Delete removed-neuron from neuron_list
        for(GSOM_Neuron element: removed_neuron_list) {
            neuron_list.removeElement(element);
        }
    }
}
