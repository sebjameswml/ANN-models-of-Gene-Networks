#include <iostream>
#include <vector>
#include <cmath>

// A Recurrent network class.
template <class Flt>
class RNet
{
public:
    int N = 100;                // The number of genes (i.e. number of nodes)

    std::vector<int> weightexistence; // Flat matrix. 1 if there's a connection at all, 0 if not
    std::vector<int> bestWE;          // The weightexistence matrix for the best weights

    std::vector<Flt> states;    // The states of N genes
    std::vector<Flt> avstates;  // Average states of N genes

    std::vector<Flt> inputs;    // Vector of inputs. size < N. Might be 2 (for x and y)
    std::vector<Flt> target;    // The target output states. Might be 4 for 4 identifiable area types.
    std::vector<Flt> result;    // A temporary variable used in the step() function. Size N.

    Flt learnrate = 0.1;        // Learning rate of the algorithm
    int bias = 2;               // Which node is the 'bias node'?

    // Initialise a random weights matrix
    std::vector<std::vector<Flt>> weights;
    std::vector<std::vector<Flt>> best; // Holds a copy of the best weights
#ifdef USE_CONVERGE
    // a matrix of small values to nudge the weights if required
    std::vector<std::vector<Flt>> nudge;
#endif

    RNet()
    {
        // Resize std::vectors
        this->states.resize (this->N);
        RNet<Flt>::randomiseStates();
        this->avstates.resize (this->N);
        this->result.resize (this->N);

        // initialise inputs as 0
        this->inputs.resize (this->N, 0);
        // initialise target as flexible (-1 is the free variable flag)
        this->target.resize (this->N, -1);

        this->weightexistence.resize (this->N*this->N, 1);
        this->bestWE.resize (this->N*this->N, 1);

        this->weights.resize (this->N);
        for (std::vector<Flt>& w_inner : this->weights) {
            w_inner.resize (this->N);
        }
        this->best.resize (this->N);
        for (std::vector<Flt>& w_inner : this->best) {
            w_inner.resize (this->N);
        }
#ifdef USE_CONVERGE
        this->nudge.resize (this->N);
        for (std::vector<Flt>& w_inner : this->nudge) {
            w_inner.resize (this->N);
        }
#endif
        // init weights with random numbers
        RNet<Flt>::randommatrix (this->weights);
    }

    // alters matrix A in memory! Does not make copy
    void randommatrix (std::vector<std::vector<Flt>>& A)
    {
        // find the size of matrices
        int aRows = A.size();
        int aColumns = A[0].size();

        // randomise
        for(int i=0; i<aRows; i++) {
            for(int j=0; j<aColumns; j++) {
                A[i][j] = (Flt)rand() / (Flt)RAND_MAX * 2 - 1;
                // zero for weights that don't exist
                A[i][j] = A[i][j] * this->weightexistence[this->N*i+j];
            }
        }
    }

    void setAllStates (Flt initstate)
    {
        for (int i=0; i<this->N; i++) {
            states[i] = initstate;
        }
    }

    void setAllTargetStates (Flt initstate)
    {
        for (int i=0; i<this->N; i++) {
            target[i] = initstate;
        }
    }

    void printTarget() const
    {
        for (int i=0; i<this->N; i++) {
            std::cout << this->target[i] << " ";
        }
        std::cout << std::endl;
    }

    void randomiseStates()
    {
        for (int i=0; i<this->N; i++){
            this->states[i] = (Flt) rand() / (Flt) RAND_MAX;
        }
    }

    void setweights (std::vector<std::vector<Flt>>& initweights) { weights = initweights; }

    void printState() const
    {
        for (int i=0; i<this->N; i++) {
            std::cout << this->states[i] << " ";
        }
        std::cout << "\n\n";
    }

    void printweights() const
    {
        for (int i=0; i<this->N; i++) {
            for (int j=0; j<this->N; j++) {
                std::cout << this->weights[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Find squared difference between state and target
    Flt error()
    {
        float sum = 0;
        for (int i=0; i<this->N; i++) {
            if (this->target[i] != -1) {
                sum = sum + (states[i]-target[i]) * (states[i]-target[i]);
            }
        }
        return sum;
    }

    // Perform the Pineda-inspired backpropagation on this recurrent network.
    void updateWeights()
    {
        std::vector<Flt> deltas(this->N, Flt{0});
        std::vector<int> visits(this->N, 0);

        // initialise the list of deltas with the errors for target indices
        std::vector<int> fixed(this->N, 0);
        for (int i=0; i<this->N; i++) {
            // If target[i] is -1 then it is free to vary. If not, it is a fixed target.
            if (this->target[i] != -1) {
                deltas[i] = this->learnrate * 2 * (this->target[i]-this->states[i]) * this->states[i] * (1-this->states[i]);
                fixed[i] = 1;
            }
        }

        // Now the backpropogation of errors, so each node is given a delta
        int visitcount;
        int test = 0;
        while (true) { // will loop until the list of visits is full

            test++;
            visitcount = 0;

            for (int j=0; j<this->N; j++) {
                if (visits[j] > 0) { continue; }
                visitcount++;
                if (fixed[j] < 1) { continue; } // target j not fixed

                // Target j is fixed, but has not been visited so compute its delta
                for (int i=0; i<this->N; i++) {
                    // Add the dot product if fixed[0] is 0
                    deltas[i] += (fixed[i] == 0) ? (this->weights[i][j] * deltas[j]) : Flt{0};
                }
                visits[j] = 1;
            }

            for (int k=0; k<this->N; k++) {
                if (fixed[k] == 0 and deltas[k] != 0) {
                    deltas[k] *= this->states[k] * (1-this->states[k]);
                    fixed[k] = 1;
                }
            }
            if (visitcount < 1) { break; } // Because nothing happened
            if (test > 10) { break; } // in the event of an island
        }

        // Once every node has a delta update all the weights (except where weightexistence matrix has zeros)
        for (int i=0; i<this->N; i++) {
            for (int j=0; j<this->N; j++) {
                this->weights[i][j] = this->weightexistence[this->N*i+j] ? (this->weights[i][j] + this->states[i] * deltas[j]) : Flt{0};
            }
        }
    }

protected:
    // Perform one forward pass step of the network
    void step()
    {
        Flt total = Flt{0};

        // generate dot product
        for (int i=0; i<this->N; i++) {
            total = 0;
            for (int j=0; j<this->N; j++) {
                total = total + this->states[j] * this->weights[j][i];
            }
            this->result[i] = total;
        }

        // squash
        for (int i=0; i<this->N; i++) {
            this->result[i] = Flt{2}/(Flt{1} + std::exp(-this->result[i])) - Flt{1}; // range 0 to 1
        }

        // add input to result and place in this->states
        for (int i=0; i<this->N; i++) {
            this->states[i] = this->result[i] + this->inputs[i];
        }
    }

#ifdef USE_CONVERGE
public:
    void converge()
    {
        // do steps until the states are the same within 1/1000
        std::vector<Flt> copy; // better as a member variable, readily sized.
        copy.resize (this->N);
        float total = 1;
        int count = 0;

        while (total > 0.01*0.01*this->N) {
            count++;
            if (count > 200) { // Re-randomise if we've not already exited the while after 200 counts
                this->randommatrix (this->weights); // rand vals between -1 and 1
                count = 0;
                break;
            }
            // store a copy of the states
            copy = states;
            // update the states
            this->step();
            // find the sum of squared differences between the two
            total=0;
            for (int i=0; i<this->N; i++) {
                total = total + (states[i]-copy[i])*(states[i]-copy[i]);
            }
        }
    }
#endif
};

// This class allows for running the network with one of the genes knocked out,
// simulating one of the gene knockout conditions.
template <class Flt>
class RNetKnock : public RNet<Flt>
{
public:
    int konode = 3; // Node index of the first knockout

    // Find the average value of each node after 50 forward iterations of the
    // network. This is one way to do the 'forward pass' of the network. The other is to
    // use the converge() function (untested by Seb)
    void average (int knockout)
    {
        // update the states 50 times
        for (int n=0; n<50; n++) { this->step(knockout); }

        // update the states 50 more times and add to the average (avstates)
        for (int n=0; n<50; n++) {
            this->step(knockout);
            for (int i=0; i<this->N; i++) {
                this->avstates[i] += this->states[i];
            }
        }

        for (int i=0; i<this->N; i++) {
            this->states[i] = this->avstates[i]/50;
            this->avstates[i] = 0;
        }

        // Final knockout to make deltas and weight updates correct
        if (knockout != 0) { this->states[knockout+this->konode-1] = 0; }
    }

protected:
    // Perform one step of the forward-pass through the network
    void step (int knockout)
    {
        Flt total = Flt{0};

        // generate dot product
        for (int i=0; i<this->N; i++) {
            total = Flt{0};
            for (int j=0; j<this->N; j++) {
                total = total + this->states[j] * this->weights[j][i];
            }
            this->result[i] = total;
        }

        // squash
        for (int i=2; i<this->N; i++) {
            this->result[i] = Flt{1} / (Flt{1} + std::exp(-this->result[i])); // range -1 to 1
        }
        this->result[0] = 0;
        this->result[1] = 0;
        this->result[this->bias] = Flt{1};
        // added in
        if (knockout != 0) { this->result[knockout+this->konode-1] = Flt{0}; }

        // add input
        for (int i=0; i<this->N; i++) {
            this->states[i] = this->result[i] + this->inputs[i];
        }
    }

#ifdef USE_CONVERGE
public:
    // do steps until the states are the same within 1/1000.
    void converge (int knockout)
    {
        std::vector<Flt> copy(this->N);
        float total = Flt{1};
        int count = 0;

        while (total > 0.01*0.01*this->N) {
            count++;
            if (count>400) {
                std::cout<<"n";
                this->weights = this->best;
                // put in a nudge.
                for (int i=0; i<this->N; i++) {
                    for (int j=0; j<this->N; j++) {
                        this->weights[i][j] += ((Flt)rand() / (Flt)RAND_MAX * 2 - 1) * 0.0001;
                        this->weights[i][j] *= this->weightexistence[this->N*i+j];
                    }
                }
                count = 0;
                break;
            }
            if (knockout != 0) {
                this->states[knockout+this->konode-1] = 0;
            }
            // store a copy of the states
            copy = this->states;
            // update the states
            this->step(knockout);
            // find the sum of squared differences between the two
            total = 0;
            for (int i=0; i<this->N; i++) {
                total = total + (this->states[i]-copy[i]) * (this->states[i]-copy[i]);
            }
        }
        // Final knockout to make deltas and weight updates correct
        if (knockout != 0) { this->states[knockout+this->konode-1] = 0; }
    }
#endif // USE_CONVERGE
};
