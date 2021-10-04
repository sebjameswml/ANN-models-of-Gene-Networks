#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>

using std::cout;
using std::endl;

//from an arbitrary initial state to an arbitrary target what does the error landscape
//look like, is it smooth and does it have many local minima?

// A Recurrent network class.
template <class Flt>
class RNet
{
public:
    //get network topolgy from h5 file
    // The number of genes
    int N = 100;
    float p = 0.3;
    float p_link = 0.1;

    std::vector<int> weightexistence;
    std::vector<int> bestWE;

    int tal = 0;
    //The states of N genes:
    std::vector<Flt> states;
    std::vector<Flt> avstates;

    std::vector<Flt> inputs;
    std::vector<Flt> target;

    Flt learnrate = 0.1;

    int konode = 3; //if the ko = 1 the node index is ko+3-1 = 3, the 4th node
    int bias = 2;

    //Initialise a random weights matrix
    std::vector<std::vector<Flt>> weights;
    std::vector<std::vector<Flt>> best; // Holds a copy of the best weights

    //a matrix of small values to nudge the weights if required
    std::vector<std::vector<Flt>> nudge;


    RNet() {
        // Resize std::vectors
        this->states.resize (this->N);
        this->avstates.resize (this->N);

        this->inputs.resize (this->N);
        this->target.resize (this->N);

        this->weightexistence.resize (this->N*this->N);
        for(int i=0;i<(this->N*this->N);i++){this->weightexistence[i] = 1;};

        this->bestWE.resize (this->N*this->N);
        for(int i=0;i<(this->N*this->N);i++){this->bestWE[i] = 1;};

        this->weights.resize (this->N);
        for (std::vector<Flt>& w_inner : this->weights) {
            w_inner.resize (this->N);
        }
        this->best.resize (this->N);
        for (std::vector<Flt>& w_inner : this->best) {
            w_inner.resize (this->N);
        }
        this->nudge.resize (this->N);
        for (std::vector<Flt>& w_inner : this->nudge) {
            w_inner.resize (this->N);
        }

        RNet<Flt>::randomiseStates();

        // init weights
        RNet<Flt>::randommatrix (this->weights);

        //initialise target as flexible (-1 is the free variable flag)
        for (int i=0; i<this->N; i++){
            target[i] = -1;
        }

        //initialise inputs as 0
        for (int i=0; i<this->N; i++){
            inputs[i] = 0;
        }
    }

    int randommatrix(std::vector<std::vector<Flt>> &A) {
        //alters matrix in memory! Does not make copy

        //find the size of matrices
        int aRows = A.size();
        int aColumns = A[0].size();

        //randomise
        for(int i=0; i<aRows; i++){
            for(int j=0; j<aColumns; j++){
                A[i][j] = (Flt) rand() / (Flt) RAND_MAX *2 -1;
                //cout<<i<< " "<< j<< " "<< A[i][j]<<endl;
                //zero for weights that don't exist
                A[i][j] = A[i][j] * this->weightexistence[this->N*i+j];

                //cout<<this->weightexistence[this->N*i+j]<<endl;
                //cout<< A[i][j]<<endl;
            }
        }
        //cout<<"randomised"<<endl;
        //this->printweights();
        return(0);
    }

    void setAllStates(Flt initstate){
        for (int i=0; i<this->N; i++) {
            states[i] = initstate;
        }
    }

    void setAllTargetStates(Flt initstate){
        for (int i=0; i<this->N; i++) {
            target[i] = initstate;
        }
    }

    void printTarget() const {
        for (int i=0; i<this->N; i++) {
            cout << this->target[i] << " ";
        }
        cout << endl;
    }

    void randomiseStates() {
        for (int i=0; i<this->N; i++){
            this->states[i] = (Flt) rand() / (Flt) RAND_MAX;
        }
    }

    void setweights(std::vector<std::vector<Flt>>initweights){
        weights = initweights;
    }

    void printState() const {
        for (int i=0; i<this->N; i++) {
            cout << this->states[i] << " ";
        }
        cout << endl << endl;
    }

    void printweights() const {
        for(int i=0;i<this->N;i++){
            for(int j=0;j<this->N;j++){
                cout << this->weights[i][j] << " ";
            }
            cout << endl;
        }
    }

    Flt error() {
        //Find squared difference between state and target

        float sum = 0;

        for(int i=0; i<this->N; i++){
            if(this->target[i]!=-1){
                sum = sum + (states[i]-target[i])*(states[i]-target[i]);
            }
        }

        return(sum);
    }

    void updateWeights() {
        std::vector<Flt> deltas;
        deltas.resize(this->N);
        for(int i=0;i<(this->N);i++){deltas[i] = 0;}

        std::vector<int> visits;
        visits.resize(this->N);
        for(int i=0;i<(this->N);i++){visits[i] = 0;}

        std::vector<int> fixed;
        fixed.resize(this->N);
        for(int i=0;i<(this->N);i++){fixed[i] = 0;}

        //initialise the list of deltas with the errors for target indices
        for(int i=0;i<(this->N);i++){
            if(this->target[i]!=-1){
                deltas[i] = this->learnrate*2*(this->target[i]-this->states[i])*this->states[i]*(1-this->states[i]);
                fixed[i] = 1;
            }
        }

        //Now the backpropogation of errors, so each node is given a delta
        int visitcount;
        int test = 0;
        while(true){ //will loop until the list of visits is full

            test++;
            visitcount = 0;

            for(int j=0;j<(this->N);j++){
                if(visits[j]==0){
                    visitcount++;
                    if(fixed[j]==1){
                        for(int i=0;i<(this->N);i++){
                            if(fixed[i]==0){
                                //dot product
                                deltas[i] += this->weights[i][j]*deltas[j];
                            }
                        }
                        visits[j]=1;
                    }
                }
            }

            for(int k=0;k<(this->N);k++){
                if(fixed[k]==0 and deltas[k]!=0){
                    deltas[k] *= this->states[k]*(1-this->states[k]);
                    fixed[k] = 1;
                }
            }
            if(visitcount==0){break;}
            if(test>10){break;} //in the event of an island

        }
        //cout<<"deltas"<<endl;
        //for(int i=0; i<this->N; i++){cout<<deltas[i]<<" ";}cout<<endl;

        //Once every node has a delta update all the weights
        for(int i=0; i<this->N; i++){
            for(int j=0; j<this->N; j++){
                this->weights[i][j] += this->states[i]*deltas[j];
                //zero for weights that don't exist
                this->weights[i][j] = this->weights[i][j] * this->weightexistence[this->N*i+j];
            }
        }
        //cout<<"weights"<<endl;RNet<Flt>::printweights();
    }

    void step() {
        //dot it and squash it

        //initialise result matrix
        std::vector<Flt> result(this->N);
        Flt total;

        //generate dot product
        for(int i=0; i<this->N; i++){
            total = 0;
            for(int j=0;j<this->N;j++){
                total = total + this->states[j] * this->weights[j][i];
            }
            result[i] = total;
        }

        //squash
        for(int i=0; i<this->N; i++){
            result[i] = 2/(1+exp(-result[i]))-1;
        }

        //add input
        for(int i=0; i<this->N; i++){
            result[i] = result[i] + inputs[i];
        }

        // At end, update states with result:
        this->states = result;
    }

    void converge() {
        //do steps until the states are the same within 1/1000
        std::vector<Flt> copy;
        float total=1;
        copy.resize (this->N);
        int count = 0;

        while(total>0.01*0.01*this->N){
            count ++;
            if(count>200){
                this->randommatrix(this->weights); //rand vals between -1 and 1
                /*
                for(int i=0; i<this->N; i++){
                    for(int j=0; j<this->N; j++){
                        this->store[i][j] = this->store[i][j] + this->nudge[i][j]/100;  //stored weights changed by small amount
                    }
                }

                this->weights = this->store;*/
                //cout<<"reset"<<endl;
                count = 0;
                break;
            }
            //store a copy of the states
            copy=states;
            //update the states
            this->step();
            //find the sum of squared differences between the two
            total=0;
            for(int i=0;i<this->N;i++){
                total = total + (states[i]-copy[i])*(states[i]-copy[i]);
            }
        }
    }
};

template <class Flt>
class RNetBin : public RNet<Flt>
{
public:
    void stepbin() {
        //dot it and squash it

        //RNet<Flt>::printState();
        //initialise result matrix
        std::vector<Flt> result(this->N);
        Flt total;

        //generate dot product
        for(int i=0; i<this->N; i++){
            total = 0;
            for(int j=0;j<this->N;j++){
                total = total + this->states[j] * this->weights[j][i];
            }
            result[i] = total;
        }
        this->states = result;
        //cout<<"dotproduct"<<endl;RNet<Flt>::printState();
        //squash
        for(int i=0; i<this->N; i++){
            result[i] = 1/(1+exp(-result[i]));
        }
        result[0] = 0;
        result[1] = 0;
        result[this->bias] = 1;


        //add input
        for(int i=0; i<this->N; i++){
            result[i] = result[i] + this->inputs[i];
        }

        //THE STOCHASTIC BINARY BIT
        //Go through each node value and collapse it to a binary value based on probability
        for(int i=0; i<this->N; i++){
            this->states[i] = 0;
            if(((Flt) rand() / (Flt) RAND_MAX )<result[i]){
                this->states[i] = 1;
            }
        }
    }

    void step() {
        //dot it and squash it

        //RNet<Flt>::printState();
        //initialise result matrix
        std::vector<Flt> result(this->N);
        Flt total;

        //generate dot product
        for(int i=0; i<this->N; i++){
            total = 0;
            for(int j=0;j<this->N;j++){
                total = total + this->states[j] * this->weights[j][i];
            }
            result[i] = total;
        }
        this->states = result;
        //cout<<"dotproduct"<<endl;RNet<Flt>::printState();
        //squash
        for(int i=0; i<this->N; i++){
            result[i] = 1/(1+exp(-result[i]));
        }
        result[0] = 0;
        result[1] = 0;
        result[this->bias] = 1;


        //add input
        for(int i=0; i<this->N; i++){
            result[i] = result[i] + this->inputs[i];
        }

        this->states = result;
    }

    void converge() {
        //do steps until the states are the same within 1/1000
        std::vector<Flt> copy;
        float total=1;
        copy.resize (this->N);
        int count = 0;

        while(total>0.01*0.01*this->N){
            count ++;
            //cout<<count<<endl;
            if(count>400){
                cout<<"n";
                this->weights=this->best;
                //put in a nudge.
                for(int i=0;i<this->N;i++){
                    for(int j=0;j<this->N;j++){
                        this->weights[i][j]=this->weights[i][j]+ ((Flt) rand() / (Flt) RAND_MAX *2 -1)*0.1;
                        this->weights[i][j] *= this->weightexistence[this->N*i+j];
                    }
                }

                //cout<<"reset"<<endl;
                count = 0;
                break;
            }
            //store a copy of the states
            copy=this->states;
            //update the states
            this->step();
            //find the sum of squared differences between the two
            total=0;
            for(int i=0;i<this->N;i++){
                total = total + (this->states[i]-copy[i])*(this->states[i]-copy[i]);
            }

        }

    }

};


// Example of a derived class with a specialisation of a method
template <class Flt>
class RNetKnock : public RNet<Flt>
{
public:

    void step (int knockout) {
        //dot it and squash it
        //RNet<Flt>::printweights();
        //RNet<Flt>::printState();
        //initialise result matrix
        std::vector<Flt> result(this->N);
        Flt total;


        //THE KNOCKOUT PART now done in converge
        //if(knockout!=0){
        //    this->states[knockout+this->konode-1] = 0;
        //}
        //cout<<knockout<<endl;
        //cout<<"knockedout"<<endl;RNet<Flt>::printState();
        //generate dot product
        for(int i=0; i<this->N; i++){
            total = 0;
            for(int j=0;j<this->N;j++){
                total = total + this->states[j] * this->weights[j][i];
            }
            result[i] = total;
        }
        this->states = result;
        //cout<<"dotproduct"<<endl;RNet<Flt>::printState();
        //squash
        for(int i=0; i<this->N; i++){
            result[i] = 1/(1+exp(-result[i]));
        }
        result[0] = 0;
        result[1] = 0;
        result[this->bias] = 1;
        //added in
        if(knockout!=0){
            result[knockout+this->konode-1] = 0;
        }

        //add input
        for(int i=0; i<this->N; i++){
            result[i] = result[i] + this->inputs[i];
        }
        this->states = result;
        //cout<<"squashed"<<endl;RNet<Flt>::printState();

        // At end, update states with result:
        this->states = result;
        //this->printState();
    }

    void average(int knockout){

    for(int n=0;n<50;n++){
        //update the states
        this->step(knockout);
        }

    for(int n=0;n<50;n++){
        //update the states
        this->step(knockout);
        //add them to average
        for(int i=0;i<this->N;i++){
            this->avstates[i] += this->states[i];
        }
    }

    for(int i=0;i<this->N;i++){
        this->states[i] = this->avstates[i]/50;
        this->avstates[i] = 0;
        }

    //Final knockout to make deltas and weight updates correct
    if(knockout!=0){
        this->states[knockout+this->konode-1] = 0;
        }

    }


    void converge(int knockout){
        //do steps until the states are the same within 1/1000
        std::vector<Flt> copy;
        float total=1;
        copy.resize (this->N);
        int count = 0;

        while(total>0.01*0.01*this->N){
            count ++;
            //cout<<count<<endl;
            if(count>400){
                cout<<"n";
                this->weights=this->best;
                //put in a nudge.
                for(int i=0;i<this->N;i++){
                    for(int j=0;j<this->N;j++){
                        this->weights[i][j]=this->weights[i][j]+ ((Flt) rand() / (Flt) RAND_MAX *2 -1)*0.0001;
                        this->weights[i][j] *= this->weightexistence[this->N*i+j];
                    }
                }

                //cout<<"reset"<<endl;
                count = 0;
                break;
            }
            if(knockout!=0){
                this->states[knockout+this->konode-1] = 0;
                }
            //store a copy of the states
            copy=this->states;
            //update the states
            this->step(knockout);
            //find the sum of squared differences between the two
            total=0;
            for(int i=0;i<this->N;i++){
                total = total + (this->states[i]-copy[i])*(this->states[i]-copy[i]);
            }


        }
        //Final knockout to make deltas and weight updates correct
        if(knockout!=0){
            this->states[knockout+this->konode-1] = 0;
        }
        //cout<<"finalknock"<<endl;RNet<Flt>::printState();
        //cout<<"converged"<<endl;
    }

};


template <class Flt>
class RNetEvolve : public RNet<Flt>
{
public:

    void updateWeights(void){
        //destroy or create links by altering each element of matrix with probability p_link
        for(int i=0; i<this->N; i++){
            for(int j=0; j<this->N; j++){
                if ((Flt) rand() / (Flt) RAND_MAX< this-> p_link){
                    this->weightexistence[this->N*i+j] = 1 - this->weightexistence[this->N*i+j];
                    this->weights[i][j] = (float) rand() / (float) RAND_MAX *2 -1 * this->weightexistence[this->N*i+j]; //randomise it for a new connection

                }
            }
        }
        //for each weight, increment or decrement by the learnrate, with probability p.


        for(int i=0; i<this->N; i++){
            for(int j=0; j<this->N; j++){
                if ((Flt) rand() / (Flt) RAND_MAX< this-> p){
                    if ((Flt) rand() / (Flt) RAND_MAX<0.5){
                        this->weights[i][j] = this->weights[i][j] + this->learnrate;
                    }
                    else{
                        this->weights[i][j] = this->weights[i][j] - this->learnrate;
                    }
                }
                //zero for weights that don't exist
                this->weights[i][j] = this->weights[i][j] * this->weightexistence[this->N*i+j];
            }
        }
        //cout<<"weights"<<endl;RNet<Flt>::printweights();

    }

    void step (void) {
        //dot it and squash it

        //RNet<Flt>::printState();
        //initialise result matrix
        std::vector<Flt> result(this->N);
        Flt total;


        //generate dot product
        for(int i=0; i<this->N; i++){
            total = 0;
            for(int j=0;j<this->N;j++){
                total = total + this->states[j] * this->weights[j][i];
            }
            result[i] = total;
        }
        this->states = result;
        //cout<<"dotproduct"<<endl;RNet<Flt>::printState();
        //squash
        for(int i=0; i<this->N; i++){
            result[i] = 1/(1+exp(-result[i]));
        }
        result[0] = 0;
        result[1] = 0;
        result[2] = 1; //bias


        //add input
        for(int i=0; i<this->N; i++){
            result[i] = result[i] + this->inputs[i];
        }
        this->states = result;
        //cout<<"squashed"<<endl;RNet<Flt>::printState();

        // At end, update states with result:
        this->states = result;
        //this->printState();
    }


    int converge(void){
        //do steps until the states are the same within 1/1000
        std::vector<Flt> copy;
        float total=1;
        copy.resize (this->N);
        int count = 0;

        while(total>0.01*0.01*this->N){
            count ++;
            //cout<<count<<endl;
            if(count>100){
                this->weights=this->best;
                //cout<<"reset"<<endl;
                count = 0;
                return(1);
                break;
            }
            //store a copy of the states
            copy=this->states;
            //update the states
            this->step();
            //find the sum of squared differences between the two
            total=0;
            for(int i=0;i<this->N;i++){
                total = total + (this->states[i]-copy[i])*(this->states[i]-copy[i]);
            }

        }
        return(0);
    }
};
