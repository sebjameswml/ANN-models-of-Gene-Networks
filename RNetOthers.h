/*
 * Two more classes that derive from RNet. Split out from RNet.h by Seb.
 */

template <class Flt>
class RNetBin : public RNet<Flt>
{
protected:
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

#ifdef USE_CONVERGE
public:
    // do steps until the states are the same within 1/1000
    void converge() {
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
#endif // USE_CONVERGE
};


template <class Flt>
class RNetEvolve : public RNet<Flt>
{
public:
    void updateWeights() {
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
    }

protected:
    void step() {
        //dot it and squash it
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
            result[i] = 1/(1+exp(-result[i]));
        }
        result[0] = 0;
        result[1] = 0;
        result[2] = 1; //bias

        //add input and place in this->states
        for(int i=0; i<this->N; i++){
            this->states[i] = result[i] + this->inputs[i];
        }
    }

#ifdef USE_CONVERGE
public:
    //do steps until the states are the same within 1/1000
    int converge() {
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
                return 1;
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
        return 0;
    }
#endif // USE_CONVERGE
};
