#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <sstream>
#include <cstdio>
#include <string>
#include <stdexcept>
#include <morph/HdfData.h>
#include "RNet.h"

#include "bitmap_image.hpp"

int main(int argc, char** argv){

    if (argc < 2){
        cout << "args required" << endl;
        return 1;
    }

    int ID = atoi(argv[1]);
    float learnrate = 0.05;

    int N = atoi(argv[2]);

    int trials = 1000000; //5,000,000 should produce extremely accurate reproductions and take a couple hours

    int npixels = 1000;
    int sampling = 1000;
    int decay = 0;


    //Set random seed
    srand(ID+1);

    std::vector<int> inputs, outputs, weightexistence;

    weightexistence.resize(N*N);

    outputs = {2,3,4,5};
    inputs = {0,1};
    int bias = 6;
    int konode = 7; //the first knockout will be this node, the second is konode + 1 and so on.

    //This is the section where network structure is determined, in this case its a recurrent net.
    //weightexistence is a flattened NxN matrix with 0s where there is no connection and 1 where there is.

    for(int i=0; i<N*N; i++){
        weightexistence[i] = 1;
    }
    //no weights back to bias node
    for(int i=0; i<N; i++){
        weightexistence[i*N+bias] = 0;
    }
    //no self interaction
    for(int i=0; i<N; i++){
        weightexistence[i*N+i] = 0;
    }

    RNetKnock<float> rNet;
    rNet.N = N;
    rNet.konode = konode;
    rNet.bias = bias;

    for(int i=0;i<N*N;i++){
        rNet.weightexistence[i] = weightexistence[i];
    }


    rNet.randommatrix(rNet.weights);
    rNet.randommatrix(rNet.best);

    std::vector<bitmap_image*> images; images.reserve(5);
    bitmap_image image0("knockoutimgs/0.bmp");
    bitmap_image image1("knockoutimgs/1.bmp");
    bitmap_image image2("knockoutimgs/2.bmp");
    bitmap_image image3("knockoutimgs/3.bmp");
    bitmap_image image4("knockoutimgs/4.bmp");
    images[0] = &image0; images[1] = &image1; images[2] = &image2; images[3] = &image3; images[4] = &image4;

    int imageWidth = image0.width();
    int imageHeight = image0.height();
    for (int i = 1; i < 5; i++) {
        if (images[i]->width() != imageWidth || images[i]->height() != imageHeight) {
            throw std::runtime_error ("Check bitmaps");
        }
    }

    float rx;
    float ry;
    int rk;

    float x;
    float y;

    float besterror = 10000;
    float sum;
    int bestcolor = 0;

    //flag to say if a pixel has been picked which doesn't have an appropriate colour.
    int flag;

    std::vector<std::vector<int>> areacolours{{255,255,255},{255,0,0},{0,0,255},{0,255,0},{255,0,255}};
    std::vector<std::vector<float>> targetmappings{{0,0,0,0},{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};

    std::vector<float> errors;

    rNet.learnrate = learnrate;
    for(int trial = 0;trial<trials;trial++){
        //f(trial%(trials/100)==0){cout<<trial<<" "<<besterror<<endl;}

        //decrease learnrate over time if option selected in configs
        if(decay==1){rNet.learnrate = learnrate*exp(-trial*1/trials);}


        //Pick a random pixel and check its within the bounds of the ellipse
        rx = rand() % imageWidth;
        ry = rand() % imageHeight;
        x =  rx /  imageWidth;
        y =  ry /  imageHeight;

        //Pick a random knockout image
        rk = rand() % 4;

        //set the target based on the colour of the pixel

        //first get the colour
        rgb_t colour;
        rgb_t match;
        images[rk]->get_pixel(int(rx), int(ry), colour);

        flag = 0;

        for (int i=0; i<5; i++) {
            match = make_colour(areacolours[i][0],areacolours[i][1],areacolours[i][2]);
            if (colour==match) {
                for (int n=0; n<4; n++) {
                    rNet.target[outputs[0]+n] = targetmappings[i][n];
                }
                rNet.inputs[inputs[0]] = x;
                rNet.inputs[inputs[1]] = y;
                flag = 1;
            }
        }

        if (flag == 1) {
            rNet.randomiseStates();
            //make the bias term 1
            rNet.states[bias]=1;
            rNet.states[0] = x;
            rNet.states[1] = y;
            //the argument here tells the network which node to knockout
            rNet.average(rk); //settles the network

            rNet.updateWeights(); //implements backpropagation algorithm
        }


        //over sample of trials, pick npixels and measure the error over them
        if(trial%sampling==0){//
            int tally=0;
            float error=0;
            while(tally<npixels){

                while(true){
                    //Pick a random pixel and check its within the bounds of the ellipse
                    rx = rand() % imageWidth;
                    ry = rand() % imageHeight;
                    //equation of ellipse<=1
                    if (((rx-imageWidth/2)/(imageWidth/2))*((rx-imageWidth/2)/(imageWidth/2))+
                        ((ry-imageHeight/2)/(imageHeight/2))*((ry-imageHeight/2)/(imageHeight/2))<=1){
                        break;
                    }
                }

                x = (float) rx / (float) imageWidth;
                y = (float) ry / (float) imageHeight;

                rk = rand() % 4;

                //set the target based on the colour of the pixel

                //first get the colour
                rgb_t colour;
                rgb_t match;
                images[rk]->get_pixel(int(rx), int(ry), colour);


                flag = 0;


                for(int i=0;i<5;i++){
                    match = make_colour(areacolours[i][0],areacolours[i][1],areacolours[i][2]);
                    if(colour==match){
                        for(int n=0;n<4;n++){
                            rNet.target[outputs[0]+n] = targetmappings[i][n];
                        }
                        rNet.inputs[inputs[0]] = x;
                        rNet.inputs[inputs[1]] = y;
                        flag = 1;
                    }
                }

                if(flag == 1){

                    rNet.randomiseStates();
                    rNet.states[bias]=1;
                    rNet.states[0] = x;
                    rNet.states[1] = y;
                    rNet.average(rk);
                    error = error + rNet.error();
                    tally = tally + 1;
                }

            }

            errors.push_back(error);
            if(error<besterror){
                //cout<<error<<endl;
                besterror=error;
                rNet.best = rNet.weights;
                if(error<1){break;}
            }
        }






    }
    if(besterror<10000){
        std::string file = "results/" + std::to_string(ID) + ".h5";
        morph::HdfData d(file);

        d.add_contained_vals("/x",errors);
        d.add_val("/besterror",besterror);
        d.add_val("/learnrate",learnrate);
        d.add_val("/numberofnodes",rNet.N);

        //flatten the weights matrix into a vector
        std::vector<float> bestweights;

        for(int i=0;i<rNet.N;i++){
            for(int j=0;j<rNet.N;j++){
                bestweights.push_back(rNet.best[i][j]);
            }
        }

        d.add_contained_vals("/bestweights",bestweights);
        d.add_contained_vals("/structure",weightexistence);

        rNet.weights = rNet.best;
        //cout<<besterror<<endl;

        std::vector<float> outputStates;
        for (int rk=0;rk<4;rk++){
            //At the end generate an image from the set of weights
            bitmap_image generated(150,100);

            // set background to white
            generated.clear();

            rgb_t inputcolour;

            for(float x=0;x<imageWidth;x++){
                for(float y=0;y<imageHeight;y++){
                    rNet.inputs[inputs[0]] = x/imageWidth;
                    rNet.inputs[inputs[1]] = y/imageHeight;

                    rNet.randomiseStates();
                    rNet.states[bias]=1;
                    rNet.states[0] = x/imageWidth;
                    rNet.states[1] = y/imageHeight;
                    rNet.average(rk);
                    if(rk==0){
                        outputStates.push_back(x);
                        outputStates.push_back(y);
                        //log all the values for all nodes, one long vector of form [x0,y0,n0,n1...]
                        for(int n = 0;n<rNet.N; n++){
                            outputStates.push_back(rNet.states[n]);}
                    }



                    besterror = 20;
                    //find out what colour the pixel is by finding the least distance to the 5 options
                    for(int i =0;i<5;i++){
                        sum = 0;
                        for(int node=0;node<4;node++){
                            sum = sum + (rNet.states[outputs[0]+node]-targetmappings[i][node])*(rNet.states[outputs[0]+node]-targetmappings[i][node]);
                        }
                        if (sum < besterror){
                            besterror = sum;
                            bestcolor = i;
                        }
                    }
                    inputcolour = make_colour(areacolours[bestcolor][0],areacolours[bestcolor][1],areacolours[bestcolor][2]);
                    generated.set_pixel(x,y,inputcolour);

                }
            }

            std::string bmpfile = "results/"+std::to_string(ID)+"-"+std::to_string(rk)+ ".bmp";
            generated.save_image(bmpfile);


        }

        d.add_contained_vals("/outputstates",outputStates);
        //rNet.printweights();
    }
    return(0);
}
