#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <string>
#include <stdexcept>
#include <morph/HdfData.h>
#include <morph/Config.h>
#include "RNet.h"
#include "bitmap_image.hpp"

using std::cout;
using std::endl;

int main (int argc, char** argv)
{
    if (argc < 3) {
        cout << "2 args required" << endl;
        return 1;
    }

    // Parameters given on command line
    const int ID = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int N2 = N*N;
    // Set random seed
    srand(ID+1);

    // Parameters taken from morph::Config
    morph::Config conf("dans_genenet.json");
    if (!conf.ready) { throw std::runtime_error ("Failed to open dans_genenet.json for config"); }
    const float learnrate = conf.getFloat ("learnrate", 0.05f);
    const int trials = conf.getInt ("trials", 1000000); // 5,000,000 should produce extremely accurate reproductions and take 2.5 mins
    const int npixels = conf.getInt ("npixels", 1000);
    const int sampling = conf.getInt ("sampling", 1000);
    const bool decay = conf.getBool ("decay_learnrate", false);

    // Define which nodes are inputs, outputs, the bias, and the knockout node range
    std::vector<int> inputs = {0,1};
    std::vector<int> outputs = {2,3,4,5};
    const int bias = conf.getInt ("bias_node", 6);
    const int konode = conf.getInt ("knockout_node", 7); // the first knockout node

    // weightexistence is a flattened NxN matrix with 0s where there is no connection
    // and 1 where there is. Initialise with 1s
    std::vector<int> weightexistence (N2, 1);

    // no weights back to bias node
    for (int i=0; i<N; i++) { weightexistence[i*N+bias] = 0; }
    // no self interaction
    for (int i=0; i<N; i++) { weightexistence[i*N+i] = 0; }

    RNetKnock<float> rNet;
    rNet.N = N;
    rNet.konode = konode;
    rNet.bias = bias;
    rNet.weightexistence = weightexistence;
    rNet.learnrate = learnrate;
    // These two not necessary, but removal changes the results.
    rNet.randommatrix(rNet.weights);
    rNet.randommatrix(rNet.best);

    std::vector<bitmap_image*> images; images.reserve(5);
    bitmap_image image0("knockoutimgs/0.bmp"); // Wildtype
    bitmap_image image1("knockoutimgs/1.bmp"); // Manipulation - green enlarged
    bitmap_image image2("knockoutimgs/2.bmp"); // Manipulation - red reduced
    bitmap_image image3("knockoutimgs/3.bmp"); // Manipulation - Red enlarged, blue, green reduced
    bitmap_image image4("knockoutimgs/4.bmp"); // Manipulation - Cyan regions
    images[0] = &image0; images[1] = &image1; images[2] = &image2; images[3] = &image3; images[4] = &image4;

    const unsigned int imageWidth = image0.width();
    const unsigned int imageHeight = image0.height();
    for (int i = 1; i < 5; i++) {
        if (images[i]->width() != imageWidth || images[i]->height() != imageHeight) {
            throw std::runtime_error ("Check bitmaps");
        }
    }

    float rx = 0.0f;
    float ry = 0.0f;
    int rk = 0;
    float x = 0.0f;
    float y = 0.0f;
    float besterror = 10000;
    int bestcolor = 0;
    // flag to say if a pixel has been picked which has an appropriate colour.
    int flag = 0;

    std::vector<rgb_t> areacolours (5);
    areacolours[0] = make_colour (255,255,255); // White
    areacolours[1] = make_colour (255,  0,  0); // Red
    areacolours[2] = make_colour (  0,  0,255); // Blue
    areacolours[3] = make_colour (  0,255,  0); // Green
    areacolours[4] = make_colour (255,  0,255); // Magenta
    std::vector<std::vector<float>> targetmappings{ {0,0,0,0},   // White: No out nodes active
                                                    {1,0,0,0},   // Red: 1st node active
                                                    {0,1,0,0},   // Blue: 2nd node active
                                                    {0,0,1,0},   // Green: 3rd node active
                                                    {0,0,0,1} }; // Magenta: 4th node active
    std::vector<float> errors;

    cout << "Looping over " << trials << " trials\n";
    for (int trial = 0; trial<trials; trial++) {
        // decrease learnrate over time if option selected in configs
        if (decay == true) { rNet.learnrate = learnrate * std::exp(-trial*1/trials); }

        // Pick a random pixel and check its within the bounds of the ellipse
        rx = rand() % imageWidth;
        ry = rand() % imageHeight;
        x =  rx / imageWidth;
        y =  ry / imageHeight;

        // Pick a random knockout image
        rk = rand() % 4;

        // Set the target based on the colour of the pixel. First get the colour.
        rgb_t colour;
        images[rk]->get_pixel (int(rx), int(ry), colour);

        flag = 0; // colour match flag
        for (int i=0; i<5; i++) {
            if (colour == areacolours[i]) {
                for (int n=0; n<4; n++) {
                    rNet.target[outputs[0]+n] = targetmappings[i][n];
                }
                rNet.inputs[inputs[0]] = x;
                rNet.inputs[inputs[1]] = y;
                flag = 1;
            }
        }

        if (flag == 1) { // The colour matched for at least one area!
            rNet.randomiseStates();
            rNet.states[bias] = 1; // One node in the network is the 'bias' node and is set to 1
            rNet.states[0] = x;
            rNet.states[1] = y;
            rNet.average(rk); // settles the network. The argument here tells the network which node to knockout
            rNet.updateWeights(); // implements backpropagation algorithm
        }

        // over sample of trials (every sample-th trial), pick npixels and measure the error over them
        if (trial%sampling == 0) {
            int tally = 0; // A tally of random pixels drawn from the images
            float error = 0.0f;
            while (tally < npixels) {

                while (true) {
                    // Pick a random pixel and check its within the bounds of the ellipse
                    rx = rand() % imageWidth;
                    ry = rand() % imageHeight;
                    // equation of ellipse<=1
                    if (((rx-imageWidth/2)/(imageWidth/2))*((rx-imageWidth/2)/(imageWidth/2))+
                        ((ry-imageHeight/2)/(imageHeight/2))*((ry-imageHeight/2)/(imageHeight/2))<=1) {
                        break;
                    }
                } // Now have an in-bounds random pixel defined by rx, ry

                x = rx / (float) imageWidth;
                y = ry / (float) imageHeight;
                rk = rand() % 4; // Randomly select one of the training images to pick a pixel from

                // set the target based on the colour of the pixel
                rgb_t colour;
                images[rk]->get_pixel (int(rx), int(ry), colour);

                flag = 0;
                for (int i=0;i<5;i++) {
                    if (colour == areacolours[i]) {
                        for (int n=0; n<4; n++) {
                            rNet.target[outputs[0]+n] = targetmappings[i][n];
                        }
                        rNet.inputs[inputs[0]] = x;
                        rNet.inputs[inputs[1]] = y;
                        flag = 1;
                    }
                }

                if (flag == 1) { // colour match
                    rNet.randomiseStates();
                    rNet.states[bias]=1;
                    rNet.states[0] = x;
                    rNet.states[1] = y;
                    rNet.average(rk);
                    error = error + rNet.error();
                    tally = tally + 1;
                }
            }

            errors.push_back (error);
            if (error < besterror) {
                besterror = error;
                rNet.best = rNet.weights;
                if (error < 1) { break; }
            }
        }
    } // end of loop over trials

    if (besterror < 10000) { // Save results

        std::string file = "results/" + std::to_string(ID) + ".h5";
        morph::HdfData d(file);

        d.add_contained_vals ("/x", errors);
        d.add_val ("/besterror", besterror);
        d.add_val ("/learnrate", learnrate);
        d.add_val ("/numberofnodes", rNet.N);

        // flatten the weights matrix into a vector
        std::vector<float> bestweights;
        for (int i=0; i<rNet.N; i++) {
            for (int j=0; j<rNet.N; j++) {
                bestweights.push_back (rNet.best[i][j]);
            }
        }
        d.add_contained_vals ("/bestweights", bestweights);
        d.add_contained_vals ("/structure", weightexistence);

        rNet.weights = rNet.best;
        cout << "The best error value achieved for these trials was: " << besterror << endl;

        std::vector<float> outputStates;
        for (int rk=0; rk<4; rk++) {
            // At the end generate an image from the set of weights
            bitmap_image generated(150, 100, true);

            for (float x=0; x<imageWidth; x++) {
                for (float y=0; y<imageHeight; y++) {
                    rNet.inputs[inputs[0]] = x/imageWidth;
                    rNet.inputs[inputs[1]] = y/imageHeight;

                    rNet.randomiseStates(); // Start with random states
                    rNet.states[bias] = 1;  // Set the bias state to its usual value of 1
                    rNet.states[0] = x/imageWidth;  // Set the input
                    rNet.states[1] = y/imageHeight; // input
                    rNet.average(rk);       // Run for 50 steps and find the average values of the states
                    if (rk == 0) { // rk==0 is the wildtype, for which we save outputStates
                        outputStates.push_back(x);
                        outputStates.push_back(y);
                        // log all the values for all nodes *for each pixel*. One long vector of form [x0,y0,n0,n1...]
                        for (int n = 0; n<rNet.N; n++) { outputStates.push_back (rNet.states[n]); }
                    }

                    besterror = std::numeric_limits<float>::max();
                    // find out what colour the pixel is by finding the least distance to the 5 options
                    for (int c=0; c<5; c++) { // loop over 5 colours
                        float sum = 0.0f;
                        for (int node=0; node<4; node++) { // loop over 4 output nodes
                            int oi = outputs[0] + node; // current output index
                            sum += (rNet.states[oi] - targetmappings[c][node]) * (rNet.states[oi] - targetmappings[c][node]);
                        }
                        if (sum < besterror) {
                            besterror = sum;
                            bestcolor = c;
                        }
                    }
                    generated.set_pixel (x, y, areacolours[bestcolor]);
                }
            }

            std::string bmpfile = "results/" + std::to_string(ID) + "-" + std::to_string(rk) + ".bmp";
            generated.save_image (bmpfile);
        }
        d.add_contained_vals ("/outputstates", outputStates);
    }
    return 0;
}
