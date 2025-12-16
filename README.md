
This is the corresponding code for the paper “3D stochastic reconstruction of microstructures in porous materials based on a multi-conditional denoising diffusion implicit model.”

Python implementation code for the paper titled,

Authors: Ting Zhang1,2, Boyu Zhang1, Lei Wang2,4, Yi Du3,*

1.College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

2.State Key Laboratory of Oil and Gas Reservoir Geology and Exploitation, Chengdu University of Technology, Chengdu 610059, China

3.College of Engineering, Shanghai Polytechnic University, Shanghai 201209, China

4.State Key Laboratory of Oil and Gas Reservoir Geology and Exploitation, Chengdu University of Technology, Chengdu 610059, China

(*corresponding author, E-mail: duyi0701@126.com. Tel.: 86 - 21- 50214252. Fax: 86 - 21- 50214252. )

Ting Zhang Email: tingzh@shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Boyu Zhang: y23208090@mail.shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Yi Du E-mail: duyi0701@126.com, Affiliation: College of Engineering, Shanghai Polytechnic University, Shanghai 201209, China

#MCDDIM
1.requirements
pytorch == 2.1.1
To run the code, an NVIDIA GeForce RTX4090 Super GPU video card with 24GB video memory is required.

Software development environment should be any Python integrated development environment used on an NVIDIA video card.

Programming language: Python 3.8.

2.How to use？

First, preprocess the image: Cut the porous media slice into 64 * 64 * 64 size pictures. Each training image consists of 64 pictures of size 64 * 64, stored in a separate folder. Then use use scripts/preparedata.py to convert the image into .npy format.

Secondly, set the network parameters such as batchsize, learning rate and storage location. The executable .py file of IWGAN-GP, the path is: IWGAN-GP/IWGAN-GP.py. After configuring the parameters and environment, you can run directly: python IWGAN-GP.py

Finally, in IWGAN-GP/savepoint/Test, find the loss images during the training process and the .npy format of the porous media three-dimensional structure images of different rounds. Use scripts/loadnpy to convert .npy to .txt format for later analysis and processing.
