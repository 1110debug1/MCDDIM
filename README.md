This is the corresponding code for the paper “3D stochastic reconstruction of microstructures in porous materials based on a multi-conditional denoising diffusion implicit model.”

Python implementation code for the paper titled,

Authors: Ting Zhang a,b, Boyu Zhang a, Lei Wang b,c, Yi Du d,*

a. College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China 

b. State Key Laboratory of Oil and Gas Reservoir Geology and Exploitation, Chengdu University of Technology, Chengdu 610059, China

c. College of Energy, Chengdu University of Technology, Chengdu 610059, China.

d. School of Computer and Information Engineering, Institute for Artificial Intelligence, Shanghai Polytechnic University, Shanghai 201209, China

(*corresponding author, E-mail: duyi0701@126.com. Tel.: 86 - 21- 50214252. Fax: 86 - 21- 50214252. )

Ting Zhang Email: tingzh@shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Boyu Zhang: y23208090@mail.shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Lei Wang: wanglei@cdut.edu.cn, Affiliation:College of Energy, Chengdu University of Technology, Chengdu 610059, China.

Yi Du E-mail: duyi0701@126.com, Affiliation: College of Engineering, Shanghai Polytechnic University, Shanghai 201209, China


#  Usage
Python 3.8+pytorch  
Hardware requirements: RTX 4090.
Program language: Python/Pytorch.
Software required:Pycharm.

# Installation
   pip install -r 'requirement.txt'
   
# 1.Train
  # 1.1
  Please run 'MainCondition_new.py' and change the 'state' to 'train' and 'path' to the location of your dataset, you can adjust the hyper parameters as needed.  
 # 1.2 
 Preprocess the image: Cut the porous media slice into 64 * 64 * 64 size pictures. The image can be find at https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-374/digital_dataset/f56f99e7-8600-4d26-badf-541dd940a773.
  # 1.3
  Regarding the training parameters in 'MainCondition_new.py', 'epoch' represents the number of training iterations, 'batch_size' refers to the number of training batches, and 'T' represents the time step in the diffusion model equation, typically set to 1000. A value of 500 will result in lower resolution effects. 'channel' represents the number of channels to adjust based on hardware requirements. 'labels' corresponds to the porosity parameter, 'label1' represents the average pore diameter, 'label2' represents the standard deviation of pore diameter, 'label3' represents the average throat diameter, 'label4' represents the standard deviation of throat diameter,'label5' represents the average coordination number, 'label6' represents the standard deviation of coordination number.For the description of other parameters, please refer to the paper.
  
# 2.Test
   After the training process, if you want to try to generate porous media, you can run 'MainCondition_new.py' file , and change state to eval and a series of locations.The constructions will be saved in the 'npydata' folder.
  
# Acknowledgements
   Although we have proposed a relatively new approach, the initial idea and design were inspired by https://github.com/ermongroup/ddim, and the code structure was inspired by https://github.com/luoxinggyyy/MCDDPM.
   

# License
   meanderpy is licensed under the Apache License 2.0.
   



