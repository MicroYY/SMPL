'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]


About the Script:
=================
This script demonstrates a few basic functions to help users get started with using 
the SMPL model. The code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Save the resulting body as a mesh in .OBJ format


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python hello_smpl.py

'''
import sys
sys.path.append('C:/Users/jianh/Desktop/SMPL/SMPL_python_v.1.0.0.zip/smpl')
from smpl_webuser.serialization import load_model
import numpy as np
import chumpy as ch
from chumpy.ch import MatVecMult

## Load SMPL model (here we load the female model)
## Make sure path is correct
m = load_model( 'C:/Users/jianh/Desktop/SMPL/SMPL_python_v.1.0.0.zip/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl' )
#m = load_model( '../models/basicModel_f_lbs_10_207_0_v1.0.0.pkl' )


## Assign random pose and shape parameters
#m.pose[:] = np.random.rand(m.pose.size) * .2
m.betas[:] = np.random.rand(m.betas.size) * .03
print("pose:")
print(m.pose[:])
print("shape:")
print(m.betas[:])

# W = np.zeros(6890)
# for i in range(6890):
#     for j in range(24):
#         W[i] += m.weights.r[i][j]
J = np.zeros((6890,10))
shape = np.zeros(10)
fx = np.zeros(6890)
for i in range(1000):
    for N in range(6890):
        x = m.v_template.r[N][0] - m.r[N][0]
        y = m.v_template.r[N][1] - m.r[N][1]
        z = m.v_template.r[N][2] - m.r[N][2]
        for n in range(10):
            x += m.shapedirs.r[N][0][n] * shape[n]
            y += m.shapedirs.r[N][1][n] * shape[n]
            z += m.shapedirs.r[N][2][n] * shape[n]
        for n in range(10):
            J[N][n] = 2 * x * m.shapedirs.r[N][0][n] + 2 * y * m.shapedirs.r[N][1][n] + 2 * z * m.shapedirs.r[N][2][n]
        fx[N] = x * x + y * y + z * z
    H = np.dot(J.T, J)
    B = np.dot(-J.T, fx)
    delta = np.linalg.lstsq(H,B)
    loss = np.linalg.norm(delta[0],2)
    if loss<1e-9:
        break
    shape += delta[0]
    print('Iter: %i' % i)
    print("Given shape:")
    print(m.betas[:])
    print("Result:")
    print(shape[:])
    print('Loss: %f' %loss)
    print('')

        


# Bx = MatVecMult(m.J_regressor, m.v_template[:,0])
# By = MatVecMult(m.J_regressor, m.v_template[:,1])
# Bz = MatVecMult(m.J_regressor, m.v_template[:,2])
# B = ch.vstack((Bx, By, Bz)).T    
# J = np.reshape(m.J, (72,1))
# B = J - np.reshape(B,(72,1))
# A1 = m.J_regressor.dot(m.shapedirs[:,:,0])
# A1 = np.reshape(A1, (72,1))
# A2 = m.J_regressor.dot(m.shapedirs[:,:,1])
# A2 = np.reshape(A2, (72,1))
# A3 = m.J_regressor.dot(m.shapedirs[:,:,2])
# A3 = np.reshape(A3, (72,1))
# A4 = m.J_regressor.dot(m.shapedirs[:,:,3])
# A4 = np.reshape(A4, (72,1))
# A5 = m.J_regressor.dot(m.shapedirs[:,:,4])
# A5 = np.reshape(A5, (72,1))
# A6 = m.J_regressor.dot(m.shapedirs[:,:,5])
# A6 = np.reshape(A6, (72,1))
# A7 = m.J_regressor.dot(m.shapedirs[:,:,6])
# A7 = np.reshape(A7, (72,1))
# A8 = m.J_regressor.dot(m.shapedirs[:,:,7])
# A8 = np.reshape(A8, (72,1))
# A9 = m.J_regressor.dot(m.shapedirs[:,:,8])
# A9 = np.reshape(A9, (72,1))
# A10 = m.J_regressor.dot(m.shapedirs[:,:,9])
# A10 = np.reshape(A10, (72,1))
# C = np.hstack((A1,A2,A3,A4,A5,A6,A7,A8,A9,A10))
# beta = np.linalg.lstsq(C,B)

## Write to an .obj file
outmesh_path = './hello_smpl.obj'
with open( outmesh_path, 'w') as fp:
    for v in m.r:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

## Print message
#print '..Output mesh saved to: ', outmesh_path 
print('..Output mesh saved to: ' + outmesh_path)