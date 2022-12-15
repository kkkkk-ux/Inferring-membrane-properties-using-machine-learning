
import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv
import sys

def eqr(r,r1,r2,r3,r4,z1,z2,z3,z4,kappa,sigma,p):
    '''Variational equation corresponding to r'''
    temp = (r**2*(-1.*r1**2*sigma*(r1**2+z1**2)**17.5+1.*sigma*(r1**2+z1**2)**18.5+r*(1.*r1**2*r2*sigma*(r1**2+z1**2)**16.5-1.*r2*sigma*(r1**2+z1**2)**17.5-1.*p*z1*(r1**2+z1**2)**18.+1.*r1*sigma*z1*(r1**2+z1**2)**16.5*z2))+kappa*(z1**2*(-0.5*r1**2*(r1**2+z1**2)**16.5-0.5*(r1**2+z1**2)**17.5)+r*z1*(-1.5*r1**2*r2*z1*(r1**2+z1**2)**15.5+0.5*r2*z1*(r1**2+z1**2)**16.5-1.5*r1*z1**2*(r1**2+z1**2)**15.5*z2+1.*r1*(r1**2+z1**2)**16.5*z2)+r**2*(1.5*r2**2*z1**2*(r1**2+z1**2)**15.5-10.*r1**3*r2*z1*(r1**2+z1**2)**14.5*z2+2.5*r1**4*(r1**2+z1**2)**14.5*z2**2-15.*z1**4*(r1**2+z1**2)**14.5*z2**2+18.*z1**2*(r1**2+z1**2)**15.5*z2**2-3.*(r1**2+z1**2)**16.5*z2**2+r1*z1*(2.*r3*z1*(r1**2+z1**2)**15.5-25.*r2*z1**2*(r1**2+z1**2)**14.5*z2+16.*r2*(r1**2+z1**2)**15.5*z2)+3.*z1**3*(r1**2+z1**2)**15.5*z3-3.*z1*(r1**2+z1**2)**16.5*z3+r1**2*(-7.5*r2**2*z1**2*(r1**2+z1**2)**14.5-5.*z1**2*(r1**2+z1**2)**14.5*z2**2+0.5*(r1**2+z1**2)**15.5*z2**2+1.*z1*(r1**2+z1**2)**15.5*z3))+r**3*(-17.5*r1**4*r2*(r1**2+z1**2)**13.5*z2**2+r1**2*r2*(17.5*r2**2*z1**2*(r1**2+z1**2)**13.5-35.*z1**2*(r1**2+z1**2)**13.5*z2**2+17.5*(r1**2+z1**2)**14.5*z2**2+5.*z1*(r1**2+z1**2)**14.5*z3)+r1**3*(-17.5*z1*(r1**2+z1**2)**13.5*z2**3+5.*(r1**2+z1**2)**14.5*z2*z3)+z1*(-2.5*r2**3*z1*(r1**2+z1**2)**14.5+1.*r4*z1*(r1**2+z1**2)**15.5-10.*r3*z1**2*(r1**2+z1**2)**14.5*z2+4.*r3*(r1**2+z1**2)**15.5*z2+r2*(35.*z1**3*(r1**2+z1**2)**13.5*z2**2-20.*z1*(r1**2+z1**2)**14.5*z2**2-5.*z1**2*(r1**2+z1**2)**14.5*z3+1.*(r1**2+z1**2)**15.5*z3))+r1*(-10.*r2*r3*z1**2*(r1**2+z1**2)**14.5-35.*z1**3*(r1**2+z1**2)**13.5*z2**3+20.*z1*(r1**2+z1**2)**14.5*z2**3+r2**2*(52.5*z1**3*(r1**2+z1**2)**13.5*z2-15.*z1*(r1**2+z1**2)**14.5*z2)+15.*z1**2*(r1**2+z1**2)**14.5*z2*z3-5.*(r1**2+z1**2)**15.5*z2*z3-1.*z1*(r1**2+z1**2)**15.5*z4))))/(r**2*(r1**2+z1**2)**18.)
    return temp
def eqz(r, r1, r2, r3, r4, z1, z2, z3, z4,kappa,sigma,p):
    '''Variational equation corresponding to z'''
    temp=(r**2*(-1.*r1*sigma*z1*(r1**2+z1**2)**17.5+r*(1.*r1*r2*sigma*z1*(r1**2+z1**2)**16.5+p*r1*(r1**2+z1**2)**18.+1.*sigma*z1**2*(r1**2+z1**2)**16.5*z2-1.*sigma*(r1**2+z1**2)**17.5*z2))+kappa*(-0.5*r1*z1**3*(r1**2+z1**2)**16.5+1.*r1*z1*(r1**2+z1**2)**17.5+r*(r1*r2*(-1.5*z1**3*(r1**2+z1**2)**15.5+1.*z1*(r1**2+z1**2)**16.5)+(-1.5*z1**4*(r1**2+z1**2)**15.5+2.5*z1**2*(r1**2+z1**2)**16.5-1.*(r1**2+z1**2)**17.5)*z2)+r**2*(r1*r2**2*z1*(17.5*z1**2*(r1**2+z1**2)**14.5-19.*(r1**2+z1**2)**15.5)+r3*(-3.*z1**3*(r1**2+z1**2)**15.5+3.*z1*(r1**2+z1**2)**16.5)-10.*r1**4*r2*(r1**2+z1**2)**14.5*z2+r2*(15.*z1**4*(r1**2+z1**2)**14.5-18.*z1**2*(r1**2+z1**2)**15.5+3.*(r1**2+z1**2)**16.5)*z2+r1**2*(-5.*r3*z1*(r1**2+z1**2)**15.5+r2*(20.*z1**2*(r1**2+z1**2)**14.5+1.*(r1**2+z1**2)**15.5)*z2)+r1**3*(25.*r2**2*z1*(r1**2+z1**2)**14.5-7.5*z1*(r1**2+z1**2)**14.5*z2**2+2.*(r1**2+z1**2)**15.5*z3))+r**3*(35.*r1**4*r2**2*(r1**2+z1**2)**13.5*z2+r2*z1*(5.*r3*z1**2*(r1**2+z1**2)**14.5-5.*r3*(r1**2+z1**2)**15.5-17.5*r2*z1**3*(r1**2+z1**2)**13.5*z2+17.5*r2*z1*(r1**2+z1**2)**14.5*z2)+r1**3*(-35.*r2**3*z1*(r1**2+z1**2)**13.5-5.*r3*(r1**2+z1**2)**14.5*z2+52.5*r2*z1*(r1**2+z1**2)**13.5*z2**2-10.*r2*(r1**2+z1**2)**14.5*z3)+r1*(-1.*r4*z1*(r1**2+z1**2)**15.5+r2**3*(-17.5*z1**3*(r1**2+z1**2)**13.5+20.*z1*(r1**2+z1**2)**14.5)+5.*r3*z1**2*(r1**2+z1**2)**14.5*z2+1.*r3*(r1**2+z1**2)**15.5*z2+r2*(-15.*z1*(r1**2+z1**2)**14.5*z2**2+4.*(r1**2+z1**2)**15.5*z3))+r1**2*(15.*r2*r3*z1*(r1**2+z1**2)**14.5+17.5*z1**2*(r1**2+z1**2)**13.5*z2**3-2.5*(r1**2+z1**2)**14.5*z2**3+r2**2*(-35.*z1**2*(r1**2+z1**2)**13.5*z2-20.*(r1**2+z1**2)**14.5*z2)-10.*z1*(r1**2+z1**2)**14.5*z2*z3+1.*(r1**2+z1**2)**15.5*z4))))/(r**2*(r1**2+z1**2)**18.)
    return temp

def transform(u,z0,r_1):
    ''' Calculation of coefficients of transformation equations under Hard Constraint '''
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(u)  # Trace network input for derivation
        output1 = tf.cast(model(u), dtype=tf.float64)

        r = tf.reshape(output1[:, 0], (-1, 1))
        z = tf.reshape(output1[:, 1], (-1, 1))

        r2 = tape1.gradient(tape1.gradient(r, u), u)  # r''
        z1 = tape1.gradient(z, u)  # z'

        # coefficients of r's transformation equation
        b1 = -r2[0] / 2
        a1 = -r2[-1] / 6 - b1 / 3
        d1 = 0.001 - r[0]
        c1 = r_1 - r[-1] - a1 - b1 - d1  # r_1 means r(u)|u=1, the boundary condition

        # coefficients of z's transformation equation
        a2 = z0 - z[0]
        b2 = 0 - z1[0]
        c2 = -3 * a2 - 2 * b2 - 3 * z[-1] + z1[-1]
        d2 = -a2 - b2 - c2 - z[-1]
    return a1,b1,c1,d1,a2,b2,c2,d2

def interp(r_ref,z_ref):
    '''linear interpolation of r at the same z'''
    b=tf.where(z_ref>z_end,1,0)  # z>z_end set to 1，else set to 0

    z_ref_filter=z_ref[:tf.reduce_sum(b) + 1]
    r_ref_filter=r_ref[:tf.reduce_sum(b) + 1]
    z_interp=tf.linspace(tf.reduce_max(z_ref_filter),z_end,50)
    d=tf.reshape(z_interp,(1,-1))-tf.reshape(z_ref,(-1,1))
    b=tf.where(d<0,1,0)  # After subtraction, set d<0 to 1, and set others to 0

    index=tf.reduce_sum(b,axis=0)
    r_i=tf.gather(r_ref_filter,indices=index[1:])
    r_i_sub_1=tf.gather(r_ref_filter,indices=index[1:]-1)
    z_i = tf.gather(z_ref_filter, indices=index[1:])
    z_i_sub_1 = tf.gather(z_ref_filter, indices=index[1:]-1)

    r_interp=((r_i-r_i_sub_1)*z_interp[1:]+z_i*r_i_sub_1-z_i_sub_1*r_i)/(z_i-z_i_sub_1)

    return tf.concat([tf.reshape(r_ref_filter[0],(-1,)),r_interp],axis=0)

def loss_function(u,solution,z0,r_1,kappa,sigma,p):
    a1, b1, c1, d1, a2, b2, c2, d2=transform(u,z0,r_1)  # get the coefficients of transformation equations
    '''calculate loss function'''
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(u)
        output = tf.cast(model(u), dtype=tf.float64)

        # add Hard Constraint
        r = tf.reshape(output[:, 0], (-1, 1))+a1*u**3+b1*u**2+c1*u+d1
        z = tf.reshape(output[:, 1], (-1, 1))+a2+b2*u+c2*u**2+d2*u**3

        r1 = tape.gradient(r,u)  # the fist order of r
        r2 = tape.gradient(r1,u)  # the second order of r
        r3 = tape.gradient(r2, u)
        r4 = tape.gradient(r3, u)

        z1 = tape.gradient(z, u)  # the fist order of z
        z2 = tape.gradient(z1, u)
        z3 = tape.gradient(z2, u)
        z4 = tape.gradient(z3, u)

    a = (r1 ** 2 + z1 ** 2) ** (0.5)
    a_FirstOrder =  (r1 * r2 + z1 * z2)/a  # the fist order of a

    eq1 = eqr(r,r1,r2,r3,r4,z1,z2,z3,z4,kappa,sigma,p)  # the variational equation of r, derived from Mathematica
    eq2 = eqz(r, r1, r2, r3, r4, z1, z2, z3, z4,kappa,sigma,p)  # the variational equation of z

    loss_equation=(tf.reduce_sum(eq1[1:-1]**2)+tf.reduce_sum(eq2[1:-1]**2))/tf.cast(
                    tf.shape(eq1[1:-1])[0],dtype=tf.float64)

    loss_constraint=tf.reduce_sum(a_FirstOrder[1:-1]**2)/tf.cast(
                    tf.shape(eq1[1:-1])[0],dtype=tf.float64)


    interp_network = interp(tf.reshape(r, (-1,)), tf.reshape(z, (-1,)))  # interpolation of ML's outputs z(r)
    loss_BVP = tf.reduce_sum((interp_network[1:] - solution[1:]) ** 2)/tf.reduce_sum(solution[1:] ** 2) # Ldata mentioned in article
    loss_BVP2=tf.reduce_sum(  ( (interp_network[1:] - solution[1:])/solution[1:] )** 2  ) # another kinds of Ldata just for test no for fitting


    loss = loss_equation+loss_constraint+loss_BVP  # total loss

    return loss, 1*loss_equation, 1*loss_constraint,1*loss_BVP,r,z,loss_BVP2


@tf.function
def train1(input_tf,solution,optimizer,z0,r_1,kappa,sigma,p):

    with tf.GradientTape(persistent=True) as tape:
        loss=loss_function(input_tf,solution,z0,r_1,kappa,sigma*1e-4,p*1e-5)
        trainable_params=model.variables
        trainable_params.append(r_1) # only set Rb to trainable
    grad=tape.gradient(loss[0], trainable_params)  # the Derivatives of trainable parameters to total loss function
    optimizer.apply_gradients(zip(grad, trainable_params))  # update trainable parameters
    return loss

@tf.function
def train2(input_tf,solution,optimizer,z0,r_1,kappa,sigma,p):

    with tf.GradientTape(persistent=True) as tape:
        loss=loss_function(input_tf,solution,z0,r_1,kappa,sigma*1e-4,p*1e-5)
        trainable_params=model.variables
        # set sigma and p to trainable
        trainable_params.append(sigma)
        trainable_params.append(p)
    grad=tape.gradient(loss[0], trainable_params)
    optimizer.apply_gradients(zip(grad, trainable_params))
    return loss


def multipoint_train(profile_params,arrange_hight,domainsize=15,epochs=4*100000):
    kappa = 1
    sigma = tf.Variable(0 , trainable=True, dtype=tf.float64)  # Initialize trainable parameter sigma
    p = tf.Variable(0, trainable=True, dtype=tf.float64)
    r_1 = tf.Variable(0, trainable=True, dtype=tf.float64)

    optimizer1 = keras.optimizers.Adam(learning_rate=1e-2)
    optimizer2 = keras.optimizers.Adam(learning_rate=1e-3)
    # the input of NN, convert to tensor data and float 64 to avoid overflow
    u_tf = tf.convert_to_tensor(np.linspace(0, 1, domainsize + 1).reshape(-1, 1),
                                dtype=tf.float64)


    for k in range(10):#8 groups, 10 shapes for each group
        if k * 8 + int(group) > 78:
            break
        i = arrange_hight[k * 8 + int(group)]

        sigma.assign(1.)
        p.assign(1.)
        #Rb started from 50nm to fit
        r_1.assign(50)
        z0=profile_params[2*i+1][0]

        global z_end
        # we use z0/3 as the minimum of Interpolation height when some shapes' heights is too low to let z0/3<4.
        z_end=np.min([z0/3,4])

        # Interpolate the experimental data
        interp_experiment = interp(tf.constant(profile_params[2*i], dtype=tf.float64),
                                   tf.constant(profile_params[2*i+1], dtype=tf.float64))

        # save the print content to specified file
        log = open("./save_16group_" + total_times + "/print_" + group + ".txt", mode="a", encoding="utf-8")
        print("Start trainning,z0:%f,domainsize:%f" % (z0,domainsize),file=log)

        for epoch in range(epochs + 1):

            if epoch<=(epochs/4) or (epoch>(epochs/2) and epoch<=(epochs*0.75)):  # every 10 000 epoches shift trainable parameters
                shift=1
            else:
                shift=2

            if shift==1:  # Rb is trainable
                loss, eq, a_1, loss_BVP, r, z,loss_BVP2 = train1(u_tf, solution=interp_experiment, optimizer=optimizer1,
                                                             z0=z0, r_1=r_1, kappa=kappa, sigma=sigma, p=p)
            else:  # sigma and p are trainable
                loss, eq, a_1, loss_BVP, r, z,loss_BVP2 = train2(u_tf, solution=interp_experiment, optimizer=optimizer2,
                                                          z0=z0, r_1=r_1, kappa=kappa, sigma=sigma, p=p)

            if epoch % (epochs/4e1) == 0:
                print("Cost after epoch %i: %e , sigma=%f,p=%f,rb=%f,loss_BVP=%f,loss_BVP2=%f" % (
                epoch, loss, sigma, p, r_1, loss_BVP,loss_BVP2),file=log)

        print("Trainning ended,loss_eq:%f,loss_a1:%f\n" % (eq, a_1),file=log)

        # write ML's outputs into csv
        f=open('./save_16group_'+total_times+'/Profile_u_all_'+group+'_net.csv', 'a',newline='')
        writer = csv.writer(f)
        writer.writerow(p.numpy().reshape(-1).tolist())
        writer.writerow(sigma.numpy().reshape(-1).tolist())
        writer.writerow(r_1.numpy().reshape(-1).tolist())
        writer.writerow([z0])
        writer.writerow(r.numpy().reshape(-1).tolist())
        writer.writerow(z.numpy().reshape(-1).tolist())
        f.close()


def profile_plot():
    '''read symmetried shapes '''
    f=open("./symmetried_shapes.csv", "r")
    profile = list(csv.reader(f))
    for i in range(len(profile)):
        profile[i] = list(map(float, list(filter(lambda x: x != '', profile[i]))))
    f.close()
    return profile

def DefineModel():
    '''build the structure of neuron network'''
    model = keras.models.Sequential([
        keras.layers.Dense(32, activation='sigmoid', input_dim=1),
        keras.layers.Dense(32, activation='sigmoid'),
        keras.layers.Dense(2, use_bias=True)
    ])
    return model


if __name__ == "__main__":
    total_times = sys.argv[1]  # Get the fitting times parameter from other script settings
    group= sys.argv[2]  # Get the group number from other script settings


    model=DefineModel()

    # the index of Symmetrize shapes from small to large height
    arrange_hight=[74,66,65,58,69,75,70,73,7,77,63,64,61,24,50,67,53,
                   68,76,71,60,62,52,1,3,23,20,54,2,59,21,4,14,15,55,51,
                   10,56,17,72,19,37,22,35,78,44,12,11,57,47,40,42,33,0,13,
                   18,43,41,32,6,9,27,26,45,8,36,38,46,30,25,39,5,16,48,34,28,29,49,31]

    multipoint_train(profile_params=profile_plot(),arrange_hight=arrange_hight)

