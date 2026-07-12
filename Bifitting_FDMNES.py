
from pymoo.core.problem import ElementwiseProblem
import nlopt


import math
import numpy as np
from scipy.interpolate import interp1d


import os
import shutil
import time

setting_norm=32
setting_EFermi=7112#Fe
setting_Gamma_hole=1.25#Fe
setting_fpdb="Fe_prm.pdb"
setting_f_exp="Fephen3_using.exp"#using exp energy as interp


global NRUN
NRUN=0
global mu_pred,mu_energy
mu_pred=np.zeros(100)#100 dosent matter
mu_energy=np.zeros(100)#100 dosent matter
global exp
exp = np.loadtxt(setting_f_exp,comments="#")




dict_elem={"Fe":26,"C":6,"N":7,"O":8,"H":1}
b1=np.array([ -1.131,  -1.579,   0.287])
b2=np.array([1.361,  -1.349,  -0.410])
atomb=0.5*(b1+b2)/np.linalg.norm(0.5*(b1+b2))
c1=np.array([ -1.357,   1.312,   0.559])
c2=np.array([0.477,   0.053,   1.912])
atomc=0.5*(c1+c2)/np.linalg.norm(0.5*(c1+c2))
d1=np.array([1.190,   1.513,  -0.425])
d2=np.array([ -0.458,   0.156,  -1.918])
atomd=0.5*(d1+d2)/np.linalg.norm(0.5*(d1+d2))


with open(setting_fpdb, 'r') as file:
    lines = [line.strip() for line in file if line.strip()]
z_list=[ i.split()[2] for i in lines]
z_num_list=[int(dict_elem[i]) for i in z_list]
group_list=[ i.split()[3][0] for i in lines];group_list=np.array(group_list)
x_arr= [float(i.split()[5]) for i in lines]
y_arr= [float(i.split()[6]) for i in lines]
z_arr= [float(i.split()[7]) for i in lines]
xyz_mat0=np.vstack([x_arr,y_arr,z_arr]);xyz_mat0=xyz_mat0.T
xyz_mat=np.copy(xyz_mat0)

print("PAUSE")


def integral(x,y):
    my = (y[1:]+y[:-1])/2
    dx = x[1:]-x[:-1]
    return np.sum(my*dx)
def kernelCauchy(x, a, sigma): return sigma/2/math.pi/((x-a)**2+sigma**2/4)
def kernelGauss(x, a, sigma): return 1/sigma/math.sqrt(2*math.pi)*np.exp(-(x-a)**2/2/sigma**2)
def YvesWidth(e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
    ee = (e-Efermi)/Ecent
    ee[ee==0] = 1e-5
    return Gamma_hole + Gamma_max*(0.5+1/math.pi*np.arctan( math.pi/3*Gamma_max/Elarg*(ee-1/ee**2) ))
def smooth_fdmnes(e, xanes, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
    xanes = np.copy(xanes)
    lastValueInd = xanes.size - int(xanes.size*0.05)
    #lastValue = utils.integral(e[lastValueInd:], xanes[lastValueInd:])/(e[-1] - e[lastValueInd])
    lastValue = integral(e[lastValueInd:], xanes[lastValueInd:]) / (e[-1] - e[lastValueInd])
    E_interval = e[-1] - e[0]
    xanes[e<Efermi] = 0
    sigma = YvesWidth(e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    virtualStartEnergy = e[0]-E_interval; virtualEndEnergy = e[-1]+E_interval
    norms = 1.0/math.pi*( np.arctan((virtualEndEnergy-e)/sigma*2) - np.arctan((virtualStartEnergy-e)/sigma*2) )
    toAdd = 1.0/math.pi*( np.arctan((virtualEndEnergy-e)/sigma*2) - np.arctan((e[-1]-e)/sigma*2) ) * lastValue
    kern = kernelCauchy(e.reshape(-1,1), e.reshape(1,-1), sigma.reshape(-1,1))
    assert (kern.shape[0]==e.size) and (kern.shape[1]==e.size)
    de = (e[1:]-e[:-1]).reshape(1,-1);
    f = xanes.reshape(1,-1) * kern
    new_xanes = (0.5*np.sum((f[:,1:]+f[:,:-1])*de, axis=1).reshape(-1) + toAdd)/norms
    return e, new_xanes

def gjf(pos,z,fout="opt_res.gjf"):
    feffpar=[]
    with open("model.gjf") as f:
        feffpar.extend(f.readlines())
    atom_lines=[]
    natom=len(z)
    for i in range(natom):
        tmp="%d %.5f %.5f %.5f \n"%( z[i],pos[i,0],pos[i,1],pos[i,2]   )
        atom_lines.append(tmp)
    feffpar.extend(atom_lines)
    with open(fout,"w") as f:
        f.write( "".join(feffpar) )
    return feffpar



def res_r2(y,yp):
    aa = np.square(y-yp)
    bb = np.square(y)
    tmp=sum(aa)/sum(bb)
    return tmp
def mae_weight(y,yp):
    global weight_obj
    tmp=np.abs(y-yp)
    tmp=weight_obj*tmp
    res=np.mean(tmp)
    return res

def diff_exp_the_save(energy_exp,muexp,energy,mu):
    # fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    # fexp = interp1d(xexp,yexp,kind="cubic",fill_value='extrapolate')
    # energy_use=xexp#using exp energy grid
    # muexp=fexp(energy_use)
    fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    muthe=fthe(energy_exp)
    # res = mae_weight(muexp, muthe)
    # res=np.mean(np.abs(muexp - muthe))
    res = res_r2(muexp, muthe)
    # res=F.l1_loss( torch.Tensor(muexp),torch.Tensor(muthe) )#ERROR
    return res

def diff_exp_the(energy_exp,muexp,energy,mu):
    # fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    # fexp = interp1d(xexp,yexp,kind="cubic",fill_value='extrapolate')
    # energy_use=xexp#using exp energy grid
    # muexp=fexp(energy_use)
    fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    muthe=fthe(energy_exp)
    # res = mae_weight(muexp, muthe)
    res=np.mean(np.abs(muexp - muthe))
    # res = res_r2(muexp, muthe)
    # res=F.l1_loss( torch.Tensor(muexp),torch.Tensor(muthe) )#ERROR

    return res

def diff_exp_the_weight(energy_exp,muexp,energy,mu):
    fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    muthe=fthe(energy_exp)
    res=mae_weight(muexp,muthe)
    return res


def diff_exp_the_obj(energy_exp,muexp,energy,mu):
    res=diff_exp_the(energy_exp,muexp,energy,mu)
    return res
def diff_exp_the_obj2(energy_exp,muexp,energy,mu):
    res=diff_exp_the(energy_exp,muexp,energy,mu)
    return res

def diff_exp_the_pearson(energy_exp,muexp,energy,mu):
    fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    # fexp = interp1d(xexp,yexp,kind="cubic",fill_value='extrapolate')
    # energy_use=xexp#using exp energy grid
    muthe=fthe(energy_exp)
    # muexp=fexp(energy_use)
    res=-pearson(muexp,muthe)
    return res

def pearson(arr1, arr2):
    return np.corrcoef(arr1, arr2)[0][1]



def opt_sub_conv():
    norm = setting_norm
    opt = nlopt.opt(nlopt.GN_DIRECT,6)  # LN_COBYLA  LN_BOBYQA
    E0=setting_EFermi
    opt.set_lower_bounds( np.array( [norm*0.7,-5,E0-5,15-10,30-10,30-10] ) )  # [-float('inf'), 0]
    opt.set_upper_bounds( np.array( [norm*1.5, 5,E0+5,15+10,30+10,30+10] ) )
    # opt.set_lower_bounds( np.array( [norm*0.7,-5,7113,17,21,39] ) )  # [-float('inf'), 0]
    # opt.set_upper_bounds( np.array( [norm*1.5, 5,7113,17,21,39] ) )

    # opt.set_lower_bounds( np.array( [norm*1.01,1.73] ) )  # [-float('inf'), 0]
    # opt.set_upper_bounds( np.array( [norm*1.01,1.73] ) )

    opt.set_min_objective(obj_sub_conv)
    # opt.set_maxtime(300)#2 minuit
    opt.set_maxeval(1000)
    # x0=np.array([norm*1.0,0,7112,15,30,30])
    x0=np.array([norm*1.0,0,E0,15,30,30])
    
    x = opt.optimize(x0)
    minf = opt.last_optimum_value()
    # print("Sub optimization results",x)
    return x

def obj_sub_conv(x,grad=None):
    global mu_pred,mu_energy
    global exp


    mu=mu_pred.copy()
    mu_energy=mu_energy.copy()

    Gamma_hole = setting_Gamma_hole


    norm = x[0]
    es = x[1]
    Efermi = x[2]
    Gamma_max = x[3];
    Ecent = x[4]
    Elarg = x[5]
    ee, mu = smooth_fdmnes(mu_energy, mu, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    mu=mu*norm
    energy_es = mu_energy+es#energy_es = energy+es
    res=diff_exp_the(exp[:,0],exp[:,1],energy_es,mu)

    return res
def cal_conv(x,mu):
    global mu_energy
    Gamma_hole = setting_Gamma_hole
    norm = x[0]
    es = x[1]
    Efermi = x[2]
    Gamma_max = x[3];
    Ecent = x[4]
    Elarg = x[5]
    ee, mu = smooth_fdmnes(mu_energy, mu, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    mu=mu*norm
    energy_es = mu_energy+es#energy_es = energy+es

    return energy_es,mu

def change_structure(x):
    xyz_mat=np.copy(xyz_mat0)#deep copy
    xyzB=xyz_mat[group_list=='B'];xyz_mat[group_list=='B']=xyzB+np.repeat(x[0]*atomb.reshape([1,-1]),xyzB.shape[0],axis=0);
    xyzC=xyz_mat[group_list=='C'];xyz_mat[group_list=='C']=xyzC+np.repeat(x[1]*atomc.reshape([1,-1]),xyzC.shape[0],axis=0);
    xyzD=xyz_mat[group_list=='D'];xyz_mat[group_list=='D']=xyzD+np.repeat(x[2]*atomd.reshape([1,-1]),xyzD.shape[0],axis=0);
    
    # tmpname="_".join([str(i) for i in x])
    # with open(tmpname+".xyz","w") as f:
    #     [print(z_list[i],xyz_mat[i,0],xyz_mat[i,1],xyz_mat[i,2],file=f)  for i in range(xyz_mat.shape[0]) ]
    # refb=0.5*(xyz_mat[1]+xyz_mat[2])
    # refc=0.5*(xyz_mat[3]+xyz_mat[4])
    # refd=0.5*(xyz_mat[5]+xyz_mat[6])
    # with open(tmpname+".xyz","a") as f:
    #     print("Ru",refb[0],refb[1],refb[2],file=f)
    #     print("Ru",refc[0],refc[1],refc[2],file=f)
    #     print("Ru",refd[0],refd[1],refd[2],file=f)
    return xyz_mat
# change_structure([0,0,0])
# change_structure([1,2,3])
# change_structure([0.1,-0.2,-0.3])

def fdminp(pos,z,fout="fdm.inp",nameout="out"):
    feffpar=[]
    feffpar.extend("Filout\n")#output file name
    feffpar.extend(nameout+"\n")
    with open("fdmnesmodel.inp") as f:
        feffpar.extend(f.readlines())
    atom_lines=[]
    natom=len(z)
    for i in range(natom):
        tmp="%d %.5f %.5f %.5f \n"%( z[i],pos[i,0],pos[i,1],pos[i,2]   )
        atom_lines.append(tmp)
    feffpar.extend(atom_lines)
    feffpar.append("END\n")
    with open(fout,"w") as f:
        f.write( "".join(feffpar) )
    return feffpar
def cal_mu(xyz_mat,z_num_list,dirname="RUN_FDMNES"):
    global mu_pred,mu_energy
    finp=os.path.join(dirname,dirname+".fdmnes")
    fdminp(xyz_mat,z_num_list,finp,nameout=dirname)
    with open(os.path.join(dirname,"fdmfile.txt"),"w") as f:
        f.write("1"+"\n")
        f.write(dirname+".fdmnes"+"\n")
    os.chdir(dirname)
    os.system("fdmnes_win32.exe")
    
    try:
        fmu=dirname+".txt"
        dat=np.loadtxt(fmu,skiprows=2)
        mu_energy=dat[:,0];mu_pred=dat[:,1]

        x_conv=opt_sub_conv()#norm_opt=1.0;es_opt=0#
        energy_es,mu=cal_conv(x_conv,mu_pred)
        



        res=diff_exp_the_obj(exp[:,0],exp[:,1],energy_es,mu)#mae now
        res_save=diff_exp_the_save(exp[:,0],exp[:,1],energy_es,mu)#r2 now
    except:
        res=10000*10000
    

    os.chdir("..")#return to main dir
    np.savetxt("saveconv/"+dirname,np.vstack((energy_es,mu)).T)
    np.savetxt("savemu/"+dirname,np.vstack((mu_energy,mu_pred)).T)
    return res
def cal_energy(xyz_mat,z_list,dirname="RUN_MOPAC"):
    energy=10000*10000
    finp=os.path.join(dirname,dirname+".inp")
    with open(finp,"w") as f:
        f.write("PM6"+"\n")
        f.write("molecule"+"\n")
        f.write("All coordinates are Cartesian"+"\n")
        for i in range(xyz_mat.shape[0]):
            print(z_list[i],xyz_mat[i,0],0,xyz_mat[i,1],0,xyz_mat[i,2],0,file=f)
    os.chdir(dirname)
    os.system("mopac "+dirname+".inp")
    os.chdir("..")#return to main dir
    fout=os.path.join(dirname,dirname+".inp"+".out")
    
    with open(fout,"r") as f:
        line=f.readline()
        while line:#FINAL HEAT OF FORMATION =        648.73736 KCAL/MOL =    2714.31712 KJ/MOL
            line=line.strip()
            if line.find("FINAL HEAT OF FORMATION")>=0:
                energy=line.split("=")[1]
                energy=float( energy.split("KCAL")[0].strip() )
                break
            line=f.readline()


    # print("PAUSE")

    
    return energy








class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=np.array([-0.3,-0.3,-0.3]),
                         xu=np.array([0.3,0.3,0.3]))

    def _evaluate(self, x, out, *args, **kwargs):
        
        dirname="_".join( str(i) for i in x)
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)

        xyz_mat=change_structure(x)



        f1 = cal_mu(xyz_mat,z_num_list,dirname=dirname)
        f2 = cal_energy(xyz_mat,z_list,dirname=dirname)#cal_energy(xyz_mat,z_list,dirname=dirname)

        with open("log_mu_energy.txt","a") as f:
            print(f1,f2,x[0],x[1],x[2],file=f)
        if os.path.exists(dirname):# rm run dir after two calculations
            shutil.rmtree(dirname)
        

        # g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        # g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8
        # out["G"] = [g1]

        out["F"] = [f1, f2]



problem = MyProblem()

# problem._evaluate([0.1,0.2,0.3],[])

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

algorithm = NSGA2(
    pop_size=40,#40
    n_offsprings=10,#10
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)
from pymoo.termination import get_termination

termination = get_termination("n_gen", 10)#15+10*4=55

from pymoo.optimize import minimize

#clear log before opt
# with open("log_energy.txt","w") as f:
#     pass

t0=time.time()
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

X = res.X
F = res.F
np.savetxt("res_X.txt",X)
np.savetxt("res_F.txt",F)
print("Time using(s):",time.time()-t0)

import matplotlib.pyplot as plt
xl, xu = problem.bounds()
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
plt.xlim(xl[0], xu[0])
plt.ylim(xl[1], xu[1])
plt.title("Design Space")
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()

print("PAUSE")