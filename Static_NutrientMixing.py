## ------------------------------------------------------------------------ ##
## -- Copyright (c) 2024 Stefan Streif, stefan.streif@etit.tu-chemnitz.de - ##
""" 
Licensed under the EUPL-1.2-or-later

-----------------------------------------------------------------------------------------
The Work is a work in progress, which is continuously improved by numerous
Contributors. It is not a finished work and may therefore contain defects or
‘bugs’ inherent to this type of development.

For the above reason, the Work is provided under the Licence on an ‘as is’ basis
and without warranties of any kind concerning the Work, including without
limitation merchantability, fitness for a particular purpose, absence of defects
or errors, accuracy, non-infringement of intellectual property rights other than
copyright as stated in Article 6 of this Licence.

This disclaimer of warranty is an essential part of the Licence and a condition
for the grant of any rights to the Work.
-----------------------------------------------------------------------------------------

See https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12 for full license text. """
## ------------------------------------------------------------------------ ##

import numpy as np
import math
import pandas as pd
import casadi as cs
import matplotlib.pyplot as plt
import matplotlib as mpl
from tkinter import messagebox 

## ----------------------------------------------------------------------- ##
## --- extracting data from csv-files and saving these in dictionaries --- ##
## ----------------------------------------------------------------------- ##

Nut_frame                   = pd.read_xml('Input_Files/Nutrient_Concentrations.xml',parser='lxml')
Nut_frame.set_index('Nutrient',inplace=True)
Nut_list                    = Nut_frame.index.values
Number_of_Nutrients         = Nut_list.size
Nut_dict                    = Nut_frame.to_dict('dict')

WaterSource_frame           = pd.read_xml('Input_Files/Water_use.xml',parser='lxml')
WaterSource_frame.set_index('Water_Source',inplace=True)
j = 0
for i in range(0,len(WaterSource_frame.index)):
    if WaterSource_frame["Use_Source"][WaterSource_frame.index[j]] != 1:
        WaterSource_frame.drop(WaterSource_frame.index[j])
    else:
        j+=1
WaterSource_list            = WaterSource_frame.index.values
Number_of_WaterSources      = WaterSource_list.size
WaterSource_frame           = WaterSource_frame.assign(Indexes=range(0, Number_of_WaterSources))
WaterSource_dict            = WaterSource_frame.to_dict('dict')

Fertilizer_frame            = pd.read_xml('Input_Files/Fertilizers.xml',parser='lxml')
Fertilizer_frame.set_index('Fertilizers',inplace=True)
j = 0
for i in range(0,len(Fertilizer_frame.index)):
    if Fertilizer_frame["Use_Source"][Fertilizer_frame.index[j]] != 1:
        Fertilizer_frame    = Fertilizer_frame.drop(Fertilizer_frame.index[j])
    else:
        j+=1
Fertilizer_list             = Fertilizer_frame.index.values
Number_of_Fertilizers       = Fertilizer_list.size
Fertilizer_frame            = Fertilizer_frame.assign(Indexes=range(Number_of_WaterSources,Number_of_WaterSources+Number_of_Fertilizers))
Fertilizer_dict             = Fertilizer_frame.to_dict('dict')

Tank_frame                  = pd.read_xml('Input_Files/Mixing_Tank_Parameters.xml',parser='lxml')
Tank_dict                   = Tank_frame.to_dict('dict')

## ------------------------------------------- ##
## --- setting up the optimization problem --- ##
## ------------------------------------------- ##

# initialization of starting guess and max/min bounds of optimization variables
u_0   = np.zeros(Number_of_WaterSources+Number_of_Fertilizers)
u_max = np.zeros(Number_of_WaterSources+Number_of_Fertilizers)
u_min = np.zeros(Number_of_WaterSources+Number_of_Fertilizers)

# setting starting guess and max/min bounds of optimization variables
for ind_water in range(0, Number_of_WaterSources):
    if math.isnan(WaterSource_dict["Max_Value"][WaterSource_list[ind_water]]) == 0:
        u_max[WaterSource_dict["Indexes"][WaterSource_list[ind_water]]] = WaterSource_dict["Max_Value"][WaterSource_list[ind_water]]
    else:
        u_max[WaterSource_dict["Indexes"][WaterSource_list[ind_water]]] = np.inf
    if math.isnan(WaterSource_dict["Min_Value"][WaterSource_list[ind_water]]) == 0:
        u_min[WaterSource_dict["Indexes"][WaterSource_list[ind_water]]] = WaterSource_dict["Min_Value"][WaterSource_list[ind_water]]
    else:
        u_min[WaterSource_dict["Indexes"][WaterSource_list[ind_water]]] = 0

for ind_fert in range(Number_of_WaterSources, Number_of_WaterSources+Number_of_Fertilizers):      
    if math.isnan(Fertilizer_dict["Max_Value"][Fertilizer_list[ind_fert-Number_of_WaterSources]]) == 0:
        u_max[Fertilizer_dict["Indexes"][Fertilizer_list[ind_fert-Number_of_WaterSources]]] = Fertilizer_dict["Max_Value"][Fertilizer_list[ind_fert-Number_of_WaterSources]]
    else:
        u_max[Fertilizer_dict["Indexes"][Fertilizer_list[ind_fert-Number_of_WaterSources]]] = np.inf
    u_min[Fertilizer_dict["Indexes"][Fertilizer_list[ind_fert-Number_of_WaterSources]]] = 0

# initialization of optimization variables, objective function, error terms and constraints
u_opt=cs.SX.sym('u_opt',Number_of_WaterSources+Number_of_Fertilizers,1)
J_mix = 0
e_NutRef = 0
e_NutRef_appr = 0
e_NutMin = 0
e_Water = 0
e_Fert = 0
h_mix=[]
num_constraints=0
c_ref_sum= 0
c_0_sum= 0
delta_c_abs = []
num_ref = 0
num_min = 0
sum_fertil = 0
V_max_MT=0
V_min_MT=0
V_max_fallback = 1000 #liters
c_max_fallback = 0.2  #gram per liters

# symbolic calculation of V_post
V_0 = Tank_dict["Current_Value"][0]
V_post=V_0
for ind_water in range(0, Number_of_WaterSources):
    V_post = V_post + u_opt[WaterSource_dict["Indexes"][WaterSource_list[ind_water]]]

# bounds of constraints of V_post
if math.isnan(Tank_dict["Target_Value"][0]) == 0:
    V_max_MT=1.01 * Tank_dict["Target_Value"][0]
    V_min_MT=0.99 * Tank_dict["Target_Value"][0]
else:
    if math.isnan(Tank_dict["Max_Value"][0]) == 0:
        V_max_MT = Tank_dict["Max_Value"][0]
    elif math.isnan(Tank_dict["Max_Tank_Volume"][0]) == 0:
        V_max_MT = Tank_dict["Max_Tank_Volume"][0]
    else:
        V_max_MT = max(V_0+100,V_max_fallback) 
    if math.isnan(Tank_dict["Min_Value"][0]) == 0:
        V_min_MT = Tank_dict["Min_Value"][0]
    else:
        V_min_MT = V_0

if V_max_MT < V_0:
    V_max_MT = V_0
if V_min_MT < V_0:
    V_min_MT = V_0
    
h_mix.append(V_max_MT - V_post)
h_mix.append(V_post - V_min_MT)
num_constraints += 2  
V_post_appr=(V_max_MT+V_min_MT)/2 

for ind_water in range(0, Number_of_WaterSources):
    u_0[WaterSource_dict["Indexes"][WaterSource_list[ind_water]]] = (V_max_MT-V_0)/Number_of_WaterSources

# symbolically calculation m_post,nut for every nutrient 
# calculate error term e_N,ref
for ind_nut in range(0, Number_of_Nutrients):
    if math.isnan(Nut_dict["Target_Concentration"][Nut_list[ind_nut]]) == 0 and Nut_dict["Target_Concentration"][Nut_list[ind_nut]]!=0 and Nut_dict["Nutrient_Weight"][Nut_list[ind_nut]]!=0:
        c_ref_sum += Nut_dict["Target_Concentration"][Nut_list[ind_nut]]
        c_0_sum += Nut_dict["Start_Concentration"][Nut_list[ind_nut]]
        delta_c_abs.append(Nut_dict["Target_Concentration"][Nut_list[ind_nut]]-Nut_dict["Start_Concentration"][Nut_list[ind_nut]])
        num_ref +=1
    if math.isnan(Nut_dict["Target_Concentration"][Nut_list[ind_nut]]) == 0 and Nut_dict["Target_Concentration"][Nut_list[ind_nut]]==0:
        num_min+=1

for ind_nut in range(0, Number_of_Nutrients):
    
    N_post_water= V_0 * Nut_dict["Start_Concentration"][Nut_list[ind_nut]]    
    for ind_water in range(0, Number_of_WaterSources):
        if math.isnan(Nut_dict[WaterSource_list[ind_water]+"_Concentration"][Nut_list[ind_nut]]) == 0:
            N_post_water=N_post_water + u_opt[WaterSource_dict["Indexes"][WaterSource_list[ind_water]]] * Nut_dict[WaterSource_list[ind_water]+"_Concentration"][Nut_list[ind_nut]]
    
    N_post_fertil = 0  
    for ind_fert in range(0, Number_of_Fertilizers):
        if Nut_list[ind_nut] in Fertilizer_dict and math.isnan(Fertilizer_dict[Nut_list[ind_nut]][Fertilizer_list[ind_fert]]) == 0:
            N_post_fertil=N_post_fertil + u_opt[Fertilizer_dict["Indexes"][Fertilizer_list[ind_fert]]] * Fertilizer_dict[Nut_list[ind_nut]][Fertilizer_list[ind_fert]]/100

    sum_fertil += N_post_fertil    
    N_post_nut = N_post_water + N_post_fertil

    if math.isnan(Nut_dict["Target_Concentration"][Nut_list[ind_nut]]) == 0:
        N_ref_appr=Nut_dict["Target_Concentration"][Nut_list[ind_nut]] * (V_max_MT+V_min_MT)/2
        N_ref = Nut_dict["Target_Concentration"][Nut_list[ind_nut]] * V_post
        w_Nut=1
        if math.isnan(Nut_dict["Nutrient_Weight"][Nut_list[ind_nut]]) == 0:
            w_Nut=Nut_dict["Nutrient_Weight"][Nut_list[ind_nut]]
        
        if Nut_dict["Target_Concentration"][Nut_list[ind_nut]]!=0:
            e_NutRef_appr = e_NutRef_appr + w_Nut * (N_post_nut-N_ref)**2/(N_ref_appr**2)
            e_NutRef = e_NutRef + w_Nut * (N_post_nut-N_ref)**2/(N_ref**2)
        elif math.isnan(Nut_dict["Max_Concentration"][Nut_list[ind_nut]]) == 0:
            e_NutMin = e_NutMin + w_Nut * ((N_post_nut-0)/(V_post_appr*Nut_dict["Max_Concentration"][Nut_list[ind_nut]]))**2
        else:
            e_NutMin = e_NutMin + w_Nut * ((N_post_nut-0)/(V_post_appr*c_max_fallback))**2
            
    if math.isnan(Nut_dict["Max_Concentration"][Nut_list[ind_nut]]) == 0:
        h_mix.append(Nut_dict["Max_Concentration"][Nut_list[ind_nut]] * V_post - N_post_nut)
        num_constraints+=1
    if math.isnan(Nut_dict["Min_Concentration"][Nut_list[ind_nut]]) == 0:
        h_mix.append(N_post_nut - Nut_dict["Min_Concentration"][Nut_list[ind_nut]] * V_post)
        num_constraints+=1
        
# calculate error term e_water
for ind_water in range(0, Number_of_WaterSources):
    w_water = 1
    if math.isnan(WaterSource_dict["Source_Weight"][WaterSource_list[ind_water]]) == 0:
        w_water = WaterSource_dict["Source_Weight"][WaterSource_list[ind_water]]
    if math.isnan(WaterSource_dict["MaxTank_Value"][WaterSource_list[ind_water]])==0 and math.isnan(WaterSource_dict["Current_Value"][WaterSource_list[ind_water]])==0:
        V_curr = WaterSource_dict["Current_Value"][WaterSource_list[ind_water]]
        V_max = WaterSource_dict["MaxTank_Value"][WaterSource_list[ind_water]]
        if w_water > 0:
            w_water=w_water * (2*(V_max-V_curr)/V_max)
        if w_water < 0:
            w_water=w_water * (2 - 2*(V_max-V_curr)/V_max)
        
    e_Water = e_Water + w_water * u_opt[WaterSource_dict["Indexes"][WaterSource_list[ind_water]]]

# calculate error term e_fert
for ind_fert in range(0, Number_of_Fertilizers):
    w_fert = 1
    if math.isnan(Fertilizer_dict["Fertilizer_Weight"][Fertilizer_list[ind_fert]]) == 0:
        w_fert = Fertilizer_dict["Fertilizer_Weight"][Fertilizer_list[ind_fert]]
    e_Fert = e_Fert + w_fert * u_opt[Fertilizer_dict["Indexes"][Fertilizer_list[ind_fert]]]


# set the balancing weights
one_fourth_of_volume = (V_post_appr-V_0)/4

if num_ref == 0:
    w_bal_c_ref = 0
else:
    w_bal_c_ref = 10000/num_ref
if num_min==0:
    w_bal_c_min = 0
else:
    w_bal_c_min = 4*100/num_min

w_bal_fert  = 100/((V_post_appr*c_ref_sum-V_0*c_0_sum))
if max(u_max[0:Number_of_WaterSources])> 0 and V_max_MT > V_0:
    w_bal_water = 100/one_fourth_of_volume
else:
    w_bal_water = 0

#error message 
if (max(u_max[0:Number_of_WaterSources])==0 or V_max_MT==0) and max(delta_c_abs)==0:
    messagebox.showerror("Error", "Solution already at reference concentration and max volume. Use of program is not possible.\nClose the Message to end the program.")
    exit()

   
# -------------------- Approximated OBJECTIVE FUNCTION - equals sum of error terms ----------------------------- #
J_mix_appr = w_bal_c_ref * e_NutRef_appr + w_bal_c_min * e_NutMin + w_bal_water * e_Water + w_bal_fert * e_Fert  #
# -------------------------------------------------------------------------------------------------------------- #  

# -------------------------- OBJECTIVE FUNCTION - equals sum of error terms -------------------------- #
J_mix = w_bal_c_ref * e_NutRef + w_bal_c_min * e_NutMin + w_bal_water * e_Water + w_bal_fert * e_Fert  #
# -----------------------------------------------------------------------------------------------------#  

# lower and upper bounds of constraints
h_mix = cs.vertcat(*h_mix)
h_max = np.inf * np.ones(num_constraints)
h_min = np.zeros(num_constraints)

# initialization of the optimization problem
nlp_mix = {
    "f": J_mix,
    "x": u_opt,
    "g": h_mix
}
nlp_mix_appr = {
    "f": J_mix_appr,
    "x": u_opt,
    "g": h_mix
}

## --------------------------------------------- ##
## --- solve optimization problem with IPOPT --- ##
## --------------------------------------------- ##

# solver options
opts_sol = {'ipopt.print_level': 5, 'print_time': 0}

# solver initialization
solver_mix = cs.nlpsol("solver","ipopt",nlp_mix,opts_sol)
solver_mix_appr = cs.nlpsol("solver","ipopt",nlp_mix_appr,opts_sol)

# solver call
sol_mix_prob_appr = solver_mix_appr(x0=u_0, lbx=u_min, ubx=u_max, lbg=h_min, ubg=h_max)
u_0 = sol_mix_prob_appr["x"]
sol_mix_prob = solver_mix(x0=u_0, lbx=u_min, ubx=u_max, lbg=h_min, ubg=h_max)
stats=solver_mix.stats()
if stats["return_status"]!='Solve_Succeeded':
    messagebox.showerror("Error", "Solving not succesful. Stated problem might be infeasible.\nClose the Message to end the program.")
    exit()
# results for the optimal choice of optimization variables
u_sol = sol_mix_prob["x"]
u_sol = np.array(u_sol)
u_solved = u_sol.tolist()
for i in range(0,Number_of_WaterSources+Number_of_Fertilizers):
    u_solved[i]=u_solved[i][0]

# rounding the resulting optimization variables
for ind_wat in range(0, Number_of_WaterSources):
    if u_solved[ind_wat] < 0:
        u_solved[ind_wat] = 0
for ind_fert in range(Number_of_WaterSources, Number_of_WaterSources + Number_of_Fertilizers):
    if u_solved[ind_fert] < 0:
        u_solved[ind_fert] = 0       
for i in range(0, Number_of_WaterSources+Number_of_Fertilizers):
    if i < Number_of_WaterSources:
        u_solved[i]=round(u_solved[i],3)
    else:
        u_solved[i]=round(u_solved[i],3)

# calculate resulting nutrient concentrations in mixing tank after applying the 'optimal' mixing procedure
V_post_sol=Tank_dict["Current_Value"][0]
for ind_water in range(0, Number_of_WaterSources):
    V_post_sol = V_post_sol + u_solved[WaterSource_dict["Indexes"][WaterSource_list[ind_water]]]

N_post_sol = []
c_post_sol = []
c_post_sol_norm = []
c_max_rel = []
sum_fertil_sol = 0
for ind_nut in range(0, Number_of_Nutrients):
        N_post_water_sol=Tank_dict["Current_Value"][0] * Nut_dict["Start_Concentration"][Nut_list[ind_nut]]
        
        for ind_water in range(0, Number_of_WaterSources):
            if math.isnan(Nut_dict[WaterSource_list[ind_water]+"_Concentration"][Nut_list[ind_nut]]) == 0:
                N_post_water_sol=N_post_water_sol + u_solved[WaterSource_dict["Indexes"][WaterSource_list[ind_water]]] * Nut_dict[WaterSource_list[ind_water]+"_Concentration"][Nut_list[ind_nut]]
       
        N_post_fertil_sol = 0
        for ind_fert in range(Number_of_WaterSources, Number_of_WaterSources+Number_of_Fertilizers):
            if Nut_list[ind_nut] in Fertilizer_dict and math.isnan(Fertilizer_dict[Nut_list[ind_nut]][Fertilizer_list[ind_fert-Number_of_WaterSources]]) == 0:
                N_post_fertil_sol=N_post_fertil_sol + u_solved[Fertilizer_dict["Indexes"][Fertilizer_list[ind_fert-Number_of_WaterSources]]] * Fertilizer_dict[Nut_list[ind_nut]][Fertilizer_list[ind_fert-Number_of_WaterSources]]/100
        
        sum_fertil_sol += N_post_fertil_sol

        N_post_sol.append(N_post_water_sol + N_post_fertil_sol)
        c_post_sol.append(round((N_post_water_sol + N_post_fertil_sol)/V_post_sol,6))


## ---------------------------------------------- ##
## --- Evaluation of the optimization results --- ##
## ---------------------------------------------- ##

# print resulting concentrations and optimization variables in python terminal
print("\n")
print("****************************************************************************** \n Solution of Optimization Problem: \n \n")
print("Optimization Variables:\n")
print(u_solved)
print(" \n \nResulting nutrient Concentrations:\n")
print(c_post_sol)
print(" \n \nResulting Water Volume in Mixing Tank:\n")
print(sum(u_solved[0:Number_of_WaterSources])+V_0)
print("\n \nResulting use of fertilizers:\n")
print(sum(u_solved[Number_of_WaterSources:Number_of_WaterSources+Number_of_Fertilizers-1]))
print("\n \nNutrient Mass by fertilizers:\n")
print(sum_fertil_sol)
print("\n \nFor further information see the results of the optimization variables in the output-file <working-directory>\Output_Files\Output_Values.csv.\n")
print("****************************************************************************** \n")

# save optimization results in a csv file
WaterSource_list=WaterSource_list.tolist()
Fertilizer_list=Fertilizer_list.tolist()
NutrientSources = WaterSource_list+Fertilizer_list
Output_data = {'Variable':NutrientSources,
               'Value':u_solved
               }
Output_df = pd.DataFrame(Output_data)
Output_df.to_xml('Output_Files\Output_Values.xml',index=False)

# plot normalized nutrient concentrations as bar chart
num_bars = 0
pos = 0
x_pos = 0
x_pos_norm = []
x_pos_max = []
x_pos_ticks = []

mpl.rcParams['font.sans-serif'] = "Times New Roman"
mpl.rcParams['font.family'] = "sans-serif"

Nutrients = []
for ind_nut in range(0, Number_of_Nutrients):
   if math.isnan(Nut_dict["Target_Concentration"][Nut_list[ind_nut]]) == 0 or math.isnan(Nut_dict["Max_Concentration"][Nut_list[ind_nut]]) == 0:
        Nutrients.append(Nut_list[ind_nut]) 
        if math.isnan(Nut_dict["Target_Concentration"][Nut_list[ind_nut]]) == 0 and Nut_dict["Target_Concentration"][Nut_list[ind_nut]] != 0:
            c_post_sol_norm.append(round(100*abs(c_post_sol[ind_nut])/Nut_dict["Target_Concentration"][Nut_list[ind_nut]],1))
            x_pos_norm.append(pos)
            num_bars += 1
            if math.isnan(Nut_dict["Max_Concentration"][Nut_list[ind_nut]]) == 0:
                pos += 1
            else:
                pos += 1.25
        if math.isnan(Nut_dict["Max_Concentration"][Nut_list[ind_nut]]) == 0:
            c_max_rel.append(round(100*c_post_sol[ind_nut]/Nut_dict["Max_Concentration"][Nut_list[ind_nut]],1))  
            x_pos_max.append(pos)
            num_bars +=1
            pos+=1.25
        if math.isnan(Nut_dict["Max_Concentration"][Nut_list[ind_nut]]) == 0 and math.isnan(Nut_dict["Target_Concentration"][Nut_list[ind_nut]]) == 0 and Nut_dict["Target_Concentration"][Nut_list[ind_nut]] != 0:
            x_pos += 0.5
            x_pos_ticks.append(x_pos)
            x_pos +=1.75
        else:
            x_pos_ticks.append(x_pos)
            x_pos +=1.25
            
Nutrients = tuple(Nutrients)
c_max_rel = tuple(c_max_rel)
c_post_sol_norm = tuple(c_post_sol_norm)
c_post_sol_1 = tuple(c_post_sol_norm)
nut_concentrations = {
    "c_norm": c_post_sol_norm,
    "c_max_ref": c_max_rel
}

c_plot = np.arange(len(Nutrients))
width = 0.9
multiplier = 0
cm = 1/2.54
fig_width=num_bars*2
fig_height=fig_width*0.7

fig_norm, ax_norm = plt.subplots(figsize=(fig_width*cm, fig_height*cm))
for attribute, measurement in nut_concentrations.items():
    
    if attribute == "c_norm":
        offset = width * multiplier/2
        rects1 = ax_norm.bar(x_pos_norm, measurement, width, label=attribute,  color = 'cornflowerblue')
        ax_norm.bar_label(rects1, padding=-1, fontsize= 12,fontweight='bold',zorder=10)
        multiplier += 1
    else:
        offset = width * multiplier/2
        rects2 = ax_norm.bar(x_pos_max, measurement, width, label=attribute,  color = 'tomato')
        ax_norm.bar_label(rects2, padding=-1, fontsize= 12,fontweight='bold',zorder=10)
        multiplier += 1

ax_norm.set_ylabel('Normalized concentrations [%]',fontsize = 12)
ax_norm.set_xlabel('Nutrient',fontsize = 12)
ax_norm.set_title('Resulting concentrations after mixing the nutrient solution',fontsize = 12)
ax_norm.set_xticks(x_pos_ticks, Nutrients,fontsize = 12)
plt.yticks(fontsize = 12)
ax_norm.axhline(y = 100, color = 'k', linestyle = 'dashed',alpha=0.6,zorder=0) 
ax_norm.set_ylim(0, max(120,30+round(10*round(max(c_post_sol_norm)/10))))
ax_norm.grid(alpha=0.5, which='major')
ax_norm.set_axisbelow(True)
ax_norm.legend((rects1, rects2), ('c_post/c_ref', 'c_post/c_max'), loc='upper right',fontsize = 12, ncol=2)

# plot.show() interrupts program execution
# plot commented if program should immediatly end after calculating the optimization results
# uncomment to see bar chart
plt.show()
