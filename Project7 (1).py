# Student Starter File — Complete all TODOs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
DATAFILE = 'pk_sample.csv'
def load_params(path):
    df = pd.read_csv(path)
   
    return df
path=DATAFILE
# TODO load CSV containing Dose, Vd, k per patient
pass
def concentration_curve(dose, vd, k, t):
    df["C(t)"]=(
        (df["dose"])/df["vd"])*
    np.exp(-df["k"]*df["t"]
    )
    return df

# TODO return C(t) = (Dose/Vd) * exp(-k*t)
pass
def compute_metrics(df, t):

# TODO compute Cmax, Tmax, and AUC for each patient
pass
def create_plots(df):
# TODO concentration_curves.png ( title conctrtaion time curves and instead of site a b or c we have p1 p2 p3 for [atient id)
plt.figure(figsize=(8,4))
sns.lineplot(data=df, hue="PatientID")
plt.title("Concentration-time curves")
plt.xlabel("Time (h)")
plt.ylabel("Concentration")
plt.savefig(r"C:/Users/ubaidullah/Downloads/7Pharmacokinetics SimulatorTeamProjectQues/concentration_curves.png") 
plt.close()

# TODO cmax_hist.png (cmax is x and count is y) title is cmax distribution
plt.figure(figsize=(8,4))
sns.histplot(
        data=df,
        x="Cmax",
        kde=True,
        stat="count",
        bins=5,
        alpha=0.4
    )
plt.xlabel("Cmax")
plt.ylabel("Count")
plt.title("Cmax Distribution")
plt.savefig(r"C:/Users/ubaidullah/Downloads/7Pharmacokinetics SimulatorTeamProjectQues/concentration_curves.png")
plt.close()




# TODO dose_vs_cmax.png ( dose is x and cmax is y, patient id instead of site id)
plt.figure(figsize=(6,4))
sns.scatterplot(data=df,x="Dose",y="Cmax", hue="PatientID")
plt.title("\Dose vs Cmax")
plt.savefig(r"C:/Users/ubaidullah/Desktop/7Pharmacokinetics SimulatorTeamProjectQues/dose_vs_cmax.png") 
plt.close()
# TODO auc_boxplot.png
plt.figure(figsize(7,5))
sns.boxplot(data=df,x="PatientID",y="AUC")
plt.title("\AUC Box plot")
plt.savefig(r"C:/Users/ubaidullah/Desktop/7Pharmacokinetics SimulatorTeamProjectQues/auc_boxplot.png") 
plt.close()
create_plots(df)

pass


def make_animation(df, t):
# TODO animate the concentration curves over time
pass
def export_outputs(df):
# TODO export pk_summary.csv and patient_auc.csv'''