# Student Starter File — Complete all TODOs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation

sns.set_theme(style="whitegrid")

DATAFILE = 'pk_sample.csv'

def load_params(path):
    # TODO load CSV containing Dose, Vd, k per patient
    df = pd.read_csv(path)
    return df

def concentration_curve(dose, vd, k, t):
    # TODO return C(t) = (Dose/Vd) * exp(-k*t)
    t = np.array(t)
    C_t = (dose/vd) * np.exp(-k*t)
    return C_t

def compute_metrics(df, t):
    # Create empty lists to store metrics
    cmax_list = []
    tmax_list = []
    auc_list = []
    
    # Compute metrics for each patient
    for i, row in df.iterrows():
        C = concentration_curve(row["Dose"], row["Vd"], row["k"], t)
        cmax = np.max(C)
        tmax = t[np.argmax(C)]
        auc = np.trapz(C, t)
        cmax_list.append(cmax)
        tmax_list.append(tmax)
        auc_list.append(auc)
    
    # add metrics to dataframe
    df["Cmax"] = cmax_list
    df["Tmax"] = tmax_list
    df["AUC"] = auc_list
    return df

def create_plots(df, t):
    # TODO concentration_curves.png
    # Transform data for seaborn lineplot
    conc_list = []
    for idx, row in df.iterrows():
        C = concentration_curve(row['Dose'], row['Vd'], row['k'], t)
        for i in range(len(t)):
            conc_list.append({
                'Time': t[i],
                'Concentration': C[i],
                'PatientID': row['PatientID']
            })
    conc_df = pd.DataFrame(conc_list)
    
    plt.figure(figsize=(8,4))
    sns.lineplot(data=conc_df, x="Time", y="Concentration", hue="PatientID")
    plt.title("Concentration-time curves")
    plt.xlabel("Time (h)")
    plt.ylabel("Concentration")
    plt.savefig("concentration_curves.png") 
    plt.close()
    
    # TODO cmax_hist.png
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
    plt.savefig("cmax_hist.png")
    plt.close()
    
    # TODO dose_vs_cmax.png
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x="Dose", y="Cmax", hue="PatientID")
    plt.title("Dose vs Cmax")
    plt.savefig("dose_vs_cmax.png") 
    plt.close()
    
    # TODO auc_boxplot.png
    plt.figure(figsize=(7,5))
    sns.boxplot(data=df, y="AUC")
    plt.title("AUC Boxplot")
    plt.savefig("auc_boxplot.png") 
    plt.close()

def make_animation(df, t):
    # TODO animate the concentration curves over time
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 5)) 
    ax.set_xlim(0, t.max())
    ax.set_ylim(0, df["Cmax"].max() + 5)
    
    # Create curves for each patient
    curves = []
    for i, row in df.iterrows():
        C = concentration_curve(row["Dose"], row["Vd"], row["k"], t)
        curves.append(C)
    
    lines = []
    for i, row in df.iterrows():
        line, = ax.plot([], [], label=f'Patient {row["PatientID"]}')
        lines.append(line)
    ax.legend()
    
    # Animation update function
    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(t[:frame], curves[i][:frame])
        ax.set_title(f'Time: {t[frame]:.2f} h')
        return lines
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(t), blit=True, repeat=False)
    anim.save("pk_animation.gif", writer='pillow', fps=10) 
    plt.close()

def export_summary(df, t):
    # TODO export pk_summary.csv and patient_auc.csv
    # Export summary with Cmax, Tmax, AUC
    summary_df = df[["PatientID", "Cmax", "Tmax", "AUC"]]
    summary_df.to_csv("pk_summary.csv", index=False)
    
    # Export concentration data over time
    conc_data = {}
    for idx, row in df.iterrows():
        C = concentration_curve(row['Dose'], row['Vd'], row['k'], t)
        conc_data[row['PatientID']] = C
    
    conc_df = pd.DataFrame(conc_data, index=t).T
    conc_df.to_csv("patient_auc.csv")

if __name__ == "__main__":
    df = load_params(DATAFILE)
    
    # Generate time vector
    t = np.linspace(0, 24, 241)  # every 0.1 hour
    
    df = compute_metrics(df, t)
    create_plots(df, t)
    make_animation(df, t)
    export_summary(df, t)
    
    print("Project 7 completed successfully!")
