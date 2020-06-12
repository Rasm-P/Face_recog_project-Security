import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
    

csv_path = os.path.join('facerec/csv_data/')

if(not(Path(f'{Path.cwd()}/facerec/csv_data/').is_dir())):
        Path.mkdir(Path(f'{Path.cwd()}/facerec/csv_data/'))

def csv_writer(linalg_norm, name, file_name):
    information = {'Date': [], 'Linalg norm': [], 'Name': []}
    df = pd.DataFrame(information, columns= ['Date', 'Linalg norm', 'Name'])

    if os.path.isfile(csv_path + file_name):
        df = open(csv_path + file_name)
        df = pd.read_csv(df, sep = ',')

    date = datetime.datetime.now()
    date = datetime.datetime.strftime(date, '%Y/%m/%d/%H/%M/%S')
    data = {'Date': date, 'Linalg norm': linalg_norm, 'Name': name}
    df = df.append(data, ignore_index = True) 

    df.to_csv(csv_path + file_name, index = False, header=True)


def plot_csv_data(name, file_name):
    df = open(csv_path + file_name)
    df = pd.read_csv(df, sep = ',')

    df1 = df[df['Name'] == name]
    df2 = df[df['Name'] != name]

    ax = plt.gca()
    ax.set_ylim([0,1])
    ax.scatter(df1.index, df1["Linalg norm"], s=3, label=name, color='blue', zorder=3)
    ax.scatter(df2.index, df2["Linalg norm"], s=3, label='False positive', color='red', zorder=3)
    plt.legend(loc=1)
    plt.title('Benchmark graph')
    plt.xlabel('Index')
    plt.ylabel('Percentage')
    plt.grid(zorder=0)
    plt.savefig(os.path.join('facerec/csv_data/' + file_name[:-4] + '.png'))
    plt.show()