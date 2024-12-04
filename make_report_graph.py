from cProfile import label
from functools import reduce
from numpy import dtype
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(palette="twilight_shifted")


def graph1():
    df_jeff = pd.DataFrame({
        "Ray": [True, False],
        "cpu_inf": [208.66, 784.42],
        "cpu_bp": [287.23, 1208.2],
        "gpu_inf": [223.53, 809.22],
        "gpu_bp": [447.05, 825.67]
    })

    df_jerry = pd.DataFrame({
        "Ray": [True, False],
        "cpu_inf": [371.0, 705.0],
        "cpu_bp": [725.0, 1309.5],
        "gpu_inf": [226.0, 550.0],
        "gpu_bp": [444.5, 574.5]
    })

    df_yibin = pd.DataFrame({
        "Ray": [True, False],
        "cpu_inf": [461.7, 873.41],
        "cpu_bp": [816.77, 1358.37],
        "gpu_inf": [282.9, 684.47],
        "gpu_bp": [383.44, 614.49]
    })

    df_yuanhao = pd.DataFrame({
        "Ray": [True, False],
        "cpu_inf": [847.0, 2539.0],
        "cpu_bp": [1250.0, 2728.0],
        "gpu_inf": [1011.0, 2006.0],
        "gpu_bp": [1720.0, 2184.0]
    })

    df_merged = reduce(
        lambda left, right: pd.merge(left, right, how='outer'),
        [df_jeff, df_jerry, df_yibin, df_yuanhao]
    )
    # calculate averages of the columns
    averages = df_merged.groupby("Ray").mean().reset_index()
    df_melted = pd.melt(averages, id_vars=[
                        "Ray"], var_name="Category", value_name="Energy")
    df_melted.rename(columns={"Energy": "Energy (Joules)",
                     "Category": "Task Type"}, inplace=True)
    sns.barplot(data=df_melted, x="Task Type",
                y="Energy (Joules)", hue="Ray", errorbar="sd")
    plt.title("Average Energy Consumption By Task Type")
    plt.savefig("./out/energy_consumption_by_task_type.png", dpi=300)
    # plt.show()
    plt.clf()


def graph2():
    df_jeff = pd.DataFrame({
        "Chip": ["M1 Pro", "M1 Pro"],
        "Ray": [True, False],
        "cpu_inf": [208.66, 784.42],
        "cpu_bp": [287.23, 1208.2],
        "gpu_inf": [223.53, 809.22],
        "gpu_bp": [447.05, 825.67]
    })

    df_jerry = pd.DataFrame({
        "Chip": ["M1", "M1"],
        "Ray": [True, False],
        "cpu_inf": [371.0, 705.0],
        "cpu_bp": [725.0, 1309.5],
        "gpu_inf": [226.0, 550.0],
        "gpu_bp": [444.5, 574.5]
    })

    df_yibin = pd.DataFrame({
        "Chip": ["M1 Pro", "M1 Pro"],
        "Ray": [True, False],
        "cpu_inf": [461.7, 873.41],
        "cpu_bp": [816.77, 1358.37],
        "gpu_inf": [282.9, 684.47],
        "gpu_bp": [383.44, 614.49]
    })

    df_yuanhao = pd.DataFrame({
        "Chip": ["Intel i9-9880H", "Intel i9-9880H"],
        "Ray": [True, False],
        "cpu_inf": [847.0, 2539.0],
        "cpu_bp": [1250.0, 2728.0],
        "gpu_inf": [1011.0, 2006.0],
        "gpu_bp": [1720.0, 2184.0]
    })

    df_merged = reduce(
        lambda left, right: pd.merge(left, right, how='outer'),
        [df_jeff, df_jerry, df_yibin, df_yuanhao]
    )
    # calculate averages of the columns
    averages = df_merged.groupby(["Chip", "Ray"]).mean().reset_index()
    df_melted = pd.melt(averages, id_vars=[
                        "Chip", "Ray"], var_name="Category", value_name="Energy")
    df_melted.rename(columns={"Energy": "Energy (Joules)",
                     "Category": "Task Type"}, inplace=True)
    sns.barplot(data=df_melted, x="Chip",
                y="Energy (Joules)", hue="Ray", errorbar="sd")
    plt.title("Average Energy Consumption By Chip")
    plt.savefig("./out/energy_consumption_by_chip.png", dpi=300)
    # plt.show()
    plt.clf()


def graph3():
    def to_float(df):
        for col in df.columns:
            if df[col].dtype == 'int64' and col != 'Repeat':
                df[col] = df[col].astype('float64')
        return df
    df_jeff = pd.DataFrame({
        'Repeat': [50, 100, 150, 200],
        'Ray (cpu_inf)': [75.5, 151.33, 164.59, 206.39],
        'Not Ray (cpu_inf)': [193.65, 387.58, 594.30, 790.73],
        'Ray (cpu_bp)': [96.10, 159.52, 230.98, 289.34],
        'Not Ray (cpu_bp)': [278.71, 528.90, 804.10, 1211.8]
    })

    df_jerry = pd.DataFrame({
        'Repeat': [50, 100, 150, 200],
        'Ray (cpu_inf)': [110.5, 197, 285, 371],
        'Not Ray (cpu_inf)': [168, 347.5, 530.5, 705],
        'Ray (cpu_bp)': [195, 366, 543, 725],
        'Not Ray (cpu_bp)': [317, 651, 974.5, 1309.5]
    })

    df_yibin = pd.DataFrame({
        'Repeat': [50, 100, 150, 200],
        'Ray (cpu_inf)': [87.33, 183.5, 268.84, 380.07],
        'Not Ray (cpu_inf)': [194.07, 451.70, 587.83, 968.66],
        'Ray (cpu_bp)': [191.57, 383.54, 595.05, 795.14],
        'Not Ray (cpu_bp)': [281.28, 637.72, 937.37, 1330.16]
    })

    df_haoyuan = pd.DataFrame({
        'Repeat': [50, 100, 150, 200],
        'Ray (cpu_inf)': [392, 511, 637, 847],
        'Not Ray (cpu_inf)': [480, 1137, 1778, 2539],
        'Ray (cpu_bp)': [519, 760, 1000, 1250],
        'Not Ray (cpu_bp)': [665, 1364, 2004, 2728]
    })

    df_jerry = to_float(df_jerry)
    df_haoyuan = to_float(df_haoyuan)

    df = reduce(
        lambda left, right: pd.merge(left, right, how='outer'),
        [df_jeff, df_jerry, df_yibin, df_haoyuan]
    )
    # Melt the DataFrame
    melted = pd.melt(df, id_vars=["Repeat"],
                     var_name="Ray_Metric", value_name="Value")

    # Extract "Ray" status and metric
    melted["Ray"] = melted["Ray_Metric"].apply(lambda x: "Not Ray" not in x)
    melted["Metric"] = melted["Ray_Metric"].apply(
        lambda x: x.split(" ")[-1].strip("()"))

    # Drop the original Ray_Metric column and reorder
    final_df = melted.drop(columns=["Ray_Metric"]).rename(
        columns={"Value": "Measurement"})
    final_df = final_df[["Repeat", "Ray", "Metric", "Measurement"]]
    print(final_df)
    sns.relplot(
        data=final_df, x="Repeat", y="Measurement",
        hue="Ray", style="Ray", kind="line", markers=True
    )
    plt.xlabel("Number of Tasks")
    plt.ylabel("Energy Consumption (Joules)")
    plt.title("Energy Consumption by Number of Tasks")
    plt.savefig("./out/energy_consumption_by_tasks.png",
                dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()


def graph4():
    def to_float(df):
        for col in df.columns:
            if df[col].dtype == 'int64' and col != 'Intensity':
                df[col] = df[col].astype('float64')
        return df
    df_jeff = pd.DataFrame(
        {
            'Intensity': [1, 2, 4, 8],
            'Ray (cpu_inf)': [183.33, 210.40, 334.76, 355.67],
            'Not Ray (cpu_inf)': [729.36, 822.36, 886.04, 1087.16],
            'Ray (cpu_bp)': [226.69, 293.14, 418.78, 760.18],
            'Not Ray (cpu_bp)': [875.73, 1073.46, 1293.16, 1861.23]
        }
    )

    df_jerry = pd.DataFrame(
        {
            'Intensity': [1, 2, 4, 8],
            'Ray (cpu_inf)': [266.5, 371, 542, 1000],
            'Not Ray (cpu_inf)': [638, 705, 1016.5, 1489],
            'Ray (cpu_bp)': [441, 725, 1284, 2493],
            'Not Ray (cpu_bp)': [855, 1309.5, 1995, 3279]
        }
    )
    df_jerry = to_float(df_jerry)

    df_yibin = pd.DataFrame(
        {
            'Intensity': [1, 2, 4, 8],
            'Ray (cpu_inf)': [261.21, 388.23, 591.38, 1019.63],
            'Not Ray (cpu_inf)': [800.62, 957.40, 1015.49, 1857.47],
            'Ray (cpu_bp)': [489.89, 781.13, 1282.59, 2594.08],
            'Not Ray (cpu_bp)': [884.34, 1361.35, 1829.60, 3390.82]
        }
    )

    df_yuanhao = pd.DataFrame(
        {
            'Intensity': [1, 2, 4, 8],
            'Ray (cpu_inf)': [660, 847, 958, 1372],
            'Not Ray (cpu_inf)': [2017, 2539, 2966, 4357],
            'Ray (cpu_bp)': [920, 1250, 1792, 2980],
            'Not Ray (cpu_bp)': [2229, 2728, 3630, 5704]
        }
    )
    df_yuanhao = to_float(df_yuanhao)

    df_merged = reduce(
        lambda left, right: pd.merge(left, right, how='outer'),
        [df_jeff, df_jerry, df_yibin, df_yuanhao]
    )

    melted = pd.melt(df_merged, id_vars=["Intensity"],
                     var_name="Ray_Metric", value_name="Value")
    melted["Ray"] = melted["Ray_Metric"].apply(lambda x: "Not Ray" not in x)
    melted["Metric"] = melted["Ray_Metric"].apply(
        lambda x: x.split(" ")[-1].strip("()"))

    # Drop the original Ray_Metric column and reorder
    final_df = melted.drop(columns=["Ray_Metric"]).rename(
        columns={"Value": "Measurement"})
    final_df = final_df[["Intensity", "Ray", "Metric", "Measurement"]]

    sns.relplot(
        data=final_df, x="Intensity", y="Measurement",
        hue="Ray", style="Ray", kind="line", markers=True
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Energy Consumption (Joules)")
    plt.title("Energy Consumption by Batch Size")
    plt.savefig("./out/energy_consumption_by_batch_size.png",
                dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()
    print(final_df)


def graph5():
    # https://ourworldindata.org/grapher/carbon-intensity-electricity?tab=chart&country=~USA
    co2_per_kwh = 369
    joules_to_kwh_factor = 2.78 * 10**(-7)
    joules_per_task_no_ray = 7
    joules_per_task_ray = 3
    kwh_per_task_no_ray = joules_per_task_no_ray * joules_to_kwh_factor
    kwh_per_task_ray = joules_per_task_ray * joules_to_kwh_factor
    number_of_tasks = list(range(10_000, 10_000_000, 10_000))
    kwh_for_tasks_no_ray = [
        kwh_per_task_no_ray * n for n in number_of_tasks
    ]
    kwh_for_tasks_ray = [
        kwh_per_task_ray * n for n in number_of_tasks
    ]
    co2_for_tasks_no_ray = [
        co2_per_kwh * kwh for kwh in kwh_for_tasks_no_ray
    ]
    co2_for_tasks_ray = [
        co2_per_kwh * kwh for kwh in kwh_for_tasks_ray
    ]
    df = pd.DataFrame({
        "Number of Tasks": number_of_tasks,
        "No Ray (CO2)": co2_for_tasks_no_ray,
        "Ray (CO2)": co2_for_tasks_ray
    })
    sns.lineplot(data=df, x="Number of Tasks",
                 y="No Ray (CO2)", label="No Ray")
    sns.lineplot(data=df, x="Number of Tasks", y="Ray (CO2)", label="Ray")
    plt.xlabel("Number of Tasks")
    plt.ylabel("CO2 Emissions (gCO2)")
    plt.legend()
    plt.title("CO2 Emissions by Number of Tasks (GPU Training, CO2 per KWH = 369)")
    plt.savefig("./out/co2_emissions_by_tasks.png",
                dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    # graph1()
    # graph2()
    # graph3()
    # graph4()
    graph5()
