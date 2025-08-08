"""Main code for simulating closed-loop MPC
    Created by Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2024-04-24
    Modified: 2025-08-08
"""

# Standard Library Imports
from typing import Union
from pathlib import Path
from datetime import datetime

# Imports from 3rd party libraries
import numpy as np
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt

# Repository Imports
from models.train_model import ModelTraining

print(Path().absolute())

def generate_report(
    model_report_obj: ModelTraining,
    output_folder: Union[Path, str],
    metadata: dict,
    xlim=None,
    ylim=True,
):
    """
    Save multiple plots to a PDF file, organized in a 4x4 matrix on each page.

    Parameters:
    - datas: List of data for each plot.
    - output_pdf: Name of the output PDF file.
    """
    output_path = Path(output_folder, f"{metadata['Training Data Study']}_report.pdf")
    figures_filepath = Path(output_folder, "report_figures")
    figures_filepath.mkdir(parents=True, exist_ok=True)
    logo_filepath = str(Path(Path().absolute(), "reports", "report_info", "GSK_logo_2022.png"))

    simulation_train_dict, train_dict = model_report_obj.get_model_data_dict(data_agg="train")
    simulation_test_dict, test_dict = model_report_obj.get_model_data_dict(data_agg="test")

    # tables to plot in report
    df_rmse = model_report_obj.get_rmse_table().round(2)
    df_r2 = model_report_obj.get_r2_table().round(2)
    df_corrcoef = model_report_obj.get_corrcoef_table().round(2)

    dict_keys_train = list(simulation_train_dict.keys())
    dict_keys_test = list(simulation_test_dict.keys())
    cols = 2
    rows = 4
    ppg = rows * cols # subplot dimensions
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")


    df_a_matrix = pd.DataFrame(
        data=model_report_obj.a_matrix, index=model_report_obj.states, columns=model_report_obj.states
    ).round(5).reset_index()

    # Plotting the table and removing all axes
    fig_table_a, ax_a = plt.subplots(
        figsize=(12,6)
    )  # set the size that you'd like (width, height)
    ax_a.axis("off")

    tbl_a = ax_a.table(
        cellText=df_a_matrix.values,
        colLabels=list(df_a_matrix.columns),
        cellLoc="center",
        loc="center",
    )

    # Create the table and scale it to fit the fig
    for (i, j), cell in tbl_a.get_celld().items():
        if i == 0:  # header cells
            cell.set_fontsize(16)
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#F25D18")
    
    tbl_a.auto_set_font_size(False)
    tbl_a.set_fontsize(16)
    tbl_a.scale(2,4)  # may need to adjust this for your data
    plt.savefig(rf"{str(figures_filepath)}\a_matrix_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig_table_a)


    df_b_matrix = pd.DataFrame(
        data=model_report_obj.b_matrix, index=model_report_obj.states, columns=model_report_obj.inputs
    ).round(5).reset_index()

    # Plotting the table and removing all axes
    fig_table_b, ax_b = plt.subplots(
        figsize=(12,6)
    )  # set the size that you'd like (width, height)
    ax_b.axis("off")

    tbl_b = ax_b.table(
        cellText=df_b_matrix.values,
        colLabels=list(df_b_matrix.columns),
        cellLoc="center",
        loc="center",
    )

    # Create the table and scale it to fit the fig
    for (i, j), cell in tbl_b.get_celld().items():
        if i == 0:  # header cells
            cell.set_fontsize(16)
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#F25D18")
    
    tbl_b.auto_set_font_size(False)
    tbl_b.set_fontsize(16)
    tbl_b.scale(2,4)  # may need to adjust this for your data
    plt.savefig(rf"{str(figures_filepath)}\b_matrix_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig_table_b)


    # Plotting the table and removing all axes
    fig_table, ax = plt.subplots(
        figsize=(12,6)
    )  # set the size that you'd like (width, height)
    ax.axis("off")

    scaler_dict = {}
    for key, value in metadata["scaler"].items():
        if isinstance(value,list) and len(value) == len(metadata["scaler"]["feature_names_in_"]):
            scaler_dict[key] = value
    scaler_df = pd.DataFrame.from_dict(scaler_dict).round(3)

    tbl = ax.table(
        cellText=scaler_df.values,
        colLabels=list(scaler_df.columns),
        cellLoc="center",
        loc="center",
    )

    # Create the table and scale it to fit the fig
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:  # header cells
            cell.set_fontsize(16)
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#F25D18")
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(16)
    tbl.scale(2,4)  # may need to adjust this for your data
    plt.savefig(rf"{str(figures_filepath)}\scaler_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig_table)


    # Plotting the table and removing all axes
    fig_table_rmse, ax_rmse = plt.subplots(
        figsize=(12,6)
    )  # set the size that you'd like (width, height)
    ax_rmse.axis("off")

    tbl_rmse = ax_rmse.table(
        cellText=df_rmse.values,
        colLabels=list(df_rmse.columns),
        cellLoc="center",
        loc="center",
    )

    # Create the table and scale it to fit the fig
    for (i, j), cell_rmse in tbl_rmse.get_celld().items():
        if i == 0:  # header cells
            cell_rmse.set_fontsize(16)
            cell_rmse.set_text_props(weight="bold", color="white")
            cell_rmse.set_facecolor("#F25D18")
    
    tbl_rmse.auto_set_font_size(False)
    tbl_rmse.set_fontsize(16)
    tbl_rmse.scale(2,4)  # may need to adjust this for your data
    plt.savefig(rf"{str(figures_filepath)}\rmse_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig_table_rmse)


    # Plotting the table and removing all axes
    fig_table_r2, ax_r2 = plt.subplots(
        figsize=(12,6)
    )  # set the size that you'd like (width, height)
    ax_r2.axis("off")

    tbl_r2 = ax_r2.table(
        cellText=df_r2.values,
        colLabels=list(df_r2.columns),
        cellLoc="center",
        loc="center",
    )

    # Create the table and scale it to fit the fig
    for (i, j), cell_r2 in tbl_r2.get_celld().items():
        if i == 0:  # header cells
            cell_r2.set_fontsize(16)
            cell_r2.set_text_props(weight="bold", color="white")
            cell_r2.set_facecolor("#F25D18")
    
    tbl_r2.auto_set_font_size(False)
    tbl_r2.set_fontsize(16)
    tbl_r2.scale(2,4)  # may need to adjust this for your data
    plt.savefig(rf"{str(figures_filepath)}\r2_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig_table_r2)


    # Plotting the table and removing all axes
    fig_table_corrcoef, ax_corrcoef = plt.subplots(
        figsize=(12,6)
    )  # set the size that you'd like (width, height)
    ax_corrcoef.axis("off")

    tbl_corrcoef = ax_corrcoef.table(
        cellText=df_corrcoef.values,
        colLabels=list(df_corrcoef.columns),
        cellLoc="center",
        loc="center",
    )

    # Create the table and scale it to fit the fig
    for (i, j), cell_corrcoef in tbl_corrcoef.get_celld().items():
        if i == 0:  # header cells
            cell_corrcoef.set_fontsize(16)
            cell_corrcoef.set_text_props(weight="bold", color="white")
            cell_corrcoef.set_facecolor("#F25D18")

    tbl_corrcoef.auto_set_font_size(False)
    tbl_corrcoef.set_fontsize(16)
    tbl_corrcoef.scale(2,4)  # may need to adjust this for your data
    plt.savefig(rf"{str(figures_filepath)}\corrcoef_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig_table_corrcoef)


    pdf = FPDF(format="A4")  # A4 (210 by 297 mm)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 8)
    pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
    pdf.image(logo_filepath, w=30, h=10, x=10, y=277)

    # Set the title of the document
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(15)
    pdf.write(5, "BDSD MPC Training Report")

    # Specify the reference number of document
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(15)
    try:
        pdf.write(7, f"IDBS Reference: {metadata['IDBS Number']}")
    except KeyError:
        pdf.write(7, "IDBS Reference: EXXXXX")

    # Write other data to the front cover
    pdf.set_font("helvetica", "", 16)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.set_text_color(r=0, g=0, b=0)
    pdf.ln(8)
    pdf.write(7, f"Report Generated for {metadata['Asset']}")
    pdf.ln(8)

    pdf.set_font("helvetica", "B", 16)
    pdf.write(7, "Model Training Dataset: ")
    pdf.set_font("helvetica", "", 16)
    pdf.write(7, f"{metadata['Training Data Study']}")

    pdf.ln(8)

    pdf.set_font("helvetica", "B", 16)
    pdf.write(7, "States in Model: ")
    pdf.set_font("helvetica", "", 16)
    pdf.write(7, f"{(', ').join(model_report_obj.states)}")

    pdf.ln(8)

    pdf.set_font("helvetica", "B", 16)
    pdf.write(7, "Inputs in Model: ")
    pdf.set_font("helvetica", "", 16)
    pdf.write(7, f"{(', ').join(model_report_obj.inputs)}")

    pdf.set_text_color(r=0, g=0, b=0)
    # pdf.ln(135)
    pdf.ln(8)
    pdf.set_font("helvetica", "B", 16)
    pdf.write(7, "Report Author: ")
    pdf.set_font("helvetica", "", 16)
    pdf.write(7, "Zach Hatzenbeller")
    # pdf.ln(8)
    # pdf.write(7, "GitHub Link: https://github.com/gsk-tech/state-space-model")
    pdf.ln(8)
    pdf.set_font("helvetica", "B", 16)
    pdf.write(7, "Report Date: ")
    pdf.set_font("helvetica", "", 16)
    pdf.write(7, f"{now}")
    
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(15)
    pdf.write(7, "Scaling Parameters")

    pdf.image(rf"{str(figures_filepath)}\scaler_table.png", w=160, h=100, x=25, y=120)

    pdf.add_page()
    pdf.set_font("helvetica", "B", 8)
    pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
    pdf.image(logo_filepath, w=30, h=10, x=10, y=277)

    # Set the title of the document
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(127)
    pdf.write(5, "Model Training Dataset")
    df_sim_concat = pd.concat(simulation_train_dict.values(), ignore_index=True)
    df_train_concat = pd.concat(train_dict.values(), ignore_index=True)
    for test_label in model_report_obj.states:
        pdf.add_page()
        pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
        pdf.image(logo_filepath, w=30, h=10, x=10, y=277)
        # Set the title of the document
        pdf.set_font("helvetica", "B", 24)
        pdf.set_text_color(r=242, g=93, b=24)
        pdf.ln(5)
        pdf.write(5, f"State: {test_label}")
        if df_sim_concat[test_label].max() > df_train_concat[test_label].max():
            max_value = df_sim_concat[test_label].max()
        else:
            max_value = df_train_concat[test_label].max()

        if df_sim_concat[test_label].min() < df_sim_concat[test_label].min():
            min_value = df_sim_concat[test_label].min()
        else:
            min_value = df_sim_concat[test_label].min()

        for i in range(0, len(simulation_train_dict.keys()), ppg):
            if i != 0:
                pdf.add_page()
                pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
                pdf.image(logo_filepath, w=30, h=10, x=10, y=277)
            fig, axs = plt.subplots(rows, cols, figsize=(8, 10), squeeze=False)
            fig.subplots_adjust(top=0.8)

            for count, ax_test in enumerate(axs.reshape(-1)):
                if count + i < len(simulation_train_dict.keys()):
                    key = dict_keys_train[count + i]
                    time = np.arange(0, len(simulation_train_dict[key][test_label]), 1)
                    ax_test.plot(
                        time,
                        simulation_train_dict[key][test_label],
                        "ro-",
                        label="Simulated Data",
                        markersize=3.5,
                    )
                    ax_test.plot(
                        time,
                        train_dict[key][test_label],
                        "bo-",
                        label="Experimental Data",
                        markersize=3.5,
                    )
                    ax_test.set_title(key, size="medium", weight="bold")
                    ax_test.grid()
                    if ylim:
                        if min_value > 200:
                            ax_test.set_ylim(
                                min_value - (min_value * 0.2),
                                max_value + (max_value * 0.2),
                            )
                        else:
                            ax_test.set_ylim(0, max_value + (max_value * 0.2))
                    if xlim is not None:
                        ax_test.set_xlim(-1.5, xlim)

            # If on the last page and there are fewer than 12 plots, remove extra subplots
            if len(simulation_train_dict.keys()) - i < ppg:
                for j in range(len(simulation_train_dict.keys()) - i, len(axs.flatten())):
                    axs.ravel()[j].remove()

            axs[rows - 1][cols - 1].legend()
            fig.supxlabel("Day", size="x-large", weight="bold")
            fig.supylabel(f"{test_label}", size="x-large", weight="bold")
            fig.tight_layout()

            plt.savefig(rf"{str(figures_filepath)}\training_{test_label}_{i}.png", dpi=200)
            plt.close(fig)
            pdf.image(
                rf"{str(figures_filepath)}\training_{test_label}_{i}.png",
                w=190,
                h=250,
                x=10,
                y=22,
            )
    
    pdf.add_page()
    pdf.set_font("helvetica", "B", 8)
    pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
    pdf.image(logo_filepath, w=30, h=10, x=10, y=277)

    # Set the title of the document
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(127)
    pdf.write(5, "Model Testing Dataset")
    df_sim_concat = pd.concat(simulation_test_dict.values(), ignore_index=True)
    df_test_concat = pd.concat(test_dict.values(), ignore_index=True)
    for test_label in model_report_obj.states:
        pdf.add_page()
        pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
        pdf.image(logo_filepath, w=30, h=10, x=10, y=277)
        # Set the title of the document
        pdf.set_font("helvetica", "B", 24)
        pdf.set_text_color(r=242, g=93, b=24)
        pdf.ln(5)
        pdf.write(5, f"State: {test_label}")
        if df_sim_concat[test_label].max() > df_test_concat[test_label].max():
            max_value = df_sim_concat[test_label].max()
        else:
            max_value = df_test_concat[test_label].max()

        if df_sim_concat[test_label].min() < df_sim_concat[test_label].min():
            min_value = df_sim_concat[test_label].min()
        else:
            min_value = df_sim_concat[test_label].min()

        for i in range(0, len(simulation_test_dict.keys()), ppg):
            if i != 0:
                pdf.add_page()
                pdf.image(logo_filepath, w=30, h=10, x=170, y=10)
                pdf.image(logo_filepath, w=30, h=10, x=10, y=277)
            fig, axs = plt.subplots(rows, cols, figsize=(8, 10), squeeze=False)
            fig.subplots_adjust(top=0.8)

            for count, ax_test in enumerate(axs.reshape(-1)):
                if count + i < len(simulation_test_dict.keys()):
                    key = dict_keys_test[count + i]
                    time = np.arange(0, len(simulation_test_dict[key][test_label]), 1)
                    ax_test.plot(
                        time,
                        simulation_test_dict[key][test_label],
                        "ro-",
                        label="Simulated Data",
                        markersize=3.5,
                    )
                    ax_test.plot(
                        time,
                        test_dict[key][test_label],
                        "bo-",
                        label="Experimental Data",
                        markersize=3.5,
                    )
                    ax_test.set_title(key, size="medium", weight="bold")
                    ax_test.grid()
                    if ylim:
                        if min_value > 200:
                            ax_test.set_ylim(
                                min_value - (min_value * 0.2),
                                max_value + (max_value * 0.2),
                            )
                        else:
                            ax_test.set_ylim(0, max_value + (max_value * 0.2))
                    if xlim is not None:
                        ax_test.set_xlim(-1.5, xlim)

            # If on the last page and there are fewer than 12 plots, remove extra subplots
            if len(simulation_test_dict.keys()) - i < ppg:
                for j in range(len(simulation_test_dict.keys()) - i, len(axs.flatten())):
                    axs.ravel()[j].remove()

            axs[rows - 1][cols - 1].legend()
            fig.supxlabel("Day", size="x-large", weight="bold")
            fig.supylabel(f"{test_label}", size="x-large", weight="bold")
            fig.tight_layout()

            plt.savefig(rf"{str(figures_filepath)}\testing_{test_label}_{i}.png", dpi=200)
            plt.close(fig)
            pdf.image(
                rf"{str(figures_filepath)}\testing_{test_label}_{i}.png",
                w=190,
                h=250,
                x=10,
                y=22,
            )
    
    pdf.add_page()
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(7)
    pdf.write(7, "A Matrix")
    pdf.image(rf"{str(figures_filepath)}\a_matrix_table.png", w=180, h=100, x=25, y=40)

    pdf.add_page()
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(7)
    pdf.write(7, "B Matrix")
    pdf.image(rf"{str(figures_filepath)}\b_matrix_table.png", w=180, h=100, x=25, y=40)
    
    pdf.add_page()
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(7)
    pdf.write(7, "Model RMSE Table")
    pdf.image(rf"{str(figures_filepath)}\rmse_table.png", w=160, h=160, x=25, y=40)

    pdf.add_page()
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(7)
    pdf.write(7, "Model R2 Table")
    pdf.image(rf"{str(figures_filepath)}\r2_table.png", w=160, h=160, x=25, y=40)

    pdf.add_page()
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(r=242, g=93, b=24)
    pdf.ln(7)
    pdf.write(7, "Model Correlation Coefficient Table")
    pdf.image(rf"{str(figures_filepath)}\corrcoef_table.png", w=160, h=160, x=25, y=40)


    pdf.output(str(output_path))
